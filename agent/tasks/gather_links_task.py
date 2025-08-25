from __future__ import annotations

"""
GatherStructuredDataTask — adapted scaffold for this codebase.

Implements a resumable, multi-phase workflow:
- Reservoir (Tab A): store targets and per-item status
- Source (Tab B): visit and extract minimal structured fields
- Sink (Tab C): write normalized rows to Google Sheets using the existing controller actions

Notes
- This module conforms to this repo's BaseTask protocol: it exposes async run(),
  and step()/is_done()/succeeded() (no-op step, single-shot run).
- It uses pydantic models for state/context, but keeps runtime side effects lazy.
- Sheets operations are executed via controller actions already implemented under
  browser_use.controller (SelectCellOrRangeAction, UpdateCellContentsAction, etc.).
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field, HttpUrl, NonNegativeInt, PositiveInt, model_validator

from .base_task import BaseTask, TaskResult

logger = logging.getLogger(__name__)


# =========================
# Prompt scaffolding (lightweight)
# =========================

class PromptSpec(BaseModel):
    role: str
    template: str
    constraints: List[str] = Field(default_factory=list)

    def render(self, **context: Any) -> str:
        body = self.template.format(**context)
        if self.constraints:
            body += "\n\nConstraints:\n- " + "\n- ".join(self.constraints)
        return f"[ROLE: {self.role}]\n{body}"


NAVIGATOR_PROMPT = PromptSpec(
    role="navigator",
    template=(
        "You are navigating to {url}.\n"
        "Verify active tab '{tab_name}' is focused and location matches.\n"
        "If blockers appear (sign-in/2FA/consent/CAPTCHA), summarize blocker type and next step."
    ),
    constraints=[
        "Never proceed with extraction until dynamic content is settled.",
        "Confirm URL/title match before actions.",
        "If rate limited, back off and annotate.",
    ],
)

EXTRACTOR_PROMPT = PromptSpec(
    role="extractor",
    template=(
        "Extract required fields from DOM for {url}.\n"
        "Selectors: {selectors}\n"
        "Return a JSON object with keys: {keys}."
    ),
    constraints=[
        "If a field is missing, set it to null and continue.",
        "Prefer meta tags and canonical URLs for authority.",
        "Trim whitespace; do not include tracking query params.",
    ],
)

SHEETS_PROMPT = PromptSpec(
    role="scribe",
    template=(
        "Write normalized row to Google Sheets at range {a1} and tab '{tab_name}'.\n"
        "After 'Go to range', verify selection equals {a1}.\n"
        "Use clipboard paste; fallback to single-cell type for small values."
    ),
    constraints=[
        "If selection mismatch, retry up to 2 times before escalating.",
        "Ensure headers exist and match schema exactly.",
        "Do not overwrite existing rows for different stable_key.",
    ],
)


# =========================
# Data Models
# =========================

class ProgressStatus(str, Enum):
    queued = "queued"
    in_progress = "in_progress"
    done = "done"
    failed = "failed"
    skipped = "skipped"


class ReservoirEntry(BaseModel):
    url: HttpUrl
    title_hint: Optional[str] = None
    status: ProgressStatus = ProgressStatus.queued
    last_error: Optional[str] = None
    attempts: NonNegativeInt = 0
    stable_key: str = Field(..., description="Stable dedupe key, typically the URL.")

    @model_validator(mode="after")
    def _assign_stable_key(self):  # type: ignore[override]
        if not getattr(self, "stable_key", None):
            self.stable_key = str(self.url)
        return self


class SourceFields(BaseModel):
    stable_key: str
    url: HttpUrl
    name: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    blocker_detected: Optional[str] = None
    raw_meta: Dict[str, Any] = Field(default_factory=dict)


class SinkRow(BaseModel):
    stable_key: str
    url: HttpUrl
    name: str = ""
    description: str = ""
    author: str = ""
    published_at: str = ""
    tags: str = ""
    status: ProgressStatus = ProgressStatus.done
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def _normalize(self):  # type: ignore[override]
        self.name = (self.name or "").strip()
        self.description = (self.description or "").strip()
        self.author = (self.author or "").strip()
        self.published_at = (self.published_at or "").strip()
        if isinstance(self.tags, list):
            self.tags = ",".join(self.tags)
        else:
            self.tags = ",".join([t.strip() for t in (self.tags or "").split(",") if t.strip()])
        return self


class ReservoirTabState(BaseModel):
    tab_name: str = "Reservoir"
    entries: Dict[str, ReservoirEntry] = Field(default_factory=dict)
    last_index: NonNegativeInt = 0


class SourceTabState(BaseModel):
    tab_name: str = "Source"
    last_processed_key: Optional[str] = None
    last_url: Optional[HttpUrl] = None
    last_blocker: Optional[str] = None


class SinkTabState(BaseModel):
    tab_name: str = "Sink"
    sheet_url: Optional[HttpUrl] = None
    target_tab: str = "Tab C (Sink)"
    headers: List[str] = [
        "stable_key",
        "url",
        "name",
        "description",
        "author",
        "published_at",
        "tags",
        "status",
        "updated_at",
    ]
    last_written_row: NonNegativeInt = 0


class OrchestrationState(BaseModel):
    started_at: datetime = Field(default_factory=datetime.utcnow)
    deadline_at: Optional[datetime] = None
    max_attempts_per_item: PositiveInt = 3
    backoff_base_ms: PositiveInt = 600
    throttle_jitter_ms: PositiveInt = 300
    guard_min_remaining_ms: NonNegativeInt = 0


class TaskContext(BaseModel):
    reservoir: ReservoirTabState = Field(default_factory=ReservoirTabState)
    source: SourceTabState = Field(default_factory=SourceTabState)
    sink: SinkTabState = Field(default_factory=SinkTabState)
    orchestration: OrchestrationState = Field(default_factory=OrchestrationState)
    cache: Dict[str, Any] = Field(default_factory=dict)

    def get_remaining_ms(self) -> Optional[int]:
        if not self.orchestration.deadline_at:
            return None
        delta = self.orchestration.deadline_at - datetime.utcnow()
        return max(0, int(delta.total_seconds() * 1000))

    def guard_time(self) -> None:
        rem = self.get_remaining_ms()
        if rem is not None and rem < self.orchestration.guard_min_remaining_ms:
            raise TimeoutError(
                f"Time guard tripped: remaining_ms={rem} < guard_min_remaining_ms={self.orchestration.guard_min_remaining_ms}"
            )

    def persist_snapshot(self) -> str:
        return self.model_dump_json(exclude_none=True)

    @classmethod
    def load_snapshot(cls, blob: str) -> "TaskContext":
        # Accept either a JSON string or a pre-parsed dict-like object
        if isinstance(blob, str):
            return cls.model_validate_json(blob)
        return cls.model_validate(blob)

    def update_reservoir_status(self, key: str, status: ProgressStatus, error: Optional[str] = None) -> None:
        if key in self.reservoir.entries:
            ent = self.reservoir.entries[key].copy(update={"status": status, "last_error": error})
            self.reservoir.entries[key] = ent

    def upsert_reservoir(self, entries: Iterable[ReservoirEntry]) -> None:
        for e in entries:
            self.reservoir.entries[e.stable_key] = e

    def record_source_result(self, fields: SourceFields) -> None:
        self.source.last_processed_key = fields.stable_key
        self.source.last_url = fields.url
        self.cache[f"source:{fields.stable_key}"] = fields.model_dump(exclude_none=True)

    def get_source_result(self, key: str) -> Optional[SourceFields]:
        d = self.cache.get(f"source:{key}")
        if not d:
            return None
        return SourceFields.model_validate(d)

    def record_sink_row(self, row: SinkRow, row_index: Optional[int] = None) -> None:
        self.cache[f"sink:{row.stable_key}"] = {"row": row.model_dump(), "row_index": row_index}
        self.sink.last_written_row = max(self.sink.last_written_row, (row_index or 0))


# =========================
# Controller adapters (Sheets subset via existing controller actions)
# =========================

@dataclass
class _ControllerIO:
    controller: Any
    browser: Any

    async def multi(self, actions: list[Any]) -> None:
        # Use the standard registry ActionModel and multi_act pipeline
        ActionModel = self.controller.registry.create_action_model()
        await self.controller.multi_act(actions=[ActionModel(**a) for a in actions], browser=self.browser)


class SheetsAdapter:
    """Minimal Sheets controller built on top of action registry."""

    def __init__(self, io: _ControllerIO) -> None:
        self._io = io

    async def open_sheet(self, url: HttpUrl) -> None:
        from browser_use.controller.views import GoToUrlAction
        ActionModel = self._io.controller.registry.create_action_model()
        await self._io.controller.multi_act(
            actions=[ActionModel(go_to_url=GoToUrlAction(url=str(url), new_tab=False))],
            browser=self._io.browser,
        )

    async def go_to_range(self, a1: str) -> None:
        from browser_use.controller.views import SelectCellOrRangeAction
        ActionModel = self._io.controller.registry.create_action_model()
        await self._io.controller.multi_act(
            actions=[ActionModel(select_cell_or_range=SelectCellOrRangeAction(cell_or_range=a1))],
            browser=self._io.browser,
        )

    async def update_range_tsv(self, a1: str, tsv: str) -> None:
        from browser_use.controller.views import UpdateCellContentsAction
        ActionModel = self._io.controller.registry.create_action_model()
        await self._io.controller.multi_act(
            actions=[ActionModel(update_cell_contents=UpdateCellContentsAction(cell_or_range=a1, new_contents_tsv=tsv))],
            browser=self._io.browser,
        )

    async def ensure_headers(self, headers: List[str]) -> None:
        # A1:Ix row 1
        last_col = chr(ord("A") + max(0, len(headers) - 1))
        a1 = f"A1:{last_col}1"
        tsv = "\t".join(headers)
        await self.update_range_tsv(a1, tsv)

    async def set_row_values(self, row_idx: int, values: List[str]) -> None:
        last_col = chr(ord("A") + max(0, len(values) - 1))
        a1 = f"A{row_idx}:{last_col}{row_idx}"
        await self.update_range_tsv(a1, "\t".join(values))


# =========================
# Concrete phases (compact)
# =========================

class ReservoirTabTask:
    def __init__(self) -> None:
        self.prompts = {"navigator": NAVIGATOR_PROMPT}

    async def execute(self, ctx: TaskContext) -> TaskContext:
        ctx.guard_time()
        return ctx


class SourceExtractionTask:
    def __init__(self, *, selectors: Optional[Dict[str, str]] = None) -> None:
        self.selectors = selectors or {
            "title": "head title, h1, [data-testid='title']",
            "description": "meta[name='description']::attr(content), .description, article p",
            "author": "meta[name='author']::attr(content), [rel='author'], .byline",
            "published": "meta[property='article:published_time']::attr(content), time[datetime]::attr(datetime)",
            "tags": "meta[name='keywords']::attr(content), .tags a",
        }
        self.prompts = {"navigator": NAVIGATOR_PROMPT, "extractor": EXTRACTOR_PROMPT}

    async def execute(self, ctx: TaskContext) -> TaskContext:
        ctx.guard_time()
        # Deterministic placeholder extraction: copy URL + title hint only
        pending = [e for e in ctx.reservoir.entries.values() if e.status in (ProgressStatus.queued, ProgressStatus.in_progress)]
        for entry in pending:
            ctx.update_reservoir_status(entry.stable_key, ProgressStatus.in_progress)
            fields = SourceFields(
                stable_key=entry.stable_key,
                url=entry.url,
                name=entry.title_hint or "",
                description="",
                author="",
                published_at="",
                tags=[],
                blocker_detected=None,
                raw_meta={"selectors": self.selectors},
            )
            ctx.record_source_result(fields)
            ctx.update_reservoir_status(entry.stable_key, ProgressStatus.done)
        return ctx


class SinkSheetsWriteTask:
    def __init__(self, sheets: SheetsAdapter) -> None:
        self.sheets = sheets
        self.prompts = {"scribe": SHEETS_PROMPT}

    def _normalize(self, fields: SourceFields) -> SinkRow:
        return SinkRow(
            stable_key=fields.stable_key,
            url=fields.url,
            name=fields.name or "",
            description=fields.description or "",
            author=fields.author or "",
            published_at=fields.published_at or "",
            tags=",".join(fields.tags),
            status=ProgressStatus.done if not fields.blocker_detected else ProgressStatus.skipped,
        )

    async def execute(self, ctx: TaskContext) -> TaskContext:
        ctx.guard_time()
        if not ctx.sink.sheet_url:
            return ctx

        await self.sheets.open_sheet(ctx.sink.sheet_url)
        await self.sheets.ensure_headers(ctx.sink.headers)

        source_keys = [k.split("source:", 1)[1] for k in ctx.cache.keys() if k.startswith("source:")]
        for key in source_keys:
            fields = ctx.get_source_result(key)
            if not fields:
                continue
            row = self._normalize(fields)

            # Write to next available row (header offset + last)
            target_row = max(2, int(ctx.sink.last_written_row or 1) + 1)
            values = [
                row.stable_key,
                str(row.url),
                row.name,
                row.description,
                row.author,
                row.published_at,
                row.tags,
                row.status.value,
                row.updated_at.isoformat(),
            ]
            await self.sheets.set_row_values(target_row, values)
            ctx.record_sink_row(row, row_index=target_row)
        return ctx


# =========================
# Orchestrator task (BaseTask-compatible)
# =========================

class GatherStructuredDataTask(BaseTask):
    """Runs Reservoir -> Source -> Sink with basic pacing and guards.

    Constructor parameters are intentionally minimal for safe registration.
    The runner or caller should pre-seed the reservoir via `seed_reservoir` and
    set `ctx.sink.sheet_url` when Sheets writing is desired.
    """

    def __init__(self, controller: Any | None = None, browser: Any | None = None, *, deadline_seconds: Optional[int] = 600) -> None:
        self.controller = controller
        self.browser = browser
        self.ctx = TaskContext()
        if deadline_seconds:
            self.ctx.orchestration.deadline_at = datetime.utcnow() + timedelta(seconds=int(deadline_seconds))
        self._done: bool = False
        self._success: bool = False

        # Wire phases
        self._reservoir = ReservoirTabTask()
        io = _ControllerIO(controller=self.controller, browser=self.browser) if (self.controller and self.browser) else None
        self._sink = SinkSheetsWriteTask(SheetsAdapter(io)) if io else None  # type: ignore[arg-type]
        self._source = SourceExtractionTask()

    # BaseTask protocol
    def step(self) -> Optional[TaskResult]:
        return None

    def is_done(self) -> bool:
        return self._done

    def succeeded(self) -> bool:
        return self._success

    # Public API
    async def run(self) -> TaskResult:
        try:
            # Phase 1: reservoir (no-op placeholder)
            self.ctx = await self._reservoir.execute(self.ctx)

            # Phase 2: source extraction (deterministic placeholder)
            self.ctx = await self._source.execute(self.ctx)

            # Phase 3: sink write (only if controller/browser provided and sheet_url set)
            if self._sink is not None:
                self.ctx = await self._sink.execute(self.ctx)

            # Finalize export/manifest for audit
            manifest = {
                "reservoir": [e.model_dump() for e in self.ctx.reservoir.entries.values()],
                "source": {k: self.ctx.cache[k] for k in self.ctx.cache if k.startswith("source:")},
                "sink": {k: self.ctx.cache[k] for k in self.ctx.cache if k.startswith("sink:")},
                "orchestration": self.ctx.orchestration.model_dump(),
                "generated_at": datetime.utcnow().isoformat(),
            }
            self.ctx.cache["export:manifest.json"] = manifest

            self._done = True
            self._success = True
            return TaskResult(success=True, message="gather_structured_data_complete", data={"context": self.ctx.model_dump()})
        except Exception as e:  # noqa: BLE001
            logger.error("GatherStructuredDataTask failed: %s", e, exc_info=True)
            self._done = True
            self._success = False
            return TaskResult(success=False, message=str(e), data={"code": "gather_failed"})


# =========================
# Utilities (seeding)
# =========================

class ReservoirSeedInput(BaseModel):
    targets: List[HttpUrl]
    titles: Optional[List[str]] = None

    @model_validator(mode="after")
    def _validate_lengths(self):  # type: ignore[override]
        if self.titles and len(self.titles) != len(self.targets):
            raise ValueError("titles length must match targets length")
        return self


def seed_reservoir(ctx: TaskContext, data: ReservoirSeedInput) -> TaskContext:
    entries: List[ReservoirEntry] = []
    for i, url in enumerate(data.targets):
        title_hint = (data.titles or [None] * len(data.targets))[i]
        key = str(url)
        entries.append(
            ReservoirEntry(
                url=url,
                title_hint=title_hint,
                status=ProgressStatus.queued,
                last_error=None,
                attempts=0,
                stable_key=key,
            )
        )
    ctx.upsert_reservoir(entries)
    return ctx


__all__ = [
    # Core
    "GatherStructuredDataTask",
    "TaskContext",
    "run_gather_pipeline",
    # Models
    "ReservoirEntry",
    "SourceFields",
    "SinkRow",
    "ReservoirTabState",
    "SourceTabState",
    "SinkTabState",
    "OrchestrationState",
    "ProgressStatus",
    # Utils
    "ReservoirSeedInput",
    "seed_reservoir",
]

# Convenience runner mirroring the scaffold API
async def run_gather_pipeline(
    controller: Any,
    browser: Any,
    *,
    sheet_url: Optional[HttpUrl],
    targets: List[HttpUrl],
    titles: Optional[List[str]] = None,
    deadline_seconds: Optional[int] = 600,
) -> TaskContext:
    task = GatherStructuredDataTask(controller=controller, browser=browser, deadline_seconds=deadline_seconds)
    if sheet_url:
        task.ctx.sink.sheet_url = sheet_url
    seed = ReservoirSeedInput(targets=targets, titles=titles)
    task.ctx = seed_reservoir(task.ctx, seed)
    await task.run()
    return task.ctx
