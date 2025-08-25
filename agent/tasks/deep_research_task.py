from __future__ import annotations

"""
DeepResearchSynthesisTask — lightweight, production-safe scaffold.

Goals
- Provide a composable research/synthesis task that fits this repo's BaseTask protocol
- Zero side effects on import; lazy LLM usage inside run()
- Minimal but structured inputs/outputs using pydantic models

Notes
- This is a scaffold: it focuses on LLM-driven planning/synthesis; browsing/search can be
  integrated later by extending the class with controller/browser actions.

Contract
- Constructor: (controller, browser, llm)
- Methods: step() (not used here), is_done(), succeeded(), async run() -> TaskResult
"""

import json
import logging
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from .base_task import BaseTask, TaskResult
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import UserMessage, ContentPartTextParam

logger = logging.getLogger(__name__)


# --------- I/O models (kept intentionally small) ---------

class ResearchConstraints(BaseModel):
    recency_window_days: int = Field(365, ge=1)
    regions: List[str] = Field(default_factory=lambda: ["global"])  # str tags for simplicity
    sectors: List[str] = Field(default_factory=list)


class ResearchInput(BaseModel):
    research_question: str
    scope_in: List[str] = Field(default_factory=list)
    scope_out: List[str] = Field(default_factory=list)
    constraints: ResearchConstraints = Field(default_factory=ResearchConstraints)


class ExecutiveSummary(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)
    top_uncertainties: List[str] = Field(default_factory=list)


class ResearchOutput(BaseModel):
    executive_summary: ExecutiveSummary
    key_findings: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    final: bool = True


# --------- Task implementation ---------


class DeepResearchSynthesisTask(BaseTask):
    """LLM-driven research and synthesis scaffold.

    This minimal version relies on the provided LLM to expand queries and produce
    an executive summary. It returns a TaskResult with a structured payload.
    """

    def __init__(
        self,
        controller: Any | None = None,
        browser: Any | None = None,
        llm: BaseChatModel | None = None,
        research_input: ResearchInput | None = None,
    ) -> None:
        self.controller = controller
        self.browser = browser
        self.llm = llm
        self.input = research_input or ResearchInput(research_question="What is the current state of the art?")
        self._done = False
        self._success = False
        self._last_output: ResearchOutput | None = None

    # BaseTask contract
    def step(self) -> Optional[TaskResult]:
        return None  # This task runs in one async shot via run()

    def is_done(self) -> bool:
        return self._done

    def succeeded(self) -> bool:
        return self._success

    # Public entry point
    async def run(self) -> TaskResult:
        if not self.llm:
            # Without an LLM, produce a deterministic failure result (diagnostic-friendly)
            self._done = True
            self._success = False
            return TaskResult(success=False, message="No LLM provided", data={"code": "missing_llm"})

        # 1) Query planning (concise prompt for scaffold)
        plan_prompt = (
            "You are a research planner.\n"
            f"Question: {self.input.research_question}\n"
            f"Scope(in): {'; '.join(self.input.scope_in) or 'n/a'}\n"
            f"Scope(out): {'; '.join(self.input.scope_out) or 'n/a'}\n"
            f"Constraints: recency={self.input.constraints.recency_window_days}d, "
            f"regions={', '.join(self.input.constraints.regions)}, sectors={', '.join(self.input.constraints.sectors) or 'n/a'}\n"
            "Return JSON with keys: broad (<=4), narrow (<=4), adversarial (<=3)."
        )
        msg1 = UserMessage(content=[ContentPartTextParam(text=plan_prompt)])
        try:
            plan_resp = await self.llm.ainvoke([msg1])
            plan = json.loads(plan_resp.completion if isinstance(plan_resp.completion, str) else "{}")
        except Exception as e:
            logger.debug("DeepResearch: plan JSON parse failed: %s", e, exc_info=True)
            plan = {"broad": [], "narrow": [], "adversarial": []}

        # 2) Synthesis (LLM-based). Keep contract minimal: text, confidence, uncertainties
        coverage = {
            "queries": {k: len(v) for k, v in plan.items() if isinstance(v, list)},
            "recency_window_days": self.input.constraints.recency_window_days,
        }
        syn_prompt = (
            "You are a senior analyst. Produce a concise executive summary (<=250 words) "
            "with a confidence score (0-1) and top 3 uncertainties.\n"
            f"Question: {self.input.research_question}\n"
            f"Planned queries: {json.dumps(plan)}\n"
            f"Coverage hints: {json.dumps(coverage)}\n"
            "Return JSON with keys: text, confidence, uncertainties (list)."
        )
        msg2 = UserMessage(content=[ContentPartTextParam(text=syn_prompt)])
        try:
            syn_resp = await self.llm.ainvoke([msg2])
            syn = json.loads(syn_resp.completion if isinstance(syn_resp.completion, str) else "{}")
            es = ExecutiveSummary(
                text=syn.get("text", ""),
                confidence=float(syn.get("confidence", 0.6)),
                top_uncertainties=list(syn.get("uncertainties", []))[:3],
            )
        except Exception as e:
            logger.debug("DeepResearch: synthesis JSON parse failed: %s", e, exc_info=True)
            es = ExecutiveSummary(
                text="Executive summary could not be generated.",
                confidence=0.55,
                top_uncertainties=["LLM failure", "Coverage unknown", "Data availability"],
            )

        output = ResearchOutput(
            executive_summary=es,
            key_findings=[],
            contradictions=[],
            coverage=coverage,
            final=True,
        )
        self._last_output = output
        self._done = True
        self._success = True
        return TaskResult(success=True, message="deep_research_synthesis_complete", data={"output": output.model_dump()})


__all__ = [
    "ResearchConstraints",
    "ResearchInput",
    "ExecutiveSummary",
    "ResearchOutput",
    "DeepResearchSynthesisTask",
]
