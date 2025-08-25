from __future__ import annotations

"""
CDAD planner stub.

This minimal implementation exists to keep imports stable during refactors and
in environments where the full planner strategy isn't required. It provides the
expected symbols and safe no-op behavior.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class PlanRow:
    """Parsed plan row representation expected by tests.

    - task_id: hierarchical id like "1.1"
    - task_name: class name inside square brackets, e.g., "LowLevelExecutionTask"
    - dependencies: list of task ids this row depends on
    - args: optional kv overrides for instantiation (unused for now)
    """

    task_id: str
    task_name: str
    dependencies: List[str]
    args: Dict[str, Any] | None = None


def _parse_markdown_table(markdown: str) -> List[PlanRow]:
    """Parse a markdown table into plan rows.

    Expected columns (by tests): Phase | Task ID | Subtask Description | Dependencies | ...
    We extract Task ID, the bracketed task class from Subtask Description, and Dependencies.
    """
    rows: List[PlanRow] = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        # skip header separator lines
        if set(line.replace("|", "").strip()) <= {"-", " "}:
            continue
        parts = [p.strip() for p in line.split("|")][1:-1]
        if len(parts) < 4:
            continue
        task_id = parts[1]
        desc = parts[2]
        deps_cell = parts[3]
        # Extract class name in brackets, e.g., "... [LowLevelExecutionTask] ..."
        task_name = None
        lb = desc.rfind("[")
        rb = desc.rfind("]")
        if lb != -1 and rb != -1 and rb > lb:
            task_name = desc[lb + 1 : rb].strip()
        if not task_name:
            continue
        deps: List[str] = []
        if deps_cell and deps_cell.lower() != "none":
            deps = [d.strip() for d in deps_cell.split(",") if d.strip()]
        rows.append(PlanRow(task_id=task_id, task_name=task_name, dependencies=deps, args=None))
    return rows


def _instantiate_tasks_from_rows(rows: List[PlanRow], orchestrator: Any) -> List[Any]:
    """Map PlanRows to BaseTask instances using the task registry and orchestrator deps."""
    try:
        from browser_use.agent.tasks import TASK_REGISTRY
    except Exception:
        return []

    # Map class name -> class for convenience
    name_to_cls = {cls.__name__: cls for cls in TASK_REGISTRY.values()}
    tasks: List[Any] = []
    for r in rows:
        cls = name_to_cls.get(r.task_name)
        if not cls:
            continue
        # Heuristic construction based on known task signatures
        try:
            if r.task_name == "LowLevelExecutionTask":
                inst = cls(orchestrator=orchestrator)
            elif r.task_name == "SolveCaptchaTask":
                # Pull deps from orchestrator in tests
                controller = getattr(orchestrator, "controller", None)
                browser = getattr(orchestrator, "browser_session", None)
                llm = getattr(getattr(orchestrator, "settings", object()), "llm", None)
                inst = cls(controller=controller, browser=browser, page_extraction_llm=llm)
            else:
                # Default: try no-arg construction
                inst = cls()  # type: ignore[call-arg]
        except Exception:
            # Best-effort: skip on constructor errors
            continue
        tasks.append(inst)
    return tasks


async def plan_tasks_with_cdad(orchestrator: Any, goal: str, mission_state: Any) -> List[Any]:
    """Return a list of planned tasks for the given goal.

    Minimal stub: no real planning; return a single LowLevelExecutionTask if available.
    """
    try:
        from browser_use.agent.tasks import TASK_REGISTRY
        cls = next((c for c in TASK_REGISTRY.values() if c.__name__ == "LowLevelExecutionTask"), None)
        return [cls(orchestrator=orchestrator)] if cls else []  # type: ignore[misc]
    except Exception:
        return []


def preflight_validate_tasks(tasks: List[Any]) -> Tuple[bool, List[str]]:
    """Validate that tasks are registered and have required deps.

    Returns (ok, errors). ok is False if any errors found.
    """
    errors: List[str] = []
    try:
        from browser_use.agent.tasks import TASK_REGISTRY
        registered = set(TASK_REGISTRY.values())
    except Exception:
        registered = set()

    for t in tasks:
        t_cls = t.__class__
        if t_cls not in registered:
            errors.append(f"Task class {t_cls.__name__} is not registered")
            continue
        # Dependency checks for known tasks
        name = t_cls.__name__
        if name == "SolveCaptchaTask":
            # Must have controller, browser, llm
            for attr in ("controller", "browser", "llm"):
                if getattr(t, attr, None) is None:
                    errors.append(f"SolveCaptchaTask missing dependency: {attr}")
        elif name == "LowLevelExecutionTask":
            # Should reference an orchestrator
            if getattr(t, "_orchestrator", None) is None:
                errors.append("LowLevelExecutionTask missing dependency: orchestrator")

    return (len(errors) == 0, errors)


__all__ = [
    "PlanRow",
    "_parse_markdown_table",
    "_instantiate_tasks_from_rows",
    "plan_tasks_with_cdad",
    "preflight_validate_tasks",
]
