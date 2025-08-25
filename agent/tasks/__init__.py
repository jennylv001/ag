from __future__ import annotations

"""
Task registry and discovery helpers.

Exposes a stable mapping of task names to their implementing classes so that
the Agent/Orchestrator can enumerate and construct tasks dynamically without
tight coupling. This module is additive and does not change existing behavior.
"""

from typing import Dict, Iterable, Tuple, Type

from .base_task import BaseTask
# LowLevelExecutionTask removed - use orchestrator directly
from .solve_captcha_task import SolveCaptchaTask
from .deep_research_task import DeepResearchSynthesisTask
from .trading_agent_task import TradingSymbolOrchestratorTask

# Public registry of available tasks. Keys are canonical task names used by planners.
TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    # "low_level_execution" removed - use orchestrator directly
    "solve_captcha": SolveCaptchaTask,
    "deep_research_synthesis": DeepResearchSynthesisTask,
    "trading_symbol_orchestrator": TradingSymbolOrchestratorTask,
}

# Note: GatherStructuredDataTask is imported lazily by controller.service action to
# keep this package import stable for smoke tests. It is not registered here.


def available_tasks() -> Iterable[str]:
    """Return iterable of task names available for planning/selection."""

    return TASK_REGISTRY.keys()


def get_task_class(name: str) -> Type[BaseTask]:
    """Resolve a task class by name, raising KeyError if not found."""

    return TASK_REGISTRY[name]


def task_signatures() -> Dict[str, str]:
    """Return a lightweight mapping of task name -> constructor signature string.

    Kept as strings to avoid importing inspect.Signature types at call sites and to
    remain friendly for JSON/telemetry export.
    """

    import inspect

    sigs: Dict[str, str] = {}
    for name, cls in TASK_REGISTRY.items():
        try:
            sig = inspect.signature(getattr(cls, "__init__"))
            sigs[name] = str(sig)
        except Exception:
            sigs[name] = "(unavailable)"
    return sigs


__all__ = [
    "BaseTask",
    "SolveCaptchaTask",
    "DeepResearchSynthesisTask",
    "TradingSymbolOrchestratorTask",
    "TASK_REGISTRY",
    "available_tasks",
    "task_signatures",
]
# GatherStructuredDataTask is intentionally not imported here to keep package imports stable.
