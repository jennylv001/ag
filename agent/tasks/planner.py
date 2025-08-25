"""
Task planner facade.

Public import surface for planning helpers used by the task layer.

Re-exports CDAD planner APIs from `browser_use.agent.planner.cdad` while
allowing future strategy swaps without changing call sites.

Preferred import path for agent code:

    from browser_use.agent.tasks.planner import plan_tasks_with_cdad, preflight_validate_tasks

Internal details (subject to change) live under `agent/planner/cdad.py`.
"""

from __future__ import annotations

# Re-export CDAD strategy APIs from sibling package agent.planner
try:
    from ..planner.cdad import (
        PlanRow,
        _parse_markdown_table,
        _instantiate_tasks_from_rows,
        plan_tasks_with_cdad,
        preflight_validate_tasks,
    )
except Exception:
    # Fallback: import via package __init__ which may provide relaxed exports
    from ..planner import (  # type: ignore
        PlanRow,
        _parse_markdown_table,
        _instantiate_tasks_from_rows,
        plan_tasks_with_cdad,
        preflight_validate_tasks,
    )

__all__ = [
    "PlanRow",
    "_parse_markdown_table",
    "_instantiate_tasks_from_rows",
    "plan_tasks_with_cdad",
    "preflight_validate_tasks",
]
