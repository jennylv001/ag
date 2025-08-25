from __future__ import annotations

# Package init for agent.planner
# Re-export commonly used CDAD planner APIs for convenience
try:
    from .cdad import (
        PlanRow,
        _parse_markdown_table,
        _instantiate_tasks_from_rows,
        plan_tasks_with_cdad,
        preflight_validate_tasks,
    )
except Exception:
    # Avoid import-time errors during partial installs; callers should import directly when needed.
    PlanRow = None  # type: ignore
    _parse_markdown_table = None  # type: ignore
    _instantiate_tasks_from_rows = None  # type: ignore
    plan_tasks_with_cdad = None  # type: ignore
    preflight_validate_tasks = None  # type: ignore

__all__ = [
    'PlanRow',
    '_parse_markdown_table',
    '_instantiate_tasks_from_rows',
    'plan_tasks_with_cdad',
    'preflight_validate_tasks',
]
