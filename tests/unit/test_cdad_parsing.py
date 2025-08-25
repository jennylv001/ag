from __future__ import annotations

import pytest

from browser_use.agent.tasks.planner import _parse_markdown_table, _instantiate_tasks_from_rows, PlanRow
from browser_use.agent.tasks import TASK_REGISTRY


MD = """
## Abstracted Task Plan
| Phase | Task ID | Subtask Description | Dependencies | Required Resources/Actors | Rationale |
|------|---------|----------------------|--------------|---------------------------|-----------|
| 1 | 1.1 | Run a low-level step [LowLevelExecutionTask] | None | Agent | Warm up |
| 1 | 1.2 | Attempt captcha if present [SolveCaptchaTask] | 1.1 | LLM | Unblock flow |
"""


def test_cdad_parse_table_basic():
    rows = _parse_markdown_table(MD)
    assert len(rows) == 2
    assert rows[0].task_id == "1.1"
    assert rows[1].dependencies == ["1.1"]


class _Orch:
    def __init__(self):
        self.controller = object()
        self.browser_session = object()
        class _S: pass
        self.settings = _S()
        self.settings.llm = object()


def test_cdad_map_rows_to_tasks():
    rows = _parse_markdown_table(MD)
    orch = _Orch()
    tasks = _instantiate_tasks_from_rows(rows, orch)
    # Should instantiate known tasks present in the registry by name
    assert any(t.__class__.__name__ == 'LowLevelExecutionTask' for t in tasks)
    assert any(t.__class__.__name__ == 'SolveCaptchaTask' for t in tasks)
