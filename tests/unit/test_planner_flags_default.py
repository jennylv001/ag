from __future__ import annotations

from browser_use.agent.settings import AgentSettings


def test_planner_flags_default_false():
    # Defaults should be conservative to avoid behavior drift
    # use_task_planner and use_replanning default to False
    s = AgentSettings(task="x", llm=object())
    assert getattr(s, 'use_task_planner', False) is False
    assert getattr(s, 'use_replanning', False) is False
