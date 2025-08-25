from browser_use.agent.settings import AgentSettings


def test_single_flag_cascade_enabled():
    s = AgentSettings(task="t", llm=object())  # llm typed at runtime; object() is fine for this check
    assert s.task_layer_enabled is False
    assert s.replanning_enabled is False

    s.use_task_planner = True
    assert s.task_layer_enabled is True
    # When single flag is on, replanning_enabled should be True regardless of use_replanning
    assert s.replanning_enabled is True


def test_single_flag_cascade_respects_legacy_toggle():
    s = AgentSettings(task="t", llm=object(), use_replanning=True)
    # With planner off, legacy use_replanning still applies
    assert s.task_layer_enabled is False
    assert s.replanning_enabled is True
