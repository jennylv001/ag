from __future__ import annotations

from browser_use.agent.state_manager import (
    AgentStatus,
    LoadStatus,
    _TransitionEngine,
    TransitionInputs,
)


def _base_inputs(**overrides):
    base = dict(
        status=AgentStatus.RUNNING,
        n_steps=0,
        consecutive_failures=0,
        modes=0,
        load_status=LoadStatus.NORMAL,
        step_completed=True,
        had_failure=False,
        missed_heartbeats=0,
        io_timeouts_recent_count=0,
        max_steps=10,
        max_failures=3,
        reflect_on_error=True,
        use_planner=True,
        reflect_cadence=0,
        reflect_cooldown_seconds=0.0,
        seconds_since_last_reflect=None,
        reflection_requested_this_epoch=False,
    )
    base.update(overrides)
    return TransitionInputs(**base)


def test_success_running_and_no_modes():
    eng = _TransitionEngine()
    d = eng.decide(_base_inputs())
    assert d.next_status == AgentStatus.RUNNING
    assert d.set_modes == 0 and d.clear_modes == 0
    assert "success_running" in d.reason
    assert d.reflection_intent is False


def test_failure_under_threshold_stays_running():
    eng = _TransitionEngine()
    d = eng.decide(_base_inputs(had_failure=True, consecutive_failures=0))
    assert d.next_status == AgentStatus.RUNNING
    assert "failure_but_under_threshold" in d.reason


def test_failures_threshold_trips_failed():
    eng = _TransitionEngine()
    d = eng.decide(_base_inputs(had_failure=True, consecutive_failures=2, max_failures=3))
    assert d.next_status == AgentStatus.FAILED
    assert "failures_threshold" in d.reason


def test_max_steps_reached():
    eng = _TransitionEngine()
    d = eng.decide(_base_inputs(n_steps=9, max_steps=10))
    assert d.next_status == AgentStatus.MAX_STEPS_REACHED
    assert "max_steps_reached" in d.reason


def test_heartbeat_sets_degraded_and_stalling():
    eng = _TransitionEngine()
    d1 = eng.decide(_base_inputs(missed_heartbeats=1))
    assert d1.set_modes != 0  # DEGRADED set
    d3 = eng.decide(_base_inputs(missed_heartbeats=3))
    # Should include STALLING in desired modes
    assert d3.set_modes != 0


def test_io_timeouts_set_uncertain_and_stalling():
    eng = _TransitionEngine()
    d1 = eng.decide(_base_inputs(io_timeouts_recent_count=1))
    assert d1.set_modes != 0  # UNCERTAIN set
    d3 = eng.decide(_base_inputs(io_timeouts_recent_count=3))
    assert d3.set_modes != 0  # STALLING set as well


def test_high_load_sets_mode_and_suppresses_reflection():
    eng = _TransitionEngine()
    d = eng.decide(_base_inputs(had_failure=True, load_status=LoadStatus.SHEDDING))
    assert d.set_modes != 0
    assert "shed_suppresses_reflection" in d.reason
    assert d.reflection_intent is False


def test_cooldown_suppresses_reflection():
    eng = _TransitionEngine()
    d = eng.decide(
        _base_inputs(
            had_failure=True,
            reflect_cooldown_seconds=10.0,
            seconds_since_last_reflect=3.0,
        )
    )
    assert d.reflection_intent is False
    assert "cooldown_guard" in d.reason


def test_cadence_gate():
    eng = _TransitionEngine()
    # n_steps=0 -> next is step 1; cadence=2 blocks at step 1
    d = eng.decide(_base_inputs(had_failure=True, reflect_cadence=2))
    assert d.reflection_intent is False
    assert "cadence_gate" in d.reason


def test_epoch_guard():
    eng = _TransitionEngine()
    d = eng.decide(_base_inputs(had_failure=True, reflection_requested_this_epoch=True))
    assert d.reflection_intent is False
    assert "epoch_guard" in d.reason


def test_reflect_on_error_when_allowed():
    eng = _TransitionEngine()
    d = eng.decide(_base_inputs(had_failure=True))
    assert d.reflection_intent is True
    assert "reflect_on_error" in d.reason
