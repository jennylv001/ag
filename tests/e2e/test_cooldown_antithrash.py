import time
import pytest

from browser_use.agent.state_manager import StateManager, AgentState, AgentStatus, AgentMode, TransitionInputs, _TransitionEngine, LoadStatus


def make_inputs(**overrides):
    base = dict(
        status=AgentStatus.RUNNING,
        n_steps=5,
        consecutive_failures=1,
        modes=int(AgentMode.NONE),
        load_status=LoadStatus.NORMAL,
        step_completed=True,
        had_failure=True,
        missed_heartbeats=0,
        io_timeouts_recent_count=0,
        max_steps=100,
        max_failures=3,
        reflect_on_error=True,
        use_planner=True,
        reflect_cadence=0,
        reflect_cooldown_seconds=10.0,
        seconds_since_last_reflect=1.0,
        reflection_requested_this_epoch=False,
    )
    base.update(overrides)
    return TransitionInputs(**base)


def test_cooldown_blocks_on_first_failure_only():
    eng = _TransitionEngine()
    # First failure after last reflect within cooldown
    dec = eng.decide(make_inputs(consecutive_failures=1))
    assert dec.cooldown_blocked is True
    assert dec.reflection_intent is False
    assert 'cooldown_guard' in dec.reason

    # Second consecutive failure should bypass cooldown
    dec2 = eng.decide(make_inputs(consecutive_failures=2))
    assert dec2.cooldown_blocked is False
    # Reflection allowed unless other guards
    assert ('reflect_on_error' in dec2.reason) or dec2.reflection_intent is True


def test_cooldown_bypassed_when_uncertain_or_stalling():
    eng = _TransitionEngine()
    # UNCERTAIN mode should bypass cooldown block
    dec_uncertain = eng.decide(make_inputs(consecutive_failures=1, modes=int(AgentMode.UNCERTAIN)))
    assert dec_uncertain.cooldown_blocked is False
    # STALLING mode should bypass cooldown block
    dec_stalling = eng.decide(make_inputs(consecutive_failures=1, modes=int(AgentMode.STALLING)))
    assert dec_stalling.cooldown_blocked is False


@pytest.mark.asyncio
async def test_cooldown_blocks_counter_increments():
    st = AgentState(task="t")
    sm = StateManager(
        initial_state=st,
        file_system=None,
        max_failures=3,
        lock_timeout_seconds=5.0,
        use_planner=True,
        reflect_on_error=True,
        max_history_items=10,
        enable_modes=True,
        reflect_cadence=0,
        reflect_cooldown_seconds=10.0,
        io_timeout_window=5,
    )

    # Simulate a failed step with recent reflection inside cooldown
    st.last_reflect_exit_ts = time.monotonic()
    # Build a minimal actuation-like result object for decide_and_apply_after_step
    from browser_use.agent.views import ActionResult, StepMetadata
    class R:
        def __init__(self):
            self.action_results = [ActionResult(success=False, error='boom')]
            self.llm_output = None
            self.browser_state = None
            self.step_metadata = StepMetadata(step_number=0, step_start_time=time.monotonic(), step_end_time=time.monotonic())
    r = R()

    outcome = await sm.decide_and_apply_after_step(r, max_steps=100, step_completed=True)
    assert sm.state.cooldown_blocks >= 1
    # Reflection should be blocked due to cooldown on first failure
    assert outcome.reflection_intent in (False, 0)
