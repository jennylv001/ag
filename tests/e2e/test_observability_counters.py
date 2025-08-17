import asyncio
import time
import pytest

from browser_use.agent.state_manager import AgentState, StateManager, AgentStatus, LoadStatus
from browser_use.agent.views import AgentHistoryList
from browser_use.agent.events import ActuationResult
from browser_use.agent.views import ActionResult, StepMetadata


@pytest.mark.asyncio
async def test_reflection_intent_increments_counter():
    state = AgentState(task="t", history=AgentHistoryList())
    sm = StateManager(
        initial_state=state,
        file_system=None,
        max_failures=3,
        lock_timeout_seconds=5.0,
        use_planner=True,
        reflect_on_error=True,
        max_history_items=50,
        enable_modes=True,
        reflect_cadence=0,
        reflect_cooldown_seconds=0.0,
    )
    # Simulate a failed step to trigger reflection intent
    ar = ActionResult(success=False, error="boom")
    result = ActuationResult(
        action_results=[ar],
        llm_output=None,
        browser_state=None,
        step_metadata=StepMetadata(step_number=0, step_start_time=time.monotonic(), step_end_time=time.monotonic()),
    )
    outcome = await sm.decide_and_apply_after_step(result, max_steps=10, step_completed=True)
    snapshot = await sm.get_health_snapshot()
    assert snapshot["reflections_requested"] >= 1
    assert outcome.reflection_intent in (True, False)  # just ensuring path executed


@pytest.mark.asyncio
async def test_suppressed_by_load_increments_counter():
    state = AgentState(task="t", history=AgentHistoryList())
    sm = StateManager(
        initial_state=state,
        file_system=None,
        max_failures=3,
        lock_timeout_seconds=5.0,
        use_planner=True,
        reflect_on_error=True,
        max_history_items=50,
        enable_modes=True,
        reflect_cadence=0,
        reflect_cooldown_seconds=0.0,
    )
    # Set load to shedding
    await sm.set_load_status(LoadStatus.SHEDDING)
    # Simulate a failed step; reflection should be suppressed and counter incremented
    ar = ActionResult(success=False, error="boom")
    result = ActuationResult(
        action_results=[ar],
        llm_output=None,
        browser_state=None,
        step_metadata=StepMetadata(step_number=0, step_start_time=time.monotonic(), step_end_time=time.monotonic()),
    )
    await sm.decide_and_apply_after_step(result, max_steps=10, step_completed=True)
    snapshot = await sm.get_health_snapshot()
    assert snapshot["reflections_suppressed_by_load"] >= 1


@pytest.mark.asyncio
async def test_cooldown_block_increments_counter():
    state = AgentState(task="t", history=AgentHistoryList())
    sm = StateManager(
        initial_state=state,
        file_system=None,
        max_failures=5,
        lock_timeout_seconds=5.0,
        use_planner=True,
        reflect_on_error=True,
        max_history_items=50,
        enable_modes=True,
        reflect_cadence=0,
        reflect_cooldown_seconds=60.0,
    )
    # Pretend a reflection just happened recently
    async with asyncio.timeout(1):
        async with sm._lock:
            sm.state.last_reflect_exit_ts = time.monotonic()
    # Now cause a single failure; cooldown should block reflection
    ar = ActionResult(success=False, error="boom")
    result = ActuationResult(
        action_results=[ar],
        llm_output=None,
        browser_state=None,
        step_metadata=StepMetadata(step_number=0, step_start_time=time.monotonic(), step_end_time=time.monotonic()),
    )
    await sm.decide_and_apply_after_step(result, max_steps=10, step_completed=True)
    snapshot = await sm.get_health_snapshot()
    assert snapshot["cooldown_blocks"] >= 1


@pytest.mark.asyncio
async def test_downshift_guard_counter():
    state = AgentState(task="t", history=AgentHistoryList())
    # start in RUNNING, then propose a downshift to PENDING via a signal decision that is lower priority
    sm = StateManager(
        initial_state=state,
        file_system=None,
        max_failures=3,
        lock_timeout_seconds=5.0,
        use_planner=True,
        reflect_on_error=True,
        max_history_items=50,
        enable_modes=True,
    )
    # Force status to RUNNING
    await sm.set_status(AgentStatus.RUNNING, force=True)
    # Emit a signal that results in no step change but attempt to set a lower priority status
    # We'll simulate via a fake decision by calling private method under lock (acceptable for test)
    from browser_use.agent.state_manager import TransitionDecision
    async with asyncio.timeout(1):
        async with sm._lock:
            # Craft a decision suggesting PENDING (lower priority) without counters cleared
            decision = TransitionDecision(
                next_status=AgentStatus.PENDING,
                set_modes=0,
                clear_modes=0,
                reason="test_downshift",
                reflection_intent=False,
            )
            sm._state.missed_heartbeats = 1  # not cleared
            sm._apply_decision(decision, source="test")
    snapshot = await sm.get_health_snapshot()
    assert snapshot["downshift_guard_prevented"] >= 1


@pytest.mark.asyncio
async def test_flags_off_no_modes_behavior():
    state = AgentState(task="t", history=AgentHistoryList())
    sm = StateManager(
        initial_state=state,
        file_system=None,
        max_failures=3,
        lock_timeout_seconds=5.0,
        use_planner=True,
        reflect_on_error=True,
        max_history_items=50,
        enable_modes=False,
    )
    # Ingest signals should be no-ops
    await sm.ingest_signal('io_timeout')
    await sm.ingest_signal('heartbeat_miss')
    snapshot = await sm.get_health_snapshot()
    assert snapshot["modes"] == 0
    assert snapshot["missed_heartbeats"] == 0
    assert snapshot["io_timeouts_recent"] == 0
