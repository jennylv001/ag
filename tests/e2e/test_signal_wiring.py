import asyncio
import time
import pytest

from browser_use.agent.state_manager import StateManager, AgentState, LoadStatus, AgentMode


@pytest.mark.asyncio
async def test_heartbeat_signals_toggle_modes():
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
        reflect_cooldown_seconds=0.0,
        io_timeout_window=5,
    )

    # Simulate heartbeat miss and ok
    await sm.ingest_signal('heartbeat_miss')
    assert AgentMode.DEGRADED & AgentMode(sm.state.modes)
    await sm.ingest_signal('heartbeat_ok')
    # Miss counter reset; modes may clear on subsequent engine decisions
    assert st.missed_heartbeats == 0


@pytest.mark.asyncio
async def test_io_timeout_emits_modes():
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
        reflect_cooldown_seconds=0.0,
    io_timeout_window=3,
    )

    # Timeouts should mark UNCERTAIN at 1, STALLING at 3 by threshold
    await sm.ingest_signal('io_timeout')
    assert AgentMode.UNCERTAIN & AgentMode(sm.state.modes)
    await sm.ingest_signal('io_timeout')
    assert not (AgentMode.STALLING & AgentMode(sm.state.modes))
    await sm.ingest_signal('io_timeout')
    assert AgentMode.STALLING & AgentMode(sm.state.modes)

    # io_ok should decay the window
    await sm.ingest_signal('io_ok')
    assert len(st.io_timeouts_recent) == 2
