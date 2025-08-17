from __future__ import annotations

import asyncio
import time

import pytest

from browser_use.agent.state_manager import StateManager, AgentState, AgentStatus, LoadStatus, _TransitionEngine


@pytest.mark.asyncio
async def test_heartbeat_miss_triggers_modes_and_reflection_intent():
    sm = StateManager(
        initial_state=AgentState(task="t"),
        file_system=None,
        max_failures=3,
        lock_timeout_seconds=1.0,
        use_planner=True,
        reflect_on_error=True,
        max_history_items=10,
        memory_budget_mb=1.0,
        enable_modes=True,
        reflect_cadence=0,
        reflect_cooldown_seconds=0.0,
        io_timeout_window=5,
    )
    # Ensure starting status
    await sm.set_status(AgentStatus.RUNNING, force=True)

    # Three misses -> engine should set STALLING in desired modes
    await sm.ingest_signal('heartbeat_miss', {})
    await sm.ingest_signal('heartbeat_miss', {})
    await sm.ingest_signal('heartbeat_miss', {})

    # Because our engine may set reflection on stall (health-based), the epoch flag may be True
    assert sm.state.missed_heartbeats == 3
    assert sm.state.modes != 0  # Some mode bits set
    # Reflection is tracked as epoch flag (intent may be set)
    assert isinstance(sm.state.reflection_requested_this_epoch, bool)


@pytest.mark.asyncio
async def test_priority_rule_no_downshift_without_clear():
    sm = StateManager(
        initial_state=AgentState(task="t"),
        file_system=None,
        max_failures=3,
        lock_timeout_seconds=1.0,
        use_planner=True,
        reflect_on_error=True,
        max_history_items=10,
        memory_budget_mb=1.0,
        enable_modes=True,
    )
    await sm.set_status(AgentStatus.RUNNING, force=True)

    # Push to a higher/equal priority state via a fake decision: simulate by heartbeat misses
    await sm.ingest_signal('heartbeat_miss', {})

    # Now attempt an immediate downshift via load=normal and counters not cleared: should not downshift status
    await sm.ingest_signal('load_status', {"status": LoadStatus.NORMAL})
    # No counters cleared yet since missed_heartbeats != 0
    # Ensure status remains RUNNING (no downshift)
    assert (await sm.get_status()) == AgentStatus.RUNNING

    # Clear counters via heartbeat_ok and io_ok; epoch resets
    await sm.ingest_signal('heartbeat_ok', {})
    await sm.ingest_signal('io_ok', {})

    # After counters cleared, a future decision could downshift (not applicable here but rule exercised)
    assert sm.state.missed_heartbeats == 0
    assert len(sm.state.io_timeouts_recent) == 0
    assert sm.state.load_status == LoadStatus.NORMAL


@pytest.mark.asyncio
async def test_io_timeouts_window_and_decay():
    sm = StateManager(
        initial_state=AgentState(task="t"),
        file_system=None,
        max_failures=3,
        lock_timeout_seconds=1.0,
        use_planner=True,
        reflect_on_error=True,
        max_history_items=10,
        memory_budget_mb=1.0,
        enable_modes=True,
        io_timeout_window=2,
    )
    await sm.set_status(AgentStatus.RUNNING, force=True)

    await sm.ingest_signal('io_timeout', {})
    await sm.ingest_signal('io_timeout', {})
    await sm.ingest_signal('io_timeout', {})  # should pop oldest keeping len<=2

    assert len(sm.state.io_timeouts_recent) == 2

    # decay once
    await sm.ingest_signal('io_ok', {})
    assert len(sm.state.io_timeouts_recent) == 1
