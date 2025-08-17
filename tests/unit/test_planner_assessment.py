import asyncio
import types
import pytest

from browser_use.agent.settings import AgentSettings
from browser_use.agent.planner import Planner
from browser_use.agent.events import AssessmentUpdate, LLMRequest


class DummyState:
    def __init__(self):
        # Minimal fields used by Planner
        self.n_steps = 1
        self.agent_id = "agent-test"
        self.last_error = None
        self.task = "t"
        # Minimal history interface used by _emit_planning_request
        class H:
            def __init__(self):
                self.history = []
            def screenshots(self, n_last: int, return_none_if_not_screenshot: bool = False):
                return []
        self.history = H()

class DummyStateManager:
    def __init__(self):
        self.state = DummyState()
    async def get_status(self):
        # Always RUNNING-like; not terminal
        class S: pass
        return object()
    async def get_current_task_id(self):
        return "root"
    async def get_task_stack_summary(self):
        return []
    async def record_error(self, *args, **kwargs):
        pass
    async def update_last_history_with_reflection(self, *args, **kwargs):
        pass
    async def clear_error_and_failures(self, *args, **kwargs):
        pass
    async def mark_reflection_exit(self):
        pass


@pytest.mark.asyncio
async def test_assessment_trigger_with_dwell_and_cooldown():
    # Settings: enable planner; low dwell and short cooldown to test
    settings = AgentSettings(task="t", llm=types.SimpleNamespace(), use_planner=True)
    settings.assessor_dwell_seconds = 0.05
    settings.assessor_cooldown_seconds = 0.2
    settings.planner_interval = 0

    agent_bus: asyncio.Queue = asyncio.Queue(maxsize=10)
    heartbeat_bus: asyncio.Queue = asyncio.Queue(maxsize=10)

    planner = Planner(settings=settings, state_manager=DummyStateManager(), agent_bus=agent_bus, heartbeat_bus=heartbeat_bus)

    # Simulate stable proactive signals over dwell and ensure exactly one LLMRequest emitted
    evt = AssessmentUpdate(step_token=1, risk=0.2, opportunity=0.8, confidence=0.8, contributors=[])
    # First event starts dwell; no emit
    await planner._maybe_plan_on_assessment(evt)
    assert agent_bus.qsize() == 0
    # After dwell elapsed with same mode, another event should emit
    await asyncio.sleep(settings.assessor_dwell_seconds + 0.02)
    await planner._maybe_plan_on_assessment(evt)
    assert agent_bus.qsize() == 1
    req = await agent_bus.get()
    assert isinstance(req, LLMRequest)
    # Second immediate event within cooldown should not emit
    await planner._maybe_plan_on_assessment(evt)
    assert agent_bus.qsize() == 0


@pytest.mark.asyncio
async def test_time_based_cadence_emits_requests():
    settings = AgentSettings(task="t", llm=types.SimpleNamespace(), use_planner=True)
    settings.planner_interval_seconds = 0.05
    settings.planner_interval = 0

    agent_bus: asyncio.Queue = asyncio.Queue(maxsize=50)
    heartbeat_bus: asyncio.Queue = asyncio.Queue(maxsize=10)

    planner = Planner(settings=settings, state_manager=DummyStateManager(), agent_bus=agent_bus, heartbeat_bus=heartbeat_bus)

    loop_task = asyncio.create_task(planner.run())
    try:
        # Let the planner run enough to tick the time-based cadence across multiple 0.5s poll cycles
        await asyncio.sleep(1.2)
        # Drain any heartbeats from the queue if accidentally placed there
        emitted = []
        while not agent_bus.empty():
            item = await agent_bus.get()
            if isinstance(item, LLMRequest):
                emitted.append(item)
        assert len(emitted) >= 2  # Expect at least a couple of cadence ticks
    finally:
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass
