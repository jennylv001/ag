import asyncio
import types
import time
from typing import Any

import pytest

from browser_use.agent.orchestrator import AgentOrchestrator


class DummySettings:
    # Minimal settings the orchestrator touches
    max_steps = 3
    reflect_on_error = True
    use_planner = False
    reflect_cadence = 0
    reflect_cooldown_seconds = 0.0
    default_action_timeout_seconds = 1.0  # intentionally tiny default
    wait_timeout_guard_seconds = 0.5

    # Controller stub with a registry that can create an action model and hold a wait action
    class _Ctrl:
        def __init__(self):
            class _Registry:
                def create_action_model(self, include_actions: list[str] | None = None):
                    # Dynamic pydantic-like model is overkill; we just need an object with model_dump
                    class _Action:
                        def __init__(self, seconds: int = 0):
                            self.wait = {"seconds": seconds}

                        def model_dump(self, exclude_unset: bool = True):
                            return {"wait": self.wait}

                    return _Action
            self.registry = _Registry()
    controller = _Ctrl()


class DummyState:
    # Orchestrator expects .state with attributes used across calls
    def __init__(self):
        self.status = types.SimpleNamespace(name='RUNNING')
        self.n_steps = 0
        self.task = "test"
        self.consecutive_failures = 0
        self.last_error = None
        self.history = []
        self.task_stack = None
        self.agent_id = "A1"


class DummyStateManager:
    def __init__(self):
        self.state = DummyState()


class DummyMessageManager:
    def update_history_representation(self, *_args, **_kwargs):
        pass

    def add_local_note(self, *_args, **_kwargs):
        pass

    async def prepare_messages(self, **_kwargs):
        return []


class DummyBrowserState:
    url = "https://example.com"
    title = "Example"


class DummyBrowser:
    browser_profile = types.SimpleNamespace(wait_between_actions=0, keep_alive=False)

    async def get_state_summary(self, **_kwargs):
        # minimal OrchestratorState expectations
        class _Summary:
            url = "https://example.com"
            title = "Example"
            selector_map = {}
            screenshot = None
            page_info = None
        return _Summary()

    async def stop(self, *_args, **_kwargs):
        pass


class DummyController:
    # matches Controller.multi_act signature used in orchestrator.execute
    async def multi_act(self, actions, browser_session, **_kwargs):
        # emulate the wait action by sleeping for given seconds in the params
        first = actions[0]
        seconds = first.model_dump(exclude_unset=True)["wait"]["seconds"]
        await asyncio.sleep(seconds)
        # return a single ActionResult-like object with required attributes
        class _R:
            success = True
            error = None
            is_done = False
        return [_R()]


class DummyAgentOutput:
    def __init__(self, seconds: int):
        # Emulate AgentOutput with a dynamic action instance
        Action = DummySettings.controller.registry.create_action_model()
        self.action = [Action(seconds=seconds)]


class DummyOrchestratorState:
    def __init__(self):
        self.step_number = 0
        self.step_start_time = time.monotonic()
        class _BS:
            url = "https://example.com"
            title = "Example"
        self.browser_state = _BS()
        self.oscillation_score = 0.0
        self.no_progress_score = 0.0


@pytest.mark.asyncio
async def test_wait_action_extends_timeout_scoped():
    settings = DummySettings()
    state_mgr = DummyStateManager()
    msg_mgr = DummyMessageManager()
    browser = DummyBrowser()
    controller = DummyController()

    orch = AgentOrchestrator(settings=settings, state_manager=state_mgr, message_manager=msg_mgr, browser_session=browser, controller=controller)

    orch_state = DummyOrchestratorState()
    out = DummyAgentOutput(seconds=2)

    # Ensure selected timeout >= seconds + guard > default
    timeout = orch._select_timeout(out, orch_state)
    assert timeout >= 2.5

    # Also verify execute completes without asyncio.wait_for timeout
    start = time.monotonic()
    res = await orch.execute(agent_output=out, orchestrator_state=orch_state)
    elapsed = time.monotonic() - start
    assert elapsed >= 2.0
    assert hasattr(res, "action_results") and len(res.action_results) == 1
