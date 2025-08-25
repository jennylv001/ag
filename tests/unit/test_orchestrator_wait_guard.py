import asyncio
import time

import pytest


class DummyLLM:
    def __init__(self, payload):
        self.payload = payload

    async def ainvoke(self, messages, output_format=None):
        # Return payload as if parsed by the schema
        class R:
            def __init__(self, p):
                self.completion = p
        return R(self.payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,seconds",
    [
        ("Please wait for 10 minutes and then finish", 600),
        ("Wait 600 seconds before completing", 600),
        ("This is a 10-minute wait, then done", 600),
        ("ok wait 10m then done", 600),
        ("for 120 s just wait then done", 120),
    ],
)
async def test_wait_injection_on_done(text, seconds):
    from browser_use.agent.state import AgentState, AgentStatus
    from browser_use.agent.settings import AgentSettings
    from browser_use.agent.orchestrator import AgentOrchestrator

    # Minimal controller/browser to capture actions
    class DummyController:
        class _Reg:
            def create_action_model(self, include_actions=None):
                # Build a simple dynamic object for wait/done actions
                class A:
                    def __init__(self):
                        pass

                    def model_dump(self, exclude_unset=True):
                        return getattr(self, "_dump", {})

                    def __setattr__(self, k, v):
                        if k in ("wait", "done"):
                            super().__setattr__("_dump", {k: v})
                        super().__setattr__(k, v)

                return A

        registry = _Reg()

        async def multi_act(self, actions, **kwargs):
            # Should receive a leading wait action before done
            dumps = [a.model_dump(exclude_unset=True) for a in actions]
            return [type("R", (), {"success": True, "extracted_content": str(dumps[0])})()]

    class DummyBrowser:
        browser_profile = type("P", (), {})()

        async def get_state_summary(self, **kwargs):
            return type("S", (), {"url": "http://example.com", "title": "t", "screenshot": None, "selector_map": {}})()

        async def stop(self, _hint=None):
            return None

        async def get_affordances_summary(self):
            return []

    # Seed agent state
    st = AgentState(task="Task: " + text)
    st.status = AgentStatus.RUNNING

    # Build orchestrator; we'll bypass decide() and feed a crafted AgentOutput-like object
    settings = AgentSettings(task=text, llm=DummyLLM({}))
    settings.wait_timeout_guard_seconds = 0.1
    orch = AgentOrchestrator(
        settings=settings,
        state_manager=type("SM", (), {"state": st})(),
        message_manager=type("MM", (), {"prepare_messages": lambda *a, **k: [] , "add_local_note": lambda *a, **k: None, "update_history_representation": lambda *a, **k: None})(),
        browser_session=DummyBrowser(),
        controller=DummyController(),
    )

    # Perceive to obtain orchestrator_state
    state = await orch._perceive()

    # Craft an AgentOutput-like object with text fields and only a 'done' action
    class DoneAction:
        def __init__(self):
            self._dump = {"done": {"success": True, "text": "complete"}}

        def model_dump(self, exclude_unset=True):
            return self._dump

    class Out:
        def __init__(self, t):
            self.thinking = t
            self.next_goal = t
            self.task_log = t
            self.prior_action_assessment = t
            self.action = [DoneAction()]

    out = Out(text)
    result = await orch.execute(out, state)

    # Verify a wait was injected: the first action result content should contain 'wait' with seconds
    msg = getattr(result.action_results[0], "extracted_content", "")
    assert "wait" in msg
    assert str(seconds if seconds <= 300 else 300) in msg  # chunked at controller cap 300s
