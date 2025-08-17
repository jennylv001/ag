import asyncio
import pytest

from browser_use.agent.settings import AgentSettings
from browser_use.agent.supervisor import Supervisor
from browser_use.agent.state_manager import AgentStatus
from browser_use.llm.base import BaseChatModel
from browser_use.browser.views import BrowserStateSummary


class DummyLLM:
    def __init__(self, model: str = "dummy"):
        self.model = model

    async def ainvoke(self, messages, output_format):
        class R:
            completion = output_format() if output_format else None
        return R()


class FakeBrowserSession:
    def __init__(self):
        self.browser_profile = type("P", (), {"downloads_path": None})()
        self.downloaded_files = []

    async def get_state_summary(self, cache_clickable_elements_hashes: bool = True) -> BrowserStateSummary:
        return BrowserStateSummary(url="about:blank", title="Test", tabs=[], screenshot=None, selector_map={}, element_tree=None)

    async def get_minimal_state_summary(self) -> BrowserStateSummary:
        return await self.get_state_summary()

    async def stop(self):
        return None


@pytest.mark.asyncio
async def test_human_guidance_queue_and_consume():
    settings = AgentSettings(task="test task", llm=DummyLLM(model="dummy"), max_steps=1, use_planner=False)
    settings.browser_session = FakeBrowserSession()

    sup = Supervisor(settings)

    # Start pause handler loop
    pause_task = asyncio.create_task(sup._pause_handler())

    # Put agent in PAUSED and enqueue guidance
    await sup.state_manager.set_status(AgentStatus.PAUSED, force=True)
    await sup.state_manager.add_human_guidance("Click the login button then resume")

    # Allow handler to consume
    await asyncio.sleep(0.6)

    # Verify guidance surfaced into message manager state
    notes = sup.message_manager.state.local_system_notes
    assert any("Human guidance received:" in n.system_message for n in notes)

    # Stop loop and cleanup
    await sup.state_manager.set_status(AgentStatus.STOPPED, force=True)
    await asyncio.sleep(0.1)
    pause_task.cancel()
    try:
        await pause_task
    except asyncio.CancelledError:
        pass
    await sup.close()
