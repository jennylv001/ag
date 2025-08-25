#!/usr/bin/env python3

"""
Test script to reproduce the 'str' object has no attribute 'content' error.
"""

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.message_manager.views import MessageManagerSettings
from browser_use.agent.state_manager.service import StateManager
from browser_use.browser.session import BrowserSession
from browser_use.browser.profile import BrowserProfile, CdpSettings, StealthSettings
from browser_use.agent.state_manager.models import AgentState

async def test_content_error():
    """Test to reproduce the content attribute error."""

    # Create basic test settings
    message_settings = MessageManagerSettings()

    # Create a minimal state manager
    state_manager = StateManager()

    # Create a message manager
    message_manager = MessageManager(
        settings=message_settings,
        state_manager=state_manager,
        system_message=None,
        task="Test task"
    )

    # Try to prepare messages which should trigger the error
    try:
        # Create a minimal browser session to pass requirements
        browser_profile = BrowserProfile()
        browser_session = BrowserSession(
            browser_profile=browser_profile,
            keep_open=False
        )

        # Mock browser state
        class MockBrowserState:
            def __init__(self):
                self.url = "https://example.com"
                self.screenshot = None

        browser_state = MockBrowserState()

        # This should trigger the content error
        messages = await message_manager.prepare_messages(
            state_manager=state_manager,
            browser_state=browser_state,
        )

        print(f"✅ Success: Got {len(messages)} messages")
        for i, msg in enumerate(messages):
            print(f"  Message {i}: {type(msg).__name__}")

    except AttributeError as e:
        if "content" in str(e):
            print(f"❌ Reproduced the content error: {e}")
            import traceback
            traceback.print_exc()
        else:
            print(f"❌ Different AttributeError: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_content_error())
