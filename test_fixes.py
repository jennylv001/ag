#!/usr/bin/env python3
"""
Simple test to verify our browser session and long-running mode fixes.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from browser_use.browser.session import BrowserSession
from browser_use.agent.settings import AgentSettings

async def test_browser_session_creation():
    """Test that browser session can be created without the _tab_visibility_binding_name error."""
    try:
        settings = AgentSettings()
        browser_session = BrowserSession(config=settings.browser_config)
        print("✅ BrowserSession created successfully")

        # Try to initialize the browser session
        await browser_session.get_current_page()
        print("✅ BrowserSession get_current_page() works")

        # Clean up
        await browser_session.close()
        print("✅ BrowserSession closed successfully")

        return True
    except Exception as e:
        print(f"❌ BrowserSession test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

async def test_long_running_integration():
    """Test that long-running integration can be imported and created."""
    try:
        from browser_use.agent.long_running_integration import LongRunningIntegration
        from browser_use.agent.state_manager import StateManager

        state_manager = StateManager(agent_id="test", max_steps=1)
        integration = LongRunningIntegration(state_manager=state_manager)
        print("✅ LongRunningIntegration created successfully")

        return True
    except Exception as e:
        print(f"❌ LongRunningIntegration test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("Running browser session and long-running mode fix tests...")

    browser_test = await test_browser_session_creation()
    integration_test = await test_long_running_integration()

    if browser_test and integration_test:
        print("\n🎉 All tests passed! Fixes are working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Issues remain.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
