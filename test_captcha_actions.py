#!/usr/bin/env python3

"""
Quick test to validate that solve_captcha can create proper ActionModel instances
without encountering controller execution errors.
"""

import asyncio
from unittest.mock import Mock, AsyncMock
from controller.service import Controller
from controller.views import ClickElementAction

async def test_action_model_creation():
    """Test that our ActionModel creation works correctly"""

    # Create a controller instance
    controller = Controller()

    # Get the ActionModel class from the controller's registry
    ActionModel = controller.registry.create_action_model()

    # Test creating ActionModel with click_element_by_index action using dictionary format
    try:
        # Try dictionary format first
        test_action = ActionModel(click_element_by_index={'index': 1})

        # Verify the action model has the expected structure
        assert hasattr(test_action, 'click_element_by_index')

        # Test model_dump() which is used by controller
        action_dump = test_action.model_dump(exclude_unset=True)
        assert 'click_element_by_index' in action_dump
        assert action_dump['click_element_by_index']['index'] == 1

        print("✅ ActionModel creation successful!")
        print(f"ActionModel structure: {action_dump}")
        return True

    except Exception as e:
        print(f"❌ ActionModel creation failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_action_model_creation())
    if success:
        print("\n🎉 CAPTCHA ActionModel test PASSED - controller execution should work!")
    else:
        print("\n💥 CAPTCHA ActionModel test FAILED - needs more work")
