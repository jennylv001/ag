#!/usr/bin/env python3
"""Quick test for action_history method implementation."""

import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_action_history_import():
    """Test that we can import and instantiate AgentHistoryList."""
    try:
        from browser_use.agent.views import AgentHistoryList

        # Create an empty instance
        history_list = AgentHistoryList()

        # Test the action_history method exists and returns correct type
        result = history_list.action_history()
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) == 0, f"Expected empty list for empty history, got {result}"

        print("✅ action_history method imported and basic functionality verified")
        return True

    except Exception as e:
        print(f"❌ Import/instantiation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_action_history_import()
    sys.exit(0 if success else 1)
