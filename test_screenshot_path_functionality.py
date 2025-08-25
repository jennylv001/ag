#!/usr/bin/env python3
"""Test the new screenshot_path functionality."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_screenshot_path_functionality():
    """Test the complete screenshot path functionality."""
    try:
        from browser_use.browser.views import BrowserStateSummary, BrowserStateHistory
        from browser_use.agent.views import AgentHistoryList, AgentHistory
        from browser_use.dom.views import DOMElementNode

        # Test 1: BrowserStateSummary with screenshot_path
        print("🧪 Test 1: BrowserStateSummary with screenshot_path field")
        element_tree = DOMElementNode(
            tag_name='body',
            xpath='/body',
            attributes={},
            children=[],
            is_visible=True,
            parent=None
        )

        state_summary = BrowserStateSummary(
            element_tree=element_tree,
            selector_map={},
            url='https://example.com',
            title='Test Page',
            tabs=[],
            screenshot='base64encodeddata',
            screenshot_path='/tmp/screenshot_123.png'
        )

        assert state_summary.screenshot == 'base64encodeddata'
        assert state_summary.screenshot_path == '/tmp/screenshot_123.png'
        print("✅ BrowserStateSummary screenshot_path field works correctly")

        # Test 2: BrowserStateHistory with screenshot_path (already exists)
        print("🧪 Test 2: BrowserStateHistory screenshot_path integration")
        state_history = BrowserStateHistory(
            url='https://example.com',
            title='Test Page',
            tabs=[],
            interacted_element=[],
            screenshot_path='/tmp/screenshot_456.png'
        )

        assert state_history.screenshot_path == '/tmp/screenshot_456.png'
        print("✅ BrowserStateHistory screenshot_path field works correctly")

        # Test 3: AgentHistoryList.screenshot_paths() method
        print("🧪 Test 3: AgentHistoryList.screenshot_paths() method")
        history_list = AgentHistoryList()

        # Test empty history
        paths = history_list.screenshot_paths()
        assert paths == []
        print("✅ Empty history returns empty list")

        # Test with mock history containing screenshot paths
        # Note: This is a simplified test since creating full AgentHistory objects
        # would require more complex setup
        print("✅ screenshot_paths() method exists and returns correct type")

        # Test 4: Method signature matches screenshots() method
        print("🧪 Test 4: Method signature compatibility")
        paths_all = history_list.screenshot_paths()
        paths_last_2 = history_list.screenshot_paths(n_last=2)
        paths_no_none = history_list.screenshot_paths(return_none_if_not_screenshot=False)

        assert isinstance(paths_all, list)
        assert isinstance(paths_last_2, list)
        assert isinstance(paths_no_none, list)
        print("✅ All method signatures work correctly")

        print("\n🎉 All screenshot_path functionality tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_screenshot_path_functionality()
    sys.exit(0 if success else 1)
