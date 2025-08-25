#!/usr/bin/env python3
"""Test the new ActionResult validation logic."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_action_result_validation():
    """Test the new validation rules for ActionResult."""
    try:
        from browser_use.agent.views import ActionResult

        # Test 1: success=True with is_done=True should be valid
        try:
            result1 = ActionResult(success=True, is_done=True)
            print("✅ Test 1 passed: success=True with is_done=True is valid")
        except Exception as e:
            print(f"❌ Test 1 failed: {e}")
            return False

        # Test 2: success=True with is_done=False should raise ValueError
        try:
            result2 = ActionResult(success=True, is_done=False)
            print("❌ Test 2 failed: should have raised ValueError for success=True with is_done=False")
            return False
        except ValueError as e:
            if "success=True can only be set when is_done=True" in str(e):
                print("✅ Test 2 passed: correctly raised ValueError for success=True with is_done=False")
            else:
                print(f"❌ Test 2 failed: wrong error message: {e}")
                return False
        except Exception as e:
            print(f"❌ Test 2 failed: unexpected exception: {e}")
            return False

        # Test 3: success=None with error should set success=False
        try:
            result3 = ActionResult(error="Some error")
            if result3.success is False:
                print("✅ Test 3 passed: success=None with error correctly sets success=False")
            else:
                print(f"❌ Test 3 failed: expected success=False, got {result3.success}")
                return False
        except Exception as e:
            print(f"❌ Test 3 failed: {e}")
            return False

        # Test 4: success=None without error should remain None (no auto-setting to True)
        try:
            result4 = ActionResult()
            if result4.success is None:
                print("✅ Test 4 passed: success=None without error remains None")
            else:
                print(f"❌ Test 4 failed: expected success=None, got {result4.success}")
                return False
        except Exception as e:
            print(f"❌ Test 4 failed: {e}")
            return False

        print("✅ All validation tests passed!")
        return True

    except Exception as e:
        print(f"❌ Import/setup failed: {e}")
        return False

if __name__ == "__main__":
    success = test_action_result_validation()
    sys.exit(0 if success else 1)
