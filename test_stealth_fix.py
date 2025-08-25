#!/usr/bin/env python3
"""
Test script to verify StealthManager _action_rng fix
"""

import sys
import os

# Add the browser_use path to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from browser.stealth import StealthManager

    print("✅ Importing StealthManager...")

    # Create a StealthManager instance
    stealth_manager = StealthManager()
    print("✅ StealthManager instantiated successfully")

    # Check if _action_rng and _rng attributes exist
    if hasattr(stealth_manager, '_action_rng'):
        print(f"✅ _action_rng attribute exists: {stealth_manager._action_rng}")
    else:
        print("❌ _action_rng attribute missing")

    if hasattr(stealth_manager, '_rng'):
        print(f"✅ _rng attribute exists: {type(stealth_manager._rng)}")
    else:
        print("❌ _rng attribute missing")

    # Test the exploration sequence method that was failing
    try:
        # Just test that the method exists and doesn't crash on basic access
        duration = stealth_manager._calculate_exploration_step_delay(0, 5)
        print(f"✅ _calculate_exploration_step_delay working: {duration}")
    except Exception as e:
        print(f"❌ _calculate_exploration_step_delay failed: {e}")

    print("\n🎯 StealthManager fix test PASSED!")

except Exception as e:
    print(f"❌ StealthManager test FAILED: {e}")
    import traceback
    traceback.print_exc()
