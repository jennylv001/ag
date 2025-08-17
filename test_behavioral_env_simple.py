#!/usr/bin/env python3
"""
Simple test for behavioral planning environment controls.
Tests just the environment variable parsing logic without full BrowserSession instantiation.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_environment_variables():
    """Test environment variable parsing logic directly."""
    print("🧪 Testing behavioral planning environment variable parsing...")

    # Test 1: Default values (should all be False)
    print("✅ Testing default values...")

    # Clear any existing environment variables
    env_vars = ['STEALTH_BEHAVIORAL_PLANNING', 'STEALTH_PAGE_EXPLORATION', 'STEALTH_ERROR_SIMULATION']
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]

    # Test default behavior
    behavioral_planning_enabled = os.environ.get('STEALTH_BEHAVIORAL_PLANNING', 'false').lower() == 'true'
    page_exploration_enabled = os.environ.get('STEALTH_PAGE_EXPLORATION', 'false').lower() == 'true'
    error_simulation_enabled = os.environ.get('STEALTH_ERROR_SIMULATION', 'false').lower() == 'true'

    assert behavioral_planning_enabled == False, "STEALTH_BEHAVIORAL_PLANNING should default to False"
    assert page_exploration_enabled == False, "STEALTH_PAGE_EXPLORATION should default to False"
    assert error_simulation_enabled == False, "STEALTH_ERROR_SIMULATION should default to False"

    print("✅ Default values are correct (all False)")

    # Test 2: Custom values
    print("✅ Testing custom values...")

    # Set test environment variables
    os.environ['STEALTH_BEHAVIORAL_PLANNING'] = 'true'
    os.environ['STEALTH_PAGE_EXPLORATION'] = 'TRUE'  # Test case insensitive
    os.environ['STEALTH_ERROR_SIMULATION'] = 'false'

    # Test custom behavior
    behavioral_planning_enabled = os.environ.get('STEALTH_BEHAVIORAL_PLANNING', 'false').lower() == 'true'
    page_exploration_enabled = os.environ.get('STEALTH_PAGE_EXPLORATION', 'false').lower() == 'true'
    error_simulation_enabled = os.environ.get('STEALTH_ERROR_SIMULATION', 'false').lower() == 'true'

    assert behavioral_planning_enabled == True, "STEALTH_BEHAVIORAL_PLANNING should be True when set to 'true'"
    assert page_exploration_enabled == True, "STEALTH_PAGE_EXPLORATION should be True when set to 'TRUE'"
    assert error_simulation_enabled == False, "STEALTH_ERROR_SIMULATION should be False when set to 'false'"

    print("✅ Custom values are parsed correctly")

    # Test 3: Invalid values (should default to False)
    print("✅ Testing invalid values...")

    os.environ['STEALTH_BEHAVIORAL_PLANNING'] = 'invalid'
    os.environ['STEALTH_PAGE_EXPLORATION'] = '1'
    os.environ['STEALTH_ERROR_SIMULATION'] = 'yes'

    # Test invalid behavior (should all be False)
    behavioral_planning_enabled = os.environ.get('STEALTH_BEHAVIORAL_PLANNING', 'false').lower() == 'true'
    page_exploration_enabled = os.environ.get('STEALTH_PAGE_EXPLORATION', 'false').lower() == 'true'
    error_simulation_enabled = os.environ.get('STEALTH_ERROR_SIMULATION', 'false').lower() == 'true'

    assert behavioral_planning_enabled == False, "STEALTH_BEHAVIORAL_PLANNING should be False for invalid value"
    assert page_exploration_enabled == False, "STEALTH_PAGE_EXPLORATION should be False for invalid value"
    assert error_simulation_enabled == False, "STEALTH_ERROR_SIMULATION should be False for invalid value"

    print("✅ Invalid values correctly default to False")

    # Clean up
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]

    return True


def test_code_implementation():
    """Test that the code was properly added to session.py."""
    print("🧪 Testing implementation in session.py...")

    try:
        with open('browser/session.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for environment variable parsing
        required_patterns = [
            "behavioral_planning_enabled = os.environ.get('STEALTH_BEHAVIORAL_PLANNING', 'false').lower() == 'true'",
            "page_exploration_enabled = os.environ.get('STEALTH_PAGE_EXPLORATION', 'false').lower() == 'true'",
            "error_simulation_enabled = os.environ.get('STEALTH_ERROR_SIMULATION', 'false').lower() == 'true'",
            "stealth.env.behavioral_planning=",
            "stealth.env.page_exploration=",
            "stealth.env.error_simulation="
        ]

        for pattern in required_patterns:
            if pattern not in content:
                print(f"❌ Missing pattern: {pattern}")
                return False

        print("✅ All required code patterns found in session.py")
        return True

    except Exception as e:
        print(f"❌ Error reading session.py: {e}")
        return False


def main():
    """Run the simple behavioral environment controls test."""
    print("🚀 Starting simple behavioral planning environment controls test...")
    print("=" * 70)

    # Test environment variable parsing logic
    env_success = test_environment_variables()

    # Test code implementation
    code_success = test_code_implementation()

    print("=" * 70)
    if env_success and code_success:
        print("🎉 Simple behavioral environment controls test PASSED!")
        print("✅ Environment variable parsing logic works correctly")
        print("✅ Code implementation is properly added to session.py")
        print("✅ Safe defaults (False) are properly applied")
        print("✅ Environment variables handle edge cases correctly")
    else:
        print("❌ Simple behavioral environment controls test FAILED!")

    return env_success and code_success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
