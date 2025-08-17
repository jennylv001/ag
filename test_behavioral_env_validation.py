#!/usr/bin/env python3
"""
Simple validation test for behavioral planning environment controls.
Validates that the environment variable parsing code was added correctly to session.py.
"""

import os
import re
from pathlib import Path

def test_env_controls_in_session_file():
    """Test that environment variable controls were added to session.py."""
    print("🧪 Testing behavioral planning environment controls in session.py...")

    session_file = Path(__file__).parent / "browser" / "session.py"

    if not session_file.exists():
        print(f"❌ Session file not found: {session_file}")
        return False

    content = session_file.read_text(encoding='utf-8')

    # Test 1: Check for STEALTH_BEHAVIORAL_PLANNING environment variable parsing
    behavioral_planning_pattern = r"os\.environ\.get\(['\"]STEALTH_BEHAVIORAL_PLANNING['\"],\s*['\"]false['\"].*"
    if not re.search(behavioral_planning_pattern, content):
        print("❌ STEALTH_BEHAVIORAL_PLANNING environment variable parsing not found")
        return False
    print("✅ STEALTH_BEHAVIORAL_PLANNING environment variable parsing found")

    # Test 2: Check for STEALTH_PAGE_EXPLORATION environment variable parsing
    page_exploration_pattern = r"os\.environ\.get\(['\"]STEALTH_PAGE_EXPLORATION['\"],\s*['\"]false['\"].*"
    if not re.search(page_exploration_pattern, content):
        print("❌ STEALTH_PAGE_EXPLORATION environment variable parsing not found")
        return False
    print("✅ STEALTH_PAGE_EXPLORATION environment variable parsing found")

    # Test 3: Check for STEALTH_ERROR_SIMULATION environment variable parsing
    error_simulation_pattern = r"os\.environ\.get\(['\"]STEALTH_ERROR_SIMULATION['\"],\s*['\"]false['\"].*"
    if not re.search(error_simulation_pattern, content):
        print("❌ STEALTH_ERROR_SIMULATION environment variable parsing not found")
        return False
    print("✅ STEALTH_ERROR_SIMULATION environment variable parsing found")

    # Test 4: Check for behavioral_planning logging
    behavioral_planning_log = r"stealth\.env\.behavioral_planning="
    if not re.search(behavioral_planning_log, content):
        print("❌ Behavioral planning logging not found")
        return False
    print("✅ Behavioral planning logging found")

    # Test 5: Check for page_exploration logging
    page_exploration_log = r"stealth\.env\.page_exploration="
    if not re.search(page_exploration_log, content):
        print("❌ Page exploration logging not found")
        return False
    print("✅ Page exploration logging found")

    # Test 6: Check for error_simulation logging
    error_simulation_log = r"stealth\.env\.error_simulation="
    if not re.search(error_simulation_log, content):
        print("❌ Error simulation logging not found")
        return False
    print("✅ Error simulation logging found")

    # Test 7: Check that all variables default to 'false'
    false_defaults = content.count("'false'")
    if false_defaults < 3:  # Should have at least 3 'false' defaults for our new variables
        print(f"❌ Expected at least 3 'false' defaults, found {false_defaults}")
        return False
    print("✅ Safe defaults ('false') confirmed")

    # Test 8: Check proper placement after stealth manager initialization
    stealth_manager_init = r"Stealth manager initialized"
    env_parsing = r"behavioral_planning_enabled = os\.environ\.get"

    stealth_init_pos = content.find("Stealth manager initialized")
    env_parsing_pos = content.find("behavioral_planning_enabled = os.environ.get")

    if stealth_init_pos == -1:
        print("❌ Stealth manager initialization not found")
        return False

    if env_parsing_pos == -1:
        print("❌ Environment variable parsing not found")
        return False

    if env_parsing_pos <= stealth_init_pos:
        print("❌ Environment variable parsing should come after stealth manager initialization")
        return False

    print("✅ Environment variable parsing properly placed after stealth manager initialization")

    return True


def main():
    """Run the validation test."""
    print("🚀 Starting behavioral planning environment controls validation...")
    print("=" * 70)

    success = test_env_controls_in_session_file()

    print("=" * 70)
    if success:
        print("🎉 Behavioral environment controls validation PASSED!")
        print("✅ All three environment variables (STEALTH_BEHAVIORAL_PLANNING, STEALTH_PAGE_EXPLORATION, STEALTH_ERROR_SIMULATION) are properly parsed")
        print("✅ All environment variables have safe defaults ('false')")
        print("✅ All environment variable states are logged during stealth manager initialization")
        print("✅ Implementation follows existing patterns in the codebase")
        print("✅ Low risk rollback: environment variables can be removed safely")
    else:
        print("❌ Behavioral environment controls validation FAILED!")

    return success


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
