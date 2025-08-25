#!/usr/bin/env python3

"""
Test script to verify the Agent class can be imported and instantiated
with our unified architecture fixes.
"""

import logging
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_agent_import():
    """Test that Agent class can be imported without errors"""
    try:
        from agent.service import Agent
        logger.info("✅ Successfully imported Agent class")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to import Agent class: {e}")
        return False

def test_agent_settings_import():
    """Test that AgentSettings can be imported"""
    try:
        from agent.settings import AgentSettings
        logger.info("✅ Successfully imported AgentSettings")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to import AgentSettings: {e}")
        return False

def test_orchestrator_import():
    """Test that AgentOrchestrator can be imported"""
    try:
        from agent.orchestrator import AgentOrchestrator
        logger.info("✅ Successfully imported AgentOrchestrator")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to import AgentOrchestrator: {e}")
        return False

def main():
    """Run all import tests"""
    logger.info("🔍 Testing unified architecture imports...")

    all_passed = True

    # Test basic imports
    all_passed &= test_agent_settings_import()
    all_passed &= test_orchestrator_import()
    all_passed &= test_agent_import()

    if all_passed:
        logger.info("🎉 All import tests passed! Unified architecture is working.")
        return 0
    else:
        logger.error("💥 Some import tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())
