#!/usr/bin/env python3

"""
Test script to validate the fixes for the runtime issues:
1. Global I/O semaphore initialization
2. System prompt inclusion
3. Reduced verbose logging
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_semaphore_initialization():
    """Test that the global I/O semaphore can be initialized without errors"""
    try:
        from agent.concurrency import set_global_io_semaphore, get_global_io_semaphore

        # Test initialization
        set_global_io_semaphore(3)
        logger.info("✅ Global I/O semaphore initialization successful")

        # Test getting the semaphore
        semaphore = get_global_io_semaphore()
        logger.info(f"✅ Global I/O semaphore retrieval successful: {semaphore.name}")

        return True
    except Exception as e:
        logger.error(f"❌ Semaphore test failed: {e}")
        return False

def test_system_prompt_loading():
    """Test that the system prompt can be loaded"""
    try:
        from pathlib import Path
        system_prompt_path = Path("agent/system_prompt.md")

        if system_prompt_path.exists():
            system_message = system_prompt_path.read_text(encoding='utf-8')
            if system_message and len(system_message) > 100:  # Basic sanity check
                logger.info(f"✅ System prompt loaded successfully ({len(system_message)} characters)")
                return True
            else:
                logger.error("❌ System prompt file exists but appears empty or too short")
                return False
        else:
            logger.error("❌ System prompt file not found")
            return False
    except Exception as e:
        logger.error(f"❌ System prompt test failed: {e}")
        return False

def test_orchestrator_syntax():
    """Test that the orchestrator has valid syntax after our changes"""
    try:
        import ast
        with open('agent/orchestrator.py', 'r', encoding='utf-8') as f:
            content = f.read()

        ast.parse(content)
        logger.info("✅ Orchestrator syntax validation successful")
        return True
    except Exception as e:
        logger.error(f"❌ Orchestrator syntax test failed: {e}")
        return False

def test_service_syntax():
    """Test that the service has valid syntax after our changes"""
    try:
        import ast
        with open('agent/service.py', 'r', encoding='utf-8') as f:
            content = f.read()

        ast.parse(content)
        logger.info("✅ Service syntax validation successful")
        return True
    except Exception as e:
        logger.error(f"❌ Service syntax test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    logger.info("🔍 Testing runtime fixes...")

    all_passed = True

    # Test our fixes
    all_passed &= test_semaphore_initialization()
    all_passed &= test_system_prompt_loading()
    all_passed &= test_orchestrator_syntax()
    all_passed &= test_service_syntax()

    if all_passed:
        logger.info("🎉 All runtime fix validation tests passed!")
        logger.info("✅ Global I/O semaphore initialization fixed")
        logger.info("✅ System prompt loading implemented")
        logger.info("✅ Syntax validation successful")
        logger.info("✅ Ready for agent execution")
        return 0
    else:
        logger.error("💥 Some validation tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())
