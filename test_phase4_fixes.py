#!/usr/bin/env python3

"""
Test to verify that our Phase 4 cleanup fixes work correctly.
This test focuses on the specific imports that were broken.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_no_dead_imports():
    """Test that there are no imports to deleted modules"""
    dead_modules = ['agent.loop', 'agent.events', 'agent.decision_maker', 'agent.actuator', 'agent.supervisor']

    # Test orchestrator.py
    try:
        with open('agent/orchestrator.py', 'r', encoding='utf-8') as f:
            orchestrator_content = f.read()

        for module in dead_modules:
            if f'from {module}' in orchestrator_content or f'import {module}' in orchestrator_content:
                logger.error(f"❌ Found import of deleted module {module} in orchestrator.py")
                return False

        logger.info("✅ orchestrator.py has no dead imports")
    except Exception as e:
        logger.error(f"❌ Error checking orchestrator.py: {e}")
        return False

    # Test service.py
    try:
        with open('agent/service.py', 'r', encoding='utf-8') as f:
            service_content = f.read()

        for module in dead_modules:
            if f'from {module}' in service_content or f'import {module}' in service_content:
                logger.error(f"❌ Found import of deleted module {module} in service.py")
                return False

        logger.info("✅ service.py has no dead imports")
    except Exception as e:
        logger.error(f"❌ Error checking service.py: {e}")
        return False

    return True

def test_semantic_page_capture_class_exists():
    """Test that _SemanticPageCapture class exists in orchestrator"""
    try:
        with open('agent/orchestrator.py', 'r', encoding='utf-8') as f:
            content = f.read()

        if 'class _SemanticPageCapture:' in content:
            logger.info("✅ _SemanticPageCapture class found in orchestrator.py")
            return True
        else:
            logger.error("❌ _SemanticPageCapture class not found in orchestrator.py")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking _SemanticPageCapture: {e}")
        return False

def test_deleted_files_actually_deleted():
    """Test that the files we were supposed to delete are actually gone"""
    deleted_files = ['agent/loop.py', 'agent/events.py', 'agent/decision_maker.py', 'agent/actuator.py']

    for file_path in deleted_files:
        if Path(file_path).exists():
            logger.error(f"❌ File {file_path} still exists but should be deleted")
            return False
        else:
            logger.info(f"✅ File {file_path} properly deleted")

    return True

def test_unified_architecture_files_exist():
    """Test that the core unified architecture files exist"""
    core_files = ['agent/orchestrator.py', 'agent/state.py', 'agent/service.py', 'agent/step_summary.py']

    for file_path in core_files:
        if Path(file_path).exists():
            logger.info(f"✅ Core file {file_path} exists")
        else:
            logger.error(f"❌ Core file {file_path} missing")
            return False

    return True

def main():
    """Run all validation tests"""
    logger.info("🔍 Testing Phase 4 cleanup fixes...")

    all_passed = True

    # Test our specific fixes
    all_passed &= test_deleted_files_actually_deleted()
    all_passed &= test_unified_architecture_files_exist()
    all_passed &= test_no_dead_imports()
    all_passed &= test_semantic_page_capture_class_exists()

    if all_passed:
        logger.info("🎉 All Phase 4 cleanup validation tests passed!")
        logger.info("✅ No dead imports to deleted modules")
        logger.info("✅ _SemanticPageCapture implemented locally")
        logger.info("✅ Unified architecture files in place")
        return 0
    else:
        logger.error("💥 Some validation tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())
