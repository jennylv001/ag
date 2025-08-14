#!/usr/bin/env python3
"""
Simple test runner for TaskMonitor functionality.
"""

import asyncio
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.task_monitor import TaskMonitor


async def test_basic_functionality():
    """Test basic TaskMonitor functionality."""
    print("Testing basic TaskMonitor functionality...")
    
    monitor = TaskMonitor(max_restart_attempts=3, base_backoff=0.1, max_backoff=1.0)
    
    # Test 1: Basic registration
    async def sample_task():
        await asyncio.sleep(0.1)
        
    task = asyncio.create_task(sample_task())
    monitor.register("test_component", task)
    
    assert "test_component" in monitor._tasks
    print("✓ Task registration works")
    
    # Test 2: Factory setup
    def task_factory():
        return asyncio.create_task(sample_task())
    
    monitor.set_restart_factories({"test_component": task_factory})
    monitor.enable(["test_component"])
    
    assert "test_component" in monitor._enabled_components
    print("✓ Factory setup and enable works")
    
    # Test 3: Stats
    stats = monitor.get_component_stats()
    assert "test_component" in stats
    print("✓ Stats retrieval works")
    
    # Test 4: Close
    await monitor.close()
    assert monitor._closed
    print("✓ Clean close works")
    
    print("All basic tests passed!")


async def test_restart_functionality():
    """Test restart functionality with a failing task."""
    print("\nTesting restart functionality...")
    
    monitor = TaskMonitor(max_restart_attempts=3, base_backoff=0.1, max_backoff=1.0)
    
    failure_count = 0
    
    async def failing_task():
        nonlocal failure_count
        failure_count += 1
        if failure_count <= 2:
            raise RuntimeError(f"Simulated failure {failure_count}")
        await asyncio.sleep(0.1)  # Success on third attempt
    
    def task_factory():
        return asyncio.create_task(failing_task())
    
    # Set up restart monitoring
    monitor.set_restart_factories({"failing_component": task_factory})
    monitor.enable(["failing_component"])
    
    # Register failing task
    initial_task = task_factory()
    monitor.register("failing_component", initial_task)
    
    # Wait for failures and restarts
    await asyncio.sleep(0.8)
    
    # Check that restarts occurred
    assert monitor._restart_counts["failing_component"] >= 1
    assert failure_count >= 2
    print("✓ Restart functionality works")
    
    await monitor.close()
    print("Restart test passed!")


async def test_io_semaphore():
    """Test io_semaphore functionality."""
    print("\nTesting io_semaphore functionality...")
    
    from concurrency.io import io_semaphore, get_io_semaphore_stats, set_io_semaphore_count
    
    # Test basic semaphore usage
    async with io_semaphore():
        print("✓ io_semaphore context manager works")
    
    # Test stats
    stats = get_io_semaphore_stats()
    assert stats['initialized'] == True
    print("✓ io_semaphore stats work")
    
    # Test count setting
    set_io_semaphore_count(5)
    stats = get_io_semaphore_stats()
    assert stats['count'] == 5
    print("✓ io_semaphore count setting works")
    
    print("io_semaphore test passed!")


async def main():
    """Run all tests."""
    print("Running TaskMonitor and io_semaphore tests...\n")
    
    try:
        await test_basic_functionality()
        await test_restart_functionality()
        await test_io_semaphore()
        print("\n🎉 All tests passed!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)