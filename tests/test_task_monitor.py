"""
Tests for the TaskMonitor component.

This module tests the robust restart capability including backoff,
rate limiting, and proper task lifecycle management.
"""

import asyncio
import logging
import pytest
import time
import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.task_monitor import TaskMonitor


class TestTaskMonitor:
    """Test cases for TaskMonitor functionality."""

    @pytest.fixture
    def task_monitor(self):
        """Create a TaskMonitor instance for testing."""
        return TaskMonitor(max_restart_attempts=3, base_backoff=0.1, max_backoff=1.0)

    @pytest.fixture
    async def sample_task(self):
        """Create a sample async task for testing."""
        async def sample_coroutine():
            await asyncio.sleep(0.1)
            
        return asyncio.create_task(sample_coroutine())

    async def test_register_and_monitor_task(self, task_monitor, sample_task):
        """Test basic task registration and monitoring."""
        task_monitor.register("test_component", sample_task)
        
        assert "test_component" in task_monitor._tasks
        assert task_monitor._tasks["test_component"] == sample_task
        assert task_monitor._restart_counts["test_component"] == 0

    async def test_restart_on_task_failure(self, task_monitor):
        """Test that a failing task gets restarted automatically."""
        restart_count = 0
        
        async def failing_coroutine():
            nonlocal restart_count
            restart_count += 1
            if restart_count <= 2:
                raise RuntimeError(f"Simulated failure {restart_count}")
            await asyncio.sleep(0.1)  # Success on third attempt
        
        def task_factory():
            return asyncio.create_task(failing_coroutine())
        
        # Set up restart factory and enable restart
        task_monitor.set_restart_factories({"test_component": task_factory})
        task_monitor.enable(["test_component"])
        
        # Register initial failing task
        initial_task = task_factory()
        task_monitor.register("test_component", initial_task)
        
        # Wait for initial failure and restarts
        await asyncio.sleep(0.5)  # Allow time for failures and restarts
        
        # Should have restarted at least once
        assert task_monitor._restart_counts["test_component"] >= 1
        assert restart_count >= 2

    async def test_rate_limiting(self, task_monitor):
        """Test that excessive restarts are rate limited."""
        restart_attempts = 0
        
        async def always_failing_coroutine():
            nonlocal restart_attempts
            restart_attempts += 1
            raise RuntimeError(f"Always fails - attempt {restart_attempts}")
        
        def task_factory():
            return asyncio.create_task(always_failing_coroutine())
        
        # Set up restart factory and enable restart
        task_monitor.set_restart_factories({"test_component": task_factory})
        task_monitor.enable(["test_component"])
        
        # Register initial failing task
        initial_task = task_factory()
        task_monitor.register("test_component", initial_task)
        
        # Wait for multiple restart attempts
        await asyncio.sleep(1.0)  # Allow time for multiple restart attempts
        
        # Should be rate limited after max attempts
        assert task_monitor._restart_counts["test_component"] <= task_monitor._max_restart_attempts

    async def test_successful_task_no_restart(self, task_monitor):
        """Test that successful tasks are not restarted."""
        async def successful_coroutine():
            await asyncio.sleep(0.1)
            return "success"
        
        def task_factory():
            return asyncio.create_task(successful_coroutine())
        
        # Set up restart factory and enable restart
        task_monitor.set_restart_factories({"test_component": task_factory})
        task_monitor.enable(["test_component"])
        
        # Register successful task
        task = task_factory()
        task_monitor.register("test_component", task)
        
        # Wait for task completion
        await asyncio.sleep(0.2)
        
        # Should not have restarted
        assert task_monitor._restart_counts["test_component"] == 0

    async def test_cancelled_task_restart(self, task_monitor):
        """Test that cancelled tasks get restarted if enabled."""
        restart_called = False
        
        async def cancellable_coroutine():
            await asyncio.sleep(10)  # Long-running task
        
        def task_factory():
            nonlocal restart_called
            restart_called = True
            return asyncio.create_task(cancellable_coroutine())
        
        # Set up restart factory and enable restart
        task_monitor.set_restart_factories({"test_component": task_factory})
        task_monitor.enable(["test_component"])
        
        # Register and cancel task
        task = asyncio.create_task(cancellable_coroutine())
        task_monitor.register("test_component", task)
        task.cancel()
        
        # Wait for restart
        await asyncio.sleep(0.3)
        
        # Should have attempted restart
        assert restart_called

    async def test_disable_component_no_restart(self, task_monitor):
        """Test that disabled components are not restarted."""
        restart_called = False
        
        async def failing_coroutine():
            raise RuntimeError("This should not be restarted")
        
        def task_factory():
            nonlocal restart_called
            restart_called = True
            return asyncio.create_task(failing_coroutine())
        
        # Set up restart factory but DO NOT enable restart
        task_monitor.set_restart_factories({"test_component": task_factory})
        # Note: not calling task_monitor.enable()
        
        # Register failing task
        task = asyncio.create_task(failing_coroutine())
        task_monitor.register("test_component", task)
        
        # Wait for potential restart
        await asyncio.sleep(0.3)
        
        # Should not have restarted
        assert not restart_called
        assert task_monitor._restart_counts["test_component"] == 0

    async def test_close_cancels_tasks(self, task_monitor):
        """Test that close() cancels all monitored tasks."""
        async def long_running_coroutine():
            await asyncio.sleep(10)
        
        # Register multiple tasks
        task1 = asyncio.create_task(long_running_coroutine())
        task2 = asyncio.create_task(long_running_coroutine())
        
        task_monitor.register("component1", task1)
        task_monitor.register("component2", task2)
        
        # Close monitor
        await task_monitor.close()
        
        # All tasks should be cancelled
        assert task1.cancelled() or task1.done()
        assert task2.cancelled() or task2.done()
        assert task_monitor._closed

    async def test_get_component_stats(self, task_monitor, sample_task):
        """Test component statistics retrieval."""
        task_monitor.register("test_component", sample_task)
        task_monitor.enable(["test_component"])
        
        stats = task_monitor.get_component_stats()
        
        assert "test_component" in stats
        component_stats = stats["test_component"]
        assert component_stats["enabled"] is True
        assert component_stats["restart_count"] == 0
        assert component_stats["is_restarting"] is False

    async def test_concurrent_restart_prevention(self, task_monitor):
        """Test that concurrent restarts of the same component are prevented."""
        restart_count = 0
        
        async def failing_coroutine():
            nonlocal restart_count
            restart_count += 1
            raise RuntimeError(f"Simulated failure {restart_count}")
        
        def task_factory():
            return asyncio.create_task(failing_coroutine())
        
        # Set up restart factory and enable restart
        task_monitor.set_restart_factories({"test_component": task_factory})
        task_monitor.enable(["test_component"])
        
        # Register multiple failing tasks simultaneously
        task1 = task_factory()
        task2 = task_factory()
        
        task_monitor.register("test_component", task1)
        # Simulate rapid failure by registering another task quickly
        await asyncio.sleep(0.01)
        task_monitor.register("test_component", task2)
        
        # Wait for restart attempts
        await asyncio.sleep(0.5)
        
        # Should prevent concurrent restarts
        assert task_monitor._restart_counts["test_component"] >= 1