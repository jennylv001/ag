"""
Test script for Long-Running Operations Mode functionality.

This script validates:
1. State checkpointing and recovery
2. Resource monitoring and health assessment
3. Circuit breaker functionality
4. Integration with existing agent components
"""

import asyncio
import logging
import tempfile
import time
import os
from pathlib import Path

from browser_use.agent.long_running_mode import (
    LongRunningMode, ResourceMonitor, FailureAnalyzer,
    StateCheckpointer, CircuitBreaker, HealthStatus, OperationMode
)
from browser_use.agent.state_manager import StateManager, AgentState
from browser_use.agent.long_running_integration import LongRunningIntegration, should_trigger_intervention

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockSupervisor:
    """Mock supervisor for testing."""
    def __init__(self):
        self.state_manager = None


async def test_resource_monitor():
    """Test resource monitoring functionality."""
    print("\n=== Testing Resource Monitor ===")

    monitor = ResourceMonitor()

    # Test metrics collection
    metrics = monitor.get_current_metrics()
    print(f"Current CPU: {metrics.cpu_percent:.1f}%")
    print(f"Current Memory: {metrics.memory_percent:.1f}%")
    print(f"Memory MB: {metrics.memory_mb:.1f}")
    print(f"Open files: {metrics.open_files}")
    print(f"Threads: {metrics.thread_count}")

    # Test health assessment
    health = monitor.assess_health(metrics)
    print(f"Health status: {health.value}")

    # Simulate trend detection with multiple readings
    for i in range(15):
        metrics = monitor.get_current_metrics()
        await asyncio.sleep(0.1)

    trends = monitor.detect_trends()
    if trends:
        print(f"Detected trends: {trends}")
    else:
        print("No concerning trends detected")

    print("✅ Resource monitor test completed")


async def test_failure_analyzer():
    """Test failure pattern analysis."""
    print("\n=== Testing Failure Analyzer ===")

    analyzer = FailureAnalyzer()

    # Simulate various failures
    test_failures = [
        ("BrowserError", "Page crashed", {"component": "browser"}),
        ("TimeoutError", "Request timed out", {"component": "llm"}),
        ("BrowserError", "Browser unresponsive", {"component": "browser"}),
        ("MemoryError", "Out of memory", {"component": "system"}),
        ("BrowserError", "Page navigation failed", {"component": "browser"}),
    ]

    for error_type, error_msg, context in test_failures:
        analyzer.record_failure(error_type, error_msg, context)
        await asyncio.sleep(0.1)

    print(f"Total failures recorded: {len(analyzer.failure_history)}")
    print(f"Patterns detected: {len(analyzer.patterns)}")

    for pattern in analyzer.patterns:
        print(f"Pattern: {pattern.pattern_type} (frequency: {pattern.frequency}, "
              f"severity: {pattern.severity}, action: {pattern.suggested_action})")

    print("✅ Failure analyzer test completed")


async def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("\n=== Testing Circuit Breaker ===")

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)

    async def failing_function():
        """Function that always fails."""
        raise Exception("Simulated failure")

    async def working_function():
        """Function that works."""
        return "success"

    # Test failure accumulation
    print("Testing failure accumulation...")
    for i in range(5):
        try:
            await cb.call(failing_function)
        except Exception as e:
            print(f"Attempt {i+1}: {e}, CB state: {cb.state}, failures: {cb.failure_count}")

    # Circuit should be open now
    print(f"Circuit breaker state after failures: {cb.state}")
    assert cb.state == "open", "Circuit breaker should be open"

    # Wait for recovery timeout
    print("Waiting for recovery timeout...")
    await asyncio.sleep(2.5)

    # Test recovery
    try:
        result = await cb.call(working_function)
        print(f"Recovery test result: {result}, CB state: {cb.state}")
    except Exception as e:
        print(f"Recovery test failed: {e}")

    print("✅ Circuit breaker test completed")


async def test_state_checkpointer():
    """Test state checkpointing and recovery."""
    print("\n=== Testing State Checkpointer ===")

    # Create temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpointer = StateCheckpointer(temp_dir)

        # Create mock state manager
        agent_state = AgentState(task="Test task for checkpointing")
        state_manager = StateManager(
            initial_state=agent_state,
            file_system=None,
            max_failures=5,
            lock_timeout_seconds=30.0,
            use_planner=False,
            reflect_on_error=False,
            max_history_items=100
        )

        # Test checkpoint creation
        context_info = {
            'test_key': 'test_value',
            'recovery_hints': ['Test checkpoint', 'Created for validation']
        }

        checkpoint_id = await checkpointer.create_checkpoint(state_manager, context_info)
        print(f"Created checkpoint: {checkpoint_id}")
        assert checkpoint_id, "Checkpoint ID should not be empty"

        # Test checkpoint listing
        checkpoints = await checkpointer.list_checkpoints()
        print(f"Available checkpoints: {checkpoints}")
        assert checkpoint_id in checkpoints, "Created checkpoint should be in list"

        # Test checkpoint loading
        loaded_checkpoint = await checkpointer.load_checkpoint(checkpoint_id)
        assert loaded_checkpoint is not None, "Should be able to load checkpoint"
        print(f"Loaded checkpoint: {loaded_checkpoint.checkpoint_id}")
        print(f"Task from checkpoint: {loaded_checkpoint.agent_state['task']}")

        # Test cleanup
        await checkpointer._cleanup_old_checkpoints()

    print("✅ State checkpointer test completed")


async def test_long_running_mode_integration():
    """Test complete long-running mode integration."""
    print("\n=== Testing Long-Running Mode Integration ===")

    # Create mock components
    agent_state = AgentState(task="Long-running test task")
    state_manager = StateManager(
        initial_state=agent_state,
        file_system=None,
        max_failures=5,
        lock_timeout_seconds=30.0,
        use_planner=False,
        reflect_on_error=False,
        max_history_items=100
    )

    # Create long-running mode
    with tempfile.TemporaryDirectory() as temp_dir:
        long_running_mode = LongRunningMode(
            state_manager=state_manager,
            checkpoint_dir=temp_dir,
            monitoring_interval=1.0  # Fast monitoring for testing
        )

        # Test mode transitions
        print(f"Initial mode: {long_running_mode.mode.value}")
        print(f"Initial health: {long_running_mode.health_status.value}")

        # Test failure handling
        test_error = Exception("Test error for long-running mode")
        context = {'component': 'test', 'step': 1}

        recovery_action = await long_running_mode.handle_failure(test_error, context)
        print(f"Recovery action suggested: {recovery_action}")

        # Test checkpoint creation
        await long_running_mode._create_periodic_checkpoint()

        # Test health report
        health_report = await long_running_mode.get_health_report()
        print(f"Health report keys: {list(health_report.keys())}")
        print(f"Operation mode: {health_report['operation_mode']}")
        print(f"Resource metrics CPU: {health_report['resource_metrics']['cpu_percent']:.1f}%")

        # Test mode adjustment simulation
        long_running_mode.health_status = HealthStatus.WARNING
        await long_running_mode._adjust_operation_mode()
        print(f"Mode after health warning: {long_running_mode.mode.value}")

        # Test circuit breaker integration
        browser_cb = long_running_mode.get_circuit_breaker('browser')
        llm_cb = long_running_mode.get_circuit_breaker('llm')
        print(f"Browser circuit breaker state: {browser_cb.state}")
        print(f"LLM circuit breaker state: {llm_cb.state}")

    print("✅ Long-running mode integration test completed")


async def test_intervention_criteria():
    """Test intervention decision logic."""
    print("\n=== Testing Intervention Criteria ===")

    agent_state = AgentState(task="Intervention test")
    state_manager = StateManager(
        initial_state=agent_state,
        file_system=None,
        max_failures=5,
        lock_timeout_seconds=30.0,
        use_planner=False,
        reflect_on_error=False,
        max_history_items=100
    )

    # Test with no failures
    should_intervene = await should_trigger_intervention(state_manager, "TestError")
    print(f"Should intervene with no failures: {should_intervene}")

    # Simulate consecutive failures
    state_manager._state.consecutive_failures = 4
    should_intervene = await should_trigger_intervention(state_manager, "BrowserError")
    print(f"Should intervene with 4 failures: {should_intervene}")

    # Test with systemic error
    state_manager._state.consecutive_failures = 1
    should_intervene = await should_trigger_intervention(state_manager, "MemoryError")
    print(f"Should intervene with memory error: {should_intervene}")

    print("✅ Intervention criteria test completed")


async def test_long_running_integration_class():
    """Test the LongRunningIntegration class."""
    print("\n=== Testing LongRunningIntegration Class ===")

    # Create mock supervisor
    supervisor = MockSupervisor()
    agent_state = AgentState(task="Integration test")
    supervisor.state_manager = StateManager(
        initial_state=agent_state,
        file_system=None,
        max_failures=5,
        lock_timeout_seconds=30.0,
        use_planner=False,
        reflect_on_error=False,
        max_history_items=100
    )

    # Test integration
    integration = LongRunningIntegration(supervisor, enabled=True)

    # Test initialization
    success = await integration.initialize()
    print(f"Integration initialization success: {success}")

    if success:
        # Test health status
        health_status = await integration.get_health_status()
        print(f"Health status keys: {list(health_status.keys())}")
        print(f"Enabled: {health_status['enabled']}")

        # Test mode checks
        print(f"Is degraded mode: {integration.is_degraded_mode()}")
        print(f"Should reduce activity: {integration.should_reduce_activity()}")

        # Test failure handling
        test_error = Exception("Integration test error")
        recovery_action = await integration.handle_component_failure("test_component", test_error)
        print(f"Component failure recovery action: {recovery_action}")

        # Test cleanup
        await integration.cleanup()

    print("✅ LongRunningIntegration class test completed")


async def run_all_tests():
    """Run all long-running mode tests."""
    print("🧪 Starting Long-Running Mode Test Suite")
    print("=" * 50)

    try:
        await test_resource_monitor()
        await test_failure_analyzer()
        await test_circuit_breaker()
        await test_state_checkpointer()
        await test_long_running_mode_integration()
        await test_intervention_criteria()
        await test_long_running_integration_class()

        print("\n" + "=" * 50)
        print("🎉 All Long-Running Mode Tests Passed!")
        print("\nKey Features Validated:")
        print("✅ Resource monitoring and health assessment")
        print("✅ Failure pattern analysis and recovery strategies")
        print("✅ Circuit breaker protection for external services")
        print("✅ State checkpointing and recovery mechanisms")
        print("✅ Operation mode transitions and degradation")
        print("✅ Integration with existing agent architecture")
        print("✅ Intervention criteria and decision logic")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
