#!/usr/bin/env python3
"""
Simple integration test for Supervisor with TaskMonitor.
"""

import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.supervisor import Supervisor
from agent.state_manager import AgentStatus


class MockSettings:
    """Mock AgentSettings for testing."""
    
    def __init__(self):
        self.task = "Test task"
        self.injected_agent_state = None
        self.file_system = None
        self.file_system_path = None
        self.max_failures = 3
        self.lock_timeout_seconds = 30.0
        self.use_planner = False
        self.reflect_on_error = False
        self.max_history_items = 10
        self.browser_session = None
        self.browser = MagicMock()
        self.browser_context = None
        self.browser_profile = None
        self.page = None
        self.controller = MagicMock()
        self.llm = MagicMock()
        self.planner_llm = None
        self.page_extraction_llm = None
        self.output_model = None
        self.on_run_start = None
        self.on_run_end = None
        self.on_step_start = None
        self.on_step_end = None
        self.generate_gif = False
        self.initial_actions = []
        self.max_steps = 10
        self.planner_interval = 5
        self.sensitive_data = []
        self.available_file_paths = []
        self.context = ""
        
        # Mock controller registry
        self.controller.registry = MagicMock()
        self.controller.registry.create_action_model.return_value = MagicMock()
        self.controller.registry.get_prompt_description.return_value = "Mock prompt"
        
    def parse_initial_actions(self, action_model):
        return []


async def test_supervisor_task_monitor_integration():
    """Test that Supervisor properly integrates with TaskMonitor."""
    print("Testing Supervisor TaskMonitor integration...")
    
    # Create mock settings
    settings = MockSettings()
    
    # Create supervisor - this should work without TaskGroup now
    supervisor = Supervisor(settings)
    
    # Verify supervisor was created with task monitor
    assert hasattr(supervisor, '_task_monitor')
    print("✓ Supervisor created with TaskMonitor attribute")
    
    # Mock the state manager to control the run loop
    supervisor.state_manager.get_status = AsyncMock()
    supervisor.state_manager.set_status = AsyncMock()
    supervisor.state_manager.state = MagicMock()
    supervisor.state_manager.state.agent_id = "test_agent"
    supervisor.state_manager.state.task = "Test task"
    supervisor.state_manager.state.n_steps = 0
    supervisor.state_manager.state.history = []
    
    # Mock the perception component
    supervisor.perception.run = AsyncMock()
    supervisor.perception.watchdog = AsyncMock()
    supervisor.perception._get_browser_state_with_recovery = AsyncMock()
    
    # Mock the decision maker
    supervisor.decision_maker.decide = AsyncMock()
    
    # Mock the actuator
    supervisor.actuator.execute = AsyncMock()
    
    # Mock browser session
    supervisor.browser_session = MagicMock()
    supervisor.browser_session.stop = AsyncMock()
    
    # Set up status sequence: start running, then stop
    supervisor.state_manager.get_status.side_effect = [
        AgentStatus.RUNNING,  # Initial check
        AgentStatus.STOPPED   # Stop the run loop
    ]
    
    try:
        # Run the supervisor
        result = await supervisor.run()
        
        # Verify basic functionality
        assert result == []  # Should return empty history
        print("✓ Supervisor run completed successfully")
        
        # Verify TaskMonitor was created during run
        assert supervisor._task_monitor is None  # Should be None after close
        print("✓ TaskMonitor was properly closed")
        
    except Exception as e:
        print(f"❌ Supervisor run failed: {e}")
        raise
    
    print("Supervisor integration test passed!")


async def main():
    """Run the integration test."""
    print("Running Supervisor integration test...\n")
    
    try:
        await test_supervisor_task_monitor_integration()
        print("\n🎉 Integration test passed!")
        return 0
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)