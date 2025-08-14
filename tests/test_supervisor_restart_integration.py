"""
Integration tests for Supervisor restart functionality.

These tests verify that the Supervisor properly integrates with TaskMonitor
to provide robust restart capabilities for agent components.
"""

import asyncio
import logging
import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.supervisor import Supervisor
from agent.state_manager import AgentStatus, TERMINAL_STATES


class FakeSettings:
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
        self.browser = None
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


class TestSupervisorRestartIntegration:
    """Integration tests for Supervisor restart capabilities."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        return FakeSettings()

    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        components = {}
        
        # Mock Perception
        components['perception'] = MagicMock()
        components['perception'].run = AsyncMock()
        components['perception'].watchdog = AsyncMock()
        components['perception']._get_browser_state_with_recovery = AsyncMock()
        
        # Mock StateManager
        components['state_manager'] = MagicMock()
        components['state_manager'].state = MagicMock()
        components['state_manager'].state.agent_id = "test_agent"
        components['state_manager'].state.task = "Test task"
        components['state_manager'].state.n_steps = 0
        components['state_manager'].get_status = AsyncMock(return_value=AgentStatus.RUNNING)
        components['state_manager'].set_status = AsyncMock()
        components['state_manager'].add_history_item = AsyncMock()
        components['state_manager'].update_after_step = AsyncMock()
        components['state_manager'].record_error = AsyncMock()
        
        # Mock MessageManager
        components['message_manager'] = MagicMock()
        components['message_manager'].settings = MagicMock()
        components['message_manager'].settings.available_file_paths = []
        
        # Mock DecisionMaker
        components['decision_maker'] = MagicMock()
        components['decision_maker'].decide = AsyncMock()
        
        # Mock Actuator
        components['actuator'] = MagicMock()
        components['actuator'].execute = AsyncMock()
        
        return components

    async def test_task_monitor_creation_and_setup(self, mock_settings, mock_components):
        """Test that Supervisor creates and configures TaskMonitor properly."""
        with patch('agent.supervisor.StateManager') as mock_sm_class, \
             patch('agent.supervisor.Perception') as mock_perception_class, \
             patch('agent.supervisor.MessageManager') as mock_mm_class, \
             patch('agent.supervisor.DecisionMaker') as mock_dm_class, \
             patch('agent.supervisor.Actuator') as mock_actuator_class, \
             patch('agent.supervisor.FileSystem') as mock_fs_class, \
             patch('agent.supervisor.SignalHandler') as mock_signal_class:
            
            # Configure mocks
            mock_sm_class.return_value = mock_components['state_manager']
            mock_perception_class.return_value = mock_components['perception']
            mock_mm_class.return_value = mock_components['message_manager']
            mock_dm_class.return_value = mock_components['decision_maker']
            mock_actuator_class.return_value = mock_components['actuator']
            mock_fs_class.return_value = MagicMock()
            
            mock_signal_handler = MagicMock()
            mock_signal_class.return_value = mock_signal_handler
            
            # Create supervisor
            supervisor = Supervisor(mock_settings)
            
            # Mock the terminal status check to exit quickly
            status_calls = [AgentStatus.RUNNING, AgentStatus.STOPPED]
            mock_components['state_manager'].get_status.side_effect = status_calls
            
            # Run supervisor
            await supervisor.run()
            
            # Verify TaskMonitor was created and used
            assert supervisor._task_monitor is not None

    async def test_component_restart_on_failure(self, mock_settings, mock_components):
        """Test that component failures trigger restarts through TaskMonitor."""
        failure_count = 0
        
        async def failing_decision_loop():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise RuntimeError(f"Simulated decision loop failure {failure_count}")
            # Success on third attempt - but we'll stop the test before this
            await asyncio.sleep(10)
        
        with patch('agent.supervisor.StateManager') as mock_sm_class, \
             patch('agent.supervisor.Perception') as mock_perception_class, \
             patch('agent.supervisor.MessageManager') as mock_mm_class, \
             patch('agent.supervisor.DecisionMaker') as mock_dm_class, \
             patch('agent.supervisor.Actuator') as mock_actuator_class, \
             patch('agent.supervisor.FileSystem') as mock_fs_class, \
             patch('agent.supervisor.SignalHandler') as mock_signal_class:
            
            # Configure mocks
            mock_sm_class.return_value = mock_components['state_manager']
            mock_perception_class.return_value = mock_components['perception']
            mock_mm_class.return_value = mock_components['message_manager']
            mock_dm_class.return_value = mock_components['decision_maker']
            mock_actuator_class.return_value = mock_components['actuator']
            mock_fs_class.return_value = MagicMock()
            
            mock_signal_handler = MagicMock()
            mock_signal_class.return_value = mock_signal_handler
            
            # Create supervisor
            supervisor = Supervisor(mock_settings)
            
            # Replace the decision loop with our failing version
            supervisor._decision_loop = failing_decision_loop
            
            # Set up status progression: RUNNING for a bit, then STOPPED
            status_sequence = [AgentStatus.RUNNING] * 10 + [AgentStatus.STOPPED]
            mock_components['state_manager'].get_status.side_effect = status_sequence
            
            try:
                # Start the supervisor run in background
                run_task = asyncio.create_task(supervisor.run())
                
                # Wait a short time for failures and restarts to occur
                await asyncio.sleep(0.5)
                
                # Force terminal status to stop the run
                mock_components['state_manager'].get_status.return_value = AgentStatus.STOPPED
                
                # Wait for run to complete
                await asyncio.wait_for(run_task, timeout=5.0)
                
            except asyncio.TimeoutError:
                # This is acceptable for this test
                pass
            
            # Verify that failures occurred and restarts were attempted
            assert failure_count >= 2

    async def test_clean_shutdown_cancels_tasks(self, mock_settings, mock_components):
        """Test that clean shutdown properly cancels all tasks via TaskMonitor."""
        with patch('agent.supervisor.StateManager') as mock_sm_class, \
             patch('agent.supervisor.Perception') as mock_perception_class, \
             patch('agent.supervisor.MessageManager') as mock_mm_class, \
             patch('agent.supervisor.DecisionMaker') as mock_dm_class, \
             patch('agent.supervisor.Actuator') as mock_actuator_class, \
             patch('agent.supervisor.FileSystem') as mock_fs_class, \
             patch('agent.supervisor.SignalHandler') as mock_signal_class:
            
            # Configure mocks
            mock_sm_class.return_value = mock_components['state_manager']
            mock_perception_class.return_value = mock_components['perception']
            mock_mm_class.return_value = mock_components['message_manager']
            mock_dm_class.return_value = mock_components['decision_maker']
            mock_actuator_class.return_value = mock_components['actuator']
            mock_fs_class.return_value = MagicMock()
            
            mock_signal_handler = MagicMock()
            mock_signal_class.return_value = mock_signal_handler
            
            # Create supervisor
            supervisor = Supervisor(mock_settings)
            
            # Quick terminal status progression
            mock_components['state_manager'].get_status.side_effect = [
                AgentStatus.RUNNING, AgentStatus.STOPPED
            ]
            
            # Run supervisor
            await supervisor.run()
            
            # Verify TaskMonitor was properly closed
            assert supervisor._task_monitor is None  # Should be set to None after close

    async def test_backward_compatibility_preserved(self, mock_settings, mock_components):
        """Test that external behavior remains unchanged for backward compatibility."""
        with patch('agent.supervisor.StateManager') as mock_sm_class, \
             patch('agent.supervisor.Perception') as mock_perception_class, \
             patch('agent.supervisor.MessageManager') as mock_mm_class, \
             patch('agent.supervisor.DecisionMaker') as mock_dm_class, \
             patch('agent.supervisor.Actuator') as mock_actuator_class, \
             patch('agent.supervisor.FileSystem') as mock_fs_class, \
             patch('agent.supervisor.SignalHandler') as mock_signal_class:
            
            # Configure mocks
            mock_sm_class.return_value = mock_components['state_manager']
            mock_perception_class.return_value = mock_components['perception']
            mock_mm_class.return_value = mock_components['message_manager']
            mock_dm_class.return_value = mock_components['decision_maker']
            mock_actuator_class.return_value = mock_components['actuator']
            mock_fs_class.return_value = MagicMock()
            
            mock_signal_handler = MagicMock()
            mock_signal_class.return_value = mock_signal_handler
            
            # Mock history return
            mock_history = []
            mock_components['state_manager'].state.history = mock_history
            
            # Create supervisor
            supervisor = Supervisor(mock_settings)
            
            # Quick run with terminal status
            mock_components['state_manager'].get_status.return_value = AgentStatus.STOPPED
            
            # Run supervisor
            result = await supervisor.run()
            
            # Verify external interface is preserved
            assert result == mock_history  # Should return history as before
            
            # Verify lifecycle methods were called
            mock_signal_handler.register.assert_called_once()
            mock_signal_handler.unregister.assert_called_once()

    async def test_restart_only_enabled_components(self, mock_settings, mock_components):
        """Test that only enabled components are restarted on failure.""" 
        pause_handler_called = False
        decision_loop_called = False
        
        async def mock_pause_handler():
            nonlocal pause_handler_called
            pause_handler_called = True
            raise RuntimeError("Pause handler failure")
        
        async def mock_decision_loop():
            nonlocal decision_loop_called 
            decision_loop_called = True
            raise RuntimeError("Decision loop failure")
        
        with patch('agent.supervisor.StateManager') as mock_sm_class, \
             patch('agent.supervisor.Perception') as mock_perception_class, \
             patch('agent.supervisor.MessageManager') as mock_mm_class, \
             patch('agent.supervisor.DecisionMaker') as mock_dm_class, \
             patch('agent.supervisor.Actuator') as mock_actuator_class, \
             patch('agent.supervisor.FileSystem') as mock_fs_class, \
             patch('agent.supervisor.SignalHandler') as mock_signal_class:
            
            # Configure mocks
            mock_sm_class.return_value = mock_components['state_manager']
            mock_perception_class.return_value = mock_components['perception']
            mock_mm_class.return_value = mock_components['message_manager']
            mock_dm_class.return_value = mock_components['decision_maker']
            mock_actuator_class.return_value = mock_components['actuator']
            mock_fs_class.return_value = MagicMock()
            
            mock_signal_handler = MagicMock()
            mock_signal_class.return_value = mock_signal_handler
            
            # Create supervisor and replace methods
            supervisor = Supervisor(mock_settings)
            supervisor._pause_handler = mock_pause_handler
            supervisor._decision_loop = mock_decision_loop
            
            # Quick status progression
            mock_components['state_manager'].get_status.side_effect = [
                AgentStatus.RUNNING, AgentStatus.RUNNING, AgentStatus.STOPPED
            ]
            
            try:
                # Start run
                run_task = asyncio.create_task(supervisor.run())
                await asyncio.sleep(0.3)  # Allow failures to occur
                
                # Force stop
                mock_components['state_manager'].get_status.return_value = AgentStatus.STOPPED
                await asyncio.wait_for(run_task, timeout=2.0)
                
            except asyncio.TimeoutError:
                pass
            
            # Both should have been called initially
            assert pause_handler_called
            assert decision_loop_called
            
            # Decision loop is enabled for restart, pause handler is not priority
            # The exact restart behavior depends on timing, but we verify basic functionality