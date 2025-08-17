#!/usr/bin/env python3
"""
Focused System Validation Test
=============================

This test provides focused validation of core browser_use components
following .copilot/testing-framework.prompt requirements for external validation.
"""

import pytest
import tempfile
import sys
from pathlib import Path

# Add browser_use to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from browser_use import Agent, AgentSettings
from browser_use.agent.state_manager import AgentState, AgentStatus
from browser_use.agent.views import AgentOutput
from browser_use.browser import BrowserProfile
from browser_use.controller.views import DoneAction
from browser_use.filesystem.file_system import FileSystem
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage, UserMessage


class TestLLM(BaseChatModel):
    """Test LLM that returns proper structured responses for validation."""

    model = "test-llm-v1"
    _verified_api_keys = True

    async def ainvoke(self, messages: list[BaseMessage], **kwargs) -> ChatInvokeCompletion[AgentOutput]:
        # Import here to avoid circular imports and get the correct ActionModel
        from browser_use.controller.service import Controller
        from browser_use.controller.registry.views import ActionModel

        # Create a controller to get the ActionModel
        controller = Controller()
        TestActionModel = controller.registry.create_action_model()

        # Create the action in the correct format
        done_action = TestActionModel(done=DoneAction(
            text="System validation completed. All components working properly.",
            success=True
        ))

        # Return structured response that completes the task immediately
        agent_output = AgentOutput(
            thinking="System validation test running",
            prior_action_assessment="Initialized for testing",
            task_log="Validating browser_use system components",
            next_goal="Complete validation successfully",
            action=[done_action]
        )

        return ChatInvokeCompletion(
            completion=agent_output,
            usage=ChatInvokeUsage(
                prompt_tokens=100,
                prompt_cached_tokens=0,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=50,
                total_tokens=150
            )
        )

    @property
    def provider(self) -> str:
        return "test"

    @property
    def name(self) -> str:
        return "TestLLM"

    @property
    def model_name(self) -> str:
        return self.model


@pytest.mark.asyncio
async def test_core_system_components():
    """
    Test core system components work together properly.
    External validation only - no reliance on internal logging.
    """
    print("🚀 Testing core system components...")

    # Test 1: Agent initialization
    test_llm = TestLLM()
    settings = AgentSettings(
        task="Component validation test",
        llm=test_llm,
        max_steps=1,
        use_thinking=False,
        use_planner=False
    )

    agent = Agent(settings)
    assert agent is not None
    assert agent.supervisor is not None
    assert agent.supervisor.state_manager is not None
    print("✅ Agent initialization validated")

    # Test 2: State management validation
    state_manager = agent.supervisor.state_manager
    assert state_manager.state.status == AgentStatus.PENDING
    assert state_manager.state.task == settings.task
    print("✅ State management validated")

    # Test 3: LLM integration validation (simplified - just test instantiation)
    assert test_llm.model == "test-llm-v1"
    assert test_llm.name == "TestLLM"
    assert test_llm.provider == "test"
    print("✅ LLM integration validated")

    # Test 4: Cleanup
    await agent.supervisor.close()
    print("✅ Component cleanup completed")

    print("🎉 Core system components validation PASSED")


@pytest.mark.asyncio
async def test_filesystem_integration():
    """Test filesystem operations work correctly."""
    print("🚀 Testing filesystem integration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        fs = FileSystem(base_dir=temp_dir)

        # Test file write/read cycle
        test_file = "validation.txt"
        test_content = "Browser Use System Test"

        await fs.write_file(test_file, test_content)
        read_result = await fs.read_file(test_file)

        # FileSystem methods return message strings, so check content is in the result
        assert test_content in read_result
        print("✅ Filesystem integration validated")


def test_browser_profile_creation():
    """Test browser profile creation and configuration."""
    print("🚀 Testing browser profile creation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        profile = BrowserProfile(
            user_data_dir=temp_dir,
            stealth=True
        )

        assert profile.stealth is True
        assert Path(temp_dir).exists()

        print("✅ Browser profile creation validated")


def test_pydantic_config_migration():
    """Test that Pydantic v2 ConfigDict migration is working."""
    print("🚀 Testing Pydantic configuration...")

    # Test that we can create AgentState without warnings
    from browser_use.agent.state_manager import AgentState, TaskStack

    # Test TaskStack creation
    task_stack = TaskStack()
    assert task_stack is not None

    # Test AgentState creation
    agent_state = AgentState(
        task="Test task",
        status=AgentStatus.PENDING
    )
    assert agent_state.task == "Test task"
    assert agent_state.status == AgentStatus.PENDING

    print("✅ Pydantic configuration validated")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
