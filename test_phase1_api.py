#!/usr/bin/env python3
"""Test script to verify Phase 1 MessageManager API changes."""

import asyncio
import warnings
from unittest.mock import Mock

# Capture deprecation warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    from agent.state import AgentState, StateManager, PromptContext
    from agent.message_manager.service import MessageManager, MessageManagerSettings

    async def test_new_api():
        print("🧪 Testing Phase 1 MessageManager API changes...")

        # Create test objects
        state = AgentState('test task')
        state.current_goal = 'test goal'
        state.last_error = 'test error'

        state_manager = StateManager(
            initial_state=state,
            file_system=None,
            max_failures=3,
            lock_timeout_seconds=1.0,
            use_planner=True,
            reflect_on_error=True,
            max_history_items=50
        )

        # Mock MessageManager setup
        settings = MessageManagerSettings()

        # Create a minimal system message mock
        system_message = Mock()
        system_message.content = "System message"

        message_manager = MessageManager(
            task="test task",
            system_message=system_message,
            settings=settings
        )

        # Mock browser state
        browser_state = Mock()
        browser_state.url = "https://example.com"
        browser_state.title = "Test Page"
        browser_state.screenshot = None

        # Build prompt context
        pc = await state_manager.build_prompt_context()
        print(f"✅ PromptContext created: {type(pc).__name__}")

        # Test new API
        print("🔄 Testing new prepare_messages API...")
        try:
            messages = message_manager.prepare_messages(
                prompt_context=pc,
                browser_state=browser_state
            )
            print(f"✅ New API works: returned {len(messages)} messages")
        except Exception as e:
            print(f"❌ New API failed: {e}")
            return False

        # Test old API (should show deprecation warning)
        print("🔄 Testing deprecated prepare_messages_for_llm API...")
        try:
            messages_old = message_manager.prepare_messages_for_llm(
                browser_state=browser_state,
                current_goal=pc.current_goal,
                last_error=pc.last_error,
                agent_history_list=pc.agent_history_list,
                current_task_id=pc.current_task_id,
                task_context=pc.task_context,
            )
            print(f"✅ Old API still works: returned {len(messages_old)} messages")
        except Exception as e:
            print(f"❌ Old API failed: {e}")
            return False

        # Check deprecation warning was issued
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        if deprecation_warnings:
            print(f"✅ Deprecation warning properly issued: {deprecation_warnings[0].message}")
        else:
            print("⚠️  No deprecation warning detected")

        print("🎉 Phase 1 API changes validated successfully!")
        return True

    if __name__ == "__main__":
        success = asyncio.run(test_new_api())
        exit(0 if success else 1)
