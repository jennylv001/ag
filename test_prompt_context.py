#!/usr/bin/env python3
"""Quick test script to verify the PromptContext implementation."""

import asyncio
from agent.state import AgentState, StateManager, PromptContext

async def test_prompt_context():
    # Create a StateManager with a simple test state
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

    # Test the new build_prompt_context method
    pc = await state_manager.build_prompt_context()

    print(f"✅ build_prompt_context successful: {type(pc).__name__}")
    print(f"  current_goal: {pc.current_goal!r}")
    print(f"  last_error: {pc.last_error!r}")
    print(f"  current_task_id: {pc.current_task_id!r}")
    print(f"  task_context: {pc.task_context!r}")
    print(f"  agent_history_list: {type(pc.agent_history_list).__name__}")

    # Verify it's frozen (immutable)
    try:
        pc.current_goal = "should fail"
        print("❌ PromptContext should be frozen!")
    except AttributeError:
        print("✅ PromptContext is properly frozen (immutable)")

    return pc

if __name__ == "__main__":
    pc = asyncio.run(test_prompt_context())
