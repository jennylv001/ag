"""Unit tests for semantic page perception functionality in agent/state.py"""

import pytest
from unittest.mock import AsyncMock, patch
from agent.state import AgentState, StateManager
from browser_use.browser.views import BrowserStateSummary


class TestSemanticPagePerception:
    """Test semantic page state management and prompt context integration."""

    @pytest.mark.asyncio
    async def test_semantic_page_state_update_and_retrieval(self):
        """Test updating and retrieving semantic page state."""
        # Create a StateManager with semantic page tracking
        initial_state = AgentState(task="Test semantic page tracking")
        state_manager = StateManager(
            initial_state=initial_state,
            file_system=None,
            max_failures=3,
            lock_timeout_seconds=10.0,
            use_planner=False,
            reflect_on_error=False,
            max_history_items=100
        )

        # Test initial state - no semantic data
        semantic_state = await state_manager.get_semantic_page_state()
        assert semantic_state is None

        # Update semantic page state
        test_hash = "page_hash_123"
        test_flags = {
            "has_login_form": True,
            "form_count": 2,
            "navigation_links": ["home", "about", "contact"],
            "page_type": "landing"
        }

        await state_manager.update_semantic_page_state(test_hash, test_flags)

        # Retrieve and verify semantic state
        retrieved_state = await state_manager.get_semantic_page_state()
        assert retrieved_state is not None
        retrieved_hash, retrieved_flags = retrieved_state
        assert retrieved_hash == test_hash
        assert retrieved_flags == test_flags

        # Verify the flags are a copy (no reference sharing)
        retrieved_flags["new_key"] = "new_value"
        second_retrieval = await state_manager.get_semantic_page_state()
        assert "new_key" not in second_retrieval[1]

    @pytest.mark.asyncio
    async def test_semantic_data_directly_accessible(self):
        """Test that semantic page information is directly accessible through StateManager."""
        # Create StateManager and update semantic state
        initial_state = AgentState(task="Test semantic integration")
        state_manager = StateManager(
            initial_state=initial_state,
            file_system=None,
            max_failures=3,
            lock_timeout_seconds=10.0,
            use_planner=False,
            reflect_on_error=False,
            max_history_items=100
        )

        test_hash = "semantic_hash_456"
        test_flags = {
            "error_messages": ["Invalid username"],
            "success_indicators": [],
            "form_validation": True
        }

        await state_manager.update_semantic_page_state(test_hash, test_flags)

        # Verify semantic data is directly accessible through StateManager
        semantic_state = await state_manager.get_semantic_page_state()
        assert semantic_state is not None
        retrieved_hash, retrieved_flags = semantic_state
        assert retrieved_hash == test_hash
        assert retrieved_flags == test_flags

    @pytest.mark.asyncio
    async def test_semantic_data_defaults_to_none(self):
        """Test semantic data defaults when no semantic data is available."""
        # Create StateManager without semantic updates
        initial_state = AgentState(task="Test semantic defaults")
        state_manager = StateManager(
            initial_state=initial_state,
            file_system=None,
            max_failures=3,
            lock_timeout_seconds=10.0,
            use_planner=False,
            reflect_on_error=False,
            max_history_items=100
        )

        # Verify semantic state returns None when not set
        semantic_state = await state_manager.get_semantic_page_state()
        assert semantic_state is None

    @pytest.mark.asyncio
    async def test_checkpoint_includes_semantic_state(self):
        """Test that checkpoints include semantic page state."""
        # Create agent state with semantic data
        agent_state = AgentState(task="Test checkpoint semantic state")
        agent_state.last_semantic_page_hash = "checkpoint_hash_789"
        agent_state.last_semantic_flags = {
            "modal_present": True,
            "overlay_type": "confirmation",
            "action_required": "click_confirm"
        }

        # Create checkpoint
        checkpoint = agent_state.to_checkpoint()

        # Verify semantic data is included
        assert checkpoint["last_semantic_page_hash"] == "checkpoint_hash_789"
        assert checkpoint["last_semantic_flags"] == {
            "modal_present": True,
            "overlay_type": "confirmation",
            "action_required": "click_confirm"
        }

    def test_checkpoint_restore_semantic_state(self):
        """Test that semantic state is restored from checkpoint."""
        # Create checkpoint with semantic data
        checkpoint_data = {
            "task": "Test restore semantic state",
            "status": "PENDING",
            "facts_learned": [],
            "constraints_discovered": [],
            "tried_and_failed": [],
            "successful_patterns": [],
            "last_stable_url": None,
            "semantic_anchors": [],
            "last_semantic_page_hash": "restored_hash_101",
            "last_semantic_flags": {
                "captcha_present": True,
                "captcha_type": "image_grid",
                "difficulty": "medium"
            }
        }

        # Restore from checkpoint
        restored_state = AgentState.from_checkpoint(checkpoint_data)

        # Verify semantic data is restored
        assert restored_state.last_semantic_page_hash == "restored_hash_101"
        assert restored_state.last_semantic_flags == {
            "captcha_present": True,
            "captcha_type": "image_grid",
            "difficulty": "medium"
        }

    def test_checkpoint_restore_without_semantic_data(self):
        """Test that checkpoint restore works without semantic data."""
        # Create checkpoint without semantic data (backward compatibility)
        checkpoint_data = {
            "task": "Test restore without semantic state",
            "status": "PENDING",
            "facts_learned": [],
            "constraints_discovered": [],
            "tried_and_failed": [],
            "successful_patterns": [],
            "last_stable_url": None,
            "semantic_anchors": []
            # No semantic fields
        }

        # Restore from checkpoint
        restored_state = AgentState.from_checkpoint(checkpoint_data)

        # Verify semantic fields default to None
        assert restored_state.last_semantic_page_hash is None
        assert restored_state.last_semantic_flags is None

    @pytest.mark.asyncio
    async def test_semantic_state_thread_safety(self):
        """Test thread safety of semantic state operations."""
        import asyncio

        initial_state = AgentState(task="Test thread safety")
        state_manager = StateManager(
            initial_state=initial_state,
            file_system=None,
            max_failures=3,
            lock_timeout_seconds=10.0,
            use_planner=False,
            reflect_on_error=False,
            max_history_items=100
        )

        # Simulate concurrent updates
        async def update_semantic_state(suffix: str):
            hash_val = f"concurrent_hash_{suffix}"
            flags = {"update_id": suffix, "timestamp": suffix}
            await state_manager.update_semantic_page_state(hash_val, flags)
            return await state_manager.get_semantic_page_state()

        # Run concurrent updates
        tasks = [update_semantic_state(str(i)) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Verify all operations completed successfully
        assert len(results) == 5
        for result in results:
            assert result is not None
            hash_val, flags = result
            assert hash_val.startswith("concurrent_hash_")
            assert "update_id" in flags

    @pytest.mark.asyncio
    async def test_semantic_flags_deep_copy(self):
        """Test that semantic flags are properly deep copied."""
        initial_state = AgentState(task="Test deep copy")
        state_manager = StateManager(
            initial_state=initial_state,
            file_system=None,
            max_failures=3,
            lock_timeout_seconds=10.0,
            use_planner=False,
            reflect_on_error=False,
            max_history_items=100
        )

        # Create nested dictionary structure
        original_flags = {
            "nested": {
                "level1": {
                    "level2": ["item1", "item2"]
                }
            },
            "simple": "value"
        }

        await state_manager.update_semantic_page_state("test_hash", original_flags)

        # Retrieve and modify nested structure
        retrieved_state = await state_manager.get_semantic_page_state()
        _, retrieved_flags = retrieved_state

        # Modify the retrieved copy
        retrieved_flags["nested"]["level1"]["level2"].append("item3")
        retrieved_flags["simple"] = "modified"

        # Retrieve again and verify original is unchanged
        second_retrieval = await state_manager.get_semantic_page_state()
        _, second_flags = second_retrieval

        assert len(second_flags["nested"]["level1"]["level2"]) == 2
        assert second_flags["simple"] == "value"
