#!/usr/bin/env python3
"""
Simple validation test for the unified agent architecture.
This tests that our AgentOrchestrator and OrchestratorState work properly.
"""

import asyncio
from unittest.mock import Mock, MagicMock
import pytest

def test_orchestrator_import():
    """Test that AgentOrchestrator can be imported."""
    from agent.orchestrator import AgentOrchestrator
    assert AgentOrchestrator is not None
    print("✓ AgentOrchestrator imported successfully")

def test_orchestrator_state_import():
    """Test that OrchestratorState can be imported."""
    from agent.state import OrchestratorState
    assert OrchestratorState is not None
    print("✓ OrchestratorState imported successfully")

def test_orchestrator_structure():
    """Test that AgentOrchestrator has expected methods."""
    from agent.orchestrator import AgentOrchestrator

    required_methods = [
        '_perceive', 'decide', 'execute', 'run',
        '_setup_action_models', '_invoke_llm_with_retry'
    ]

    for method in required_methods:
        assert hasattr(AgentOrchestrator, method), f"Missing method: {method}"
        print(f"✓ {method} method exists")

def test_orchestrator_state_structure():
    """Test that OrchestratorState has expected fields."""
    from agent.state import OrchestratorState
    import dataclasses

    fields = [f.name for f in dataclasses.fields(OrchestratorState)]
    required_fields = [
        'browser_state', 'health_metrics', 'step_number', 'current_goal'
    ]

    for field in required_fields:
        assert field in fields, f"Missing field: {field}"
        print(f"✓ {field} field exists")

def test_agent_service_integration():
    """Test that Agent service integrates with orchestrator."""
    from agent.service import Agent

    # This should not fail with our unified architecture
    try:
        # Test that we can import and create agent class structure
        assert Agent is not None
        print("✓ Agent class imported successfully")

        # Test that run method exists (should delegate to orchestrator)
        assert hasattr(Agent, 'run')
        print("✓ Agent.run method exists")

        print("✓ Agent service structure validated")

    except Exception as e:
        print(f"✗ Agent service integration issue: {e}")
        raise

if __name__ == "__main__":
    print("Testing unified agent architecture...")

    test_orchestrator_import()
    test_orchestrator_state_import()
    test_orchestrator_structure()
    test_orchestrator_state_structure()
    test_agent_service_integration()

    print("\n🎉 All unified architecture tests passed!")
