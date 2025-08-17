#!/usr/bin/env python3
"""
Test Enhanced Deadlock Recovery with Autonomous Continuation

This test will demonstrate how the enhanced long-running mode handles deadlocks
by automatically triggering autonomous continuation after extended deadlock periods.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from agent.service import Agent
from agent.settings import AgentSettings

async def test_deadlock_recovery():
    """Test autonomous deadlock recovery."""

    print("🧪 Testing Enhanced Deadlock Recovery with Autonomous Continuation")
    print("=" * 70)

    # Configure settings for aggressive deadlock recovery
    settings = AgentSettings(
        # Enable long-running mode with autonomous continuation
        enable_long_running_mode=True,
        long_running_enable_autonomous_continuation=True,

        # Aggressive monitoring for faster deadlock detection
        long_running_monitoring_interval=5.0,  # Check every 5 seconds
        long_running_checkpoint_interval=30.0,  # Checkpoint every 30 seconds

        # Tolerant failure settings
        long_running_max_consecutive_failures=5,
        long_running_failure_escalation_delay=60.0,  # 1 minute escalation

        # Circuit breaker settings
        long_running_circuit_breaker_failure_threshold=3,
        long_running_circuit_breaker_recovery_timeout=30.0,

        # Auto recovery enabled
        long_running_enable_auto_recovery=True,
    )

    agent = Agent(
        task="Find job postings for 'Remote Administrative Assistant' in Boston",
        settings=settings
    )

    print("🚀 Starting agent with enhanced deadlock recovery...")
    print("📋 The agent will:")
    print("   • Detect deadlocks after 60 seconds of no events")
    print("   • Trigger autonomous continuation after 180 seconds (3 minutes)")
    print("   • Automatically try different recovery strategies")
    print("   • Force step progression if deadlocked")
    print("   • Refresh pages when element issues occur")
    print("   • Continue autonomously until success or max failures")
    print()
    print("🔍 Watch for these log messages:")
    print("   • 'Potential deadlock detected'")
    print("   • 'Extended deadlock detected, triggering autonomous continuation'")
    print("   • 'Autonomous continuation triggered with recovery action'")
    print("   • 'Forcing step progression to break deadlock'")
    print()
    print("🛑 To stop: Press Ctrl+C twice")
    print()

    try:
        # Run the agent - it should automatically recover from deadlocks
        result = await agent.run()
        print(f"\n✅ Agent completed successfully: {len(result)} history items")

    except KeyboardInterrupt:
        print("\n🛑 Agent stopped by user")
    except Exception as e:
        print(f"\n❌ Agent failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_deadlock_recovery())
