#!/usr/bin/env python3
"""
Autonomous Long-Running Agent Example - Travel/Unattended Operation

This example shows how to configure the agent for autonomous operation
when you need it to run unattended (e.g., while traveling).
"""

import asyncio
import sys
import os

# Add current directory to path to import browser_use
sys.path.insert(0, os.path.dirname(__file__))

from agent.service import Agent
from agent.settings import AgentSettings

async def autonomous_travel_agent():
    """
    Example of configuring an agent for autonomous operation while traveling.
    """

    # Configure settings for autonomous operation
    settings = AgentSettings(
        # Enable long-running mode
        enable_long_running_mode=True,

        # AUTONOMOUS CONTINUATION - Key for travel scenarios
        long_running_enable_autonomous_continuation=True,
        long_running_max_consecutive_failures=5,  # Allow more failures before giving up
        long_running_failure_escalation_delay=300.0,  # 5 minutes between escalations

        # Aggressive checkpointing for recovery
        long_running_checkpoint_interval=120.0,  # Checkpoint every 2 minutes
        long_running_max_checkpoints=100,  # Keep more checkpoints

        # Health monitoring
        long_running_monitoring_interval=15.0,  # Monitor every 15 seconds

        # Circuit breaker settings for resilience
        long_running_circuit_breaker_failure_threshold=8,  # More tolerant of failures
        long_running_circuit_breaker_recovery_timeout=180.0,  # 3 minute recovery timeout

        # Resource management
        long_running_cpu_threshold_warning=85.0,
        long_running_cpu_threshold_critical=98.0,
        long_running_memory_threshold_warning=85.0,
        long_running_memory_threshold_critical=98.0,

        # Auto-recovery enabled
        long_running_enable_auto_recovery=True,
    )

    agent = Agent(
        task="Monitor job postings for 'Python Developer' in Boston every hour and save interesting ones",
        settings=settings
    )

    print("🚀 Starting autonomous travel agent...")
    print("📍 This agent will:")
    print("   • Continue running even after individual task failures")
    print("   • Automatically recover from browser crashes, network issues, etc.")
    print("   • Create frequent checkpoints for recovery")
    print("   • Retry failed operations with intelligent strategies")
    print("   • Only stop when you press Ctrl+C twice")
    print("\n💡 Perfect for travel/unattended operation!")
    print("\n🛑 To stop: Press Ctrl+C twice")

    try:
        # This will run indefinitely with autonomous recovery
        result = await agent.run()
        print(f"Agent completed: {len(result)} history items")

    except KeyboardInterrupt:
        print("\n✅ Agent stopped by user")
    except Exception as e:
        print(f"\n❌ Agent failed completely: {e}")

def create_travel_settings() -> AgentSettings:
    """
    Create pre-configured settings for travel/autonomous operation.
    """
    return AgentSettings(
        enable_long_running_mode=True,
        long_running_enable_autonomous_continuation=True,
        long_running_max_consecutive_failures=10,  # Very tolerant
        long_running_failure_escalation_delay=600.0,  # 10 minutes
        long_running_checkpoint_interval=60.0,  # Checkpoint every minute
        long_running_max_checkpoints=200,
        long_running_monitoring_interval=10.0,  # Frequent monitoring
        long_running_circuit_breaker_failure_threshold=15,
        long_running_circuit_breaker_recovery_timeout=300.0,
        long_running_enable_auto_recovery=True,
    )

async def monitor_multiple_tasks():
    """
    Example: Running multiple autonomous tasks while traveling.
    """
    travel_settings = create_travel_settings()

    tasks = [
        "Check email every 30 minutes and flag urgent messages",
        "Monitor stock prices for AAPL, GOOGL, TSLA and alert on 5%+ changes",
        "Track package delivery status and screenshot updates",
        "Check weather in destination city every 2 hours"
    ]

    agents = []
    for task in tasks:
        agent = Agent(task=task, settings=travel_settings)
        agents.append(agent)

    print(f"🌍 Starting {len(agents)} autonomous travel agents...")

    # Run all agents concurrently with autonomous recovery
    try:
        await asyncio.gather(*[agent.run() for agent in agents])
    except KeyboardInterrupt:
        print("\n✅ All agents stopped by user")

if __name__ == "__main__":
    print("=== Autonomous Long-Running Agent for Travel ===")
    print("Choose an option:")
    print("1. Single autonomous task")
    print("2. Multiple autonomous tasks")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        asyncio.run(autonomous_travel_agent())
    elif choice == "2":
        asyncio.run(monitor_multiple_tasks())
    else:
        print("Invalid choice")
