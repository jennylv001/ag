"""
Example: Using Long-Running Operations Mode for Failure-Proof Agent Operations

This example demonstrates how to configure and use the long-running mode
for extended, failure-proof agent operations that can run for hours or days.
"""

import asyncio
import logging
from browser_use.agent.service import Agent
from browser_use.agent.settings import AgentSettings
from browser_use.llm.openai import OpenAIProvider

# Configure logging to see long-running mode events
logging.basicConfig(level=logging.INFO)


async def main():
    """
    Example of running a browser agent with long-running mode enabled.
    """

    # Configure LLM provider
    llm = OpenAIProvider(
        api_key="your-openai-api-key",  # Replace with actual API key
        model="gpt-4o"
    )

    # Configure agent settings with long-running mode enabled
    settings = AgentSettings(
        task="Monitor a website for changes and report any significant updates. "
             "Check every 30 minutes for the next 24 hours.",
        llm=llm,

        # Enable long-running mode
        enable_long_running_mode=True,

        # Long-running mode configuration
        long_running_monitoring_interval=30.0,  # Monitor health every 30 seconds
        long_running_checkpoint_interval=300.0,  # Checkpoint every 5 minutes
        long_running_checkpoint_dir="./agent_checkpoints",  # Custom checkpoint directory
        long_running_max_checkpoints=100,  # Keep up to 100 checkpoints

        # Resource monitoring thresholds
        long_running_cpu_threshold_warning=75.0,  # Lower CPU threshold for extended runs
        long_running_cpu_threshold_critical=90.0,
        long_running_memory_threshold_warning=70.0,  # Lower memory threshold
        long_running_memory_threshold_critical=85.0,

        # Circuit breaker settings for external services
        long_running_circuit_breaker_failure_threshold=3,  # Open after 3 failures
        long_running_circuit_breaker_recovery_timeout=120.0,  # 2 minutes recovery

        # Enable automatic recovery from checkpoints
        long_running_enable_auto_recovery=True,

        # Other agent settings optimized for long-running operations
        max_steps=10000,  # Allow many steps for extended operations
        max_failures=10,  # Higher failure tolerance
        reflect_on_error=True,  # Enable reflection for learning from errors
        use_planner=True,  # Use planner for complex long-term tasks
        planner_interval=10,  # Plan every 10 steps

        # Enable health-aware modes for adaptive behavior
        enable_modes=True,

        # Memory management for long runs
        max_history_items=200,  # Larger history buffer
        memory_budget_mb=500.0,  # 500MB memory budget

        # I/O management
        max_concurrent_io=2,  # Reduce concurrency for stability
        lock_timeout_seconds=60.0,  # Longer lock timeout
    )

    # Create and run the agent
    agent = Agent(settings)

    try:
        print("🚀 Starting long-running agent with failure-proof mode...")
        print("📊 Features enabled:")
        print("   ✅ Automatic health monitoring")
        print("   ✅ State checkpointing every 5 minutes")
        print("   ✅ Circuit breaker protection")
        print("   ✅ Resource usage monitoring")
        print("   ✅ Automatic recovery from failures")
        print("   ✅ Degraded mode under resource pressure")
        print()

        # Run the agent
        history = await agent.run()

        print(f"✅ Agent completed successfully after {len(history.history)} steps")

    except KeyboardInterrupt:
        print("⏸️  Agent paused by user")

        # Get health status
        if hasattr(agent.supervisor, 'long_running_integration'):
            health_status = await agent.supervisor.long_running_integration.get_health_status()
            print(f"📊 Final health status: {health_status.get('health_status', 'unknown')}")
            print(f"🔧 Operation mode: {health_status.get('operation_mode', 'unknown')}")

            # List available checkpoints for potential recovery
            if agent.supervisor.long_running_integration.long_running_mode:
                checkpoints = await agent.supervisor.long_running_integration.long_running_mode.checkpointer.list_checkpoints()
                if checkpoints:
                    print(f"💾 Available checkpoints: {len(checkpoints)}")
                    print(f"   Most recent: {checkpoints[0]}")

    except Exception as e:
        print(f"❌ Agent failed: {e}")

        # Attempt recovery if long-running mode is available
        if hasattr(agent.supervisor, 'long_running_integration'):
            integration = agent.supervisor.long_running_integration
            if integration.enabled:
                print("🔄 Attempting recovery from latest checkpoint...")
                recovery_success = await integration.attempt_recovery()
                if recovery_success:
                    print("✅ Recovery successful! Agent state restored.")
                else:
                    print("❌ Recovery failed. Check logs for details.")


async def demonstrate_manual_checkpoint_recovery():
    """
    Example of manually recovering from a specific checkpoint.
    """
    print("\n" + "="*50)
    print("🔄 Manual Checkpoint Recovery Example")
    print("="*50)

    # This would be used when you want to resume from a specific checkpoint
    # after a system restart or crash

    llm = OpenAIProvider(
        api_key="your-openai-api-key",
        model="gpt-4o"
    )

    settings = AgentSettings(
        task="Resume previous long-running task",
        llm=llm,
        enable_long_running_mode=True,
        long_running_checkpoint_dir="./agent_checkpoints"
    )

    agent = Agent(settings)

    # Access the long-running integration
    integration = agent.supervisor.long_running_integration
    await integration.initialize()

    if integration.long_running_mode:
        # List available checkpoints
        checkpoints = await integration.long_running_mode.checkpointer.list_checkpoints()

        if checkpoints:
            print(f"📁 Found {len(checkpoints)} available checkpoints:")
            for i, checkpoint_id in enumerate(checkpoints[:5]):  # Show first 5
                print(f"   {i+1}. {checkpoint_id}")

            # For demo, recover from the most recent checkpoint
            latest_checkpoint = checkpoints[0]
            print(f"\n🔄 Recovering from checkpoint: {latest_checkpoint}")

            success = await integration.attempt_recovery(latest_checkpoint)
            if success:
                print("✅ Successfully recovered from checkpoint!")
                print("🚀 Agent can now continue from the restored state")
            else:
                print("❌ Failed to recover from checkpoint")
        else:
            print("📭 No checkpoints found")


async def monitor_agent_health():
    """
    Example of monitoring agent health during operation.
    """
    print("\n" + "="*50)
    print("📊 Agent Health Monitoring Example")
    print("="*50)

    llm = OpenAIProvider(
        api_key="your-openai-api-key",
        model="gpt-4o"
    )

    settings = AgentSettings(
        task="Example task for health monitoring",
        llm=llm,
        enable_long_running_mode=True,
        long_running_monitoring_interval=5.0  # Fast monitoring for demo
    )

    agent = Agent(settings)
    integration = agent.supervisor.long_running_integration
    await integration.initialize()

    if integration.long_running_mode:
        # Monitor health for a short period
        for i in range(6):  # Monitor for 30 seconds
            health_report = await integration.get_health_status()

            print(f"\n📊 Health Check {i+1}:")
            print(f"   Mode: {health_report.get('operation_mode', 'unknown')}")
            print(f"   Status: {health_report.get('health_status', 'unknown')}")

            if 'resource_metrics' in health_report:
                metrics = health_report['resource_metrics']
                print(f"   CPU: {metrics.get('cpu_percent', 0):.1f}%")
                print(f"   Memory: {metrics.get('memory_percent', 0):.1f}%")

            if 'recent_checkpoints' in health_report:
                checkpoint_count = len(health_report['recent_checkpoints'])
                print(f"   Checkpoints: {checkpoint_count}")

            if health_report.get('trends'):
                print(f"   Trends: {health_report['trends']}")

            await asyncio.sleep(5)

        print("\n✅ Health monitoring completed")


if __name__ == "__main__":
    print("🧪 Long-Running Mode Examples")
    print("=" * 50)

    # Note: These examples require a valid OpenAI API key
    # Replace "your-openai-api-key" with an actual key to run

    print("\n1. 🚀 Main Long-Running Agent Example")
    print("   Demonstrates a complete long-running agent setup")

    print("\n2. 🔄 Manual Checkpoint Recovery Example")
    print("   Shows how to recover from specific checkpoints")

    print("\n3. 📊 Health Monitoring Example")
    print("   Demonstrates real-time health monitoring")

    # Uncomment and provide API key to run:
    # asyncio.run(main())
    # asyncio.run(demonstrate_manual_checkpoint_recovery())
    # asyncio.run(monitor_agent_health())

    print("\n💡 To run these examples:")
    print("   1. Set your OpenAI API key in the code")
    print("   2. Uncomment the asyncio.run() calls above")
    print("   3. Run this script")

    print("\n🎯 Long-Running Mode Benefits:")
    print("   ✅ Automatic failure recovery")
    print("   ✅ Resource monitoring and adaptive behavior")
    print("   ✅ State preservation across crashes")
    print("   ✅ Circuit breaker protection for external services")
    print("   ✅ Intelligent degradation under resource pressure")
    print("   ✅ Comprehensive logging and observability")
