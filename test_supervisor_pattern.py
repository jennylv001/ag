#!/usr/bin/env python3
"""
Simple test to verify TaskMonitor can be used with the pattern Supervisor uses.
"""

import asyncio
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.task_monitor import TaskMonitor


async def test_supervisor_pattern():
    """Test the exact pattern that Supervisor will use with TaskMonitor."""
    print("Testing Supervisor usage pattern with TaskMonitor...")
    
    # Create monitor like Supervisor does
    task_monitor = TaskMonitor(max_restart_attempts=5, base_backoff=2.0, max_backoff=60.0)
    
    # Simulate the component tasks
    run_count = {'perception': 0, 'decision': 0, 'actuation': 0}
    
    async def mock_perception_run():
        run_count['perception'] += 1
        if run_count['perception'] <= 2:
            await asyncio.sleep(0.1)
            raise RuntimeError(f"Perception failure #{run_count['perception']}")
        await asyncio.sleep(0.2)  # Success
    
    async def mock_decision_loop():
        run_count['decision'] += 1
        await asyncio.sleep(0.3)
    
    async def mock_actuation_loop():
        run_count['actuation'] += 1
        await asyncio.sleep(0.3)
    
    # Create tasks like Supervisor does
    perception_task = asyncio.create_task(mock_perception_run())
    decision_task = asyncio.create_task(mock_decision_loop())
    actuation_task = asyncio.create_task(mock_actuation_loop())
    
    # Set up restart factories like Supervisor does
    restart_factories = {
        'perception': lambda: asyncio.create_task(mock_perception_run()),
        'decision_loop': lambda: asyncio.create_task(mock_decision_loop()),
        'actuation_loop': lambda: asyncio.create_task(mock_actuation_loop()),
    }
    task_monitor.set_restart_factories(restart_factories)
    
    # Register tasks like Supervisor does
    task_monitor.register('perception', perception_task)
    task_monitor.register('decision_loop', decision_task)
    task_monitor.register('actuation_loop', actuation_task)
    
    # Enable restart for critical components like Supervisor does
    task_monitor.enable(['perception', 'decision_loop', 'actuation_loop'])
    
    # Simulate the Supervisor's main loop
    print("Starting simulated run...")
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Run for a longer time to see restarts happen
        while asyncio.get_event_loop().time() - start_time < 5.0:
            # Check task status like Supervisor might
            all_tasks = [
                task_monitor._tasks.get('perception'),
                task_monitor._tasks.get('decision_loop'),
                task_monitor._tasks.get('actuation_loop')
            ]
            all_tasks = [t for t in all_tasks if t and not t.done()]
            
            # Print current status for debugging
            if asyncio.get_event_loop().time() - start_time > 1.0:
                stats = task_monitor.get_component_stats()
                perception_stats = stats.get('perception', {})
                if perception_stats.get('restart_count', 0) > 0:
                    print(f"Perception restart detected: {perception_stats}")
                    break
            
            await asyncio.sleep(0.2)
            
    except Exception as e:
        print(f"Error during run: {e}")
    
    # Get final stats
    stats = task_monitor.get_component_stats()
    print(f"Final stats: {stats}")
    
    # Verify restarts occurred
    assert task_monitor._restart_counts['perception'] >= 1, "Perception should have restarted"
    assert run_count['perception'] >= 2, "Perception should have run multiple times"
    
    print("✓ Perception restarted successfully")
    print(f"✓ Perception ran {run_count['perception']} times")
    print(f"✓ Decision loop ran {run_count['decision']} times") 
    print(f"✓ Actuation loop ran {run_count['actuation']} times")
    
    # Clean shutdown like Supervisor does
    await task_monitor.close()
    print("✓ Clean shutdown completed")
    
    print("Supervisor pattern test passed!")


async def main():
    """Run the pattern test."""
    print("Testing TaskMonitor with Supervisor usage pattern...\n")
    
    try:
        await test_supervisor_pattern()
        print("\n🎉 Pattern test passed!")
        return 0
    except Exception as e:
        print(f"\n❌ Pattern test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)