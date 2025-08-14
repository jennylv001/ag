# Robust Task Restart Implementation

This document describes the robust, non-destructive restart capability implemented for long-lived agent tasks under Supervisor and the ergonomic io_semaphore async context manager.

## Overview

The implementation provides automatic restart capability for critical agent components with proper observability, rate limiting, and backward compatibility. The solution removes the single point of failure from `asyncio.TaskGroup` and replaces it with individual task management through `TaskMonitor`.

## Key Components

### 1. TaskMonitor (`agent/task_monitor.py`)

The `TaskMonitor` class manages component tasks and automatically restarts them when they fail or get cancelled unexpectedly.

**Key Features:**
- **Exponential backoff**: Starts with 2s delay, doubles on each attempt, caps at 60s
- **Rate limiting**: Maximum 5 restart attempts per component per hour
- **Race safety**: `is_restarting` flag prevents concurrent restarts
- **Observability**: Structured logging with component, attempt, backoff details
- **Clean shutdown**: Proper task cancellation and resource cleanup

**API:**
```python
from agent.task_monitor import TaskMonitor

# Create monitor
monitor = TaskMonitor(max_restart_attempts=5, base_backoff=2.0, max_backoff=60.0)

# Set restart factories
restart_factories = {
    'component_name': lambda: asyncio.create_task(component_coroutine()),
}
monitor.set_restart_factories(restart_factories)

# Register tasks for monitoring
task = asyncio.create_task(component_coroutine())
monitor.register('component_name', task)

# Enable restart for specific components
monitor.enable(['component_name'])

# Get statistics
stats = monitor.get_component_stats()

# Clean shutdown
await monitor.close()
```

### 2. io_semaphore (`concurrency/io.py`)

Provides async context manager for I/O flow control with global semaphore management.

**Key Features:**
- **Global semaphore**: Shared across the application for I/O operations
- **Configurable count**: Defaults to `max(1, os.cpu_count())` or conservative 8
- **Context manager**: Easy `async with` usage pattern
- **Backward compatibility**: `with_io_semaphore()` function for direct access
- **Statistics**: Runtime semaphore usage monitoring

**Usage:**
```python
from concurrency.io import io_semaphore, set_io_semaphore_count, get_io_semaphore_stats

# Basic usage
async with io_semaphore():
    # Perform I/O operation here
    result = await some_io_operation()

# Configure semaphore count
set_io_semaphore_count(10)

# Get usage statistics
stats = get_io_semaphore_stats()
print(f"Available: {stats['available']}, Waiting: {stats['waiting']}")

# Backward compatibility
semaphore = await with_io_semaphore()
async with semaphore:
    # Direct semaphore usage
    pass
```

### 3. Supervisor Integration (`agent/supervisor.py`)

The `Supervisor` class has been modified to use `TaskMonitor` instead of `asyncio.TaskGroup` for robust task management.

**Changes Made:**
- **Replaced TaskGroup**: Individual task creation prevents cascading failures
- **Added TaskMonitor**: Automatic restart for critical components
- **Preserved lifecycle**: All existing hooks and cleanup remain unchanged
- **Enabled restart**: For perception, decision_loop, actuation_loop, perception_watchdog
- **Clean shutdown**: TaskMonitor is properly closed during finalization

**Components Monitored:**
- `perception`: Main perception loop with restart enabled
- `perception_watchdog`: Perception timeout monitoring with restart enabled  
- `decision_loop`: Decision-making loop with restart enabled
- `actuation_loop`: Action execution loop with restart enabled
- `pause_handler`: Pause state handling (restartable but lower priority)
- `load_shedding_monitor`: System load monitoring (restartable but lower priority)

## Observability

### Structured Logging

All restart events are logged with structured data:

```
INFO: Restarting component perception (attempt 2) after 4.0s delay
  - component: perception
  - attempt: 2
  - backoff: 4.0
  - reason: RuntimeError: Connection lost

INFO: Successfully restarted component perception
  - component: perception
  - new_task_id: 140234567890
  - restart_count: 2
```

### Error Handling

Rate limiting prevents restart storms:

```
ERROR: Component perception has exceeded restart limit (5 attempts per hour). Stopping auto-restart.
  - component: perception
  - restart_count: 5
  - reason: Persistent failure
```

### Statistics API

Real-time monitoring of component health:

```python
stats = task_monitor.get_component_stats()
# Returns:
{
    'perception': {
        'enabled': True,
        'restart_count': 2,
        'is_restarting': False,
        'task_done': False,
        'task_cancelled': False,
        'last_restart_time': 1755159603.34
    }
}
```

## Backward Compatibility

**External Behavior Preserved:**
- All public APIs remain unchanged
- Supervisor.run() returns the same history object
- Lifecycle hooks (on_run_start, on_run_end, etc.) work identically
- Settings and configuration remain compatible
- Exception handling and status transitions unchanged

**Internal Changes Only:**
- Task orchestration moved from TaskGroup to individual tasks + TaskMonitor
- Added TaskMonitor instance as private attribute
- Enhanced cleanup in finally block

## Testing

### Unit Tests
- **TaskMonitor**: Restart behavior, rate limiting, race conditions
- **io_semaphore**: Context manager usage, concurrency, statistics

### Integration Tests  
- **Supervisor Pattern**: Validates the exact usage pattern in Supervisor
- **Failure Recovery**: Tests restart behavior with failing components
- **Clean Shutdown**: Verifies proper resource cleanup

### Validation
Run comprehensive validation:
```bash
cd /home/runner/work/ag/ag
python final_validation.py
```

## Production Considerations

### Performance
- **Minimal overhead**: TaskMonitor only activates on task completion
- **Efficient restart**: Quick task replacement without blocking other components
- **Memory management**: Clean task cleanup prevents resource leaks

### Reliability
- **Graceful degradation**: Rate limiting prevents restart storms
- **Isolation**: Component failures don't affect siblings
- **Observability**: Comprehensive logging for debugging

### Configuration
- **Restart limits**: Configurable per-component and time-window based
- **Backoff timing**: Configurable base delay and maximum
- **Enabled components**: Selective restart enablement

## Future Enhancements

This implementation provides a solid foundation for:
- **Health checks**: Heartbeat-based stall detection
- **Circuit breakers**: Automatic component disabling on persistent failures  
- **Metrics export**: Integration with monitoring systems
- **Dynamic configuration**: Runtime adjustment of restart policies

## Migration Guide

For existing code using Supervisor:

1. **No changes required**: External usage remains identical
2. **Optional logging**: Monitor restart events in application logs
3. **Optional tuning**: Adjust TaskMonitor parameters if needed via Supervisor configuration
4. **Optional io_semaphore**: Start using for new I/O intensive operations

The implementation is fully backward compatible and requires no changes to existing agent code.