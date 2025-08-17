# Long-Running Operations Mode - Failure-Proof Agent Operations

## Overview

The Long-Running Operations Mode is a comprehensive failure-proof system designed for browser_use agents that need to operate for extended periods (hours to days) with automatic recovery, state persistence, and intelligent resource management.

## Key Features

### 🛡️ Failure Recovery
- **Automatic Component Restart**: Failed components are automatically restarted with exponential backoff
- **Circuit Breaker Protection**: Prevents cascading failures for external services (LLM, browser)
- **Intelligent Recovery Strategies**: Context-aware recovery based on failure patterns
- **Emergency Checkpointing**: Automatic state preservation during critical failures

### 📊 Resource Monitoring
- **Real-time Health Assessment**: Continuous monitoring of CPU, memory, disk, and network usage
- **Adaptive Operation Modes**: Automatic degradation under resource pressure
  - **Normal**: Full functionality
  - **Degraded**: Reduced functionality to preserve resources
  - **Minimal**: Essential operations only
  - **Recovery**: Attempting to recover from failures
- **Trend Detection**: Proactive identification of concerning resource patterns

### 💾 State Persistence
- **Automatic Checkpointing**: Periodic state snapshots with configurable intervals
- **Recovery from Checkpoints**: Resume operations from any saved checkpoint
- **State Consistency**: Atomic state operations to prevent corruption
- **Cleanup Management**: Automatic removal of old checkpoints to prevent disk exhaustion

### 🔧 Intelligent Degradation
- **Load Shedding**: Reduce non-essential operations under resource pressure
- **Priority-based Processing**: Focus on critical operations during degraded modes
- **Graceful Degradation**: Maintain core functionality while reducing resource usage

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Long-Running Mode                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Resource        │  │ Failure         │  │ State           │ │
│  │ Monitor         │  │ Analyzer        │  │ Checkpointer    │ │
│  │                 │  │                 │  │                 │ │
│  │ • CPU/Memory    │  │ • Pattern       │  │ • Periodic      │ │
│  │ • Trends        │  │   Detection     │  │   Snapshots     │ │
│  │ • Health Status │  │ • Recovery      │  │ • Recovery      │ │
│  │                 │  │   Strategies    │  │ • Cleanup       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Circuit         │  │ Operation       │  │ Integration     │ │
│  │ Breakers        │  │ Modes           │  │ Layer           │ │
│  │                 │  │                 │  │                 │ │
│  │ • LLM           │  │ • Normal        │  │ • Supervisor    │ │
│  │ • Browser       │  │ • Degraded      │  │ • ReactorVitals │ │
│  │ • Timeout       │  │ • Minimal       │  │ • StateManager  │ │
│  │ • Recovery      │  │ • Recovery      │  │ • Settings      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

### Basic Setup

```python
from browser_use.agent.service import Agent
from browser_use.agent.settings import AgentSettings

settings = AgentSettings(
    task="Your long-running task",
    llm=your_llm_provider,

    # Enable long-running mode
    enable_long_running_mode=True,

    # Basic configuration
    long_running_monitoring_interval=30.0,    # Health check every 30 seconds
    long_running_checkpoint_interval=300.0,   # Checkpoint every 5 minutes
    long_running_checkpoint_dir="./checkpoints",
)

agent = Agent(settings)
history = await agent.run()
```

### Advanced Configuration

```python
settings = AgentSettings(
    # ... other settings ...

    # Long-running mode settings
    enable_long_running_mode=True,

    # Monitoring configuration
    long_running_monitoring_interval=30.0,
    long_running_checkpoint_interval=300.0,
    long_running_checkpoint_dir="./agent_checkpoints",
    long_running_max_checkpoints=50,

    # Resource thresholds
    long_running_cpu_threshold_warning=80.0,
    long_running_cpu_threshold_critical=95.0,
    long_running_memory_threshold_warning=80.0,
    long_running_memory_threshold_critical=95.0,

    # Circuit breaker settings
    long_running_circuit_breaker_failure_threshold=5,
    long_running_circuit_breaker_recovery_timeout=60.0,

    # Auto-recovery
    long_running_enable_auto_recovery=True,

    # Optimize for long runs
    max_steps=10000,
    max_failures=10,
    enable_modes=True,
    memory_budget_mb=500.0,
)
```

## Usage Examples

### 1. Basic Long-Running Agent

```python
import asyncio
from browser_use.agent.service import Agent
from browser_use.agent.settings import AgentSettings

async def run_long_task():
    settings = AgentSettings(
        task="Monitor website for 24 hours and report changes",
        llm=your_llm,
        enable_long_running_mode=True,
        max_steps=5000,
    )

    agent = Agent(settings)

    try:
        history = await agent.run()
        print(f"Task completed successfully with {len(history.history)} steps")
    except Exception as e:
        print(f"Task failed: {e}")

        # Auto-recovery might have been attempted
        if hasattr(agent.supervisor, 'long_running_integration'):
            health = await agent.supervisor.long_running_integration.get_health_status()
            print(f"Final health status: {health}")

asyncio.run(run_long_task())
```

### 2. Manual Checkpoint Recovery

```python
async def recover_from_checkpoint():
    settings = AgentSettings(
        task="Resume previous task",
        llm=your_llm,
        enable_long_running_mode=True,
        long_running_checkpoint_dir="./my_checkpoints"
    )

    agent = Agent(settings)
    integration = agent.supervisor.long_running_integration
    await integration.initialize()

    # List available checkpoints
    if integration.long_running_mode:
        checkpoints = await integration.long_running_mode.checkpointer.list_checkpoints()
        print(f"Available checkpoints: {checkpoints}")

        # Recover from specific checkpoint
        if checkpoints:
            success = await integration.attempt_recovery(checkpoints[0])
            if success:
                print("Recovery successful!")
                # Continue with agent.run()
```

### 3. Health Monitoring

```python
async def monitor_health():
    # ... setup agent with long-running mode ...

    integration = agent.supervisor.long_running_integration

    while True:
        health_report = await integration.get_health_status()

        print(f"Mode: {health_report['operation_mode']}")
        print(f"Health: {health_report['health_status']}")
        print(f"CPU: {health_report['resource_metrics']['cpu_percent']:.1f}%")
        print(f"Memory: {health_report['resource_metrics']['memory_percent']:.1f}%")

        if health_report['trends']:
            print(f"Concerning trends: {health_report['trends']}")

        await asyncio.sleep(30)
```

## Operation Modes

### Normal Mode
- Full functionality enabled
- All components running at normal capacity
- Regular monitoring and checkpointing

### Degraded Mode
- Reduced functionality to preserve resources
- Increased monitoring intervals
- Non-essential operations deferred

### Minimal Mode
- Only essential operations
- Maximum resource conservation
- Emergency-only checkpointing

### Recovery Mode
- Attempting to recover from failures
- Increased checkpoint frequency
- Enhanced error logging

## Failure Scenarios Handled

| Failure Type | Detection | Recovery Strategy |
|--------------|-----------|-------------------|
| Browser Crash | Page unresponsiveness | Restart browser session |
| LLM Timeout | Request timeout | Circuit breaker, retry with backoff |
| Memory Leak | Increasing memory usage | Clear caches, restart components |
| CPU Overload | High CPU usage | Reduce operation frequency |
| Network Issues | Connection errors | Exponential backoff, retry |
| Component Deadlock | Missing heartbeats | Component restart |
| Disk Full | Checkpoint failures | Cleanup old checkpoints |

## Circuit Breaker States

### Closed (Normal)
- All requests pass through
- Failure counter is reset on success

### Open (Failed)
- All requests fail immediately
- Enters recovery timeout period

### Half-Open (Testing)
- Limited requests allowed
- Tests if service has recovered

## Health Status Levels

| Status | Criteria | Actions |
|--------|----------|---------|
| Healthy | Normal resource usage | Full functionality |
| Warning | CPU > 80% OR Memory > 80% | Enter degraded mode |
| Critical | CPU > 95% OR Memory > 95% | Enter minimal mode |
| Failing | Multiple component failures | Enter recovery mode |

## Checkpointing

### Automatic Checkpoints
- Created at regular intervals (default: 5 minutes)
- Include full agent state and context
- Automatic cleanup of old checkpoints

### Emergency Checkpoints
- Created during critical failures
- Include error context and recovery hints
- Highest priority for recovery

### Checkpoint Contents
```json
{
  "checkpoint_id": "checkpoint_1234567890_abcd1234",
  "timestamp": 1234567890.123,
  "agent_state": {
    "task": "Current task description",
    "status": "RUNNING",
    "n_steps": 42,
    "history": [...],
    "current_goal": "Current objective"
  },
  "browser_state": {...},
  "recovery_hints": [
    "Created during normal operation",
    "Browser was responsive",
    "No errors detected"
  ]
}
```

## Monitoring and Observability

### Metrics Collected
- **CPU Usage**: Process and system-wide
- **Memory Usage**: RSS, virtual memory, percentage
- **Disk Usage**: Free space, I/O operations
- **Network I/O**: Bytes sent/received
- **Thread Count**: Active threads
- **Open Files**: File descriptors in use

### Trend Detection
- **Memory Leaks**: Increasing memory over time
- **CPU Spikes**: Sustained high CPU usage
- **Thread Accumulation**: Growing thread count
- **Resource Exhaustion**: Approaching system limits

### Logging Integration
- Structured logging with agent ID and step context
- Health status changes logged at INFO level
- Failures and recoveries logged at WARNING/ERROR
- Debug logging for detailed troubleshooting

## Best Practices

### 1. Configuration
```python
# For 24-hour operations
settings = AgentSettings(
    enable_long_running_mode=True,
    long_running_checkpoint_interval=300.0,  # 5 minutes
    long_running_cpu_threshold_warning=75.0,  # Conservative
    long_running_memory_threshold_warning=70.0,
    max_steps=20000,  # Allow many steps
    memory_budget_mb=1000.0,  # Larger budget
)
```

### 2. Task Design
- Break large tasks into smaller, resumable chunks
- Use clear, specific task descriptions
- Design tasks to be idempotent when possible
- Include progress tracking in task descriptions

### 3. Error Handling
- Monitor agent logs for patterns
- Set appropriate failure thresholds
- Use circuit breakers for external dependencies
- Plan for graceful degradation

### 4. Resource Management
- Monitor system resources externally
- Set conservative resource thresholds
- Use SSD storage for checkpoints
- Ensure adequate disk space

### 5. Recovery Planning
- Test checkpoint recovery regularly
- Document recovery procedures
- Keep multiple checkpoint generations
- Monitor checkpoint creation success

## Troubleshooting

### Common Issues

#### High Memory Usage
```python
# Reduce memory budget
settings.memory_budget_mb = 200.0
settings.max_history_items = 50
settings.long_running_memory_threshold_warning = 60.0
```

#### Frequent Checkpointing Failures
```python
# Check disk space and permissions
settings.long_running_checkpoint_dir = "/path/with/space"
settings.long_running_max_checkpoints = 20  # Reduce retention
```

#### Circuit Breaker Always Open
```python
# Increase failure threshold or recovery timeout
settings.long_running_circuit_breaker_failure_threshold = 10
settings.long_running_circuit_breaker_recovery_timeout = 300.0
```

### Debug Commands

```python
# Get detailed health report
health = await integration.get_health_status()
print(json.dumps(health, indent=2))

# List all checkpoints
checkpoints = await integration.long_running_mode.checkpointer.list_checkpoints()
for cp in checkpoints:
    print(f"Checkpoint: {cp}")

# Check circuit breaker states
for service, cb in integration.long_running_mode.circuit_breakers.items():
    print(f"{service}: {cb.state} (failures: {cb.failure_count})")
```

## Performance Impact

### CPU Overhead
- Monitoring: ~1-2% CPU
- Checkpointing: ~5-10% CPU during checkpoint creation
- Health checks: <1% CPU

### Memory Overhead
- Monitoring data: ~50-100MB
- Checkpoint buffer: ~100-500MB (temporary)
- Failure analysis: ~10-50MB

### Disk Usage
- Checkpoint size: ~1-10MB per checkpoint
- Log files: ~100MB per day
- Cleanup removes old data automatically

## Security Considerations

### Checkpoint Security
- Checkpoints may contain sensitive task data
- Store checkpoints in secure directories
- Consider encryption for sensitive workloads
- Implement access controls on checkpoint directories

### Network Security
- Circuit breakers protect against network attacks
- Monitor for unusual network patterns
- Use secure connections for external services

## Migration Guide

### From Standard Mode
1. Add `enable_long_running_mode=True` to settings
2. Configure checkpoint directory
3. Adjust resource thresholds if needed
4. Test checkpoint recovery procedures

### Settings Migration
```python
# Old settings
settings = AgentSettings(
    task="My task",
    llm=llm,
    max_steps=100,
    max_failures=3
)

# New settings with long-running mode
settings = AgentSettings(
    task="My task",
    llm=llm,
    max_steps=5000,  # Increased for long runs
    max_failures=10,  # Higher tolerance
    enable_long_running_mode=True,
    long_running_checkpoint_interval=300.0,
    long_running_checkpoint_dir="./checkpoints"
)
```

## API Reference

### LongRunningMode Class
```python
class LongRunningMode:
    async def start_monitoring() -> None
    async def stop_monitoring() -> None
    async def handle_failure(error: Exception, context: Dict) -> str
    async def recover_from_checkpoint(checkpoint_id: Optional[str]) -> bool
    async def get_health_report() -> Dict[str, Any]
    def get_circuit_breaker(service: str) -> CircuitBreaker
```

### LongRunningIntegration Class
```python
class LongRunningIntegration:
    async def initialize() -> bool
    async def start_monitoring() -> None
    async def stop_monitoring() -> None
    async def handle_component_failure(component: str, error: Exception) -> str
    async def create_emergency_checkpoint(reason: str) -> Optional[str]
    async def attempt_recovery(checkpoint_id: Optional[str]) -> bool
    async def get_health_status() -> Dict[str, Any]
```

### Configuration Settings
```python
class AgentSettings:
    enable_long_running_mode: bool = False
    long_running_monitoring_interval: float = 30.0
    long_running_checkpoint_interval: float = 300.0
    long_running_checkpoint_dir: Optional[str] = None
    long_running_max_checkpoints: int = 50
    long_running_cpu_threshold_warning: float = 80.0
    long_running_cpu_threshold_critical: float = 95.0
    long_running_memory_threshold_warning: float = 80.0
    long_running_memory_threshold_critical: float = 95.0
    long_running_circuit_breaker_failure_threshold: int = 5
    long_running_circuit_breaker_recovery_timeout: float = 60.0
    long_running_enable_auto_recovery: bool = True
```
