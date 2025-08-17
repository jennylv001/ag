"""
Long-Running Operations Mode for Failure-Proof Agent Operation

This module implements a comprehensive failure-proof mode for browser_use agents
designed to handle extended operations (hours to days) with automatic recovery,
state persistence, and intelligent resource management.

Architecture:
- Enhanced failure detection with predictive monitoring
- Automatic state checkpointing and recovery
- Circuit breaker patterns for external services
- Progressive degradation under resource constraints
- Intelligent retry strategies with exponential backoff
- Health monitoring with automated interventions
"""

import asyncio
import json
import logging
import os
import time
import tempfile
import pickle
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, List, Any, Callable, Awaitable
import psutil

from browser_use.agent.state_manager import StateManager, AgentState, AgentStatus, agent_log
from browser_use.agent.views import AgentHistory

logger = logging.getLogger(__name__)


class OperationMode(Enum):
    """Operating modes for long-running operations."""
    NORMAL = "normal"           # Full functionality
    DEGRADED = "degraded"       # Reduced functionality to preserve resources
    MINIMAL = "minimal"         # Essential operations only
    RECOVERY = "recovery"       # Attempting to recover from failures


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILING = "failing"


@dataclass
class ResourceMetrics:
    """System resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    network_io_mb: float
    open_files: int
    thread_count: int
    timestamp: float


@dataclass
class FailurePattern:
    """Represents a detected failure pattern."""
    pattern_type: str
    frequency: int
    last_occurrence: float
    severity: str
    suggested_action: str


@dataclass
class Checkpoint:
    """Agent state checkpoint for recovery."""
    checkpoint_id: str
    timestamp: float
    agent_state: Dict[str, Any]
    step_number: int
    browser_state: Optional[Dict[str, Any]]
    task_context: str
    recovery_hints: List[str]


class CircuitBreaker:
    """Circuit breaker pattern for external service calls."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

            raise e


class ResourceMonitor:
    """Monitors system resources and detects potential issues.

    Note: Uses system-wide CPU percent to align with Supervisor load shedding.
    Thresholds can be configured via AgentSettings.long_running_* fields.
    """

    def __init__(self, settings: Optional[Any] = None):
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history = 100
        # Defaults
        self.cpu_threshold_warning = 80.0
        self.cpu_threshold_critical = 95.0
        self.memory_threshold_warning = 80.0
        self.memory_threshold_critical = 95.0
        # Smoothing state
        self._ewma_cpu: Optional[float] = None
        self._ewma_alpha: float = 0.3  # light smoothing
        # Override from settings when provided
        if settings is not None:
            try:
                self.cpu_threshold_warning = float(getattr(settings, 'long_running_cpu_threshold_warning', self.cpu_threshold_warning))
                self.cpu_threshold_critical = float(getattr(settings, 'long_running_cpu_threshold_critical', self.cpu_threshold_critical))
                self.memory_threshold_warning = float(getattr(settings, 'long_running_memory_threshold_warning', self.memory_threshold_warning))
                self.memory_threshold_critical = float(getattr(settings, 'long_running_memory_threshold_critical', self.memory_threshold_critical))
            except Exception:
                # Never fail initialization due to settings wiring
                logger.debug("ResourceMonitor settings application failed", exc_info=True)

    def get_current_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics.

        CPU percent is sampled system-wide to match Supervisor's shedding monitor.
        """
        process = psutil.Process()

        try:
            # System-wide CPU to avoid mismatch with Supervisor monitor.
            # interval=None uses the last computed value and is non-blocking; first call may yield 0.0
            # Prime psutil on first call if needed
            if self._ewma_cpu is None:
                _ = psutil.cpu_percent(interval=0.1)
            system_cpu_percent = psutil.cpu_percent(interval=None)
            # EWMA smoothing to reduce flapping
            if self._ewma_cpu is None:
                self._ewma_cpu = system_cpu_percent
            else:
                self._ewma_cpu = self._ewma_alpha * system_cpu_percent + (1 - self._ewma_alpha) * self._ewma_cpu
            # Process memory metrics
            memory_info = process.memory_info()
            # Prefer system memory pressure for health decisions while still reporting process RSS in MB
            try:
                memory_percent = psutil.virtual_memory().percent
            except Exception:
                memory_percent = process.memory_percent()

            # System-wide metrics
            try:
                disk_usage = psutil.disk_usage(os.path.abspath(os.sep))
            except Exception:
                disk_usage = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()

            metrics = ResourceMetrics(
                cpu_percent=self._ewma_cpu if self._ewma_cpu is not None else system_cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_info.rss / 1024 / 1024,
                disk_usage_percent=(disk_usage.used / disk_usage.total) * 100,
                network_io_mb=(net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024,
                open_files=len(process.open_files()),
                thread_count=process.num_threads(),
                timestamp=time.time()
            )

            # Keep history bounded
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)

            return metrics
        except Exception as e:
            logger.error(f"Failed to get resource metrics: {e}")
            return ResourceMetrics(0, 0, 0, 0, 0, 0, 0, time.time())

    def assess_health(self, metrics: ResourceMetrics) -> HealthStatus:
        """Assess system health based on metrics."""
        if (metrics.cpu_percent > self.cpu_threshold_critical or
            metrics.memory_percent > self.memory_threshold_critical):
            return HealthStatus.CRITICAL
        elif (metrics.cpu_percent > self.cpu_threshold_warning or
              metrics.memory_percent > self.memory_threshold_warning):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    def detect_trends(self) -> List[str]:
        """Detect concerning trends in resource usage."""
        if len(self.metrics_history) < 10:
            return []

        trends = []
        recent_metrics = self.metrics_history[-10:]

        # Check for increasing CPU trend, but only consider it if above warning threshold to avoid noise
        cpu_trend = sum(m.cpu_percent for m in recent_metrics[-5:]) - sum(m.cpu_percent for m in recent_metrics[:5])
        avg_recent = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        if cpu_trend > 20 and avg_recent >= self.cpu_threshold_warning:
            trends.append("CPU usage increasing rapidly")

        # Check for memory leaks
        memory_trend = sum(m.memory_mb for m in recent_metrics[-5:]) - sum(m.memory_mb for m in recent_metrics[:5])
        if memory_trend > 100:  # 100MB increase
            trends.append("Potential memory leak detected")

        # Check for thread accumulation
        thread_trend = sum(m.thread_count for m in recent_metrics[-3:]) - sum(m.thread_count for m in recent_metrics[:3])
        if thread_trend > 10:
            trends.append("Thread count increasing")

        return trends


class FailureAnalyzer:
    """Analyzes failure patterns and suggests recovery strategies."""

    def __init__(self):
        self.failure_history: List[Dict[str, Any]] = []
        self.patterns: List[FailurePattern] = []

    def record_failure(self, error_type: str, error_message: str, context: Dict[str, Any]):
        """Record a failure for pattern analysis."""
        failure_record = {
            'timestamp': time.time(),
            'error_type': error_type,
            'error_message': error_message,
            'context': context
        }
        self.failure_history.append(failure_record)

        # Keep history bounded
        if len(self.failure_history) > 1000:
            self.failure_history = self.failure_history[-500:]

        self._analyze_patterns()

    def _analyze_patterns(self):
        """Analyze failure history for patterns."""
        if len(self.failure_history) < 3:
            return

        recent_failures = self.failure_history[-10:]
        error_types = {}

        for failure in recent_failures:
            error_type = failure['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1

        # Update patterns
        for error_type, count in error_types.items():
            if count >= 3:
                existing_pattern = next((p for p in self.patterns if p.pattern_type == error_type), None)
                if existing_pattern:
                    existing_pattern.frequency += count
                    existing_pattern.last_occurrence = time.time()
                else:
                    pattern = FailurePattern(
                        pattern_type=error_type,
                        frequency=count,
                        last_occurrence=time.time(),
                        severity="high" if count >= 5 else "medium",
                        suggested_action=self._get_suggested_action(error_type)
                    )
                    self.patterns.append(pattern)

    def _get_suggested_action(self, error_type: str) -> str:
        """Get suggested recovery action for error type."""
        suggestions = {
            'BrowserError': 'Restart browser session',
            'TimeoutError': 'Increase timeout values and check network',
            'LLMException': 'Switch to backup LLM or reduce request frequency',
            'MemoryError': 'Clear caches and restart components',
            'ConnectionError': 'Check network connectivity and retry with backoff'
        }
        return suggestions.get(error_type, 'Investigate and restart affected components')


class StateCheckpointer:
    """Handles state checkpointing and recovery."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = 50

    async def create_checkpoint(self, state_manager: StateManager,
                               context_info: Dict[str, Any]) -> str:
        """Create a checkpoint of current agent state."""
        try:
            timestamp = time.time()
            checkpoint_id = f"checkpoint_{int(timestamp)}_{hashlib.md5(str(timestamp).encode()).hexdigest()[:8]}"

            # Serialize agent state
            state_dict = {
                'agent_id': state_manager.state.agent_id,
                'task': state_manager.state.task,
                'status': state_manager.state.status.value,
                'n_steps': state_manager.state.n_steps,
                'consecutive_failures': state_manager.state.consecutive_failures,
                'current_goal': state_manager.state.current_goal,
                'last_error': state_manager.state.last_error,
                'history': [self._serialize_history_item(item) for item in state_manager.state.history.history],
                'accumulated_output': state_manager.state.accumulated_output
            }

            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                timestamp=timestamp,
                agent_state=state_dict,
                step_number=state_manager.state.n_steps,
                browser_state=context_info.get('browser_state'),
                task_context=state_manager.state.task,
                recovery_hints=context_info.get('recovery_hints', [])
            )

            # Save to disk
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(asdict(checkpoint), f, indent=2, default=str)

            # Cleanup old checkpoints
            await self._cleanup_old_checkpoints()

            logger.info(f"Created checkpoint {checkpoint_id} at step {state_manager.state.n_steps}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return ""

    def _serialize_history_item(self, item: AgentHistory) -> Dict[str, Any]:
        """Serialize a history item for checkpointing."""
        try:
            return {
                'model_output': item.model_output.model_dump() if item.model_output else None,
                'result': [r.model_dump() if hasattr(r, 'model_dump') else str(r) for r in item.result],
                'state': item.state.model_dump() if item.state else None,
                'metadata': item.metadata.model_dump() if item.metadata else None
            }
        except Exception as e:
            logger.warning(f"Failed to serialize history item: {e}")
            return {'error': f'Serialization failed: {e}'}

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint from disk."""
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
            if not checkpoint_file.exists():
                return None

            with open(checkpoint_file, 'r') as f:
                data = json.load(f)

            return Checkpoint(**data)

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    async def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        try:
            checkpoints = []
            for file in self.checkpoint_dir.glob("checkpoint_*.json"):
                checkpoints.append(file.stem)
            return sorted(checkpoints, reverse=True)  # Most recent first
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    async def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to prevent disk space issues."""
        try:
            checkpoints = await self.list_checkpoints()
            if len(checkpoints) > self.max_checkpoints:
                for old_checkpoint in checkpoints[self.max_checkpoints:]:
                    checkpoint_file = self.checkpoint_dir / f"{old_checkpoint}.json"
                    if checkpoint_file.exists():
                        checkpoint_file.unlink()
                        logger.debug(f"Removed old checkpoint: {old_checkpoint}")
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")


class LongRunningMode:
    """
    Main controller for long-running operations mode.

    Orchestrates failure detection, recovery, and state management
    for extended agent operations.
    """

    def __init__(self, state_manager: StateManager,
                 checkpoint_dir: Optional[str] = None,
                 monitoring_interval: float = 30.0,
                 settings: Optional[Any] = None):
        self.state_manager = state_manager
        self.monitoring_interval = monitoring_interval
        self.mode = OperationMode.NORMAL
        self.health_status = HealthStatus.HEALTHY
        self.settings = settings  # Store settings for access to configuration

        # Initialize components
        self.resource_monitor = ResourceMonitor(settings)
        self.failure_analyzer = FailureAnalyzer()

        checkpoint_dir = checkpoint_dir or os.path.join(tempfile.gettempdir(), f'browser_use_checkpoints_{state_manager.state.agent_id}')
        self.checkpointer = StateCheckpointer(checkpoint_dir)
        # Apply settings-based limits
        try:
            if settings is not None:
                max_cps = int(getattr(settings, 'long_running_max_checkpoints', self.checkpointer.max_checkpoints))
                self.checkpointer.max_checkpoints = max(1, max_cps)
        except Exception:
            logger.debug("Failed to apply max checkpoints from settings", exc_info=True)

        # Circuit breakers for external services
        self.circuit_breakers = {
            'llm': CircuitBreaker(failure_threshold=3, recovery_timeout=60.0),
            'browser': CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
        }

        # Monitoring state
        self.last_checkpoint_time = 0
        # Use settings-specified checkpoint interval when available
        try:
            self.checkpoint_interval = float(getattr(settings, 'long_running_checkpoint_interval', 300.0)) if settings else 300.0
        except Exception:
            self.checkpoint_interval = 300.0  # 5 minutes
        self.running = False
        self.intervention_cooldown = {}

    async def start_monitoring(self):
        """Start the long-running monitoring loop."""
        self.running = True
        logger.info("Long-running mode monitoring started")

        try:
            while self.running:
                await self._monitor_cycle()
                await asyncio.sleep(self.monitoring_interval)
        except asyncio.CancelledError:
            logger.info("Long-running mode monitoring cancelled")
        except Exception as e:
            logger.error(f"Long-running mode monitoring failed: {e}", exc_info=True)
        finally:
            self.running = False

    async def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.running = False

    async def _monitor_cycle(self):
        """Execute one monitoring cycle."""
        try:
            # Collect resource metrics
            metrics = self.resource_monitor.get_current_metrics()
            health_status = self.resource_monitor.assess_health(metrics)

            # Update health status
            if health_status != self.health_status:
                logger.info(f"Health status changed: {self.health_status.value} -> {health_status.value}")
                self.health_status = health_status
                await self._adjust_operation_mode()

            # Check for concerning trends
            trends = self.resource_monitor.detect_trends()
            if trends:
                logger.warning(f"Resource trends detected: {trends}")
                for trend in trends:
                    await self._handle_trend(trend)

            # Create periodic checkpoints
            if time.time() - self.last_checkpoint_time > self.checkpoint_interval:
                await self._create_periodic_checkpoint()

            # Log health summary
            agent_log(
                logging.DEBUG,
                self.state_manager.state.agent_id,
                self.state_manager.state.n_steps,
                f"Long-running health: {health_status.value}, Mode: {self.mode.value}, "
                f"CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_percent:.1f}%"
            )

        except Exception as e:
            logger.error(f"Monitor cycle failed: {e}", exc_info=True)

    async def _adjust_operation_mode(self):
        """Adjust operation mode based on health status."""
        if self.health_status == HealthStatus.CRITICAL:
            await self._enter_mode(OperationMode.MINIMAL)
        elif self.health_status == HealthStatus.WARNING:
            await self._enter_mode(OperationMode.DEGRADED)
        elif self.health_status == HealthStatus.HEALTHY and self.mode != OperationMode.NORMAL:
            await self._enter_mode(OperationMode.NORMAL)

    async def _enter_mode(self, new_mode: OperationMode):
        """Transition to a new operation mode."""
        if new_mode == self.mode:
            return

        old_mode = self.mode
        self.mode = new_mode

        logger.info(f"Operation mode transition: {old_mode.value} -> {new_mode.value}")

        # Apply mode-specific configurations
        if new_mode == OperationMode.MINIMAL:
            await self._apply_minimal_mode()
        elif new_mode == OperationMode.DEGRADED:
            await self._apply_degraded_mode()
        elif new_mode == OperationMode.NORMAL:
            await self._apply_normal_mode()
        elif new_mode == OperationMode.RECOVERY:
            await self._apply_recovery_mode()

        agent_log(
            logging.INFO,
            self.state_manager.state.agent_id,
            self.state_manager.state.n_steps,
            f"Entered {new_mode.value} operation mode due to {self.health_status.value} health status"
        )

    async def _apply_minimal_mode(self):
        """Apply minimal operation mode settings."""
        # Reduce monitoring frequency
        self.monitoring_interval = max(60.0, self.monitoring_interval)

        # Signal to components to reduce activity
        try:
            # Use enum value to ensure proper ingestion
            from browser_use.agent.state_manager import LoadStatus
            await self.state_manager.ingest_signal('load_status',
                                                  {'status': LoadStatus.SHEDDING})
        except Exception as e:
            logger.warning(f"Failed to signal load shedding: {e}")

    async def _apply_degraded_mode(self):
        """Apply degraded operation mode settings."""
        self.monitoring_interval = max(45.0, self.monitoring_interval)

    async def _apply_normal_mode(self):
        """Apply normal operation mode settings."""
        self.monitoring_interval = 30.0

        try:
            from browser_use.agent.state_manager import LoadStatus
            await self.state_manager.ingest_signal('load_status',
                                                  {'status': LoadStatus.NORMAL})
        except Exception as e:
            logger.warning(f"Failed to signal normal load: {e}")

    async def _apply_recovery_mode(self):
        """Apply recovery operation mode settings."""
        # Increase checkpoint frequency
        self.checkpoint_interval = 60.0  # 1 minute

    async def _handle_trend(self, trend: str):
        """Handle concerning resource trends."""
        trend_key = trend.replace(' ', '_').lower()

        # Avoid excessive interventions
        if trend_key in self.intervention_cooldown:
            if time.time() - self.intervention_cooldown[trend_key] < 300:  # 5 minutes
                return

        self.intervention_cooldown[trend_key] = time.time()

        if "memory" in trend.lower():
            await self._handle_memory_pressure()
        elif "cpu" in trend.lower():
            await self._handle_cpu_pressure()
        elif "thread" in trend.lower():
            await self._handle_thread_accumulation()

        logger.warning(f"Handled resource trend: {trend}")

    async def _handle_memory_pressure(self):
        """Handle memory pressure situations."""
        try:
            # Signal components to clear caches
            await self.state_manager.ingest_signal('memory_pressure', {})

            # Force garbage collection if available
            import gc
            gc.collect()

            logger.info("Applied memory pressure mitigation")
        except Exception as e:
            logger.error(f"Failed to handle memory pressure: {e}")

    async def _handle_cpu_pressure(self):
        """Handle CPU pressure situations."""
        try:
            # Increase operation intervals to reduce CPU load
            self.monitoring_interval = min(120.0, self.monitoring_interval * 1.5)

            logger.info("Applied CPU pressure mitigation")
        except Exception as e:
            logger.error(f"Failed to handle CPU pressure: {e}")

    async def _handle_thread_accumulation(self):
        """Handle thread accumulation."""
        try:
            # Signal components to cleanup threads
            await self.state_manager.ingest_signal('thread_cleanup', {})

            logger.info("Applied thread cleanup")
        except Exception as e:
            logger.error(f"Failed to handle thread accumulation: {e}")

    async def _create_periodic_checkpoint(self):
        """Create a periodic checkpoint."""
        try:
            context_info = {
                'recovery_hints': [
                    f"Created during {self.mode.value} mode",
                    f"Health status: {self.health_status.value}",
                    f"Last checkpoint interval: {self.checkpoint_interval}s"
                ]
            }

            checkpoint_id = await self.checkpointer.create_checkpoint(
                self.state_manager, context_info
            )

            if checkpoint_id:
                self.last_checkpoint_time = time.time()
                agent_log(
                    logging.INFO,
                    self.state_manager.state.agent_id,
                    self.state_manager.state.n_steps,
                    f"Created periodic checkpoint: {checkpoint_id}"
                )
        except Exception as e:
            logger.error(f"Failed to create periodic checkpoint: {e}")

    async def handle_failure(self, error: Exception, context: Dict[str, Any]):
        """Handle a detected failure with intelligent recovery."""
        error_type = type(error).__name__
        error_message = str(error)

        # Record failure for pattern analysis
        self.failure_analyzer.record_failure(error_type, error_message, context)

        # Create recovery checkpoint
        recovery_context = {
            'error_type': error_type,
            'error_message': error_message,
            'recovery_hints': [
                f"Failure occurred at step {self.state_manager.state.n_steps}",
                f"Error type: {error_type}",
                "Created for failure recovery"
            ]
        }

        checkpoint_id = await self.checkpointer.create_checkpoint(
            self.state_manager, recovery_context
        )

        # Determine recovery strategy
        recovery_action = self._normalize_action(
            self._determine_recovery_strategy(error_type, error_message)
        )

        logger.error(
            f"Failure handled: {error_type}({error_message}). "
            f"Checkpoint: {checkpoint_id}, Recovery: {recovery_action}"
        )

        # AUTONOMOUS CONTINUATION: Try to continue after brief recovery delay
        await self._attempt_autonomous_continuation(recovery_action, context)

    async def _attempt_autonomous_continuation(self, recovery_action: str, context: Dict[str, Any]):
        """Attempt to continue operation autonomously after failure recovery."""
        # Check if autonomous continuation is enabled
        autonomous_enabled = getattr(self.settings, 'long_running_enable_autonomous_continuation', True) if self.settings else True

        if not autonomous_enabled:
            logger.info("Autonomous continuation disabled, waiting for manual intervention")
            return

        # Brief recovery delay based on failure severity
        delay = self._calculate_recovery_delay(recovery_action)
        logger.info(f"Autonomous continuation starting in {delay}s with strategy: {recovery_action}")
        await asyncio.sleep(delay)

        try:
            # Execute recovery strategy
            if recovery_action == "restart_browser_session":
                await self._restart_browser_session()
            elif recovery_action == "clear_caches_and_restart":
                await self._clear_caches_and_restart()
            elif recovery_action == "increase_timeouts_and_retry":
                await self._increase_timeouts()
            elif recovery_action == "switch_to_backup_llm":
                await self._switch_to_backup_llm()
            elif recovery_action == "force_step_progression":
                await self._force_step_progression()
            elif recovery_action == "refresh_page_and_retry":
                await self._refresh_page_and_retry()
            elif recovery_action == "restart_affected_components":
                await self._restart_affected_components()
            else:
                logger.warning(f"Unknown recovery action: {recovery_action}")

            # Signal the supervisor that we're ready to continue
            if hasattr(self.state_manager, 'bus') and self.state_manager.bus:
                continuation_event = {
                    'type': 'autonomous_continuation',
                    'timestamp': time.time(),
                    'recovery_action': recovery_action,
                    'context': context
                }
                await self.state_manager.bus.put(continuation_event)
                logger.info("Autonomous continuation signal sent to supervisor")

        except Exception as e:
            logger.error(f"Autonomous continuation failed: {e}")

    def _calculate_recovery_delay(self, recovery_action: str) -> int:
        """Calculate appropriate delay before attempting continuation."""
        delay_map = {
            "restart_browser_session": 10,  # Browser restart needs time
            "clear_caches_and_restart": 15,  # Cache clearing + restart
            "increase_timeouts_and_retry": 3,  # Quick timeout adjustment
            "switch_to_backup_llm": 5,  # LLM switch is relatively fast
            "restart_affected_components": 8,  # Component restart
            "force_step_progression": 2,  # Quick step progression
            "refresh_page_and_retry": 5,  # Page refresh is relatively fast
        }
        return delay_map.get(recovery_action, 5)  # Default 5 seconds

    async def _restart_browser_session(self):
        """Restart the browser session via supervisor hook, if available."""
        logger.info("Restarting browser session for autonomous continuation")
        try:
            supervisor = getattr(self.state_manager, '_supervisor', None)
            if supervisor and hasattr(supervisor, 'restart_browser_session'):
                await supervisor.restart_browser_session()
            else:
                logger.warning("Supervisor restart hook not available; skipping browser restart")
        except Exception:
            logger.error("Browser session restart failed", exc_info=True)

    async def _clear_caches_and_restart(self):
        """Clear system caches and restart components."""
        logger.info("Clearing caches for autonomous continuation")
        try:
            # Try to ask components to cleanup
            try:
                await self.state_manager.ingest_signal('memory_pressure', {})
            except Exception:
                pass
            # Restart critical components if possible
            supervisor = getattr(self.state_manager, '_supervisor', None)
            if supervisor and hasattr(supervisor, 'restart_components'):
                await supervisor.restart_components()
        except Exception:
            logger.debug("Cache clear/restart had issues", exc_info=True)

    async def _increase_timeouts(self):
        """Increase timeout values for subsequent operations."""
        logger.info("Increasing timeouts for autonomous continuation")
        try:
            # If settings expose timeouts, increase conservatively
            if self.settings and hasattr(self.settings, 'controller'):
                ctrl = self.settings.controller
                for name in ("default_timeout", "nav_timeout", "action_timeout"):
                    if hasattr(ctrl, name):
                        cur = getattr(ctrl, name)
                        try:
                            setattr(ctrl, name, min(cur * 2, 120.0))
                        except Exception:
                            pass
        except Exception:
            logger.debug("Timeout increase failed or not applicable", exc_info=True)

    async def _switch_to_backup_llm(self):
        """Switch to backup LLM provider if available."""
        logger.info("Switching to backup LLM for autonomous continuation")
        try:
            if self.settings and getattr(self.settings, 'backup_llm', None) is not None:
                self.settings.llm = self.settings.backup_llm
                logger.info("Backup LLM activated")
        except Exception:
            logger.debug("Backup LLM switch not available", exc_info=True)

    async def _force_step_progression(self):
        """Force progression to next step to break deadlocks via supervisor hook."""
        logger.info("Forcing step progression to break deadlock")
        try:
            supervisor = getattr(self.state_manager, '_supervisor', None)
            if supervisor and hasattr(supervisor, 'force_step_progression'):
                await supervisor.force_step_progression()
            else:
                # Fallback: emit a StepFinalized with current step
                if hasattr(self.state_manager, 'bus') and self.state_manager.bus:
                    await self.state_manager.bus.put({'type': 'autonomous_continuation', 'recovery_action': 'force_step_progression'})
        except Exception:
            logger.debug("Force step progression fallback path used", exc_info=True)

    async def _refresh_page_and_retry(self):
        """Refresh the current page and retry the operation via browser session."""
        logger.info("Refreshing page to resolve element issues")
        try:
            supervisor = getattr(self.state_manager, '_supervisor', None)
            if supervisor and getattr(supervisor, 'browser_session', None):
                try:
                    await supervisor.browser_session.refresh_page()
                except AttributeError:
                    # Fallback to refresh()
                    await supervisor.browser_session.refresh()
            # Nudge progression afterwards
            if supervisor and hasattr(supervisor, 'force_step_progression'):
                await supervisor.force_step_progression()
        except Exception:
            logger.debug("Page refresh failed or not supported", exc_info=True)

    async def _restart_affected_components(self):
        """Ask supervisor to restart components implicated by failure patterns."""
        try:
            supervisor = getattr(self.state_manager, '_supervisor', None)
            if supervisor and hasattr(supervisor, 'restart_components'):
                await supervisor.restart_components()
        except Exception:
            logger.debug("Component restart hook failed", exc_info=True)

    def _normalize_action(self, action: str) -> str:
        """Map freeform or human phrased actions to internal action keys."""
        mapping = {
            'investigate and restart affected components': 'restart_affected_components',
            'restart affected components': 'restart_affected_components',
            'restart components': 'restart_affected_components',
        }
        key = action.lower().strip()
        return mapping.get(key, action)

    def _determine_recovery_strategy(self, error_type: str, error_message: str) -> str:
        """Determine the best recovery strategy for a failure."""
        # Check failure patterns
        pattern = next((p for p in self.failure_analyzer.patterns
                       if p.pattern_type == error_type), None)

        if pattern and pattern.frequency > 5:
            return pattern.suggested_action

        # Default strategies based on error type and message
        if "deadlock" in error_message.lower():
            return "force_step_progression"
        elif "timeout" in error_message.lower():
            return "increase_timeouts_and_retry"
        elif "memory" in error_message.lower():
            return "clear_caches_and_restart"
        elif "browser" in error_type.lower():
            return "restart_browser_session"
        elif "llm" in error_type.lower():
            return "switch_to_backup_llm"
        elif "element" in error_message.lower() and "not exist" in error_message.lower():
            return "refresh_page_and_retry"
        else:
            return "restart_affected_components"

    async def recover_from_checkpoint(self, checkpoint_id: Optional[str] = None) -> bool:
        """Recover agent state from a checkpoint."""
        try:
            if not checkpoint_id:
                # Find most recent checkpoint
                checkpoints = await self.checkpointer.list_checkpoints()
                if not checkpoints:
                    logger.error("No checkpoints available for recovery")
                    return False
                checkpoint_id = checkpoints[0]

            checkpoint = await self.checkpointer.load_checkpoint(checkpoint_id)
            if not checkpoint:
                logger.error(f"Failed to load checkpoint {checkpoint_id}")
                return False

            # Enter recovery mode
            await self._enter_mode(OperationMode.RECOVERY)

            # Apply checkpoint state (partial recovery - critical fields only)
            async with self.state_manager._lock:
                self.state_manager._state.task = checkpoint.agent_state.get('task', '')
                self.state_manager._state.current_goal = checkpoint.agent_state.get('current_goal', '')
                # Note: We don't restore full history to avoid inconsistencies

            logger.info(f"Successfully recovered from checkpoint {checkpoint_id}")

            # Return to normal mode after recovery
            await asyncio.sleep(5.0)  # Brief stabilization period
            await self._enter_mode(OperationMode.NORMAL)

            return True

        except Exception as e:
            logger.error(f"Failed to recover from checkpoint: {e}", exc_info=True)
            return False

    def get_circuit_breaker(self, service: str) -> CircuitBreaker:
        """Get circuit breaker for a service."""
        return self.circuit_breakers.get(service, CircuitBreaker())

    async def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        metrics = self.resource_monitor.get_current_metrics()
        checkpoints = await self.checkpointer.list_checkpoints()

        return {
            'operation_mode': self.mode.value,
            'health_status': self.health_status.value,
            'resource_metrics': asdict(metrics),
            'failure_patterns': [asdict(p) for p in self.failure_analyzer.patterns],
            'recent_checkpoints': checkpoints[:5],
            'circuit_breaker_states': {
                name: {'state': cb.state, 'failures': cb.failure_count}
                for name, cb in self.circuit_breakers.items()
            },
            'trends': self.resource_monitor.detect_trends()
        }
