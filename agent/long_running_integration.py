"""
Integration module for Long-Running Operations Mode.

This module provides the integration points between the long-running mode
and the existing agent components (Supervisor, StateManager, etc.).
"""

import asyncio
import logging
from typing import Optional, Dict, Any, TYPE_CHECKING

from browser_use.agent.long_running_mode import LongRunningMode, OperationMode, HealthStatus
from browser_use.agent.state_manager import StateManager, agent_log

if TYPE_CHECKING:
    from browser_use.agent.supervisor import Supervisor

logger = logging.getLogger(__name__)


class LongRunningIntegration:
    """
    Integration layer for long-running mode with the existing agent architecture.

    This class manages the lifecycle of long-running mode and provides
    hooks into the agent's normal operation flow.
    """

    def __init__(self, supervisor: 'Supervisor', enabled: bool = False):
        self.supervisor = supervisor
        self.enabled = enabled
        self.long_running_mode: Optional[LongRunningMode] = None
        self.monitoring_task: Optional[asyncio.Task] = None

    async def initialize(self) -> bool:
        """Initialize long-running mode if enabled."""
        if not self.enabled:
            return False

        try:
            # Check if we have required dependencies
            try:
                import psutil
            except ImportError:
                logger.error("psutil is required for long-running mode but not installed")
                self.enabled = False
                return False

            self.long_running_mode = LongRunningMode(
                state_manager=self.supervisor.state_manager,
                monitoring_interval=getattr(self.supervisor.settings, 'long_running_monitoring_interval', 30.0),
                settings=self.supervisor.settings
            )

            agent_log(
                logging.INFO,
                self.supervisor.state_manager.state.agent_id,
                self.supervisor.state_manager.state.n_steps,
                "Long-running mode initialized successfully"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize long-running mode: {e}", exc_info=True)
            self.enabled = False
            return False

    async def start_monitoring(self):
        """Start long-running mode monitoring."""
        if not self.long_running_mode:
            return

        try:
            self.monitoring_task = asyncio.create_task(
                self.long_running_mode.start_monitoring()
            )

            agent_log(
                logging.INFO,
                self.supervisor.state_manager.state.agent_id,
                self.supervisor.state_manager.state.n_steps,
                "Long-running mode monitoring started"
            )

        except Exception as e:
            logger.error(f"Failed to start long-running monitoring: {e}", exc_info=True)

    async def stop_monitoring(self):
        """Stop long-running mode monitoring."""
        if self.long_running_mode:
            await self.long_running_mode.stop_monitoring()

        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        agent_log(
            logging.INFO,
            self.supervisor.state_manager.state.agent_id,
            self.supervisor.state_manager.state.n_steps,
            "Long-running mode monitoring stopped"
        )

    async def handle_component_failure(self, component_name: str, error: Exception) -> str:
        """Handle component failure through long-running mode."""
        if not self.long_running_mode:
            return "restart_component"  # Default fallback

        try:
            context = {
                'component': component_name,
                'step': self.supervisor.state_manager.state.n_steps,
                'agent_id': self.supervisor.state_manager.state.agent_id
            }

            recovery_action = await self.long_running_mode.handle_failure(error, context)

            agent_log(
                logging.WARNING,
                self.supervisor.state_manager.state.agent_id,
                self.supervisor.state_manager.state.n_steps,
                f"Long-running mode handled {component_name} failure: {recovery_action}"
            )

            return recovery_action

        except Exception as e:
            logger.error(f"Failed to handle component failure through long-running mode: {e}")
            return "restart_component"

    async def handle_browser_failure(self, error: Exception) -> str:
        """Handle browser session failure through long-running mode."""
        if not self.long_running_mode:
            return "restart_browser"

        try:
            # Use circuit breaker for browser operations
            circuit_breaker = self.long_running_mode.get_circuit_breaker('browser')

            context = {
                'service': 'browser',
                'circuit_breaker_state': circuit_breaker.state,
                'failure_count': circuit_breaker.failure_count
            }

            recovery_action = await self.long_running_mode.handle_failure(error, context)

            agent_log(
                logging.WARNING,
                self.supervisor.state_manager.state.agent_id,
                self.supervisor.state_manager.state.n_steps,
                f"Long-running mode handled browser failure: {recovery_action}"
            )

            return recovery_action

        except Exception as e:
            logger.error(f"Failed to handle browser failure through long-running mode: {e}")
            return "restart_browser"

    async def handle_llm_failure(self, error: Exception) -> str:
        """Handle LLM failure through long-running mode."""
        if not self.long_running_mode:
            return "retry_with_backoff"

        try:
            # Use circuit breaker for LLM operations
            circuit_breaker = self.long_running_mode.get_circuit_breaker('llm')

            context = {
                'service': 'llm',
                'circuit_breaker_state': circuit_breaker.state,
                'failure_count': circuit_breaker.failure_count
            }

            recovery_action = await self.long_running_mode.handle_failure(error, context)

            agent_log(
                logging.WARNING,
                self.supervisor.state_manager.state.agent_id,
                self.supervisor.state_manager.state.n_steps,
                f"Long-running mode handled LLM failure: {recovery_action}"
            )

            return recovery_action

        except Exception as e:
            logger.error(f"Failed to handle LLM failure through long-running mode: {e}")
            return "retry_with_backoff"

    async def create_emergency_checkpoint(self, reason: str) -> Optional[str]:
        """Create an emergency checkpoint."""
        if not self.long_running_mode:
            return None

        try:
            context_info = {
                'reason': reason,
                'emergency': True,
                'timestamp': str(asyncio.get_event_loop().time()),
                'recovery_hints': [
                    f"Emergency checkpoint due to: {reason}",
                    "Immediate recovery may be needed",
                    "Check system health before resuming"
                ]
            }

            checkpoint_id = await self.long_running_mode.checkpointer.create_checkpoint(
                self.supervisor.state_manager, context_info
            )

            if checkpoint_id:
                agent_log(
                    logging.CRITICAL,
                    self.supervisor.state_manager.state.agent_id,
                    self.supervisor.state_manager.state.n_steps,
                    f"Emergency checkpoint created: {checkpoint_id} (reason: {reason})"
                )

            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to create emergency checkpoint: {e}", exc_info=True)
            return None

    async def attempt_recovery(self, checkpoint_id: Optional[str] = None) -> bool:
        """Attempt to recover from a checkpoint."""
        if not self.long_running_mode:
            return False

        try:
            success = await self.long_running_mode.recover_from_checkpoint(checkpoint_id)

            if success:
                agent_log(
                    logging.INFO,
                    self.supervisor.state_manager.state.agent_id,
                    self.supervisor.state_manager.state.n_steps,
                    f"Successfully recovered from checkpoint: {checkpoint_id or 'latest'}"
                )
            else:
                agent_log(
                    logging.ERROR,
                    self.supervisor.state_manager.state.agent_id,
                    self.supervisor.state_manager.state.n_steps,
                    f"Failed to recover from checkpoint: {checkpoint_id or 'latest'}"
                )

            return success

        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}", exc_info=True)
            return False

    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status from long-running mode."""
        if not self.long_running_mode:
            return {
                'enabled': False,
                'status': 'disabled'
            }

        try:
            health_report = await self.long_running_mode.get_health_report()
            health_report['enabled'] = True
            return health_report

        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {
                'enabled': True,
                'status': 'error',
                'error': str(e)
            }

    def is_degraded_mode(self) -> bool:
        """Check if the agent is in degraded operation mode."""
        if not self.long_running_mode:
            return False

        return self.long_running_mode.mode in [
            OperationMode.DEGRADED,
            OperationMode.MINIMAL,
            OperationMode.RECOVERY
        ]

    def should_reduce_activity(self) -> bool:
        """Check if components should reduce their activity level."""
        if not self.long_running_mode:
            return False

        return (self.long_running_mode.mode == OperationMode.MINIMAL or
                self.long_running_mode.health_status in [HealthStatus.CRITICAL, HealthStatus.FAILING])

    async def cleanup(self):
        """Cleanup long-running mode resources."""
        await self.stop_monitoring()

        if self.long_running_mode:
            # Perform any necessary cleanup
            try:
                await self.long_running_mode.checkpointer._cleanup_old_checkpoints()
            except Exception as e:
                logger.warning(f"Failed to cleanup checkpoints: {e}")


# Decorator for circuit breaker integration
def with_circuit_breaker(service: str):
    """Decorator to wrap functions with circuit breaker protection."""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Check if long-running mode is available
            if hasattr(self, 'long_running_integration') and self.long_running_integration.long_running_mode:
                circuit_breaker = self.long_running_integration.long_running_mode.get_circuit_breaker(service)
                return await circuit_breaker.call(func, self, *args, **kwargs)
            else:
                # Fallback to normal execution
                return await func(self, *args, **kwargs)
        return wrapper
    return decorator


# Helper function to check if long-running mode should intervene
async def should_trigger_intervention(state_manager: StateManager, error_type: str) -> bool:
    """
    Check if the current error pattern warrants long-running mode intervention.
    """
    try:
        # Check consecutive failures
        if state_manager.state.consecutive_failures >= 3:
            return True

        # Check for specific error patterns that indicate systemic issues
        systemic_errors = [
            'BrowserError', 'TimeoutError', 'ConnectionError',
            'MemoryError', 'ResourceExhausted'
        ]

        if error_type in systemic_errors:
            return True

        # Check recent error frequency
        if hasattr(state_manager.state, 'last_error') and state_manager.state.last_error:
            # If the same error occurred recently, intervene
            return error_type in str(state_manager.state.last_error)

        return False

    except Exception as e:
        logger.error(f"Failed to check intervention criteria: {e}")
        return False
