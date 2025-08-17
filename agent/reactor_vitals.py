"""
ReactorVitals component for monitoring component health via heartbeat protocol.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional, TYPE_CHECKING, Callable, Awaitable

from browser_use.agent.events import ErrorEvent, Heartbeat
from browser_use.agent.state_manager import agent_log

if TYPE_CHECKING:
    from browser_use.agent.state_manager import StateManager

logger = logging.getLogger(__name__)


class ComponentMonitor:
    """Monitors a single component's heartbeat status."""

    def __init__(self, component_name: str, timeout: float = 10.0):
        self.component_name = component_name
        self.timeout = timeout
        self.last_heartbeat = time.monotonic()
        self.restart_attempts = 0
        self.max_restart_attempts = 3
        self.restart_backoff = [2.0, 4.0, 8.0]  # Exponential backoff delays
        self.is_restarting = False
        # Track whether we've entered an expired state (for HB signals)
        self.was_expired = False

    def update_heartbeat(self):
        """Update the last heartbeat timestamp."""
        self.last_heartbeat = time.monotonic()
        # Reset restart attempts on successful heartbeat
        if self.restart_attempts > 0:
            logger.info(f"Component {self.component_name} recovered, resetting restart attempts")
            self.restart_attempts = 0
            self.is_restarting = False

    def is_expired(self) -> bool:
        """Check if the heartbeat has expired."""
        return (time.monotonic() - self.last_heartbeat) > self.timeout

    def get_next_backoff_delay(self) -> float:
        """Get the next backoff delay for restart attempts."""
        if self.restart_attempts < len(self.restart_backoff):
            return self.restart_backoff[self.restart_attempts]
        return self.restart_backoff[-1]  # Use last delay for subsequent attempts


class ReactorVitals:
    """
    Monitors component health via heartbeat protocol and handles component restarts.
    """

    def __init__(self, state_manager: StateManager, agent_bus: asyncio.Queue,
                 heartbeat_bus: asyncio.Queue, heartbeat_timeout: float = 10.0):
        self.state_manager = state_manager
        self.agent_bus = agent_bus
        self.heartbeat_bus = heartbeat_bus
        self.heartbeat_timeout = heartbeat_timeout
        self.monitors: Dict[str, ComponentMonitor] = {}
        self.component_tasks: Dict[str, asyncio.Task] = {}
        # Factories to (re)start component tasks lazily on demand
        # Each factory returns a Task or a coroutine that will be wrapped into a Task
        self.restart_factories: Dict[str, Callable[[], Awaitable[asyncio.Task] | asyncio.Task]] = {}
        self.running = False

    def set_restart_factory(self, component_name: str, factory: Callable[[], Awaitable[asyncio.Task] | asyncio.Task]):
        """Register or update a restart factory for a single component."""
        self.restart_factories[component_name] = factory
        logger.debug(f"Restart factory set for component '{component_name}'")

    def set_restart_factories(self, factories: Dict[str, Callable[[], Awaitable[asyncio.Task] | asyncio.Task]]):
        """Bulk register restart factories for multiple components."""
        self.restart_factories.update(factories)
        logger.debug(f"Restart factories registered for components: {list(factories.keys())}")

    def register_component(self, component_name: str, task: asyncio.Task):
        """Register a component for monitoring."""
        self.monitors[component_name] = ComponentMonitor(component_name, self.heartbeat_timeout)
        self.component_tasks[component_name] = task
        logger.debug(f"Registered component '{component_name}' for heartbeat monitoring")

    def unregister_component(self, component_name: str):
        """Unregister a component from monitoring."""
        if component_name in self.monitors:
            del self.monitors[component_name]
        if component_name in self.component_tasks:
            del self.component_tasks[component_name]
        logger.debug(f"Unregistered component '{component_name}' from heartbeat monitoring")

    async def run(self):
        """Main monitoring loop."""
        self.running = True
        agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                 "ReactorVitals started monitoring component health")

        try:
            while self.running:
                try:
                    # Check for heartbeat events on the dedicated heartbeat bus
                    event = await asyncio.wait_for(self.heartbeat_bus.get(), timeout=1.0)

                    if isinstance(event, Heartbeat):
                        await self._handle_heartbeat(event)
                        self.heartbeat_bus.task_done()
                    else:
                        # Not our event. Put it back on the bus and yield control
                        # to give the correct consumer a chance to pick it up.
                        try:
                            self.heartbeat_bus.put_nowait(event)
                        except asyncio.QueueFull:
                            logger.warning("Heartbeat bus full, cannot requeue event. Event may be lost.")
                        finally:
                            self.heartbeat_bus.task_done()
                            # CRITICAL: Yield control to the event loop
                            await asyncio.sleep(0)

                except asyncio.TimeoutError:
                    # Timeout is expected, use it to check component health
                    await self._check_component_health()
                    continue
                except Exception as e:
                    error_msg = f"ReactorVitals monitoring loop failed: {e}"
                    agent_log(logging.ERROR, self.state_manager.state.agent_id,
                             self.state_manager.state.n_steps, error_msg, exc_info=True)
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                     "ReactorVitals monitoring cancelled")
            raise
        finally:
            self.running = False

    async def _handle_heartbeat(self, event: Heartbeat):
        """Process a heartbeat event."""
        component_name = event.component_name

        # Initialize monitor if not exists
        if component_name not in self.monitors:
            self.monitors[component_name] = ComponentMonitor(component_name, self.heartbeat_timeout)

        # Ignore heartbeats during an in-flight restart to avoid racey state flips
        monitor = self.monitors[component_name]
        if monitor.is_restarting:
            logger.debug(f"Ignoring heartbeat from {component_name} during restart window")
            return

        # Update heartbeat timestamp (resets restart attempts on recovery)
        # If we were previously expired, emit heartbeat_ok before updating
        if monitor.was_expired:
            try:
                await self.state_manager.ingest_signal('heartbeat_ok')
            except Exception:
                logger.debug("Failed to emit heartbeat_ok signal", exc_info=True)
            monitor.was_expired = False

        monitor.update_heartbeat()

        logger.debug(f"Heartbeat received from {component_name} at step {event.step_token}")

    async def _check_component_health(self):
        """Check all registered components for expired heartbeats."""
        # CRITICAL FIX: Don't restart components if agent is in terminal state
        from browser_use.agent.state_manager import TERMINAL_STATES

        current_status = await self.state_manager.get_status()
        if current_status in TERMINAL_STATES:
            # Agent is done, no need to restart components
            return

        for component_name, monitor in self.monitors.items():
            if monitor.is_expired() and not monitor.is_restarting:
                # Emit heartbeat_miss once when entering expired state
                if not monitor.was_expired:
                    try:
                        await self.state_manager.ingest_signal('heartbeat_miss')
                    except Exception:
                        logger.debug("Failed to emit heartbeat_miss signal", exc_info=True)
                    monitor.was_expired = True
                await self._handle_expired_component(component_name, monitor)

    async def _handle_expired_component(self, component_name: str, monitor: ComponentMonitor):
        """Handle a component with expired heartbeat."""
        # Double-check terminal state before restart attempt
        from browser_use.agent.state_manager import TERMINAL_STATES

        current_status = await self.state_manager.get_status()
        if current_status in TERMINAL_STATES:
            # Agent is done, don't restart components
            monitor.is_restarting = False
            return

        agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                 f"Component {component_name} heartbeat expired (timeout: {monitor.timeout}s)")

        if monitor.restart_attempts >= monitor.max_restart_attempts:
            # Escalate to supervisor
            await self._escalate_component_failure(component_name)
            return

        # Attempt restart with backoff
        monitor.is_restarting = True
        monitor.restart_attempts += 1
        backoff_delay = monitor.get_next_backoff_delay()

        agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                 f"Attempting restart #{monitor.restart_attempts} of {component_name} "
                 f"after {backoff_delay}s backoff")

        await asyncio.sleep(backoff_delay)

        # Try to restart the component
        success = await self._restart_component(component_name)

        if success:
            agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                     f"Successfully restarted component {component_name}")
            # Reset heartbeat timer for the restarted component
            monitor.last_heartbeat = time.monotonic()
        else:
            agent_log(logging.ERROR, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                     f"Failed to restart component {component_name}")

        monitor.is_restarting = False

    async def _restart_component(self, component_name: str) -> bool:
        """Attempt to restart a component."""
        try:
            # Cancel the old task if it exists
            if component_name in self.component_tasks:
                old_task = self.component_tasks[component_name]
                if not old_task.done():
                    old_task.cancel()
                    try:
                        await old_task
                    except asyncio.CancelledError:
                        pass

            # Use restart factory to create a new task
            factory = self.restart_factories.get(component_name)
            if factory is None:
                agent_log(
                    logging.WARNING,
                    self.state_manager.state.agent_id,
                    self.state_manager.state.n_steps,
                    f"No restart factory registered for {component_name}"
                )
                return False

            try:
                new_task_or_coro = factory()
                if asyncio.iscoroutine(new_task_or_coro):
                    new_task = asyncio.create_task(new_task_or_coro)  # type: ignore[arg-type]
                else:
                    new_task = new_task_or_coro  # type: ignore[assignment]
            except Exception as e:
                agent_log(logging.ERROR, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                         f"Restart factory for {component_name} raised: {e}", exc_info=True)
                return False

            def _log_task_exceptions(task: asyncio.Task):
                try:
                    exc = task.exception()
                    if exc:
                        agent_log(
                            logging.ERROR,
                            self.state_manager.state.agent_id,
                            self.state_manager.state.n_steps,
                            f"Restarted component {component_name} task crashed: {exc}",
                            exc_info=True,
                        )
                except asyncio.CancelledError:
                    # Expected on shutdown
                    pass
                except Exception as e:
                    logger.debug(f"Exception while inspecting restarted task for {component_name}: {e}")

            new_task.add_done_callback(_log_task_exceptions)
            self.component_tasks[component_name] = new_task

            # Nudge the supervisor loop by emitting a non-critical ErrorEvent
            try:
                evt = ErrorEvent(
                    step_token=self.state_manager.state.n_steps,
                    error_message=f"Component {component_name} restarted",
                    error_type="ComponentRestarted",
                    is_critical=False,
                )
                self.agent_bus.put_nowait(evt)
            except asyncio.QueueFull:
                logger.debug("Agent bus full; could not emit ComponentRestarted event")

            return True

        except Exception as e:
            agent_log(logging.ERROR, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                     f"Error restarting component {component_name}: {e}", exc_info=True)
            return False

    async def _escalate_component_failure(self, component_name: str):
        """Escalate component failure to supervisor via ErrorEvent."""
        error_msg = f"Component {component_name} failed to restart after {self.monitors[component_name].max_restart_attempts} attempts"

        agent_log(logging.CRITICAL, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                 error_msg)

        # Check if long-running mode should handle this
        try:
            # Try to get supervisor reference through state_manager
            supervisor = getattr(self.state_manager, '_supervisor', None)
            if supervisor and hasattr(supervisor, 'long_running_integration'):
                long_running = supervisor.long_running_integration
                if long_running.enabled:
                    # Let long-running mode handle the escalation
                    recovery_action = await long_running.handle_component_failure(
                        component_name,
                        Exception(error_msg)
                    )
                    agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                             f"Long-running mode suggested recovery: {recovery_action}")
        except Exception as e:
            logger.debug(f"Failed to delegate to long-running mode: {e}")

        # Create and emit critical ErrorEvent
        error_event = ErrorEvent(
            step_token=self.state_manager.state.n_steps,
            error_message=error_msg,
            error_type="ComponentFailure",
            is_critical=True
        )

        try:
            self.agent_bus.put_nowait(error_event)
        except asyncio.QueueFull:
            agent_log(logging.ERROR, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                     f"Agent bus full, could not escalate failure for {component_name}")

    def stop(self):
        """Stop the monitoring loop."""
        self.running = False
