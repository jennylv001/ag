from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Dict, Optional, Tuple
import random

from browser_use.agent.events import ActuationResult, Decision, DecisionPlan, ActionExecuted, ErrorEvent, Heartbeat
from browser_use.agent.state_manager import agent_log
from browser_use.agent.views import ActionResult, StepMetadata
from browser_use.agent.concurrency import with_io_semaphore

if TYPE_CHECKING:
    from browser_use.agent.settings import AgentSettings
    from browser_use.agent.state_manager import StateManager
    from browser_use.browser import BrowserSession
    from browser_use.controller.service import Controller

logger = logging.getLogger(__name__)


class LongIOWatcher:
    """
    Monitors long-running I/O operations and publishes ErrorEvents for timeouts/cancellations.
    Tracks futures with their associated task context for proper error reporting.
    """

    def __init__(self, agent_bus: asyncio.Queue, state_manager, shutdown_event: Optional[asyncio.Event] = None):
        self.agent_bus = agent_bus
        self.state_manager = state_manager
        self.watched_futures: Dict[str, Dict] = {}  # operation_id -> {future, task_id, timeout, description}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = shutdown_event

    def start(self):
        """Start the watcher's cleanup task."""
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def stop(self):
        """Stop the watcher and cancel cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

    def register_operation(self, future: asyncio.Future, task_id: str, timeout_seconds: float, description: str) -> str:
        """
        Register a long-running I/O operation for monitoring.
        Returns an operation ID for tracking.
        """
        operation_id = str(uuid.uuid4())

        self.watched_futures[operation_id] = {
            'future': future,
            'task_id': task_id,
            'timeout': timeout_seconds,
            'description': description,
            'start_time': time.monotonic(),
            'step_token': self.state_manager.state.n_steps
        }

        return operation_id

    def unregister_operation(self, operation_id: str):
        """Unregister a completed operation."""
        if operation_id in self.watched_futures:
            del self.watched_futures[operation_id]

    async def _cleanup_loop(self):
        """Periodically check for timed out or cancelled operations."""
        while True:
            try:
                # Support graceful shutdown: exit if shutdown flag is set
                if self._shutdown_event and self._shutdown_event.is_set():
                    logger.debug("LongIOWatcher shutting down gracefully")
                    break
                await asyncio.sleep(1.0)  # Check every second
                current_time = time.monotonic()

                # Check each watched operation
                to_remove = []
                for operation_id, operation_info in self.watched_futures.items():
                    future = operation_info['future']
                    start_time = operation_info['start_time']
                    timeout = operation_info['timeout']
                    task_id = operation_info['task_id']
                    description = operation_info['description']
                    step_token = operation_info['step_token']

                    # Check if future is done (completed, cancelled, or failed)
                    if future.done():
                        if future.cancelled():
                            # Future was cancelled
                            await self._publish_error(
                                step_token=step_token,
                                task_id=task_id,
                                error_message=f"I/O operation cancelled: {description}",
                                error_type="io_cancelled"
                            )
                        elif future.exception():
                            # Future failed with exception
                            exc = future.exception()
                            await self._publish_error(
                                step_token=step_token,
                                task_id=task_id,
                                error_message=f"I/O operation failed: {description} - {str(exc)}",
                                error_type="io_exception"
                            )

                        to_remove.append(operation_id)
                        continue

                    # Check for timeout
                    elapsed = current_time - start_time
                    if elapsed > timeout:
                        logger.warning(f"I/O operation {operation_id} timed out after {elapsed:.1f}s")

                        # Cancel the future
                        future.cancel()

                        # Publish timeout error
                        # Emit health signal for IO timeout
                        try:
                            await self.state_manager.ingest_signal('io_timeout')
                        except Exception:
                            logger.debug("Failed to emit io_timeout signal", exc_info=True)

                        await self._publish_error(
                            step_token=step_token,
                            task_id=task_id,
                            error_message=f"I/O operation timed out: {description} (timeout: {timeout}s)",
                            error_type="io_timeout",
                            is_critical=True
                        )

                        to_remove.append(operation_id)

                # Remove completed/failed operations
                for operation_id in to_remove:
                    self.unregister_operation(operation_id)

            except asyncio.CancelledError:
                logger.debug("LongIOWatcher cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in LongIOWatcher cleanup loop: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error

    async def _publish_error(self, step_token: int, task_id: str, error_message: str, error_type: str, is_critical: bool = False):
        """Publish an ErrorEvent with proper task context."""
        error_event = ErrorEvent(
            step_token=step_token,
            task_id=task_id,
            error_message=error_message,
            error_type=error_type,
            is_critical=is_critical
        )

        try:
            self.agent_bus.put_nowait(error_event)
            logger.info(f"Published {error_type} error for task {task_id}: {error_message}")
        except asyncio.QueueFull:
            logger.warning(f"Agent bus full, dropped {error_type} error event")


class Actuator:
    """
    "Async I/O periphery"
    This component is responsible for executing actions in the environment. It
    receives a `Decision` from the core and uses the `Controller` to perform
    the actions. It is an async I/O-bound component.
    """

    def __init__(
        self,
        controller: Controller,
        browser_session: BrowserSession,
        state_manager: StateManager,
        settings: AgentSettings,
        agent_bus: asyncio.Queue,
        heartbeat_bus: asyncio.Queue,
        control_bus: Optional[asyncio.Queue] = None,
    ):
        self.controller = controller
        self.browser_session = browser_session
        self.state_manager = state_manager
        self.settings = settings
        self.agent_bus = agent_bus
        self.heartbeat_bus = heartbeat_bus
        self.control_bus = control_bus or agent_bus

        # Initialize the Long I/O Watcher (shutdown flag injected later by Supervisor)
        self.shutdown_event: Optional[asyncio.Event] = None
        self.io_watcher = LongIOWatcher(agent_bus, state_manager, shutdown_event=self.shutdown_event)

    def set_shutdown_event(self, ev: asyncio.Event):
        self.shutdown_event = ev
        # Propagate to watcher
        self.io_watcher._shutdown_event = ev

    def _select_timeout(self, decision: Decision) -> float:
        """Choose a timeout based on settings, action types, and current site."""
        try:
            # 1) Site/domain override
            current_url = None
            try:
                current_url = self.browser_session.browser_state_summary.url  # fast path
            except Exception:
                try:
                    current_url = getattr(decision.browser_state, 'url', None)
                except Exception:
                    current_url = None
            timeout_site = None
            if current_url and getattr(self.settings, 'site_profile_overrides', None):
                for domain, t in self.settings.site_profile_overrides.items():
                    if domain.lower() in current_url.lower():
                        timeout_site = float(t)
                        break

            # 2) Action-type override: first action dominates for simplicity
            timeout_action = None
            if getattr(self.settings, 'action_timeout_overrides', None) and decision.llm_output and decision.llm_output.action:
                try:
                    first = decision.llm_output.action[0]
                    action_name = type(first).__name__
                    if action_name in self.settings.action_timeout_overrides:
                        timeout_action = float(self.settings.action_timeout_overrides[action_name])
                except Exception:
                    pass

            # 3) Default
            default_timeout = float(getattr(self.settings, 'default_action_timeout_seconds', 60.0))
            # Combine: prefer action override, then site, then default; take the max to be safe on uploads
            candidates = [v for v in [timeout_action, timeout_site, default_timeout] if v is not None]
            return max(candidates) if candidates else default_timeout
        except Exception:
            return float(getattr(self.settings, 'default_action_timeout_seconds', 60.0))

    def _compute_retry_schedule(self) -> list[float]:
        """Exponential backoff schedule with jitter for retries."""
        max_attempts = max(0, int(getattr(self.settings, 'max_attempts_per_action', 0)))
        base = float(getattr(self.settings, 'backoff_base_seconds', 1.0))
        jitter = float(getattr(self.settings, 'backoff_jitter_seconds', 0.3))
        # attempts represent additional tries after the first
        schedule = []
        for i in range(max_attempts):
            delay = (2 ** i) * base + random.uniform(0, jitter)
            schedule.append(delay)
        return schedule

    async def execute(self, decision: Decision) -> ActuationResult:
        """Executes the actions from a decision and returns the results."""
        state = self.state_manager.state
        step_start_time = time.monotonic()

        if not decision.llm_output or not decision.llm_output.action:
            metadata = StepMetadata(step_number=state.n_steps, step_start_time=step_start_time, step_end_time=time.monotonic())
            return ActuationResult(action_results=[], llm_output=None, browser_state=decision.browser_state, step_metadata=metadata)

        # Get current task context for monitoring
        current_task_id = await self.state_manager.get_current_task_id()

        async def _run_with_io_limit():
            # Limit concurrent browser IO using the global semaphore
            async with await with_io_semaphore():
                return await self.controller.multi_act(
                    actions=decision.llm_output.action,
                    browser_session=self.browser_session,
                    page_extraction_llm=self.settings.page_extraction_llm,
                    context=self.settings.context,
                    sensitive_data=self.settings.sensitive_data,
                    available_file_paths=self.settings.available_file_paths,
                    file_system=self.settings.file_system,
                )

        # Build description for telemetry and debugging
        action_descriptions = [str(action) for action in decision.llm_output.action[:3]]
        description = f"Browser actions: {', '.join(action_descriptions)}"
        if len(decision.llm_output.action) > 3:
            description += f" and {len(decision.llm_output.action) - 3} more"

        # Determine timeout and retry schedule
        timeout_seconds = self._select_timeout(decision)
        retry_schedule = self._compute_retry_schedule()

        last_exc: Optional[BaseException] = None
        attempts = 0
        while True:
            # Stop early if shutdown requested
            if self.shutdown_event and self.shutdown_event.is_set():
                raise asyncio.CancelledError("Actuator shutting down")

            # Create a future the LongIOWatcher can monitor
            browser_operation = asyncio.create_task(_run_with_io_limit())
            operation_id = self.io_watcher.register_operation(
                future=browser_operation,
                task_id=current_task_id,
                timeout_seconds=timeout_seconds,
                description=description
            )
            try:
                # Wait for the operation to complete
                action_results = await browser_operation
                logger.info(f"Controller call returned with {len(action_results)} action result(s)")

                # Unregister successful operation
                self.io_watcher.unregister_operation(operation_id)
                try:
                    await self.state_manager.ingest_signal('io_ok')
                except Exception:
                    logger.debug("Failed to emit io_ok signal", exc_info=True)
                break
            except Exception as e:
                attempts += 1
                last_exc = e
                self.io_watcher.unregister_operation(operation_id)
                # Only retry on timeouts/cancellation/IO-like exceptions; conservative approach
                err_str = str(e).lower()
                is_timeout = 'timeout' in err_str or isinstance(e, asyncio.TimeoutError)
                if attempts <= len(retry_schedule) and is_timeout:
                    delay = retry_schedule[attempts - 1]
                    logger.warning(f"Retrying browser operation after timeout (attempt {attempts}/{len(retry_schedule)}). Backoff {delay:.2f}s")
                    await asyncio.sleep(delay)
                    continue
                logger.info(f"Browser operation failed without retry: {type(e).__name__}: {e}")
                raise  # Re-raise for normal error handling

        metadata = StepMetadata(step_number=state.n_steps, step_start_time=step_start_time, step_end_time=time.monotonic())

        result = ActuationResult(
            action_results=action_results,
            llm_output=decision.llm_output,
            browser_state=decision.browser_state,
            step_metadata=metadata,
        )

        # Optional: lightweight auto push/pop hooks based on 'done' or explicit milestone actions
        try:
            if getattr(self.settings, 'auto_taskstack_markers', False) and decision.llm_output and decision.llm_output.action:
                # Push when a distinct 'subtask' label appears in task_log; pop on 'done' action
                task_log = getattr(decision.llm_output, 'task_log', '') or ''
                next_goal = getattr(decision.llm_output, 'next_goal', '') or ''
                if '<subtask>' in task_log and '</subtask>' in task_log:
                    import re
                    m = re.search(r'<subtask>(.*?)</subtask>', task_log, re.IGNORECASE | re.DOTALL)
                    if m:
                        subtask_desc = m.group(1).strip()[:120]
                        await self.state_manager.push_task(task_id=str(uuid.uuid4())[:8], description=subtask_desc)
                # Pop when 'done' is explicitly set true on any action
                for a in decision.llm_output.action:
                    try:
                        ad = a.model_dump(exclude_unset=True)
                        if 'done' in ad and isinstance(ad['done'], dict):
                            if ad['done'].get('success') is True:
                                await self.state_manager.pop_task()
                                break
                    except Exception:
                        continue
        except Exception:
            logger.debug("Auto taskstack markers hook failed (non-fatal)", exc_info=True)
        return result

    async def run(self):
        """Event-driven actuation loop that subscribes to DecisionPlan events."""
        from browser_use.agent.state_manager import TERMINAL_STATES
        logger.debug("Actuator component started.")

        # Start heartbeat task and I/O watcher
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.io_watcher.start()

        async def main_loop():
            """Main event processing loop."""
            status_check_counter = 0  # Check status less frequently
            consecutive_not_mine = 0
            while True:
                try:
                    event = await asyncio.wait_for(self.agent_bus.get(), timeout=0.5)  # Very short timeout to avoid blocking heartbeat

                    # Only process DecisionPlan events
                    if isinstance(event, DecisionPlan):
                        logger.info(f"Processing DecisionPlan event with step_token={event.step_token}")
                        await self._handle_decision_plan(event)
                        self.agent_bus.task_done()
                        consecutive_not_mine = 0
                    else:
                        # Not our event. Put it back on the bus and yield control
                        # to give the correct consumer a chance to pick it up.
                        try:
                            self.agent_bus.put_nowait(event)
                        except asyncio.QueueFull:
                            logger.warning(f"Agent bus full, cannot requeue event. Event may be lost.")
                            agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                                     "Agent bus full, cannot requeue event. Event may be lost.")
                        finally:
                            self.agent_bus.task_done()
                            # Give supervisor priority window for ActionExecuted/ErrorEvent
                            if isinstance(event, (ActionExecuted, ErrorEvent)):
                                await asyncio.sleep(0.01)  # 10ms priority window for supervisor
                            else:
                                # Fairness: yield; tiny backoff every 25 misses and exit if terminal
                                consecutive_not_mine += 1
                                if consecutive_not_mine % 25 == 0:
                                    await asyncio.sleep(0.01)
                                    from browser_use.agent.state_manager import TERMINAL_STATES
                                    if await self.state_manager.get_status() in TERMINAL_STATES:
                                        logger.info("Agent in terminal state, exiting main loop")
                                        break
                                else:
                                    await asyncio.sleep(0)
                        continue

                except asyncio.TimeoutError:
                    # No event received within timeout, continuing
                    continue
                except Exception as e:
                    error_msg = f"Actuator event loop failed: {e}"
                    logger.error(f"CRITICAL ERROR in main event loop: {error_msg}")
                    agent_log(logging.CRITICAL, self.state_manager.state.agent_id, self.state_manager.state.n_steps, error_msg, exc_info=True)
                    await asyncio.sleep(1)

            logger.info("Main event loop exited")

        try:
            # Run main loop and heartbeat concurrently
            await asyncio.gather(main_loop(), heartbeat_task, return_exceptions=True)
        finally:
            # Cancel heartbeat task and stop I/O watcher when shutting down
            heartbeat_task.cancel()
            self.io_watcher.stop()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        logger.info("Actuator component stopped.")

    async def _handle_decision_plan(self, event: DecisionPlan):
        """Handle DecisionPlan event by executing actions and publishing ActionExecuted/ErrorEvent."""
        from browser_use.agent.state_manager import AgentStatus, agent_log

        # CRITICAL FIX: Filter out stale events
        current_step = self.state_manager.state.n_steps
        if event.step_token < current_step:
            logger.warning(f"Ignoring stale DecisionPlan event: step_token={event.step_token} < current_step={current_step}")
            return

        status = await self.state_manager.get_status()
        if status != AgentStatus.RUNNING:
            logger.info(f"Agent status is not RUNNING, exiting early")
            return

        # Get current task context for proper event tagging
        current_task_id = await self.state_manager.get_current_task_id()

        try:
            # Convert DecisionPlan to legacy Decision for compatibility
            decision = Decision(
                messages_to_llm=event.messages_to_llm,
                llm_output=event.llm_output,
            )

            # Execute actions using existing logic (includes LongIOWatcher monitoring)
            actuation_result = await self.execute(decision)
            logger.info(f"Actions executed; results={len(actuation_result.action_results)}")

            # Check for failures and publish appropriate events
            # Treat None as success (successful actions like go_to_url return success=None)
            success = all(r.success is not False for r in actuation_result.action_results)
            failures = sum(1 for r in actuation_result.action_results if r.success is False)
            logger.info(f"Action outcomes: success={success}, failures={failures}")

            # Log detailed action results for debugging
            for i, result in enumerate(actuation_result.action_results):
                logger.info(f"Action {i}: success={result.success}, error={result.error}")

            if success:
                # Publish ActionExecuted event with task context
                action_executed = ActionExecuted(
                    step_token=event.step_token,
                    task_id=current_task_id,
                    action_results=actuation_result.action_results,
                    success=True
                )

                try:
                    # Control-path event -> prefer control_bus
                    self.control_bus.put_nowait(action_executed)
                    logger.info(f"ActionExecuted event published successfully")
                except asyncio.QueueFull:
                    logger.warning(f"Agent bus full, dropping ActionExecuted event")
                    agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                             "Agent bus full, dropping ActionExecuted event")
            else:
                # Publish ErrorEvent for failures with task context
                error_msg = next((r.error for r in actuation_result.action_results if r.error), "Action execution failed")
                error_event = ErrorEvent(
                    step_token=event.step_token,
                    task_id=current_task_id,
                    error_message=error_msg,
                    error_type="execution_failure",
                    is_critical=False
                )

                try:
                    # Control-path event -> prefer control_bus
                    self.control_bus.put_nowait(error_event)
                    logger.info(f"ErrorEvent published successfully")
                except asyncio.QueueFull:
                    logger.warning(f"Agent bus full, dropping ErrorEvent")
                    agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                             "Agent bus full, dropping ErrorEvent")

        except Exception as e:
            logger.error(f"EXCEPTION caught in _handle_decision_plan: {type(e).__name__}: {e}")
            # Publish critical ErrorEvent for execution failures with task context
            error_msg = f"Action execution failed: {e}"
            agent_log(logging.ERROR, self.state_manager.state.agent_id, self.state_manager.state.n_steps, error_msg, exc_info=True)

            error_event = ErrorEvent(
                step_token=event.step_token,
                task_id=current_task_id,
                error_message=error_msg,
                error_type="critical_execution_failure",
                is_critical=True
            )

            try:
                # Control-path event -> prefer control_bus
                self.control_bus.put_nowait(error_event)
                logger.info(f"Critical ErrorEvent published successfully")
            except asyncio.QueueFull:
                logger.warning(f"Agent bus full, dropping critical ErrorEvent")
                agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                         "Agent bus full, dropping critical ErrorEvent")

    async def _heartbeat_loop(self):
        """Send periodic heartbeat events every 2 seconds."""
        while True:
            try:
                await asyncio.sleep(2.0)  # Send heartbeat every 2 seconds

                # DEADLOCK FIX: Use cached values instead of lock-requiring calls
                # Get current task context for heartbeat without requiring locks
                current_task_id = getattr(self.state_manager._state.task_stack, 'current_task_id', 'unknown')
                step_token = 0
                try:
                    # Try to get values without waiting if lock is available
                    step_token = self.state_manager.state.n_steps
                except:
                    pass  # Use fallback if state access fails

                # Create and emit heartbeat event with task context
                heartbeat = Heartbeat(
                    step_token=step_token,
                    task_id=current_task_id,
                    component_name="actuator"
                )

                try:
                    self.heartbeat_bus.put_nowait(heartbeat)
                    logger.debug(f"Actuator heartbeat sent for task {current_task_id}")
                except asyncio.QueueFull:
                    logger.warning("Agent bus full, dropped Actuator heartbeat")

            except asyncio.CancelledError:
                logger.debug("Actuator heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in Actuator heartbeat loop: {e}")
                await asyncio.sleep(2.0)  # Continue after error
