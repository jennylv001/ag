from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from typing import TYPE_CHECKING

from browser_use.agent.actuator import Actuator
from browser_use.agent.concurrency import set_global_io_semaphore
from browser_use.agent.decision_maker import DecisionMaker
from browser_use.agent.llm_caller import LLMCaller
from browser_use.agent.reactor_vitals import ReactorVitals
from browser_use.agent.long_running_integration import LongRunningIntegration
from browser_use.agent.events import ActuationResult, Decision, PerceptionOutput, StepFinalized, ActionExecuted, ErrorEvent, PerceptionSnapshot
from browser_use.agent.gif import create_history_gif
from browser_use.agent.message_manager.service import MessageManager, MessageManagerSettings
from browser_use.agent.perception import Perception
from browser_use.agent.scheduler import Scheduler
from browser_use.agent.planner import Planner
from browser_use.agent.assessor import Assessor
from browser_use.agent.state_manager import AgentState, AgentStatus, StateManager, agent_log, LoadStatus, TERMINAL_STATES
from browser_use.agent.concurrency import bulletproof_lock
from browser_use.agent.views import AgentHistory, AgentHistoryList, AgentError, ActionResult, StepMetadata
from browser_use.browser import BrowserSession
from browser_use.browser.views import BrowserStateHistory
from browser_use.browser.session import DEFAULT_BROWSER_PROFILE
from browser_use.exceptions import AgentInterruptedError
from browser_use.filesystem.file_system import FileSystem
from browser_use.utils import SignalHandler

if TYPE_CHECKING:
    from browser_use.agent.settings import AgentSettings

logger = logging.getLogger(__name__)

CPU_SHEDDING_THRESHOLD = 98.0
CPU_NORMAL_THRESHOLD = 96.0


class Supervisor:
    """ "Concurrency with Consequences" """

    def __init__(self, settings: AgentSettings):
        self.settings = settings

        if self.settings.page_extraction_llm is None:
            self.settings.page_extraction_llm = self.settings.llm

        self._last_event_time = time.time()  # Track event coordinator health
        self._events_processed = 0  # Event processing counter

        # Initialize long-running mode integration
        try:
            self.long_running_integration = LongRunningIntegration(
                supervisor=self,
                enabled=getattr(settings, 'enable_long_running_mode', False)
            )
        except Exception as e:
            logger.warning(f"Failed to initialize long-running integration: {e}")
            # Create a disabled integration as fallback
            self.long_running_integration = LongRunningIntegration(supervisor=self, enabled=False)

        self._setup_components()


    def _setup_components(self):
        state = self.settings.injected_agent_state or AgentState(task=self.settings.task)
        self._setup_filesystem(state)
        self.state_manager = StateManager(
            initial_state=state,
            file_system=self.settings.file_system,
            max_failures=self.settings.max_failures,
            lock_timeout_seconds=self.settings.lock_timeout_seconds,
            use_planner=self.settings.use_planner,
            reflect_on_error=self.settings.reflect_on_error,
            max_history_items=self.settings.max_history_items,
            memory_budget_mb=self.settings.memory_budget_mb,
            reflect_cadence=getattr(self.settings, 'planner_interval', 0),
            reflect_cooldown_seconds=getattr(self.settings, 'reflect_cooldown_seconds', 0.0),
            enable_modes=getattr(self.settings, 'enable_modes', False)
        )
        self._setup_browser_session()

        # QoS buses: optionally split control vs work
        self.agent_bus = asyncio.Queue(maxsize=10000)  # Increased to 10k for 8+ concurrent components
        self.control_bus = (
            asyncio.Queue(maxsize=10000)  # Increased to match agent_bus capacity
            if getattr(self.settings, 'enable_control_work_split', False)
            else self.agent_bus
        )
        self.work_bus = self.agent_bus  # For now, reuse; components can be pointed to work_bus

        # Step barrier for coordinating perception/decision steps
        self.step_barrier = asyncio.Event()
        self.step_barrier.set()  # Initially set to allow first perception/decision cycle

        # Cooperative shutdown flag propagated to components
        self.shutdown_event = asyncio.Event()

        # Initialize global I/O semaphore for resource management
        set_global_io_semaphore(self.settings.max_concurrent_io)

        # Create separate heartbeat queue for ReactorVitals to avoid event competition
        self.heartbeat_bus = asyncio.Queue(maxsize=5000)  # Increased to match other queues for 8+ components

        # Wire up components
        self.perception = Perception(
            self.browser_session,
            self.state_manager,
            self.settings,
            self.work_bus,
            self.step_barrier,
            self.heartbeat_bus,
        )
        self.message_manager = MessageManager(
            task=state.task,
            system_message=None,
            settings=MessageManagerSettings.model_validate(
                self.settings.model_dump(include=MessageManagerSettings.model_fields.keys())
            ),
            state=state.message_manager_state,
            file_system=self.settings.file_system,
        )
        self.decision_maker = DecisionMaker(
            settings=self.settings,
            state_manager=self.state_manager,
            message_manager=self.message_manager,
            agent_bus=self.work_bus,
            heartbeat_bus=self.heartbeat_bus,
        )
        self.llm_caller = LLMCaller(
            llm=self.settings.llm,
            planner_llm=(self.settings.planner_llm or self.settings.llm) if self.settings.use_planner else None,
            state_manager=self.state_manager,
            agent_bus=self.work_bus,
            heartbeat_bus=self.heartbeat_bus,
        )
        self.actuator = Actuator(
            self.settings.controller,
            self.browser_session,
            self.state_manager,
            self.settings,
            self.work_bus,
            self.heartbeat_bus,
        )
        self.reactor_vitals = ReactorVitals(
            state_manager=self.state_manager,
            agent_bus=self.work_bus,
            heartbeat_bus=self.heartbeat_bus,
            heartbeat_timeout=10.0,
        )

        # Scheduler component (lightweight)
        self.scheduler = None
        if getattr(self.settings, 'scheduler_enabled', True):
            self.scheduler = Scheduler(
                state_manager=self.state_manager,
                agent_bus=self.work_bus,
                heartbeat_bus=self.heartbeat_bus,
                interval_seconds=getattr(self.settings, 'scheduler_interval_seconds', 15.0),
            )

        # Planner component (standalone)
        self.planner = None
        if getattr(self.settings, 'use_planner', False):
            self.planner = Planner(
                settings=self.settings,
                state_manager=self.state_manager,
                agent_bus=self.work_bus,
                heartbeat_bus=self.heartbeat_bus,
            )

        # Assessor component (publishes AssessmentUpdate events)
        self.assessor = None
        if getattr(self.settings, 'assessor_enabled', True):
            self.assessor = Assessor(
                state_manager=self.state_manager,
                agent_bus=self.work_bus,
                interval_seconds=getattr(self.settings, 'assessor_interval_seconds', 1.0),
            )

        if self.settings.output_model:
            self.settings.controller.use_structured_output_action(self.settings.output_model)

        # Expose supervisor and bus on state_manager for integrations (ReactorVitals/LongRunning)
        try:
            self.state_manager._supervisor = self  # Used by ReactorVitals and LongRunningMode
            # Provide a lightweight event bridge for integrations that only know the state manager
            self.state_manager.bus = self.work_bus
            # Optionally expose the step barrier for advanced recoveries
            self.state_manager._step_barrier = self.step_barrier
        except Exception:
            # Never fail setup due to debug wiring
            pass

    async def run(self) -> AgentHistoryList:
        # Register interactive controls only when explicitly enabled
        signal_handler = None
        try:
            if bool(getattr(self.settings, 'enable_interactive_controls', False)):
                signal_handler = SignalHandler(
                    loop=asyncio.get_event_loop(),
                    pause_callback=self.pause,
                    resume_callback=self.resume,
                    custom_exit_callback=self.stop,
                    guidance_callback=lambda text: asyncio.create_task(self.state_manager.add_human_guidance(text))
                )
                signal_handler.register()
        except Exception:
            signal_handler = None

        try:
            if self.settings.on_run_start: await self.settings.on_run_start(self)
            agent_log(logging.INFO, self.state_manager.state.agent_id, 0, f"🚀 Starting agent run for task: \"{self.state_manager.state.task[:70]}...\"")

            # Initialize long-running mode if enabled
            try:
                await self.long_running_integration.initialize()
            except Exception as e:
                agent_log(logging.WARNING, self.state_manager.state.agent_id, 0,
                         f"Long-running mode initialization failed, continuing without it: {e}")
                # Disable long-running mode to prevent further issues
                self.long_running_integration.enabled = False

            await self._execute_initial_actions()

            if await self.state_manager.get_status() not in TERMINAL_STATES:
                await self.state_manager.set_status(AgentStatus.RUNNING)

                try:
                    async with asyncio.TaskGroup() as tg:
                        # Create component tasks
                        perception_task = tg.create_task(self.perception.run())
                        decision_task = tg.create_task(self.decision_maker.run())
                        llm_caller_task = tg.create_task(self.llm_caller.run())
                        actuator_task = tg.create_task(self.actuator.run())
                        # Optionally start scheduler
                        if self.scheduler is not None:
                            scheduler_task = tg.create_task(self.scheduler.run())
                        # Optionally start planner
                        if self.planner is not None:
                            planner_task = tg.create_task(self.planner.run())
                        # Optionally start assessor (no heartbeat registration to avoid false timeouts)
                        if self.assessor is not None:
                            assessor_task = tg.create_task(self.assessor.run())

                        # Register components with ReactorVitals for monitoring
                        self.reactor_vitals.register_component("perception", perception_task)
                        self.reactor_vitals.register_component("decision_maker", decision_task)
                        self.reactor_vitals.register_component("llm_caller", llm_caller_task)
                        self.reactor_vitals.register_component("actuator", actuator_task)
                        if self.scheduler is not None:
                            self.reactor_vitals.register_component("scheduler", scheduler_task)
                        if self.planner is not None:
                            self.reactor_vitals.register_component("planner", planner_task)
                        # Note: assessor doesn't emit heartbeats; we intentionally do not register it here

                        # Provide restart factories so ReactorVitals can (re)create component tasks on demand
                        def _mk_perception():
                            return self.perception.run()

                        def _mk_decision():
                            return self.decision_maker.run()

                        def _mk_llm():
                            return self.llm_caller.run()

                        def _mk_actuator():
                            return self.actuator.run()

                        factories = {
                            "perception": _mk_perception,
                            "decision_maker": _mk_decision,
                            "llm_caller": _mk_llm,
                            "actuator": _mk_actuator,
                        }
                        if self.scheduler is not None:
                            def _mk_scheduler():
                                return self.scheduler.run()
                            factories["scheduler"] = _mk_scheduler
                        if self.planner is not None:
                            def _mk_planner():
                                return self.planner.run()
                            factories["planner"] = _mk_planner
                        # Assessor is lightweight and stateless enough; if needed we could add a restart factory later

                        self.reactor_vitals.set_restart_factories(factories)

                        # Provide restart factories so ReactorVitals can (re)create component tasks on demand (already set below)

                        # Publish initial events after components are started
                        # Wait a brief moment for event loops to initialize
                        await asyncio.sleep(0.1)

                        # Publish only an initial StepFinalized to let Perception emit the snapshot.
                        # This avoids duplicate PerceptionSnapshots and reduces queue pressure.
                        
                        # CRITICAL FIX: Set step_barrier before publishing initial StepFinalized
                        # This ensures Perception won't deadlock waiting for the barrier on step_token=0
                        if hasattr(self, 'step_barrier') and self.step_barrier:
                            self.step_barrier.set()
                            logger.debug("Step barrier set for initial StepFinalized event")
                        
                        try:
                            (self.control_bus if self.control_bus is not self.work_bus else self.work_bus).put_nowait(StepFinalized(step_token=0))
                            agent_log(logging.INFO, self.state_manager.state.agent_id, 0,
                                     "✅ Published initial StepFinalized event to kickstart Perception component")
                        except asyncio.QueueFull:
                            agent_log(logging.WARNING, self.state_manager.state.agent_id, 0,
                                     "Agent bus full, could not publish initial StepFinalized event")

                        # Start monitoring and coordination tasks
                        tg.create_task(self.reactor_vitals.run())
                        tg.create_task(self._event_coordinator())
                        tg.create_task(self._pause_handler())
                        tg.create_task(self._load_shedding_monitor())

                        # Start long-running mode monitoring if enabled
                        if self.long_running_integration.enabled:
                            lr_signal_handler = SignalHandler(
                                loop=asyncio.get_event_loop(),
                                pause_callback=self.pause,
                                resume_callback=self.resume,
                                custom_exit_callback=self.stop,
                                guidance_callback=lambda text: asyncio.create_task(self.state_manager.add_human_guidance(text))
                            )
                            lr_signal_handler.register()

                except Exception as e:
                    await self._record_error_in_history(e)
        except AgentInterruptedError:
            await self.state_manager.set_status(AgentStatus.STOPPED)
        finally:
            if signal_handler:
                signal_handler.unregister()
            self._log_final_status()
            if self.settings.on_run_end: await self.settings.on_run_end(self.state_manager.state.history)
            await self.close()
            self._generate_final_gif_if_enabled()

        return self.state_manager.state.history

    async def _execute_initial_actions(self):
        action_model = self.settings.controller.registry.create_action_model()
        initial_actions_parsed = self.settings.parse_initial_actions(action_model)
        if not initial_actions_parsed: return

        agent_log(logging.INFO, self.state_manager.state.agent_id, -1, "Executing initial actions...")
        action_results = await self.actuator.controller.multi_act(
            actions=initial_actions_parsed, browser_session=self.browser_session,
            page_extraction_llm=self.settings.page_extraction_llm, sensitive_data=self.settings.sensitive_data,
            available_file_paths=self.settings.available_file_paths, context=self.settings.context,
            file_system=self.settings.file_system)

        metadata = StepMetadata(step_number=-1, step_start_time=time.monotonic(), step_end_time=time.monotonic())
        browser_state = await self.perception._get_browser_state_with_recovery()

        history_state = None
        if browser_state:
            history_state = BrowserStateHistory(
                url=browser_state.url,
                title=browser_state.title,
                tabs=browser_state.tabs,
                screenshot_path=browser_state.screenshot,
                interacted_element=[], # No interacted elements in initial actions
            )
        history_item = AgentHistory(model_output=None, result=action_results, state=history_state, metadata=metadata)
        await self.state_manager.add_history_item(history_item)

    async def _event_coordinator(self):
        """
        Coordinates events between components and handles step finalization.
        Routes events from components and triggers step finalization.
        """
        logger.debug("Event coordinator started - monitoring for deadlock prevention")

        # Cache status check to reduce lock contention
        status_check_counter = 0
        while True:
            # Only check status every 10 iterations to reduce lock pressure
            if status_check_counter % 10 == 0:
                current_status = await self.state_manager.get_status()
                if current_status in TERMINAL_STATES:
                    break
            status_check_counter += 1
            try:
                # Prioritize control bus when split enabled
                if self.control_bus is self.work_bus:
                    event = await asyncio.wait_for(self.work_bus.get(), timeout=30)
                else:
                    try:
                        event = await asyncio.wait_for(self.control_bus.get(), timeout=0.05)
                    except asyncio.TimeoutError:
                        event = await asyncio.wait_for(self.work_bus.get(), timeout=30)

                # Drop clearly stale step-scoped events so they don't clog the bus
                try:
                    evt_step = getattr(event, "step_token", None)
                    cur_step = self.state_manager.state.n_steps
                    if isinstance(evt_step, int) and evt_step < (cur_step - 1):
                        logger.debug(
                            "Event coordinator: dropping stale %s step_token=%s < current_step-1=%s",
                            type(event).__name__, evt_step, cur_step - 1
                        )
                        (self.control_bus if self.control_bus is not self.work_bus else self.work_bus).task_done()
                        await asyncio.sleep(0)  # yield to rightful consumers
                        continue
                except Exception:
                    # Never let router crash on inspection
                    pass

                # Process the event we retrieved
                # Update health monitoring
                self._last_event_time = time.time()
                self._events_processed += 1

                logger.debug("Event coordinator received: %s (total processed: %d)",
                           type(event).__name__, self._events_processed)

                # Handle ActionExecuted and ErrorEvent events to trigger step finalization
                if isinstance(event, (ActionExecuted, ErrorEvent)):
                    logger.info("Processing completion event: %s", type(event).__name__)
                    await self._handle_action_completion(event)
                    (self.control_bus if self.control_bus is not self.work_bus else self.work_bus).task_done()
                    logger.debug("Completion event processed successfully")
                # Handle autonomous continuation events from long-running mode
                elif isinstance(event, dict) and event.get('type') == 'autonomous_continuation':
                    logger.info("Processing autonomous continuation event")
                    await self._handle_autonomous_continuation(event)
                    (self.control_bus if self.control_bus is not self.work_bus else self.work_bus).task_done()
                    logger.debug("Autonomous continuation event processed successfully")
                else:
                    # Not our event. Put it back on the bus and yield control
                    # to give the correct consumer a chance to pick it up.
                    try:
                        # Requeue onto appropriate bus
                        target_bus = self.work_bus
                        if isinstance(event, (ActionExecuted, ErrorEvent, StepFinalized)) and self.control_bus is not self.work_bus:
                            target_bus = self.control_bus
                        target_bus.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.error("Agent bus full - potential event loss! Event type: %s", type(event).__name__)
                    finally:
                        (self.control_bus if self.control_bus is not self.work_bus else self.work_bus).task_done()
                        await asyncio.sleep(0)  # Brief backoff on queue pressure
            except asyncio.TimeoutError:
                # Check for potential deadlock
                time_since_last_event = time.time() - self._last_event_time
                if time_since_last_event > 60:  # 1 minute without events
                    logger.warning("Potential deadlock detected: No events processed for %.1f seconds",
                                 time_since_last_event)
                    # Report both queues where applicable
                    q_main = self.work_bus.qsize()
                    q_ctrl = (self.control_bus.qsize() if self.control_bus is not self.work_bus else q_main)
                    logger.info("Event coordinator health: %d events processed, q_work=%d q_control=%d",
                              self._events_processed, q_main, q_ctrl)

                    if time_since_last_event > 120:
                        # For extended deadlocks, try autonomous continuation if enabled
                        if (hasattr(self, 'long_running_integration') and
                            self.long_running_integration.enabled and
                            time_since_last_event > 180):  # 3 minutes = serious deadlock

                            logger.warning("Extended deadlock detected, triggering autonomous continuation")
                            try:
                                # Create a deadlock recovery event
                                deadlock_error = Exception(f"Deadlock detected: No events for {time_since_last_event:.1f}s")
                                deadlock_context = {
                                    'component': 'event_coordinator',
                                    'deadlock_duration': time_since_last_event,
                                    'events_processed': self._events_processed,
                                    'queue_size': self.agent_bus.qsize(),
                                    'current_step': self.state_manager.state.n_steps
                                }

                                # Trigger long-running mode recovery
                                await self.long_running_integration.long_running_mode.handle_failure(
                                    deadlock_error, deadlock_context
                                )
                                logger.info("Autonomous deadlock recovery initiated")

                            except Exception as recovery_error:
                                logger.error(f"Autonomous deadlock recovery failed: {recovery_error}")
                                # Fall back to standard deadlock resolution
                                await self._standard_deadlock_resolution()
                        else:
                            # Standard deadlock resolution for shorter deadlocks or when long-running disabled
                            await self._standard_deadlock_resolution()

                # If no events for 30 seconds, continue to check status
                continue
            except Exception as e:
                error_msg = f"Event coordinator failed: {e}"
                agent_log(logging.CRITICAL, self.state_manager.state.agent_id, self.state_manager.state.n_steps, error_msg, exc_info=True)
                await self.state_manager.record_error(error_msg, is_critical=True)
                await asyncio.sleep(1)

    async def _handle_action_completion(self, event):
        """Handle action completion events and trigger step finalization."""
        logger.info("Processing action completion event: %s", type(event).__name__)
        try:

            if isinstance(event, ActionExecuted):
                actuation_result = ActuationResult(
                    action_results=event.action_results,
                    llm_output=None,
                    browser_state=None,
                    step_metadata=StepMetadata(
                        step_number=event.step_token,
                        step_start_time=time.monotonic(),
                        step_end_time=time.monotonic()
                    )
                )
            else:
                error_result = ActionResult(success=False, error=event.error_message)
                actuation_result = ActuationResult(
                    action_results=[error_result],
                    llm_output=None,
                    browser_state=None,
                    step_metadata=StepMetadata(
                        step_number=event.step_token,
                        step_start_time=time.monotonic(),
                        step_end_time=time.monotonic()
                    )
                )

            await self._finalize_step(actuation_result)

            if self.settings.on_step_end:
                await self.settings.on_step_end(self)

            logger.info("Action completion handling completed successfully")

        except Exception as e:
            error_msg = f"Step finalization failed: {e}"
            logger.error("Action completion error: %s", error_msg, exc_info=True)
            agent_log(logging.ERROR, self.state_manager.state.agent_id, self.state_manager.state.n_steps, error_msg, exc_info=True)
            await self.state_manager.record_error(error_msg, is_critical=True)

    async def _finalize_step(self, result: ActuationResult):
        logger.info(f"Finalizing step with {len(result.action_results)} action results")

        if getattr(self.settings, 'enable_unified_finalization', False):
            outcome = await self.state_manager.decide_and_apply_after_step(result, max_steps=self.settings.max_steps, step_completed=True)
            next_status = outcome.status
            task_completed = outcome.task_completed
            logger.debug(f"Unified finalization outcome: status={next_status}, modes={outcome.modes}, reflect_intent={outcome.reflection_intent}")
            # In long-running mode, only allow stop on Ctrl+C (STOPPED via signal) or max steps
            if getattr(self.long_running_integration, 'enabled', False):
                if next_status in {AgentStatus.FAILED, AgentStatus.COMPLETED, AgentStatus.STOPPED}:
                    if next_status != AgentStatus.MAX_STEPS_REACHED:
                        # Force RUNNING to continue autonomously
                        await self.state_manager.set_status(AgentStatus.RUNNING, force=True)
                        logger.info("Long-running mode preventing terminal stop (status=%s); continuing run", next_status.value)
                        next_status = AgentStatus.RUNNING
            if next_status in TERMINAL_STATES:
                logger.info(f"Terminal state reached: {next_status.value}. Skipping next-cycle event publication.")
                return
        else:
            # Legacy path (kept for rollback)
            # Check for task completion before processing further
            task_completed = False
            for action_result in result.action_results:
                if action_result.is_done and action_result.success is True:
                    task_completed = True
                    logger.info(f"Task completed successfully with result: {action_result.extracted_content}")
                    break

            # Batch all lock-requiring operations into a single lock acquisition
            async with bulletproof_lock(self.state_manager._lock, self.state_manager.lock_timeout_seconds):
                # All state mutations happen within this single lock

                # Treat None as success (successful actions like go_to_url return success=None)
                if any(r.success is False for r in result.action_results):
                    error_msg = next((r.error for r in result.action_results if r.error), "An action failed.")
                    self.state_manager.state.last_error = error_msg
                    logger.warning(f"Step failed with error: {error_msg}")


                history_state = None
                if result.browser_state:
                    interacted_elements = AgentHistory.get_interacted_element(result.llm_output, result.browser_state.selector_map) if result.llm_output else []
                    history_state = BrowserStateHistory(
                        url=result.browser_state.url,
                        title=result.browser_state.title,
                        tabs=result.browser_state.tabs,
                        screenshot_path=result.browser_state.screenshot,
                        interacted_element=interacted_elements,
                    )
                history_item = AgentHistory(
                    model_output=result.llm_output,
                    result=result.action_results,
                    state=history_state,
                    metadata=result.step_metadata
                )

                self.state_manager._state.history.history.append(history_item)
                pruned_count = self.state_manager._memory_enforcer.enforce_budget(self.state_manager._state.history.history)
                if pruned_count > 0:
                    logger.debug(f"Memory enforcer pruned {pruned_count} history items")

                old_n_steps = self.state_manager.state.n_steps
                self.state_manager.state.n_steps += 1
                new_n_steps = self.state_manager.state.n_steps

                if task_completed:
                    next_status = AgentStatus.COMPLETED
                    logger.info("Task completed successfully, setting status to COMPLETED")
                elif self.state_manager._state.n_steps >= self.settings.max_steps:
                    next_status = AgentStatus.MAX_STEPS_REACHED
                elif any(r.success is False for r in result.action_results if hasattr(r, 'success')):
                    self.state_manager._state.consecutive_failures += 1
                    if self.state_manager._state.consecutive_failures >= self.settings.max_failures:
                        next_status = AgentStatus.FAILED
                    else:
                        next_status = AgentStatus.RUNNING
                else:
                    # Successful step - reset failure counter
                    self.state_manager._state.consecutive_failures = 0
                    next_status = AgentStatus.RUNNING


                self.state_manager._set_status_internal(next_status, force=False)
                logger.debug(f"Step finalized: status={next_status}, n_steps={self.state_manager._state.n_steps}")

                # In long-running mode, only allow stop on Ctrl+C (STOPPED via signal) or max steps
                if getattr(self.long_running_integration, 'enabled', False):
                    if next_status in {AgentStatus.FAILED, AgentStatus.COMPLETED, AgentStatus.STOPPED}:
                        if next_status != AgentStatus.MAX_STEPS_REACHED:
                            self.state_manager._set_status_internal(AgentStatus.RUNNING, force=True)
                            logger.info("Long-running mode preventing terminal stop (status=%s); continuing run", next_status.value)
                            next_status = AgentStatus.RUNNING

                if next_status in TERMINAL_STATES:
                    logger.info(f"Terminal state reached: {next_status.value}. Skipping next-cycle event publication.")
                    return

        # Check if task was marked as completed
        if task_completed:
            logger.info("Task completed, skipping next cycle setup")
            return

        logger.info("Getting fresh browser state for next cycle")
        try:
            browser_state = await self.perception._get_browser_state_with_recovery()
            logger.info(f"Got fresh browser state: URL={browser_state.url}")
            
            # CRITICAL FIX: Set step_barrier BEFORE publishing PerceptionSnapshot to prevent deadlock
            if hasattr(self, 'step_barrier') and self.step_barrier:
                self.step_barrier.set()
                logger.debug("Step barrier set - Perception can now proceed")
            
            perception_snapshot = PerceptionSnapshot(
                step_token=self.state_manager.state.n_steps,
                browser_state=browser_state,
                new_downloaded_files=None
            )

            try:
                self.agent_bus.put_nowait(perception_snapshot)
                logger.info(f"✅ Published PerceptionSnapshot for next cycle (step_token={self.state_manager.state.n_steps})")
            except asyncio.QueueFull:
                logger.warning("Agent bus full, dropping PerceptionSnapshot for next cycle")
        except Exception as e:
            logger.error(f"Failed to get fresh browser state: {e}")


        # Publish StepFinalized event to agent_bus (outside lock)
        step_finalized_event = StepFinalized(step_token=self.state_manager.state.n_steps)
        try:
            self.agent_bus.put_nowait(step_finalized_event)
            logger.info(f"✅ Published StepFinalized event for next cycle (step_token={self.state_manager.state.n_steps})")
        except asyncio.QueueFull:
            agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                     "Agent bus full, dropping StepFinalized event")

    async def _pause_handler(self):
        while True:
            # Cache status check to reduce lock contention from 20/sec to 1/sec
            current_status = await self.state_manager.get_status()
            if current_status in TERMINAL_STATES:
                break
            if current_status == AgentStatus.PAUSED:
                guidance = await self.state_manager.get_human_guidance()
                if guidance:
                    try:
                        agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps, "Human guidance consumed during pause")
                    except Exception:
                        pass
                    # 1) Record guidance in prompt context
                    self.message_manager.add_human_guidance(guidance)
                    # 2) Immediately set current_goal so next LLM cycle follows guidance
                    try:
                        await self.state_manager.set_current_goal(guidance)
                    except Exception:
                        pass
                    # 3) If still paused, nudge by sending a StepFinalized for the current step
                    #    so Perception emits a fresh snapshot reflecting updated goal once resumed.
                    try:
                        current_step = self.state_manager.state.n_steps
                        (self.control_bus if self.control_bus is not self.work_bus else self.work_bus).put_nowait(StepFinalized(step_token=current_step))
                    except asyncio.QueueFull:
                        logger.warning("Agent bus full, dropping guidance nudge StepFinalized event")
                    except Exception:
                        pass
            await asyncio.sleep(1.0)

    async def _load_shedding_monitor(self):
        """Lightweight CPU-based load shedding monitor.

        Uses psutil if present to sample CPU percent and flips LoadStatus.
        Falls back to a no-op if psutil is not available.
        """
        try:
            import psutil  # type: ignore
            have_psutil = True
        except Exception:
            have_psutil = False

        while True:
            # Cache status check to avoid lock contention
            current_status = await self.state_manager.get_status()
            if current_status in TERMINAL_STATES:
                break
            if not have_psutil:
                await asyncio.sleep(10.0)
                continue

            try:
                # Non-blocking CPU sampling: offload psutil call without sleeping the event loop
                cpu = await asyncio.to_thread(psutil.cpu_percent, None)
                # Use configurable thresholds when available
                shed_thr = float(getattr(self.settings, 'cpu_shed_threshold', CPU_SHEDDING_THRESHOLD))
                normal_thr = float(getattr(self.settings, 'cpu_normal_threshold', CPU_NORMAL_THRESHOLD))
                load_status = await self.state_manager.get_load_status()
                if cpu >= shed_thr and load_status != LoadStatus.SHEDDING:
                    await self.state_manager.set_load_status(LoadStatus.SHEDDING)
                elif cpu <= normal_thr and load_status != LoadStatus.NORMAL:
                    await self.state_manager.set_load_status(LoadStatus.NORMAL)
                # Cadence for next sample; keep it light
                await asyncio.sleep(1.0)
            except Exception:
                # Never fail the monitor; back off
                await asyncio.sleep(5.0)
                continue

    def pause(self): asyncio.create_task(self.state_manager.set_status(AgentStatus.PAUSED))
    def resume(self):
        async def _resume_and_nudge():
            await self.state_manager.set_status(AgentStatus.RUNNING, force=True)
            # Nudge the pipeline so Perception/Decision cycle restarts after a pause
            try:
                current_step = self.state_manager.state.n_steps
                (self.control_bus if self.control_bus is not self.work_bus else self.work_bus).put_nowait(StepFinalized(step_token=current_step))
            except asyncio.QueueFull:
                logger.warning("Agent bus full, dropping resume nudge StepFinalized event")
            except Exception:
                # Never fail resume on nudge errors
                pass
        asyncio.create_task(_resume_and_nudge())
    def stop(self):
        # Cooperative shutdown: signal components, then mark STOPPED after grace period
        try:
            self.shutdown_event.set()
        except Exception:
            pass
        delay = float(getattr(self.settings, 'shutdown_grace_seconds', 2.0))
        def _do_stop():
            asyncio.create_task(self.state_manager.set_status(AgentStatus.STOPPED))
        try:
            asyncio.get_event_loop().call_later(delay, _do_stop)
        except Exception:
            _do_stop()

    async def close(self):
        # Stop long-running mode monitoring first
        if hasattr(self, 'long_running_integration'):
            await self.long_running_integration.cleanup()

        if self.browser_session: await self.browser_session.stop()

    def _log_final_status(self):
        state = self.state_manager.state
        final_message = state.accumulated_output or f"Run ended with status: {state.status.value}"
        log_level = logging.INFO if state.status != AgentStatus.FAILED else logging.ERROR
        agent_log(log_level, state.agent_id, state.n_steps, f"🏁 Agent run finished. Status: {state.status.value}. Output: {final_message[:200]}...")

    def _generate_final_gif_if_enabled(self):
        if not self.settings.generate_gif: return
        try:
            output_path = self.settings.generate_gif if isinstance(self.settings.generate_gif, str) else f"agent_run_{self.state_manager.state.agent_id}.gif"
            create_history_gif(task=self.state_manager.state.task, history=self.state_manager.state.history, output_path=output_path)
        except Exception as e:
            agent_log(logging.ERROR, self.state_manager.state.agent_id, self.state_manager.state.n_steps, f"Failed to generate GIF: {e}")

    async def _record_error_in_history(self, error: Exception):
        agent_log(logging.CRITICAL, self.state_manager.state.agent_id, self.state_manager.state.n_steps, f"Unhandled exception in TaskGroup: {error}", exc_info=error)
        error_result = ActionResult(success=False, error=AgentError.format_error(error, include_trace=True))
        metadata = StepMetadata(step_number=self.state_manager.state.n_steps, step_start_time=time.monotonic(), step_end_time=time.monotonic())

        # Safely try to get browser state, with fallback for browser failures
        browser_state_summary = None
        if self.browser_session:
            try:
                browser_state_summary = await self.perception._get_browser_state_with_recovery()
            except Exception as browser_error:
                agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                         f"Failed to get browser state during error recording: {browser_error}")
                # Don't re-raise browser errors during error recording

        history_state = None
        if browser_state_summary:
            history_state = BrowserStateHistory(
                url=browser_state_summary.url,
                title=browser_state_summary.title,
                tabs=browser_state_summary.tabs,
                screenshot_path=browser_state_summary.screenshot,
                interacted_element=[], # No interacted element in an error step
            )
        history_item = AgentHistory(
            model_output=None,
            result=[error_result],
            state=history_state,
            metadata=metadata
        )
        await self.state_manager.add_history_item(history_item)

    def _setup_browser_session(self):
        if self.settings.browser_session: self.browser_session = self.settings.browser_session; return
        if isinstance(self.settings.browser, BrowserSession): self.browser_session = self.settings.browser; return
        self.browser_session = BrowserSession(
            browser_profile=self.settings.browser_profile or DEFAULT_BROWSER_PROFILE,
            browser=self.settings.browser, browser_context=self.settings.browser_context, agent_current_page=self.settings.page)

    def _setup_filesystem(self, state: AgentState):
        if state.file_system_state and self.settings.file_system_path:
            raise ValueError("Cannot provide both file_system_state and file_system_path.")
        if state.file_system_state: fs = FileSystem.from_state(state.file_system_state)
        elif self.settings.file_system_path: fs = FileSystem(self.settings.file_system_path)
        else: fs = FileSystem(os.path.join(tempfile.gettempdir(), f'browser_use_{state.agent_id}'))
        self.settings.file_system = fs
        state.file_system_state = fs.get_state()

    async def _handle_autonomous_continuation(self, event: dict):
        """Handle autonomous continuation event from long-running mode."""
        try:
            recovery_action = event.get('recovery_action', 'unknown')
            context = event.get('context', {})

            logger.info(f"Autonomous continuation triggered with recovery action: {recovery_action}")

            # For deadlock scenarios, we need to force actual progression
            if recovery_action == "force_step_progression":
                logger.info("Forcing step progression to break deadlock")
                await self.force_step_progression()
                return
            if recovery_action == "restart_browser_session":
                logger.info("Autonomous continuation requested browser restart")
                await self.restart_browser_session()
                return

            # Check if agent is in a terminal state and can be resumed
            current_status = await self.state_manager.get_status()
            if current_status in [AgentStatus.FAILED, AgentStatus.STOPPED]:
                logger.info("Agent in terminal state, attempting autonomous restart")

                # Reset agent status to allow continuation
                await self.state_manager.set_status(AgentStatus.RUNNING)

                # Generate a continuation task based on the original task and current context
                original_task = self.state_manager.state.task
                current_goal = self.state_manager.state.current_goal or "Continue from where we left off"

                continuation_message = (
                    f"Autonomous continuation after {recovery_action}. "
                    f"Original task: {original_task}. "
                    f"Current goal: {current_goal}. "
                    f"Please retry the failed operation or adapt the approach."
                )

                # Trigger a new planning cycle
                planning_event = {
                    'type': 'autonomous_planning_request',
                    'message': continuation_message,
                    'context': context,
                    'recovery_action': recovery_action
                }

                # Put the planning event on the bus for the planner to pick up
                await self.agent_bus.put(planning_event)

                logger.info("Autonomous continuation planning request submitted")

            else:
                logger.info(f"Agent status is {current_status}, no autonomous restart needed")

        except Exception as e:
            logger.error(f"Failed to handle autonomous continuation: {e}")
            # Don't re-raise - we don't want to crash the supervisor

    async def _standard_deadlock_resolution(self):
        """Standard deadlock resolution - force a synthetic step progression."""
        logger.warning("Attempting to break deadlock by forcing step progression")
        # If Perception is blocked on step_barrier.wait(), unblock it preemptively
        try:
            if self.step_barrier is not None and not self.step_barrier.is_set():
                self.step_barrier.set()
                logger.info("Deadlock recovery: step_barrier set to unblock Perception")
        except Exception:
            logger.debug("Failed to set step_barrier during deadlock recovery", exc_info=True)

        # Finalize a synthetic no-op step to advance n_steps and publish fresh events
        await self.force_step_progression()

    async def force_step_progression(self):
        """Finalize a synthetic no-op step to advance the engine and break deadlocks."""
        try:
            # Create a no-op successful action result to drive finalization
            noop_result = ActionResult(success=True)
            metadata = StepMetadata(
                step_number=self.state_manager.state.n_steps,
                step_start_time=time.monotonic(),
                step_end_time=time.monotonic(),
            )
            actuation = ActuationResult(
                action_results=[noop_result],
                llm_output=None,
                browser_state=None,
                step_metadata=metadata,
            )
            await self._finalize_step(actuation)
            # Ensure perception can proceed
            self.step_barrier.set()
            logger.info("Synthetic step finalized to force progression; deadlock broken")
        except Exception as e:
            logger.error(f"Failed to force step progression: {e}")

    async def restart_browser_session(self):
        """Restart the browser session and rewire dependent components."""
        try:
            if self.browser_session:
                try:
                    await self.browser_session.stop()
                except Exception:
                    logger.debug("Error stopping browser session during restart", exc_info=True)
            # Recreate session
            self._setup_browser_session()
            # Rewire components to use the new session
            self.perception.browser_session = self.browser_session
            self.actuator.browser_session = self.browser_session
            logger.info("Browser session restarted successfully")
            # Nudge the loop
            await self.force_step_progression()
        except Exception as e:
            logger.error(f"Failed to restart browser session: {e}")

    async def restart_components(self, components: list[str] | None = None) -> dict[str, bool]:
        """Restart specific or all monitored components via ReactorVitals."""
        results: dict[str, bool] = {}
        try:
            if not hasattr(self, 'reactor_vitals'):
                return results
            names = components or list(self.reactor_vitals.monitors.keys())
            for name in names:
                try:
                    ok = await self.reactor_vitals._restart_component(name)
                    results[name] = bool(ok)
                except Exception:
                    logger.debug("Component restart failed for %s", name, exc_info=True)
                    results[name] = False
            # Nudge the loop
            await self.force_step_progression()
        except Exception:
            logger.debug("restart_components encountered an error", exc_info=True)
        return results
