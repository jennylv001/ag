from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from typing import TYPE_CHECKING, Optional

import psutil 
from .actuator import Actuator
from .decision_maker import DecisionMaker
from .events import ActuationResult, Decision, PerceptionOutput 
from .gif import create_history_gif
from .message_manager.service import MessageManager, MessageManagerSettings
from .perception import Perception
from .state_manager import AgentState, AgentStatus, StateManager, agent_log, LoadStatus, TERMINAL_STATES
from .task_monitor import TaskMonitor
from .views import AgentHistory, AgentHistoryList, AgentError, ActionResult, StepMetadata
from ..browser import BrowserSession
from ..browser.views import BrowserStateHistory
from ..browser.session import DEFAULT_BROWSER_PROFILE
from ..exceptions import AgentInterruptedError
from ..filesystem.file_system import FileSystem
from ..utils import SignalHandler

if TYPE_CHECKING:
    from browser_use.agent.settings import AgentSettings

logger = logging.getLogger(__name__)

CPU_SHEDDING_THRESHOLD = 98.0  # Start shedding load if CPU usage is > 96%
CPU_NORMAL_THRESHOLD = 96.0    # Return to normal if CPU usage is < 85%


class Supervisor:
    """ "Concurrency with Consequences" """

    def __init__(self, settings: AgentSettings):
        self.settings = settings
        self._setup_components()
        self._task_monitor: Optional[TaskMonitor] = None


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
            max_history_items=self.settings.max_history_items
        )
        self._setup_browser_session()
        self.perception_queue = asyncio.Queue(maxsize=5)
        self.decision_queue = asyncio.Queue(maxsize=5)
        self.perception = Perception(self.browser_session, self.state_manager, self.settings, self.perception_queue)
        self.message_manager = MessageManager(
            task=state.task, system_message=None,
            settings=MessageManagerSettings.model_validate(self.settings.model_dump(include=MessageManagerSettings.model_fields.keys())),
            state=state.message_manager_state, file_system=self.settings.file_system
        )
        self.decision_maker = DecisionMaker(
            settings=self.settings, state_manager=self.state_manager,
            message_manager=self.message_manager, llm=self.settings.llm,
            planner_llm=(self.settings.planner_llm or self.settings.llm) if self.settings.use_planner else None
        )
        self.actuator = Actuator(self.settings.controller, self.browser_session, self.state_manager, self.settings)

        if self.settings.output_model:
            self.settings.controller.use_structured_output_action(self.settings.output_model)
 
    async def run(self) -> AgentHistoryList:
        signal_handler = SignalHandler(loop=asyncio.get_event_loop(), pause_callback=self.pause, resume_callback=self.resume, custom_exit_callback=self.stop)
        signal_handler.register()

        try:
            if self.settings.on_run_start: await self.settings.on_run_start(self)
            agent_log(logging.INFO, self.state_manager.state.agent_id, 0, f"🚀 Starting agent run for task: \"{self.state_manager.state.task[:70]}...\"")
            
            await self._execute_initial_actions()

            # CORRECTED: Use the imported TERMINAL_STATES constant directly
            if await self.state_manager.get_status() not in TERMINAL_STATES:
                await self.state_manager.set_status(AgentStatus.RUNNING)
                try:
                    # Create TaskMonitor for robust restart capability
                    self._task_monitor = TaskMonitor(max_restart_attempts=5, base_backoff=2.0, max_backoff=60.0)
                    
                    # Create individual tasks instead of using TaskGroup
                    perception_watchdog_task = asyncio.create_task(self.perception.watchdog())
                    perception_run_task = asyncio.create_task(self.perception.run())
                    decision_loop_task = asyncio.create_task(self._decision_loop())
                    actuation_loop_task = asyncio.create_task(self._actuation_loop())
                    pause_handler_task = asyncio.create_task(self._pause_handler())
                    load_shedding_task = asyncio.create_task(self._load_shedding_monitor())
                    
                    # Set up restart factories for each component
                    restart_factories = {
                        'perception_watchdog': lambda: asyncio.create_task(self.perception.watchdog()),
                        'perception': lambda: asyncio.create_task(self.perception.run()),
                        'decision_loop': lambda: asyncio.create_task(self._decision_loop()),
                        'actuation_loop': lambda: asyncio.create_task(self._actuation_loop()),
                        'pause_handler': lambda: asyncio.create_task(self._pause_handler()),
                        'load_shedding_monitor': lambda: asyncio.create_task(self._load_shedding_monitor()),
                    }
                    self._task_monitor.set_restart_factories(restart_factories)
                    
                    # Register tasks with the monitor
                    self._task_monitor.register('perception_watchdog', perception_watchdog_task)
                    self._task_monitor.register('perception', perception_run_task)
                    self._task_monitor.register('decision_loop', decision_loop_task)
                    self._task_monitor.register('actuation_loop', actuation_loop_task)
                    self._task_monitor.register('pause_handler', pause_handler_task)
                    self._task_monitor.register('load_shedding_monitor', load_shedding_task)
                    
                    # Enable restart for critical components
                    self._task_monitor.enable(['perception', 'decision_loop', 'actuation_loop', 'perception_watchdog'])
                    
                    # Wait for tasks to complete or terminal status
                    all_tasks = [
                        perception_watchdog_task, perception_run_task, decision_loop_task,
                        actuation_loop_task, pause_handler_task, load_shedding_task
                    ]
                    
                    try:
                        # Wait for any task to complete or for terminal status
                        while await self.state_manager.get_status() not in TERMINAL_STATES:
                            # Check if any critical task has failed and isn't being restarted
                            done_tasks = [task for task in all_tasks if task.done()]
                            if done_tasks:
                                # Allow some time for restart mechanism to work
                                await asyncio.sleep(1.0)
                                # Update task list to track new tasks created by restarts
                                current_tasks = []
                                if self._task_monitor:
                                    for name in ['perception_watchdog', 'perception', 'decision_loop', 
                                               'actuation_loop', 'pause_handler', 'load_shedding_monitor']:
                                        task = self._task_monitor._tasks.get(name)
                                        if task and not task.done():
                                            current_tasks.append(task)
                                all_tasks = current_tasks
                            
                            if not all_tasks:
                                # All tasks are done, check if we should exit
                                break
                                
                            await asyncio.sleep(0.5)
                            
                    except Exception as e:
                        await self.state_manager.set_status(AgentStatus.FAILED, force=True)
                        await self._record_error_in_history(e)
                        
                except Exception as e:
                    await self.state_manager.set_status(AgentStatus.FAILED, force=True)
                    await self._record_error_in_history(e)
        except AgentInterruptedError:
            await self.state_manager.set_status(AgentStatus.STOPPED)
        finally:
            signal_handler.unregister()
            
            # Clean shutdown of task monitor
            if self._task_monitor:
                await self._task_monitor.close()
                self._task_monitor = None
                
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
        # CORRECTED: Use keyword args matching the model definition.
        history_item = AgentHistory(model_output=None, result=action_results, state=browser_state.to_history(), metadata=metadata)
        await self.state_manager.add_history_item(history_item)
        
        await self.state_manager.update_after_step(
            results=action_results,
            max_steps=self.settings.max_steps,
            planner_interval=self.settings.planner_interval
        )

        agent_log(logging.INFO, self.state_manager.state.agent_id, 0, "Initial actions processed.")


    async def _decision_loop(self):
        while await self.state_manager.get_status() not in TERMINAL_STATES:
            try:
                perception_output: PerceptionOutput = await asyncio.wait_for(self.perception_queue.get(), timeout=30)
                
                if perception_output.new_downloaded_files:
                    self.settings.available_file_paths.extend(
                        [f for f in perception_output.new_downloaded_files if f not in self.settings.available_file_paths]
                    )

                if self.settings.on_step_start: await self.settings.on_step_start(self)
                
                self.message_manager.settings.available_file_paths = self.settings.available_file_paths

                decision = await self.decision_maker.decide(perception_output)
                self.perception_queue.task_done()
                if decision.llm_output or decision.action_results:
                    await self.decision_queue.put(decision)
            except asyncio.TimeoutError:
                if await self.state_manager.get_status() not in TERMINAL_STATES:
                     raise RuntimeError("Decision loop timed out waiting for perception.")
            except Exception as e:
                await self.state_manager.record_error(f"Error in decision loop: {e}", is_critical=True)

    async def _actuation_loop(self):
        while await self.state_manager.get_status() not in TERMINAL_STATES:
            try:
                decision: Decision = await asyncio.wait_for(self.decision_queue.get(), timeout=30)
                step_start_time = time.monotonic()
                actuation_result = await self.actuator.execute(decision)
                self.decision_queue.task_done()
                await self._finalize_step(actuation_result)
                agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps, 
                          f"Step completed in {time.monotonic() - step_start_time:.2f}s.")
                if self.settings.on_step_end: await self.settings.on_step_end(self)
            except asyncio.TimeoutError:
                 if await self.state_manager.get_status() not in TERMINAL_STATES:
                    raise RuntimeError("Actuation loop timed out waiting for decision.")
            except Exception as e:
                await self.state_manager.record_error(f"Error in actuation loop: {e}", is_critical=True)

    async def _finalize_step(self, result: ActuationResult):
        if any(not r.success for r in result.action_results):
            error_msg = next((r.error for r in result.action_results if r.error), "An action failed.")
            self.state_manager.state.last_error = error_msg

        history_state = None
        if result.browser_state:
            # Manually construct the BrowserStateHistory object from the BrowserStateSummary
            interacted_elements = AgentHistory.get_interacted_element(result.llm_output, result.browser_state.selector_map) if result.llm_output else []
            history_state = BrowserStateHistory(
                url=result.browser_state.url,
                title=result.browser_state.title,
                tabs=result.browser_state.tabs,
                screenshot=result.browser_state.screenshot,
                interacted_element=interacted_elements,
            )
        history_item = AgentHistory(
            model_output=result.llm_output,
            result=result.action_results,
            state=history_state,
            metadata=result.step_metadata
        )
        await self.state_manager.add_history_item(history_item)
        await self.state_manager.update_after_step(
            results=result.action_results, max_steps=self.settings.max_steps,
            planner_interval=self.settings.planner_interval)

    async def _pause_handler(self):
        while await self.state_manager.get_status() not in TERMINAL_STATES:
            if await self.state_manager.get_status() == AgentStatus.PAUSED:
                guidance = await self.state_manager.get_human_guidance()
                if guidance: self.message_manager.add_human_guidance(guidance)
            await asyncio.sleep(0.1)
            
    async def _load_shedding_monitor(self):
        """Monitors system CPU usage and updates the agent's load status accordingly."""
        agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps, "Dynamic load shedding monitor started.")
        while await self.state_manager.get_status() not in TERMINAL_STATES:
            cpu_percent = psutil.cpu_percent(interval=2.0)
            current_load_status = await self.state_manager.get_load_status()

            if cpu_percent > CPU_SHEDDING_THRESHOLD and current_load_status == LoadStatus.NORMAL:
                await self.state_manager.set_load_status(LoadStatus.SHEDDING)

            elif cpu_percent < CPU_NORMAL_THRESHOLD and current_load_status == LoadStatus.SHEDDING:
                await self.state_manager.set_load_status(LoadStatus.NORMAL)
            
            await asyncio.sleep(3.0) # Check every 5 seconds (2s interval + 3s sleep)

    def pause(self): asyncio.create_task(self.state_manager.set_status(AgentStatus.PAUSED))
    def resume(self): asyncio.create_task(self.state_manager.set_status(AgentStatus.RUNNING, force=True))
    def stop(self): asyncio.create_task(self.state_manager.set_status(AgentStatus.STOPPED))

    async def close(self):
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
        # CORRECTED: Removed invalid 'action_name' kwarg from ActionResult.
        error_result = ActionResult(success=False, error=AgentError.format_error(error))
        metadata = StepMetadata(step_number=self.state_manager.state.n_steps, step_start_time=time.monotonic(), step_end_time=time.monotonic())
        browser_state_summary = await self.perception._get_browser_state_with_recovery() if self.browser_session else None
        
        history_state = None
        if browser_state_summary:
            history_state = BrowserStateHistory(
                url=browser_state_summary.url,
                title=browser_state_summary.title,
                tabs=browser_state_summary.tabs,
                screenshot=browser_state_summary.screenshot,
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
        # CORRECTED: Changed 'file_state' to 'file_system_state' to match AgentState model.
        if state.file_system_state: fs = FileSystem.from_state(state.file_system_state)
        elif self.settings.file_system_path: fs = FileSystem(self.settings.file_system_path)
        else: fs = FileSystem(os.path.join(tempfile.gettempdir(), f'browser_use_{state.agent_id}'))
        self.settings.file_system = fs
        state.file_system_state = fs.get_state()