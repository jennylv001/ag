from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Generic, Optional, TypeVar

from browser_use.agent.settings import AgentSettings
from browser_use.agent.views import AgentHistory, AgentHistoryList, ActionResult, AgentOutput
from browser_use.dom.history_tree_processor.service import DOMHistoryElement, HistoryTreeProcessor
from browser_use.controller.service import Controller
from browser_use.utils import SignalHandler

if TYPE_CHECKING:
    from browser_use.agent.views import ActionModel
    from browser_use.browser.session import BrowserSession
    from browser_use.browser.views import BrowserStateSummary

ContextT = TypeVar('ContextT')
logger = logging.getLogger(__name__)


class Agent(Generic[ContextT]):
    """Public-facing wrapper for the "Bulletproof" agent architecture."""
    settings: AgentSettings

    def __init__(self, settings: AgentSettings):
        self.settings = settings
        if not getattr(self.settings, 'controller', None):
            self.settings.controller = Controller()

        # Prepare a process-wide SignalHandler for pause/resume and guidance injection
        try:
            loop = asyncio.get_event_loop()

            def _pause_cb():
                try:
                    self.pause()
                except Exception:
                    pass

            def _resume_cb():
                try:
                    self.resume()
                except Exception:
                    pass

            def _exit_cb():
                try:
                    self.stop()
                except Exception:
                    pass

            def _guidance_cb(text: str):
                try:
                    self.inject_human_guidance(text)
                except Exception:
                    pass

            self._signal_handler = SignalHandler(
                loop=loop,
                pause_callback=_pause_cb,
                resume_callback=_resume_cb,
                custom_exit_callback=_exit_cb,
                guidance_callback=_guidance_cb,
            )
            self._signal_handler.register()
        except Exception:
            # Non-fatal; continue without interactive pause handler
            self._signal_handler = None

        # Ensure the controller has a default_search_engine configured (with safe fallback)
        try:
            engine = getattr(self.settings, 'default_search_engine', None) or os.getenv('BROWSER_USE_DEFAULT_SEARCH_ENGINE')
            if not engine:
                engine = 'duckduckgo'
            engine = str(engine).lower().strip()
            allowed = {'duckduckgo', 'google', 'bing'}
            if engine not in allowed:
                logger.warning(
                    "Unsupported default_search_engine '%s'. Falling back to 'duckduckgo' (allowed: %s)",
                    engine,
                    ", ".join(sorted(allowed)),
                )
                engine = 'duckduckgo'
            setattr(self.settings.controller, 'default_search_engine', engine)

            # Also attach the main LLM and UX flags to controller for actions that need LLM defaulting
            if getattr(self.settings, 'llm', None):
                controller_settings = type('Settings', (), {
                    'llm': self.settings.llm,
                    'pause_on_first_click': bool(getattr(self.settings, 'pause_on_first_click', False)),
                    'signal_handler': getattr(self, '_signal_handler', None),
                })()
                setattr(self.settings.controller, 'settings', controller_settings)
        except Exception:
            # Non-fatal: action will surface a clear error if needed
            pass

        # Ensure a default local file system is available
        try:
            if getattr(self.settings, 'file_system', None) is None:
                from pathlib import Path as _Path
                from browser_use.filesystem.file_system import FileSystem as _FS
                base_dir = getattr(self.settings, 'file_system_path', None) or _Path.cwd()
                self.settings.file_system = _FS(base_dir=base_dir)
        except Exception:
            pass

        # Try to use legacy Supervisor; otherwise fall back to lightweight orchestrator
        try:
            from browser_use.agent.supervisor import Supervisor as _Supervisor  # type: ignore
            self.supervisor = _Supervisor(settings)
        except (RuntimeError, ImportError, ModuleNotFoundError):
            logger.debug("Using internal orchestrator (supervisor unavailable).")

            class _StatelessSupervisor:
                def __init__(self, settings: AgentSettings):
                    self.settings = settings
                    from browser_use.agent.state import AgentState, StateManager

                    try:
                        if getattr(self.settings, 'file_system', None) is None:
                            from pathlib import Path as _Path
                            from browser_use.filesystem.file_system import FileSystem as _FS
                            base_dir = getattr(self.settings, 'file_system_path', None) or _Path.cwd()
                            self.settings.file_system = _FS(base_dir=base_dir)
                    except Exception:
                        pass

                    self.state_manager = StateManager(
                        initial_state=settings.injected_agent_state or AgentState(task=settings.task),
                        file_system=settings.file_system,
                        max_failures=settings.max_failures,
                        lock_timeout_seconds=settings.lock_timeout_seconds,
                        use_planner=False,
                        reflect_on_error=False,
                        max_history_items=settings.max_history_items,
                        memory_budget_mb=settings.memory_budget_mb,
                    )
                    # Attach a MissionState instance if task layer is enabled (no behavior change otherwise)
                    try:
                        from browser_use.agent.state import MissionState
                        if bool(getattr(settings, 'task_layer_enabled', False)):
                            if not getattr(self.state_manager.state, 'mission', None):
                                self.state_manager.state.mission = MissionState(description=str(settings.task or ''))
                    except Exception:
                        pass

                    from browser_use.agent.message_manager.service import MessageManager, MessageManagerSettings
                    from browser_use.agent.message_manager.views import MessageManagerState

                    # Load system prompt from file
                    from pathlib import Path
                    system_prompt_path = Path(__file__).parent / "system_prompt.md"
                    try:
                        system_message = system_prompt_path.read_text(encoding='utf-8')
                    except Exception:
                        system_message = None  # Fallback to None if file not found

                    self.message_manager = MessageManager(
                        task=settings.task,
                        system_message=system_message,
                        settings=MessageManagerSettings.model_validate({}),
                        state=MessageManagerState(),
                        file_system=settings.file_system,
                    )
                    from browser_use.browser.session import BrowserSession
                    if settings.browser_session:
                        self.browser_session = settings.browser_session
                    else:
                        kwargs = {}
                        try:
                            # Prefer explicit BrowserProfile on settings for end-to-end stealth wiring
                            if getattr(settings, 'browser_profile', None) is not None:
                                from browser_use.browser.profile import BrowserProfile as _BP
                                bp: _BP = settings.browser_profile  # type: ignore[assignment]
                                kwargs.update(bp.model_dump(exclude_none=True))
                            elif hasattr(settings, 'browser_config') and settings.browser_config:
                                # Back-compat: legacy field name
                                kwargs.update(settings.browser_config.model_dump(exclude_none=True))  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        self.browser_session = BrowserSession(**kwargs)

                    # Initialize the new AgentOrchestrator (unified architecture)
                    from browser_use.agent.orchestrator import AgentOrchestrator
                    self.orchestrator = AgentOrchestrator(
                        settings=settings,
                        state_manager=self.state_manager,
                        message_manager=self.message_manager,
                        browser_session=self.browser_session,
                        controller=settings.controller,
                    )

                async def run(self):
                    # Use the new unified orchestrator instead of the old loop
                    try:
                        return await self.orchestrator.run()
                    except Exception as e:
                        logger.error(f"Orchestrator run failed: {e}")
                        # Fallback to empty history on error
                        return AgentHistoryList()

                def pause(self):
                    try:
                        from browser_use.agent.state import AgentStatus
                        asyncio.create_task(self.orchestrator.set_status(AgentStatus.PAUSED, force=True))
                    except Exception:
                        pass

                def resume(self):
                    try:
                        from browser_use.agent.state import AgentStatus
                        asyncio.create_task(self.orchestrator.set_status(AgentStatus.RUNNING, force=True))
                    except Exception:
                        pass

                def stop(self):
                    try:
                        from browser_use.agent.state import AgentStatus
                        asyncio.create_task(self.orchestrator.set_status(AgentStatus.STOPPED, force=True))
                    except Exception:
                        pass

                async def close(self):
                    try:
                        await self.browser_session.stop(_hint='(agent.supervisor close)')
                    except Exception:
                        logger.debug('Error while closing browser_session in supervisor.close()', exc_info=True)

            self.supervisor = _StatelessSupervisor(self.settings)

    async def run(self) -> AgentHistoryList:
        return await self.supervisor.run()

    def pause(self): self.supervisor.pause()
    def resume(self): self.supervisor.resume()
    def stop(self): self.supervisor.stop()

    def inject_human_guidance(self, text: str):
        asyncio.create_task(self.supervisor.orchestrator.add_human_guidance(text))

    async def add_new_task(self, new_task: str):
        await self.supervisor.orchestrator.update_task(new_task)
        self.supervisor.message_manager.add_new_task(new_task)
        logger.info(f"Agent task updated to: {new_task}")

    @property
    def state(self): return self.supervisor.orchestrator.state
    @property
    def browser_session(self) -> BrowserSession: return self.supervisor.browser_session
    @property
    def orchestrator(self):
        """Access the AgentOrchestrator (placeholder for new unified architecture)."""
        return getattr(self.supervisor, 'orchestrator', None)

    async def close(self): await self.supervisor.close()

    # Expose a helper to trigger a resume/guidance prompt (for internal use by actions)
    def _prompt_resume_or_guidance(self):
        try:
            handler = getattr(self, '_signal_handler', None)
            if handler is not None:
                # Non-blocking prompt within the handler; safe to call from async flows
                # Prefer async waiter path so we don’t block the event loop
                asyncio.create_task(handler._async_wait_for_resume())
        except Exception:
            pass

    def _get_agent_output_schema(self, include_done: bool = False) -> type[AgentOutput]:
        """ ** FIX: Helper to build the correct AgentOutput schema based on settings. ** """
        action_model_source = self.settings.controller.registry
        action_model = action_model_source.create_action_model(include_actions=['done'] if include_done else [])

        if self.settings.flash_mode:
            return AgentOutput.type_with_custom_actions_flash_mode(action_model)
        elif self.settings.use_thinking:
            return AgentOutput.type_with_custom_actions(action_model)
        else:
            return AgentOutput.type_with_custom_actions_no_thinking(action_model)

    async def load_and_rerun(self, history_file: str, **kwargs) -> list[ActionResult]:
        agent_output_model = self._get_agent_output_schema(include_done=True)
        history = AgentHistoryList.load_from_file(history_file, agent_output_model)
        return await self.rerun_history(history, **kwargs)

    async def rerun_history(self, history: AgentHistoryList, max_retries: int = 3, skip_failures: bool = True, delay_between_actions: float = 2.0) -> list[ActionResult]:
        results = []
        for i, history_item in enumerate(history.history):
            if not history_item.model_output or not history_item.model_output.action: continue
            retry_count = 0
            while retry_count < max_retries:
                try:
                    result = await self._execute_history_step(history_item, delay_between_actions)
                    results.extend(result); break
                except Exception as e:
                    logger.warning(f"Failed to execute step {i} from history, retry {retry_count+1}/{max_retries}. Error: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        if not skip_failures: raise
                    else: await asyncio.sleep(delay_between_actions)
        return results

    async def _execute_history_step(self, history_item: AgentHistory, delay: float) -> list[ActionResult]:
        state = await self.browser_session.get_state_summary()
        if not state or not history_item.model_output or not history_item.state:
            raise ValueError("Invalid state or model output in history item for replay")
        updated_actions = []
        for i, action in enumerate(history_item.model_output.action):
            historical_element = history_item.state.interacted_element[i] if history_item.state.interacted_element else None
            updated_action = await self._update_action_indices(historical_element, action, state)
            if updated_action is None:
                raise ValueError(f"Could not find matching element for action {i} in current page state")
            updated_actions.append(updated_action)
        result = await self.settings.controller.multi_act(actions=updated_actions, browser_session=self.browser_session)
        await asyncio.sleep(delay)
        return result

    async def _update_action_indices(self, historical_element: DOMHistoryElement, action: ActionModel, browser_state_summary: BrowserStateSummary) -> Optional[ActionModel]:
        if not historical_element:

            if action.get_index() is not None:
                logger.warning(f"Rejecting replay of indexed action {action} due to missing historical element context.")
                return None
            return action

        if not browser_state_summary.element_tree: return None
        current_element = HistoryTreeProcessor.find_history_element_in_tree(historical_element, browser_state_summary.element_tree)
        if not current_element or current_element.highlight_index is None: return None
        action.set_index(current_element.highlight_index)
        return action
