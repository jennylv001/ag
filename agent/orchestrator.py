from __future__ import annotations

"""
Unified Agent Orchestrator - Foundation for the new architecture

This file contains the AgentOrchestrator class that will eventually contain
the unified perceive → decide → execute loop, absorbing logic from loop.py,
decision_maker.py, and actuator.py.

Currently a placeholder to establish the new architecture without breaking
existing functionality.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional, Any
from collections import deque
from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path
import base64
import anyio
from datetime import datetime

if TYPE_CHECKING:
    from .settings import AgentSettings
    from .state import StateManager, OrchestratorState
    from .message_manager.service import MessageManager
    from ..browser import BrowserSession
    from ..controller.service import Controller
    from .views import AgentHistoryList, AgentOutput

logger = logging.getLogger(__name__)


# Lightweight result container used by execute(); defined at module scope to avoid
# nested-class indentation sensitivity during import in some environments.
@dataclass
class ActuationResult:
    action_results: list | None
    llm_output: Any
    browser_state: Any
    step_metadata: Any


class _SemanticPageCapture:
    """Minimal implementation for semantic page capture (replaces deleted loop.py version)"""

    def __init__(self, browser_session: 'BrowserSession', settings: 'AgentSettings'):
        self.browser_session = browser_session
        self.settings = settings

    async def capture_semantic_page(self) -> Optional[dict[str, Any]]:
        """Capture semantic page information for perception"""
        try:
            # Get current browser state
            state = await self.browser_session.get_state_summary()
            if not state:
                return None

            # Create a simple hash from the page content
            page_content = str(state.url) + str(len(state.element_tree or []))
            page_hash = hash(page_content)

            return {
                'page_hash': str(page_hash),
                'url': str(state.url) if state.url else None,
                'element_count': len(state.element_tree or []),
            }
        except Exception as e:
            logger.debug(f"Semantic page capture failed: {e}")
            return None



# Phase 4: Move TransitionEngine classes to orchestrator (removed from state.py)
class AgentMode(IntFlag):
    NONE = 0
    DEGRADED = 1 << 0
    STALLING = 1 << 1
    UNCERTAIN = 1 << 2
    HIGH_LOAD = 1 << 3
    DRAINING = 1 << 4


@dataclass(frozen=True)
class TransitionInputs:
    status: 'AgentStatus'
    n_steps: int
    consecutive_failures: int
    modes: int
    load_status: 'LoadStatus'
    step_completed: bool
    had_failure: bool
    missed_heartbeats: int
    io_timeouts_recent_count: int
    max_steps: int
    max_failures: int
    reflect_on_error: bool
    use_planner: bool
    reflect_cadence: int
    reflect_cooldown_seconds: float
    seconds_since_last_reflect: Optional[float] = None
    reflection_requested_this_epoch: bool = False
    consecutive_action_failures: int = 0
    last_step_duration_seconds: float = 0.0
    oscillation_score: float = 0.0
    no_progress_score: float = 0.0


@dataclass(frozen=True)
class TransitionDecision:
    next_status: 'AgentStatus'
    set_modes: int
    clear_modes: int
    reason: str
    reflection_intent: bool
    cooldown_blocked: bool = False


class _TransitionEngine:
    HB_DEGRADED_THRESHOLD = 1
    HB_STALLING_THRESHOLD = 3
    IO_UNCERTAIN_THRESHOLD = 1
    IO_STALLING_THRESHOLD = 3
    ACTION_FAILURE_STREAK_REFLECT = 2
    SLOW_STEP_REFLECT_SECONDS = 20.0
    OSCILLATION_REFLECT_THRESHOLD = 0.8
    NO_PROGRESS_REFLECT_THRESHOLD = 0.8

    def decide(self, inp: TransitionInputs) -> TransitionDecision:
        from .state import AgentStatus, LoadStatus

        desired = AgentMode.NONE
        if inp.missed_heartbeats >= self.HB_DEGRADED_THRESHOLD:
            desired |= AgentMode.DEGRADED
        if inp.missed_heartbeats >= self.HB_STALLING_THRESHOLD:
            desired |= AgentMode.STALLING
        if inp.io_timeouts_recent_count >= self.IO_UNCERTAIN_THRESHOLD:
            desired |= AgentMode.UNCERTAIN
        if inp.io_timeouts_recent_count >= self.IO_STALLING_THRESHOLD:
            desired |= AgentMode.STALLING
        if inp.load_status == LoadStatus.SHEDDING:
            desired |= AgentMode.HIGH_LOAD

        current = AgentMode(inp.modes)
        set_mask = int(desired & ~current)
        clear_mask = int(current & ~desired)
        active = desired | current

        next_status = inp.status
        reasons: list[str] = []

        if inp.step_completed:
            next_step = inp.n_steps + 1
            if next_step >= inp.max_steps:
                next_status = AgentStatus.MAX_STEPS_REACHED
                reasons.append("max_steps_reached")
            else:
                if inp.had_failure:
                    if (inp.consecutive_failures + 1) >= inp.max_failures:
                        next_status = AgentStatus.FAILED
                        reasons.append("failures_threshold")
                    else:
                        next_status = AgentStatus.RUNNING
                        reasons.append("failure_but_under_threshold")
                else:
                    next_status = AgentStatus.RUNNING
                    reasons.append("success_running")
        else:
            reasons.append("no_step_change")

        reflection_intent = False
        cooldown_blocked = False
        if (
            inp.step_completed and inp.had_failure and inp.reflect_on_error and inp.use_planner
        ):
            if (
                inp.reflect_cooldown_seconds > 0
                and inp.seconds_since_last_reflect is not None
                and inp.seconds_since_last_reflect < inp.reflect_cooldown_seconds
                and inp.consecutive_failures <= 1
                and not (AgentMode.UNCERTAIN & active or AgentMode.STALLING & active)
            ):
                reasons.append("cooldown_guard")
                cooldown_blocked = True
            elif inp.load_status == LoadStatus.SHEDDING:
                reasons.append("shed_suppresses_reflection")
            elif inp.reflect_cadence > 0 and ((inp.n_steps + 1) % inp.reflect_cadence != 0):
                reasons.append("cadence_gate")
            elif inp.reflection_requested_this_epoch:
                reasons.append("epoch_guard")
            else:
                reflection_intent = True
                reasons.append("reflect_on_error")
        else:
            if (AgentMode.STALLING & active or AgentMode.UNCERTAIN & active) and inp.use_planner:
                if inp.load_status == LoadStatus.SHEDDING:
                    reasons.append("shed_suppresses_reflection")
                elif inp.reflect_cadence > 0 and ((inp.n_steps + 1) % inp.reflect_cadence != 0):
                    reasons.append("cadence_gate")
                elif inp.reflection_requested_this_epoch:
                    reasons.append("epoch_guard")
                else:
                    reflection_intent = True
                    reasons.append("reflect_on_stall")

            if not reflection_intent and inp.use_planner and inp.step_completed:
                guarded = False
                if inp.load_status == LoadStatus.SHEDDING:
                    reasons.append("shed_suppresses_reflection"); guarded = True
                elif inp.reflect_cadence > 0 and ((inp.n_steps + 1) % inp.reflect_cadence != 0):
                    reasons.append("cadence_gate"); guarded = True
                elif inp.reflection_requested_this_epoch:
                    reasons.append("epoch_guard"); guarded = True

                if not guarded:
                    if inp.oscillation_score >= self.OSCILLATION_REFLECT_THRESHOLD:
                        reflection_intent = True; reasons.append("reflect_on_oscillation")
                    elif inp.no_progress_score >= self.NO_PROGRESS_REFLECT_THRESHOLD:
                        reflection_intent = True; reasons.append("reflect_on_no_progress")
                    elif inp.consecutive_action_failures >= self.ACTION_FAILURE_STREAK_REFLECT:
                        reflection_intent = True; reasons.append("reflect_on_action_failures")
                    elif inp.last_step_duration_seconds >= self.SLOW_STEP_REFLECT_SECONDS:
                        reflection_intent = True; reasons.append("reflect_on_slow_step")

        return TransitionDecision(
            next_status=next_status,
            set_modes=set_mask,
            clear_modes=clear_mask,
            reason=",".join(reasons) if reasons else "noop",
            reflection_intent=reflection_intent,
            cooldown_blocked=cooldown_blocked,
        )


class AgentOrchestrator:
    """
    Unified orchestrator that will eventually contain the complete agent loop.

    This class will absorb the logic from:
    - agent/loop.py (perception and main loop)
    - agent/decision_maker.py (decision-making and LLM interaction)
    - agent/actuator.py (action execution and I/O handling)

    For now, it's a placeholder that establishes the new architecture
    without breaking existing functionality.
    """

    def __init__(
        self,
        settings: AgentSettings,
        state_manager: StateManager,
        message_manager: MessageManager,
        browser_session: BrowserSession,
        controller: Controller,
    ):
        """Initialize the orchestrator with all required components."""
        # Core references
        self.settings = settings
        self.state_manager = state_manager  # DEPRECATED: will be removed in unified architecture
        self.message_manager = message_manager
        self.browser_session = browser_session
        self.controller = controller

        # Unified state management (replaces StateManager)
        self._state = state_manager.state  # Direct reference to agent state
        self._unified_lock = asyncio.Lock()  # Unified state lock

        # Concurrency controls
        try:
            # Use package-relative import so tests importing 'agent.*' work without 'browser_use' prefix
            from .concurrency import set_global_io_semaphore, set_single_actuation_semaphore
            try:
                max_io = int(getattr(self.settings, 'max_concurrent_io', 3))
            except Exception:
                max_io = 3
            try:
                set_global_io_semaphore(max_io)
            except Exception:
                logger.debug('orchestrator: set_global_io_semaphore failed', exc_info=True)
            try:
                set_single_actuation_semaphore()
            except Exception:
                logger.debug('orchestrator: set_single_actuation_semaphore failed', exc_info=True)
        except Exception:
            # Non-fatal during import smoke; runtime will retry on first use
            logger.debug('orchestrator: concurrency import failed (best-effort)', exc_info=True)

        # State placeholders
        self.current_state: Optional[OrchestratorState] = None
        self.current_protocol: str = 'normal_protocol'
        self.sem_cap = _SemanticPageCapture(browser_session=self.browser_session, settings=self.settings)
        self._recent_page_hashes: deque[str] = deque(maxlen=6)

        # Flags
        self._first_step_llm_logged = False
        self._pause_triggered = False

        # Planner control state
        self._planner_requested: bool = True  # plan once at startup
        self._last_plan_step: int = -1
        self._last_plan_time: float = 0.0
        self._last_plan_url: Optional[str] = None
        self._last_plan_title: Optional[str] = None
        self._last_action_results: list | None = None

        logger.debug("AgentOrchestrator initialized with perception capabilities")

    # === UNIFIED STATE MANAGEMENT METHODS ===
    # These methods replace StateManager functionality directly in the orchestrator

    @property
    def state(self):
        """Direct access to agent state (replaces state_manager.state)."""
        return self._state

    async def get_status(self) -> 'AgentStatus':
        """Get current agent status (replaces state_manager.get_status)."""
        from .state import AgentStatus
        async with self._unified_lock:
            return self._state.status

    async def set_status(self, new_status: 'AgentStatus', force: bool = False) -> None:
        """Set agent status (replaces state_manager.set_status)."""
        from .state import AgentStatus
        STATE_PRIORITY = {
            AgentStatus.PENDING: 0,
            AgentStatus.RUNNING: 1,
            AgentStatus.PAUSED: 2,
            AgentStatus.STOPPED: 3,
            AgentStatus.COMPLETED: 4,
            AgentStatus.FAILED: 5,
            AgentStatus.MAX_STEPS_REACHED: 6,
        }
        async with self._unified_lock:
            cur_pri = STATE_PRIORITY.get(self._state.status, -1)
            new_pri = STATE_PRIORITY.get(new_status, -1)
            if force or (new_pri >= cur_pri and self._state.status != new_status):
                self._state.status = new_status

    async def record_error(self, error_msg: str, is_critical: bool = False) -> None:
        """Record error and update state (replaces state_manager.record_error)."""
        from .state import AgentStatus
        async with self._unified_lock:
            self._state.last_error = error_msg
            if is_critical:
                self._state.consecutive_failures += 1
                max_failures = getattr(self.settings, 'max_failures', 10)
                if self._state.consecutive_failures >= max_failures:
                    self._state.status = AgentStatus.FAILED

    async def set_current_goal(self, new_goal: str) -> None:
        """Set current goal (replaces state_manager.set_current_goal)."""
        async with self._unified_lock:
            self._state.current_goal = new_goal

    async def get_current_task_id(self) -> str:
        """Get current task ID (replaces state_manager.get_current_task_id)."""
        async with self._unified_lock:
            cur = self._state.task_stack.current() if hasattr(self._state.task_stack, 'current') else None
            return getattr(cur, 'title', 'root') if cur else 'root'

    async def get_task_stack_summary(self) -> str:
        """Get task stack summary (replaces state_manager.get_task_stack_summary)."""
        async with self._unified_lock:
            if not self._state.task_stack:
                return "Task Context: root (main task)"
            try:
                return self._state.task_stack.compact_summary()
            except Exception:
                return "Task Context: root (main task)"

    async def get_human_guidance(self) -> Optional[str]:
        """Get human guidance (replaces state_manager.get_human_guidance)."""
        try:
            guidance = await asyncio.wait_for(self._state.human_guidance_queue.get(), timeout=0.5)
            self._state.human_guidance_queue.task_done()
            return guidance
        except asyncio.TimeoutError:
            return None

    async def add_human_guidance(self, text: str) -> None:
        """Add human guidance (replaces state_manager.add_human_guidance)."""
        try:
            await self._state.human_guidance_queue.put(text)
        except Exception:
            pass

    async def ingest_signal(self, signal_type: str, payload: Optional[dict] = None) -> None:
        """Ingest system signals (replaces state_manager.ingest_signal)."""
        from .state import LoadStatus
        payload = payload or {}
        async with self._unified_lock:
            if signal_type == 'heartbeat_miss':
                self._state.missed_heartbeats += 1
            elif signal_type == 'heartbeat_ok':
                self._state.missed_heartbeats = 0
            elif signal_type == 'io_timeout':
                self._state.io_timeouts_recent.append(1)
                while len(self._state.io_timeouts_recent) > 10:  # window size
                    try:
                        self._state.io_timeouts_recent.popleft()
                    except Exception:
                        break
            elif signal_type == 'io_ok':
                try:
                    if self._state.io_timeouts_recent:
                        self._state.io_timeouts_recent.popleft()
                except Exception:
                    pass
            elif signal_type == 'load_status':
                new_status = payload.get('status')
                if isinstance(new_status, LoadStatus):
                    self._state.load_status = new_status

    async def mark_reflection_exit(self) -> None:
        """Mark reflection exit (replaces state_manager.mark_reflection_exit)."""
        async with self._unified_lock:
            self._state.last_reflect_exit_ts = time.monotonic()

    async def update_task(self, new_task: str) -> None:
        """Update task (replaces state_manager.update_task)."""
        async with self._unified_lock:
            self._state.task = new_task
            self._state.current_goal = new_task

    # === END UNIFIED STATE MANAGEMENT METHODS ===

    # === COMPATIBILITY LAYER FOR MESSAGE MANAGER ===
    # MessageManager expects a state_manager with specific interface

    @property
    def _lock(self):
        """Compatibility: MessageManager expects _lock attribute."""
        return self._unified_lock

    # === END COMPATIBILITY LAYER ===

    async def decide_and_apply_after_step(
        self,
        result,
        max_steps: int,
        step_completed: bool = True,
        oscillation_score: float | None = None,
        no_progress_score: float | None = None,
    ):
        """Apply step results and make status decisions (replaces state_manager.decide_and_apply_after_step)."""
        from .state import AgentStatus
        from ..browser.views import BrowserStateHistory
        from .views import AgentHistory

        # Create StepOutcome locally (no longer importing from events.py)
        class StepOutcome:
            def __init__(self, status, modes, reflection_intent, task_completed):
                self.status = status
                self.modes = modes
                self.reflection_intent = reflection_intent
                self.task_completed = task_completed

        async with self._unified_lock:
            task_completed = False
            had_failure = False
            error_msg = None
            any_success = False

            for r in result.action_results:
                if getattr(r, 'is_done', False) and getattr(r, 'success', None) is True:
                    task_completed = True
                if getattr(r, 'success', None) is False:
                    had_failure = True
                    if getattr(r, 'error', None):
                        error_msg = r.error
                if getattr(r, 'success', None) is True:
                    any_success = True

            history_state = None
            if result.browser_state is not None:
                try:
                    interacted_elements = AgentHistory.get_interacted_element(result.llm_output, result.browser_state.selector_map) if result.llm_output else []
                except Exception:
                    interacted_elements = []
                try:
                    history_state = BrowserStateHistory(
                        url=result.browser_state.url,
                        title=result.browser_state.title,
                        tabs=result.browser_state.tabs,
                        screenshot_path=result.browser_state.screenshot,
                        interacted_element=interacted_elements,
                    )
                except Exception:
                    history_state = None

            history_item = AgentHistory(
                model_output=result.llm_output,
                result=result.action_results,
                state=history_state,
                metadata=result.step_metadata,
            )

            try:
                if hasattr(self._state.history, 'append'):
                    self._state.history.append(history_item)
            except Exception:
                pass

            if had_failure:
                self._state.last_error = error_msg or self._state.last_error
                self._state.consecutive_failures += 1
            else:
                self._state.consecutive_failures = 0

            if had_failure and not any_success:
                self._state.action_failure_streak += 1
            else:
                self._state.action_failure_streak = 0

            # Apply status updates
            if task_completed:
                self._state.status = AgentStatus.COMPLETED
            elif had_failure and not any_success:
                max_failures = getattr(self.settings, 'max_failures', 10)
                if self._state.consecutive_failures >= max_failures:
                    self._state.status = AgentStatus.FAILED

            self._state.n_steps += 1

            return StepOutcome(
                status=self._state.status,
                modes=int(getattr(self._state, 'modes', 0)),
                reflection_intent=False,  # Orchestrator now handles reflection decisions
                task_completed=task_completed,
            )

    async def _perceive(self) -> OrchestratorState:
        """
        Perception logic moved from agent/loop.py.

        Populates the OrchestratorState object with the latest browser and page state,
        including semantic snapshots and oscillation tracking.
        """
        from .state import OrchestratorState

        # Get current step number for context
        step_number = self._state.n_steps
        step_start_time = time.monotonic()

        # Core browser state perception (from loop.py)
        browser_state = await self.browser_session.get_state_summary(cache_clickable_elements_hashes=True)

        # Optional: semantic snapshot (DOM+AX) for multi-sensory perception and oscillation tracking
        semantic_snapshot = None
        page_hash = None
        try:
            semantic_snapshot = await self.sem_cap.capture_semantic_page()
            page_hash = (semantic_snapshot or {}).get('page_hash')
            if page_hash:
                self._recent_page_hashes.append(str(page_hash))
        except Exception as e:
            logger.debug(f"Semantic snapshot capture failed: {e}")
            semantic_snapshot = None

        # Calculate oscillation score (from loop.py)
        def _osc_score(seq: list[str]) -> float:
            try:
                if len(seq) >= 4:
                    a, b, c, d = seq[-4:]
                    return 1.0 if a == c and b == d and a != b else 0.0
            except Exception:
                pass
            return 0.0

        oscillation_score = _osc_score(list(self._recent_page_hashes))

        # Get current goal and task context
        current_goal = getattr(self._state, 'current_goal', None) or self._state.task
        task_stack_summary = await self.get_task_stack_summary()

        # Check for human guidance
        human_guidance = await self.get_human_guidance()

        # Build health metrics
        health_metrics = {
            'consecutive_failures': self._state.consecutive_failures,
            'last_error': self._state.last_error,
            'action_failure_streak': getattr(self._state, 'action_failure_streak', 0),
            'missed_heartbeats': getattr(self._state, 'missed_heartbeats', 0),
            'load_status': getattr(self._state, 'load_status', 'NORMAL'),
        }

        # Create and populate OrchestratorState
        orchestrator_state = OrchestratorState(
            browser_state=browser_state,
            health_metrics=health_metrics,
            oscillation_score=oscillation_score,
            no_progress_score=0.0,  # TODO: Implement if needed
            step_number=step_number,
            step_start_time=step_start_time,
            current_protocol=self.current_protocol,  # Phase 3: Track current protocol
            semantic_snapshot=semantic_snapshot,
            page_hash=page_hash,
            current_goal=current_goal,
            human_guidance=human_guidance,
            task_stack_summary=task_stack_summary,
        )

        # Store current state for access by other methods
        self.current_state = orchestrator_state

        return orchestrator_state

    async def decide(self, orchestrator_state: OrchestratorState) -> 'AgentOutput':
        """
        Decision-making logic moved from agent/decision_maker.py.

        Takes OrchestratorState as input and directly calls the MessageManager
        to return the final AgentOutput. Includes complex multi-step LLM
        invocation and retry logic.
        """
        # Setup action/output schemas (cached between calls)
        cache_key = (
            getattr(self.settings, 'flash_mode', False),
            getattr(self.settings, 'use_thinking', False)
        )

        if not hasattr(self, '_action_model_cache') or self._action_model_cache.get('key') != cache_key:
            # Prepare action and output models based on configuration
            from browser_use.agent.views import AgentOutput

            # Setup action model
            ActionModel = self.settings.controller.registry.create_action_model()
            done_action_model = self.settings.controller.registry.create_action_model(include_actions=['done'])

            if getattr(self.settings, 'flash_mode', False):
                self.AgentOutput = AgentOutput.type_with_custom_actions_flash_mode(ActionModel)
                self.DoneAgentOutput = AgentOutput.type_with_custom_actions_flash_mode(done_action_model)
            elif getattr(self.settings, 'use_thinking', True):
                self.AgentOutput = AgentOutput.type_with_custom_actions(ActionModel)
                self.DoneAgentOutput = AgentOutput.type_with_custom_actions(done_action_model)
            else:
                self.AgentOutput = AgentOutput.type_with_custom_actions_no_thinking(ActionModel)
                self.DoneAgentOutput = AgentOutput.type_with_custom_actions_no_thinking(done_action_model)

            # Export ActionModel to instance for later use
            self.ActionModel = ActionModel

            # Cache the models
            self._action_model_cache = {
                'key': cache_key,
                'ActionModel': self.ActionModel,
                'AgentOutput': self.AgentOutput,
                'DoneAgentOutput': self.DoneAgentOutput
            }
        else:
            # Use cached models
            cache = self._action_model_cache
            self.ActionModel = cache['ActionModel']
            self.AgentOutput = cache['AgentOutput']
            self.DoneAgentOutput = cache['DoneAgentOutput']

        parsed = None

        # Prepare messages from state + inputs (from decision_maker.py)
        state = self._state

        # Prefer compact history (bounded) and memory summaries
        try:
            summaries = getattr(state, 'summaries_compact', None)
            if callable(summaries):
                compact = summaries()
                if compact:
                    self.message_manager.add_local_note(f"Memory summaries: {compact}")
        except Exception:
            pass

        # Ensure MessageManager receives an object with `.history`
        _hist = getattr(state, 'history', [])
        try:
            history_obj = _hist if hasattr(_hist, 'history') else type('H', (), {'history': list(_hist)})()
        except Exception:
            history_obj = type('H', (), {'history': []})()
        self.message_manager.update_history_representation(history_obj)

        # Dynamic Protocol Switching: add strategic context when reflecting
        if self.current_protocol == 'reflection_protocol':
            logger.info("Using reflection protocol for strategic decision-making")
            strategic_context = (
                "IMPORTANT: The system has detected potential stalling or oscillation. "
                "Instead of continuing with standard actions, please:\n"
                "1. Analyze what has been attempted recently\n"
                "2. Identify why progress may be blocked\n"
                "3. Try a fundamentally different approach\n"
                "4. Consider if the current goal needs to be refined\n"
                "Focus on strategic thinking rather than repeating recent actions."
            )
            self.message_manager.add_local_note(strategic_context)

    # Intentionally avoid adding CAPTCHA-specific nudges; rely on planner-driven tasks only

    # If task layer enabled, surface Task Catalog to the LLM; CDAD plan remains gated logically
        try:
            if bool(getattr(self.settings, 'task_layer_enabled', False)):
                # Build a dedicated Task Catalog section
                try:
                    from .tasks import TASK_REGISTRY, task_signatures
                    sigs = task_signatures()
                    lines = ["## Task Catalog (always available)"]
                    for name, cls in TASK_REGISTRY.items():
                        sig = sigs.get(name, "(unavailable)")
                        lines.append(f"- {name} -> {getattr(cls, '__name__', 'Task')} {sig}")
                    catalog_text = "\n".join(lines)
                    # Add as a local system note so it's rendered in a dedicated block
                    self.message_manager.add_local_note(catalog_text)
                    # Transparent log for visibility
                    from .state import agent_log
                    agent_log(logging.INFO, self._state.agent_id, self._state.n_steps,
                              f"Planner/Tasks: Showing Task Catalog to LLM ({len(TASK_REGISTRY)} tasks)")
                except Exception:
                    logger.debug('Failed to append Task Catalog (ignored)', exc_info=True)

                # Optionally run the CDAD strategic planner with intelligent cadence
                try:
                    if bool(getattr(self.settings, 'use_task_planner', False)):
                        should_plan, reason = self._should_run_planner(orchestrator_state)
                        from .state import agent_log
                        if should_plan:
                            # Attach planning context: available tasks and recent actions summary
                            try:
                                from .tasks import TASK_REGISTRY as _TR
                                available_task_names = list(_TR.keys())
                            except Exception:
                                available_task_names = []
                            # Attach recent actions (names + success)
                            recent_actions_summary = []
                            try:
                                for r in (self._last_action_results or [])[:10]:
                                    try:
                                        nm = type(r).__name__
                                    except Exception:
                                        nm = 'ActionResult'
                                    recent_actions_summary.append({
                                        'type': nm,
                                        'success': getattr(r, 'success', None),
                                        'error': getattr(r, 'error', None),
                                    })
                            except Exception:
                                recent_actions_summary = []

                            try:
                                setattr(orchestrator_state, 'planner_context', {
                                    'available_tasks': available_task_names,
                                    'recent_actions': recent_actions_summary,
                                    'current_url': getattr(orchestrator_state.browser_state, 'url', None),
                                    'current_title': getattr(orchestrator_state.browser_state, 'title', None),
                                    'step': orchestrator_state.step_number,
                                })
                            except Exception:
                                pass

                            goal = getattr(self._state, 'current_goal', None) or self._state.task
                            planned = await self._create_strategic_plan(goal, mission_state=orchestrator_state, available_tasks=available_task_names)
                            # Update planner state
                            self._planner_requested = False
                            self._last_plan_step = orchestrator_state.step_number
                            self._last_plan_time = time.monotonic()
                            self._last_plan_url = getattr(orchestrator_state.browser_state, 'url', None)
                            self._last_plan_title = getattr(orchestrator_state.browser_state, 'title', None)

                            # Stash for potential use/inspection
                            self._last_planned_tasks = planned
                            if planned:
                                names = [type(t).__name__ for t in planned]
                                agent_log(logging.INFO, self._state.agent_id, self._state.n_steps,
                                          f"Planner: Planned tasks -> {', '.join(names)}")
                        else:
                            agent_log(logging.INFO, self._state.agent_id, self._state.n_steps,
                                      f"Planner: Skipped ({reason})")
                except Exception:
                    logger.debug('Strategic planning pre-pass failed (ignored)', exc_info=True)
        except Exception:
            logger.debug('Task layer prelude failed (ignored)', exc_info=True)

        # Prepare prompt and call LLM with retry
        try:
            messages = await self.message_manager.prepare_messages(
                state_manager=self,  # Pass orchestrator as unified state manager
                browser_state=orchestrator_state.browser_state,
            )
            self._last_messages = messages
            parsed = await self._invoke_llm_with_retry(
                messages,
                max_retries=getattr(self.settings, 'llm_max_retries', 2)
            )
        except Exception as e:
            logger.error(f"Decision pipeline failed before/at LLM: {e}", exc_info=True)
            # Ensure messages are set for persistence
            if not hasattr(self, '_last_messages') or self._last_messages is None:
                self._last_messages = []
            # Safe fallback AgentOutput with a single 'done' action
            try:
                # Inline _setup_action_models for fallback path
                cache_key = (
                    getattr(self.settings, 'flash_mode', False),
                    getattr(self.settings, 'use_thinking', False)
                )

                if not hasattr(self, '_action_model_cache') or self._action_model_cache.get('key') != cache_key:
                    # Prepare action and output models based on configuration
                    from browser_use.agent.views import AgentOutput

                    # Setup action model
                    ActionModel = self.settings.controller.registry.create_action_model()
                    done_action_model = self.settings.controller.registry.create_action_model(include_actions=['done'])

                    if getattr(self.settings, 'flash_mode', False):
                        self.AgentOutput = AgentOutput.type_with_custom_actions_flash_mode(ActionModel)
                        self.DoneAgentOutput = AgentOutput.type_with_custom_actions_flash_mode(done_action_model)
                    elif getattr(self.settings, 'use_thinking', True):
                        self.AgentOutput = AgentOutput.type_with_custom_actions(ActionModel)
                        self.DoneAgentOutput = AgentOutput.type_with_custom_actions(done_action_model)
                    else:
                        self.AgentOutput = AgentOutput.type_with_custom_actions_no_thinking(ActionModel)
                        self.DoneAgentOutput = AgentOutput.type_with_custom_actions_no_thinking(done_action_model)

                    # Export ActionModel to instance for later use
                    self.ActionModel = ActionModel

                    # Cache the models
                    self._action_model_cache = {
                        'key': cache_key,
                        'ActionModel': self.ActionModel,
                        'AgentOutput': self.AgentOutput,
                        'DoneAgentOutput': self.DoneAgentOutput
                    }
                else:
                    # Use cached models
                    cache = self._action_model_cache
                    self.ActionModel = cache['ActionModel']
                    self.AgentOutput = cache['AgentOutput']
                    self.DoneAgentOutput = cache['DoneAgentOutput']

                action_instance = self.ActionModel()
                setattr(action_instance, 'done', {"success": False, "text": "Decision pipeline failed"})
                parsed = self.AgentOutput(action=[action_instance])
            except Exception:
                class _Shim:
                    def __init__(self):
                        self.action = []
                parsed = _Shim()  # type: ignore[assignment]

        # Intentionally suppress verbose first-step payload logging per request
        self._first_step_llm_logged = True

        # Enforce caps for actions per step
        try:
            max_actions_cfg = int(getattr(self.settings, 'max_actions_per_step', 3))
            hard_cap = int(getattr(self.settings, 'cap_actions_per_step', max_actions_cfg))
            max_actions = min(max_actions_cfg, hard_cap)
        except Exception:
            max_actions = 3
        try:
            if parsed is not None and hasattr(parsed, 'action') and isinstance(parsed.action, list):
                if len(parsed.action) > max_actions:
                    from .state import agent_log
                    agent_log(logging.WARNING, self._state.agent_id, self._state.n_steps,
                              f"Clipping actions from {len(parsed.action)} to cap {max_actions}")
                    parsed.action = parsed.action[:max_actions]
        except Exception:
            pass

    # Do not enforce or inject CAPTCHA actions; planner/tasks must be chosen organically

        return parsed

    async def _persist_conversation(self, orchestrator_state: OrchestratorState, agent_output: 'AgentOutput') -> None:
        """Persist in unified rolling mode: one session folder, rolling session.md, per-step PNG/JSON."""
        try:
            conv_save_target: Optional[str] = getattr(self.settings, 'save_conversation_path', None)
            if not conv_save_target:
                return

            # Initialize session folder once per run
            if not hasattr(self, '_conv_run_dir') or getattr(self, '_conv_run_dir', None) is None:
                base_path = Path(conv_save_target)
                base_dir = base_path if not base_path.suffix else (base_path.parent if str(base_path.parent) not in ("", ".") else Path("."))
                self._conv_run_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                self._conv_run_dir = base_dir / f"session_{self._conv_run_ts}"
                try:
                    await anyio.Path(str(self._conv_run_dir)).mkdir(parents=True, exist_ok=True)
                except Exception:
                    logger.debug('orchestrator: failed to create conversation dir', exc_info=True)
                    return

            step_idx = (self.state_manager.state.n_steps or 0) + 1
            session_file = self._conv_run_dir / f"session_{self._conv_run_ts}.md"

            # Build header + content
            ts_iso = datetime.utcnow().isoformat() + "Z"
            url = getattr(orchestrator_state.browser_state, 'url', '') or ''
            title = getattr(orchestrator_state.browser_state, 'title', '') or ''
            header = (
                f"\n\n---\n## Step {step_idx} — {ts_iso}\n"
                f"URL: {url}\n\nTitle: {title}\n\n"
            )
            messages = getattr(orchestrator_state, 'messages_prepared', None) or getattr(self, '_last_messages', []) or []

            # Local formatter matching loop's lightweight formatter, resilient to model variations
            async def _fmt(messages_, response_) -> str:
                import json as _json
                lines: list[str] = []

                # Messages
                try:
                    for m in messages_ or []:
                        try:
                            role = getattr(m, 'role', None) or type(m).__name__
                            text = getattr(m, 'text', None)
                            if text is None:
                                # Try common attribute fallbacks
                                content = getattr(m, 'content', None)
                                text = content if isinstance(content, str) else str(m)
                            lines.append(f" {role} ")
                            lines.append(str(text))
                            lines.append("")
                        except Exception:
                            # Never fail the whole formatter for a single message
                            lines.append(" message <<unavailable>>")
                            lines.append("")
                except Exception:
                    lines.append(" messages <<unavailable>>")
                    lines.append("")

                # Model response
                lines.append(" RESPONSE")
                try:
                    # Prefer pydantic v2 json dump if available
                    dump_json = getattr(response_, 'model_dump_json', None)
                    if callable(dump_json):
                        # Write raw JSON string to avoid round-trip parsing failures
                        payload_str = dump_json(exclude_unset=True)
                        lines.append(payload_str if isinstance(payload_str, str) else _json.dumps(payload_str, ensure_ascii=False, indent=2))
                    else:
                        dump = getattr(response_, 'model_dump', None)
                        if callable(dump):
                            obj = dump(exclude_unset=True, exclude_none=True)
                            lines.append(_json.dumps(obj, ensure_ascii=False, indent=2))
                        else:
                            # Last resort: generic JSON serialization with default=str
                            try:
                                lines.append(_json.dumps(response_, ensure_ascii=False, indent=2, default=str))
                            except Exception:
                                lines.append(str(response_))
                except Exception:
                    lines.append('<<unavailable>>')

                lines.append('\n')
                return '\n'.join(lines)

            try:
                content = await _fmt(messages, agent_output)
                async with await anyio.open_file(str(session_file), mode='a', encoding='utf-8') as f:
                    await f.write(header)
                    await f.write(content)
            except Exception:
                logger.debug('orchestrator: append session.md failed (ignored)', exc_info=True)

            # Screenshot
            try:
                shot_b64 = getattr(orchestrator_state.browser_state, 'screenshot', None)
                if shot_b64:
                    img_path = self._conv_run_dir / f"step-{step_idx:03d}_{self._conv_run_ts}.png"
                    img_bytes = base64.b64decode(shot_b64)
                    await anyio.Path(str(img_path)).write_bytes(img_bytes)
                    img_md = f"\n![Step {step_idx} Screenshot]({img_path.name})\n"
                    async with await anyio.open_file(str(session_file), mode='a', encoding='utf-8') as f:
                        await f.write(img_md)
            except Exception:
                logger.debug('orchestrator: screenshot save failed (ignored)', exc_info=True)

            # Compact JSON summary
            try:
                try:
                    affordances = await self.browser_session.get_affordances_summary()
                    try:
                        max_items = int(getattr(self.browser_session.browser_profile, 'overlay_max_items', 60) or 60)
                    except Exception:
                        max_items = 60
                    if isinstance(affordances, list):
                        affordances = sorted(
                            affordances,
                            key=lambda a: (a.get('index', 0) if isinstance(a, dict) else 0)
                        )[:max_items]
                except Exception:
                    affordances = None

                def _safe_trunc(s: str | None, n: int = 800) -> str | None:
                    try:
                        if s is None:
                            return None
                        s = str(s)
                        return s if len(s) <= n else (s[: n - 1] + "…")
                    except Exception:
                        return None

                model_output_summary = None
                try:
                    mo = agent_output
                    if mo is not None:
                        tlog_struct_dump = None
                        try:
                            ts_ = getattr(mo, 'task_log_structured', None)
                            if ts_ is not None:
                                tlog_struct_dump = ts_.model_dump(exclude_none=True)
                                for key in ("objectives", "checklist", "risks", "blockers"):
                                    if isinstance(tlog_struct_dump.get(key), list):
                                        tlog_struct_dump[key] = tlog_struct_dump[key][:20]
                        except Exception:
                            tlog_struct_dump = None

                        model_output_summary = {
                            'prior_action_assessment': _safe_trunc(getattr(mo, 'prior_action_assessment', None), 800),
                            'task_log': _safe_trunc(getattr(mo, 'task_log', None), 1200),
                            'task_log_structured': tlog_struct_dump,
                            'next_goal': _safe_trunc(getattr(mo, 'next_goal', None), 400),
                        }
                except Exception:
                    model_output_summary = None

                summary = {
                    'url': getattr(orchestrator_state.browser_state, 'url', ''),
                    'title': getattr(orchestrator_state.browser_state, 'title', ''),
                    'viewport': {
                        'width': getattr(getattr(orchestrator_state.browser_state, 'page_info', None), 'viewport_width', None),
                        'height': getattr(getattr(orchestrator_state.browser_state, 'page_info', None), 'viewport_height', None),
                        'scroll_x': getattr(getattr(orchestrator_state.browser_state, 'page_info', None), 'scroll_x', None),
                        'scroll_y': getattr(getattr(orchestrator_state.browser_state, 'page_info', None), 'scroll_y', None),
                    },
                    'affordances': affordances,
                    'semantic': {
                        'page_hash': (orchestrator_state.semantic_snapshot or {}).get('page_hash'),
                        'flags': (orchestrator_state.semantic_snapshot or {}).get('flags'),
                        'elements_count': len((orchestrator_state.semantic_snapshot or {}).get('elements') or []),
                    } if getattr(orchestrator_state, 'semantic_snapshot', None) else None,
                    'model_output': model_output_summary,
                }

                import json as _json
                json_path = self._conv_run_dir / f"step-{step_idx:03d}_{self._conv_run_ts}.json"
                async with await anyio.open_file(str(json_path), mode='w', encoding='utf-8') as f:
                    await f.write(_json.dumps(summary, ensure_ascii=False, indent=2))
            except Exception:
                logger.debug('orchestrator: json summary save failed (ignored)', exc_info=True)
        except Exception:
            # Outer guard: never fail the run due to persistence issues
            logger.debug('orchestrator: persistence outer try failed (ignored)', exc_info=True)

    async def _invoke_llm_with_retry(self, messages: list, max_retries: int = 2) -> 'AgentOutput':
        """
        LLM invocation with retry logic (moved from decision_maker.py).

        Handles retries, action list trimming, logging, and fallback insertion
        for empty output.
        """
        import random
        from ..exceptions import LLMException
        from .state import agent_log

        # Ensure action/output schemas are prepared (cached between calls)
        cache_key = (
            getattr(self.settings, 'flash_mode', False),
            getattr(self.settings, 'use_thinking', False)
        )

        if not hasattr(self, '_action_model_cache') or self._action_model_cache.get('key') != cache_key:
            # Prepare action and output models based on configuration
            from browser_use.agent.views import AgentOutput

            # Setup action model
            ActionModel = self.settings.controller.registry.create_action_model()
            done_action_model = self.settings.controller.registry.create_action_model(include_actions=['done'])

            if getattr(self.settings, 'flash_mode', False):
                self.AgentOutput = AgentOutput.type_with_custom_actions_flash_mode(ActionModel)
                self.DoneAgentOutput = AgentOutput.type_with_custom_actions_flash_mode(done_action_model)
            elif getattr(self.settings, 'use_thinking', True):
                self.AgentOutput = AgentOutput.type_with_custom_actions(ActionModel)
                self.DoneAgentOutput = AgentOutput.type_with_custom_actions(done_action_model)
            else:
                self.AgentOutput = AgentOutput.type_with_custom_actions_no_thinking(ActionModel)
                self.DoneAgentOutput = AgentOutput.type_with_custom_actions_no_thinking(done_action_model)

            # Export ActionModel to instance for later use
            self.ActionModel = ActionModel

            # Cache the models
            self._action_model_cache = {
                'key': cache_key,
                'ActionModel': self.ActionModel,
                'AgentOutput': self.AgentOutput,
                'DoneAgentOutput': self.DoneAgentOutput
            }
        else:
            # Use cached models
            cache = self._action_model_cache
            self.ActionModel = cache['ActionModel']
            self.AgentOutput = cache['AgentOutput']
            self.DoneAgentOutput = cache['DoneAgentOutput']

        output_schema = self.DoneAgentOutput if (self.state_manager.state.n_steps + 1) >= self.settings.max_steps else self.AgentOutput

        timeout_s = getattr(self.settings, 'llm_timeout_seconds', 30.0)
        base_backoff = getattr(self.settings, 'llm_backoff_base_seconds', 1.0)
        jitter = getattr(self.settings, 'llm_backoff_jitter_seconds', 0.3)

        for attempt in range(max_retries + 1):
            agent_log(
                logging.INFO,
                self._state.agent_id,
                self._state.n_steps,
                f"LLM call attempt {attempt + 1}/{max_retries + 1}"
            )
            try:
                start = time.monotonic()
                llm = getattr(self.settings, 'llm')
                response = await asyncio.wait_for(llm.ainvoke(messages, output_format=output_schema), timeout=timeout_s)
                parsed = response.completion
                duration = time.monotonic() - start

                # Trim action list to max_actions_per_step if necessary
                action_list = parsed.get("action", None) if isinstance(parsed, dict) else getattr(parsed, "action", None)
                if action_list and isinstance(action_list, list):
                    if len(action_list) > self.settings.max_actions_per_step:
                        agent_log(
                            logging.WARNING,
                            self._state.agent_id,
                            self._state.n_steps,
                            f"Trimming actions from {len(action_list)} to {self.settings.max_actions_per_step}"
                        )
                        trimmed_actions = action_list[:self.settings.max_actions_per_step]
                        if isinstance(parsed, dict):
                            parsed["action"] = trimmed_actions
                        else:
                            parsed.action = trimmed_actions

                # Log LLM output using inlined method (from _log_llm_output)
                from .state import agent_log
                state = self._state

                # Log thinking if present
                thinking = parsed.get("thinking", None) if isinstance(parsed, dict) else getattr(parsed, "thinking", None)
                if thinking and thinking.strip() and thinking.strip() != "N/A":
                    agent_log(logging.INFO, state.agent_id, state.n_steps, f"💡 Thinking:\n{thinking}")

                # Log prior action assessment with appropriate emoji
                eval_goal = parsed.get("prior_action_assessment", None) if isinstance(parsed, dict) else getattr(parsed, "prior_action_assessment", None)
                if eval_goal and eval_goal.strip() and eval_goal.strip() != "N/A":
                    if 'success' in eval_goal.lower():
                        emoji = '👍'
                    elif 'failure' in eval_goal.lower():
                        emoji = '⚠️'
                    else:
                        emoji = '❔'
                    agent_log(logging.INFO, state.agent_id, state.n_steps, f"{emoji} Eval: {eval_goal}")

                # Log task_log if present
                task_log = parsed.get("task_log", None) if isinstance(parsed, dict) else getattr(parsed, "task_log", None)
                if task_log and task_log.strip() and task_log.strip() != "N/A":
                    agent_log(logging.INFO, state.agent_id, state.n_steps, f"🧠 Task Log: {task_log}")

                # Log structured task log briefly if provided
                try:
                    tlog_struct = parsed.get("task_log_structured", None) if isinstance(parsed, dict) else getattr(parsed, "task_log_structured", None)
                    if tlog_struct is not None:
                        pct = getattr(tlog_struct, 'progress_pct', None) if hasattr(tlog_struct, 'progress_pct') else tlog_struct.get('progress_pct', None) if isinstance(tlog_struct, dict) else None
                        nxt = getattr(tlog_struct, 'next_action', None) if hasattr(tlog_struct, 'next_action') else tlog_struct.get('next_action', None) if isinstance(tlog_struct, dict) else None
                        obj = getattr(tlog_struct, 'objectives', None) if hasattr(tlog_struct, 'objectives') else tlog_struct.get('objectives', None) if isinstance(tlog_struct, dict) else None
                        if isinstance(obj, list) and obj:
                            try:
                                objs_done = sum(1 for i in obj if (getattr(i, 'status', None) if hasattr(i, 'status') else i.get('status', None) if isinstance(i, dict) else None) == 'completed')
                                objs_total = len(obj)
                                agent_log(logging.INFO, state.agent_id, state.n_steps,
                                          f"🗂️ StructuredLog: progress={pct if pct is not None else '-'}% objectives={objs_done}/{objs_total} next={nxt or '-'}")
                            except Exception:
                                agent_log(logging.INFO, state.agent_id, state.n_steps,
                                          f"🗂️ StructuredLog: progress={pct if pct is not None else '-'}% next={nxt or '-'}")
                        else:
                            agent_log(logging.INFO, state.agent_id, state.n_steps,
                                      f"🗂️ StructuredLog: progress={pct if pct is not None else '-'}% next={nxt or '-'}")
                except Exception:
                    pass

                # Log next goal if present
                next_goal = parsed.get("next_goal", None) if isinstance(parsed, dict) else getattr(parsed, "next_goal", None)
                if next_goal and next_goal.strip() and next_goal.strip() != "N/A":
                    agent_log(logging.INFO, state.agent_id, state.n_steps, f"🎯 Next goal: {next_goal}")

                agent_log(logging.INFO, self._state.agent_id, self._state.n_steps,
                          f"LLM call duration: {duration:.2f}s")

                # Validate action output for empty/malformed responses
                has_action_attr = "action" in parsed if isinstance(parsed, dict) else hasattr(parsed, "action")
                action_value = parsed.get("action", None) if isinstance(parsed, dict) else getattr(parsed, "action", None)
                is_action_list = isinstance(action_value, list) if has_action_attr else False
                actions = action_value if is_action_list else []
                empty_actions = all(getattr(a, "model_dump", lambda: {})() == {} for a in actions) if is_action_list else True

                if (
                    not has_action_attr
                    or not is_action_list
                    or empty_actions
                ):
                    # Retry logic for empty/malformed output
                    if attempt < max_retries:
                        agent_log(
                            logging.WARNING,
                            self._state.agent_id,
                            self._state.n_steps,
                            "Model returned empty action. Retrying..."
                        )
                        await asyncio.sleep(1.0 * (2 ** attempt))
                        continue
                    # Insert noop action as fallback
                    agent_log(
                        logging.WARNING,
                        self._state.agent_id,
                        self._state.n_steps,
                        "Model still returned empty after retry. Inserting safe noop action."
                    )
                    action_instance = self.ActionModel()
                    setattr(
                        action_instance,
                        "done",
                        {
                            "success": False,
                            "text": "No next action returned by LLM!",
                        },
                    )
                    parsed.action = [action_instance]

                return parsed

            except (asyncio.TimeoutError, LLMException) as e:
                agent_log(
                    logging.WARNING,
                    self._state.agent_id,
                    self._state.n_steps,
                    f"LLM call attempt {attempt+1} failed: {type(e).__name__}: {e}",
                    exc_info=True
                )
                if attempt >= max_retries:
                    raise LLMException("LLM call failed after all retries.") from e
                await asyncio.sleep(base_backoff * (2 ** attempt) + random.uniform(0, jitter))

        raise LLMException("LLM call failed.")

    async def execute(self, agent_output: 'AgentOutput', orchestrator_state: OrchestratorState) -> 'ActuationResult':
        """
        Actuation logic moved from agent/actuator.py.

        Takes AgentOutput and OrchestratorState, calls controller.multi_act
        with I/O timeout and retry logic implemented as asyncio.wait_for.
        """
        import random
        from .views import StepMetadata, ActionResult
        from .concurrency import io_semaphore
        from .state import agent_log

    # Use module-level ActuationResult container (kept lightweight)

        # Per-step state and timing
        state = self._state
        step_start_time = orchestrator_state.step_start_time or time.monotonic()
        # Get current task context for monitoring (optional)
        try:
            current_task_id = await self.get_current_task_id()
        except Exception:
            current_task_id = None
            logger.debug('Failed to fetch current_task_id', exc_info=True)

        # Do not short-circuit to run specific tasks directly; execution flows through controller actions


        # If LLM chose a task-executing action (e.g., solve_captcha), log it clearly
        try:
            from .state import agent_log
            for a in (agent_output.action or [])[:5]:
                try:
                    dump = a.model_dump(exclude_unset=True)
                    if 'solve_captcha' in dump:
                        agent_log(logging.INFO, state.agent_id, state.n_steps, "Task chosen by LLM: SolveCaptchaTask")
                        break
                except Exception:
                    continue
        except Exception:
            logger.debug('Task choice logging failed (ignored)', exc_info=True)

        def _extract_required_wait_seconds(agent_output_obj) -> Optional[int]:
            """Detect required wait duration from various model fields and task text.

            Recognizes variants like:
            - wait 600 seconds / wait for 600 seconds / wait 10 minutes / wait 10m / wait 600s
            - for 10 minutes / for 600 seconds
            - 10-minute wait / 10 min wait
            - shorthand: 10m, 600s
            Returns largest detected duration in seconds, or None.
            """
            import re
            texts: list[str] = []
            try:
                # Original task and current goal
                t = getattr(self._state, 'task', None)
                if t:
                    texts.append(str(t))
                cg = getattr(self._state, 'current_goal', None)
                if cg:
                    texts.append(str(cg))
            except Exception:
                pass
            try:
                if agent_output_obj is not None:
                    for field in ("next_goal", "task_log", "prior_action_assessment", "thinking"):
                        val = getattr(agent_output_obj, field, None)
                        if isinstance(val, str) and val:
                            texts.append(val)
            except Exception:
                pass

            if not texts:
                return None

            # Build regexes for seconds and minutes
            # Acceptable forms include optional leading 'wait', optional 'for', and units s/sec/seconds, m/min/minutes
            patterns = [
                # explicit seconds with or without 'wait'/'for'
                r"(?:wait\s+(?:for\s+)?)?(\d+)\s*(?:sec(?:ond)?s?|s)\b",
                r"(?:for\s+)?(\d+)\s*(?:sec(?:ond)?s?|s)\b",
                # explicit minutes with or without 'wait'/'for'
                r"(?:wait\s+(?:for\s+)?)?(\d+)\s*(?:min(?:ute)?s?|m)\b",
                r"(?:for\s+)?(\d+)\s*(?:min(?:ute)?s?|m)\b",
                # hyphenated minute wait (e.g., '10-minute wait')
                r"(\d+)[-\s]?min(?:ute)?s?\s+wait",
            ]

            best_seconds = 0
            for txt in texts:
                try:
                    s = txt.lower()
                    for pat in patterns:
                        for m in re.finditer(pat, s):
                            try:
                                val = int(m.group(1))
                            except Exception:
                                continue
                            # Infer unit from pattern
                            if 'min' in pat or 'm)' in pat or 'minute' in pat:
                                seconds = val * 60
                            else:
                                seconds = val
                            if seconds > best_seconds:
                                best_seconds = seconds
                except Exception:
                    continue

            return best_seconds if best_seconds > 0 else None

        def _history_wait_satisfied(required: int) -> bool:
            """Return True if history contains an explicit wait of at least 'required' seconds."""
            try:
                hist = getattr(self._state, 'history', []) or []
                import re
                pat = re.compile(r"Waited for\s+(\d+)\s+seconds")
                best = 0
                for h in hist:
                    try:
                        for r in getattr(h, 'result', []) or []:
                            msg = getattr(r, 'extracted_content', '') or ''
                            m = pat.search(str(msg))
                            if m:
                                secs = int(m.group(1))
                                if secs > best:
                                    best = secs
                    except Exception:
                        continue
                return best >= int(required)
            except Exception:
                return False

        def _maybe_inject_wait():
            """If goal mandates a wait and model only wants to 'done', inject a wait first."""
            try:
                actions = getattr(agent_output, 'action', []) or []
                if not actions:
                    return
                # Detect presence of 'done' and absence of 'wait'
                has_done = False
                has_wait = False
                for a in actions:
                    try:
                        dump = a.model_dump(exclude_unset=True)
                        if not isinstance(dump, dict) or not dump:
                            continue
                        k = next(iter(dump.keys()))
                        if k == 'done':
                            has_done = True
                        if k == 'wait':
                            has_wait = True
                    except Exception:
                        continue
                if has_done and not has_wait:
                    required = _extract_required_wait_seconds(agent_output)
                    if required and required > 0 and not _history_wait_satisfied(required):
                        # Respect controller cap (300s) per action; chunk if needed on subsequent steps
                        chunk = min(int(required), 300)
                        try:
                            # Build a dynamic wait action instance
                            ActionModel = self.controller.registry.create_action_model()
                            wait_action = ActionModel()
                            setattr(wait_action, 'wait', {'seconds': int(chunk)})
                            # Prepend wait before done
                            agent_output.action = [wait_action] + actions
                            from .state import agent_log
                            agent_log(logging.INFO, self._state.agent_id, self._state.n_steps,
                                      f"Guard: inserting wait({chunk}s) before done to honor goal requirement")
                        except Exception:
                            logger.debug('Failed to inject wait action guard', exc_info=True)
            except Exception:
                logger.debug('wait guard check failed (ignored)', exc_info=True)

        # Pre-execution guard: honor explicit wait requirement in goal before accepting 'done'
        _maybe_inject_wait()

        async def _run_with_io_limit():
            # Limit concurrent browser IO using the global semaphore
            async with io_semaphore():
                return await self.controller.multi_act(
                    actions=agent_output.action,
                    browser_session=self.browser_session,
                    page_extraction_llm=getattr(self.settings, 'page_extraction_llm', None),
                    context=getattr(self.settings, 'context', None),
                    sensitive_data=getattr(self.settings, 'sensitive_data', None),
                    available_file_paths=getattr(self.settings, 'available_file_paths', None),
                    file_system=getattr(self.settings, 'file_system', None),
                )

        # Build description for telemetry and debugging
        action_descriptions = [str(action) for action in agent_output.action[:3]]
        description = f"Browser actions: {', '.join(action_descriptions)}"
        if len(agent_output.action) > 3:
            description += f" and {len(agent_output.action) - 3} more"

        # Determine timeout and retry schedule
        timeout_seconds = self._select_timeout(agent_output, orchestrator_state)
        retry_schedule = self._compute_retry_schedule()

        last_exc: Optional[BaseException] = None
        attempts = 0

        while True:
            try:
                # Use asyncio.wait_for for I/O timeout (simplified from LongIOWatcher)
                action_results = await asyncio.wait_for(
                    _run_with_io_limit(),
                    timeout=timeout_seconds
                )

                logger.info(f"Controller call returned with {len(action_results)} action result(s)")
                # Cache recent action results for planner context
                try:
                    self._last_action_results = list(action_results) if action_results is not None else []
                except Exception:
                    self._last_action_results = []

                # Signal successful I/O
                try:
                    await self.ingest_signal('io_ok')
                except Exception:
                    logger.debug("Failed to emit io_ok signal", exc_info=True)
                break

            except Exception as e:
                attempts += 1
                last_exc = e

                # Signal I/O timeout if it's a timeout error
                if isinstance(e, asyncio.TimeoutError):
                    try:
                        await self.ingest_signal('io_timeout')
                    except Exception:
                        logger.debug("Failed to emit io_timeout signal", exc_info=True)

                # Only retry on timeouts/cancellation/IO-like exceptions
                err_str = str(e).lower()
                is_timeout = 'timeout' in err_str or isinstance(e, asyncio.TimeoutError)

                if attempts <= len(retry_schedule) and is_timeout:
                    delay = retry_schedule[attempts - 1]
                    logger.warning(f"Retrying browser operation after timeout (attempt {attempts}/{len(retry_schedule)}). Backoff {delay:.2f}s")
                    await asyncio.sleep(delay)
                    continue

                logger.info(f"Browser operation failed without retry: {type(e).__name__}: {e}")
                raise  # Re-raise for normal error handling

        metadata = StepMetadata(
            step_number=orchestrator_state.step_number,
            step_start_time=step_start_time,
            step_end_time=time.monotonic()
        )

        # Phase 3: Task 3.1 - Populate health metrics after action execution
        step_end_time = time.monotonic()
        orchestrator_state.last_step_duration = step_end_time - step_start_time

        # Track IO timeouts this step
        if isinstance(last_exc, asyncio.TimeoutError):
            orchestrator_state.io_timeouts_this_step = attempts
        else:
            orchestrator_state.io_timeouts_this_step = 0

        # Update consecutive failures based on action results
        had_failure = False
        if action_results:
            for result in action_results:
                if hasattr(result, 'success') and result.success is False:
                    had_failure = True
                    break

        if had_failure:
            orchestrator_state.consecutive_failures = self._state.consecutive_failures + 1
        else:
            orchestrator_state.consecutive_failures = 0

        result = ActuationResult(
            action_results=action_results,
            llm_output=agent_output,
            browser_state=orchestrator_state.browser_state,
            step_metadata=metadata,
        )

        return result

    def _select_timeout(self, agent_output: 'AgentOutput', orchestrator_state: OrchestratorState) -> float:
        """Choose a timeout based on settings, action types, and current site."""
        try:
            # 1) Site/domain override
            current_url = None
            try:
                current_url = getattr(orchestrator_state.browser_state, 'url', None)
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
            action_key = None
            action_params = None
            if agent_output and getattr(agent_output, 'action', None):
                try:
                    first = agent_output.action[0]
                    # Prefer detecting by the single populated field on the dynamic action model
                    dump = first.model_dump(exclude_unset=True)
                    if isinstance(dump, dict) and dump:
                        action_key = next(iter(dump.keys()))
                        action_params = dump.get(action_key, {}) if isinstance(dump.get(action_key), dict) else None
                except Exception:
                    action_key = None
                    action_params = None

            # Map overrides by detected action key (e.g., 'wait', 'click_element_by_index')
            if getattr(self.settings, 'action_timeout_overrides', None) and action_key:
                try:
                    if action_key in self.settings.action_timeout_overrides:
                        timeout_action = float(self.settings.action_timeout_overrides[action_key])
                except Exception:
                    pass

            # 3) Default
            default_timeout = float(getattr(self.settings, 'default_action_timeout_seconds', 60.0))

            # 4) Special handling for explicit wait action: honor requested seconds + small guard
            wait_timeout = None
            try:
                if action_key == 'wait' and isinstance(action_params, dict):
                    seconds = int(action_params.get('seconds', 0) or 0)
                    guard = float(getattr(self.settings, 'wait_timeout_guard_seconds', 5.0))
                    if seconds >= 0:
                        wait_timeout = float(seconds) + guard
            except Exception:
                wait_timeout = None

            # Combine: take the maximum of available candidates
            candidates = [v for v in [timeout_action, timeout_site, wait_timeout, default_timeout] if v is not None]
            return max(candidates) if candidates else default_timeout
        except Exception:
            return float(getattr(self.settings, 'default_action_timeout_seconds', 60.0))

    def _compute_retry_schedule(self) -> list[float]:
        """Exponential backoff schedule with jitter for retries."""
        import random
        max_attempts = max(0, int(getattr(self.settings, 'max_attempts_per_action', 0)))
        base = float(getattr(self.settings, 'backoff_base_seconds', 1.0))
        jitter = float(getattr(self.settings, 'backoff_jitter_seconds', 0.3))
        # attempts represent additional tries after the first
        schedule = []
        for i in range(max_attempts):
            delay = (2 ** i) * base + random.uniform(0, jitter)
            schedule.append(delay)
        return schedule

    def _should_run_planner(self, orchestrator_state) -> tuple[bool, str]:
        """Decide whether to run the CDAD planner this step.

        Rules:
        - If planner is disabled or task layer disabled: False
        - If first step and frequency is startup/every_step: True
        - If reflection protocol requested (self._planner_requested): True
        - every_step: True
        - on_change: when URL or title changes from last plan
        - interval: respect planner_interval (steps) or planner_interval_seconds (time)
        - never: False
        """
        try:
            if not bool(getattr(self.settings, 'task_layer_enabled', False)):
                return (False, 'task_layer_disabled')
            if not bool(getattr(self.settings, 'use_task_planner', False)):
                return (False, 'planner_disabled')

            freq = getattr(self.settings, 'planner_frequency', 'every_step')
            step = int(getattr(orchestrator_state, 'step_number', 0))
            now = time.monotonic()
            url = getattr(orchestrator_state.browser_state, 'url', None)
            title = getattr(orchestrator_state.browser_state, 'title', None)

            # Reflection- or stall-triggered request overrides cadence
            if self._planner_requested:
                return (True, 'requested_by_reflection_or_stall')

            if freq == 'never':
                return (False, 'frequency=never')

            if step == 0 and freq in ('startup', 'every_step'):
                return (True, 'startup')

            if freq == 'every_step':
                return (True, 'frequency=every_step')

            if freq == 'startup':
                return (False, 'startup_only')

            if freq == 'on_change':
                if url != self._last_plan_url or title != self._last_plan_title:
                    return (True, 'url_or_title_changed')
                return (False, 'no_significant_change')

            if freq == 'interval':
                try:
                    step_ivl = int(getattr(self.settings, 'planner_interval', 0) or 0)
                except Exception:
                    step_ivl = 0
                time_ivl = float(getattr(self.settings, 'planner_interval_seconds', 0.0) or 0.0)

                step_ok = step_ivl > 0 and (self._last_plan_step < 0 or (step - self._last_plan_step) >= step_ivl)
                time_ok = time_ivl > 0.0 and (self._last_plan_time <= 0.0 or (now - self._last_plan_time) >= time_ivl)
                if step_ok or time_ok:
                    return (True, 'interval_due')
                return (False, 'interval_not_due')

            return (False, 'unknown_frequency')
        except Exception as e:
            logger.debug(f"_should_run_planner failed: {e}", exc_info=True)
            return (False, 'error')

    async def _assess_and_adapt(self, orchestrator_state: OrchestratorState) -> None:
        """
        Phase 3: Task 3.2 - Reflex function for proactive self-awareness.

        Uses the local TransitionEngine (moved from state.py) to provide
        proactive adaptation based on health metrics and system state.
        """
        from .state import LoadStatus

        # Build transition inputs based on current state and orchestrator metrics
        state = self._state

        inputs = TransitionInputs(
            status=state.status,
            n_steps=state.n_steps,
            consecutive_failures=orchestrator_state.consecutive_failures,
            modes=getattr(state, 'modes', 0),
            load_status=getattr(state, 'load_status', LoadStatus.NORMAL),
            step_completed=True,
            had_failure=orchestrator_state.consecutive_failures > 0,
            missed_heartbeats=getattr(state, 'missed_heartbeats', 0),
            io_timeouts_recent_count=orchestrator_state.io_timeouts_this_step,
            max_steps=self.settings.max_steps,
            max_failures=getattr(self.settings, 'max_failures', 10),
            reflect_on_error=getattr(self.settings, 'reflect_on_error', True),
            use_planner=getattr(self.settings, 'use_planner', True),
            reflect_cadence=getattr(self.settings, 'reflect_cadence', 0),
            reflect_cooldown_seconds=getattr(self.settings, 'reflect_cooldown_seconds', 0.0),
            seconds_since_last_reflect=None,
            reflection_requested_this_epoch=getattr(state, 'reflection_requested_this_epoch', False),
            consecutive_action_failures=orchestrator_state.consecutive_failures,
            last_step_duration_seconds=orchestrator_state.last_step_duration,
            oscillation_score=orchestrator_state.oscillation_score,
            no_progress_score=orchestrator_state.no_progress_score,
        )

        # Get decision from transition engine
        engine = _TransitionEngine()
        decision = engine.decide(inputs)
        # Store decision rationale in orchestrator state for transparency
        orchestrator_state.decision_rationale = decision.reason

        logger.debug(f"_assess_and_adapt decision: {decision.reason}, reflection_intent: {decision.reflection_intent}")

        # Phase 3: Task 3.3 - Dynamic Protocol Switching
        # Detect STALLING or OSCILLATION and set protocol flag
        current_modes = AgentMode(getattr(state, 'modes', 0))

        if (
            AgentMode.STALLING in current_modes
            or orchestrator_state.oscillation_score >= engine.OSCILLATION_REFLECT_THRESHOLD
            or decision.reflection_intent
        ):
            # Switch to reflection protocol for next iteration
            self.current_protocol = 'reflection_protocol'
            logger.info(
                f"Switching to reflection protocol due to: stalling={AgentMode.STALLING in current_modes}, "
                f"oscillation={orchestrator_state.oscillation_score:.2f}, reflection_intent={decision.reflection_intent}"
            )
            # Request planner on next step
            self._planner_requested = True
        else:
            # Reset to normal protocol
            self.current_protocol = 'normal_protocol'


    async def run(self) -> AgentHistoryList:
        """
        Main unified loop that calls _perceive(), decide(), and execute() in sequence.

        This replaces the while True: loop from agent/loop.py with the unified
        perceive → decide → execute architecture.
        """
        from .state import AgentStatus, TERMINAL_STATES
        from .concurrency import ActuationLease
        from .views import AgentHistoryList

        # Start run
        try:
            await self.set_status(AgentStatus.RUNNING)
            # Announce task layer status once per run
            try:
                from .state import agent_log
                tl_enabled = bool(getattr(self.settings, 'task_layer_enabled', False))
                agent_log(logging.INFO, self._state.agent_id, self._state.n_steps,
                          f"Planner/Tasks: task_layer_enabled={tl_enabled}")
            except Exception:
                pass

            # Optional on_run_start hook
            try:
                if getattr(self.settings, 'on_run_start', None):
                    await self.settings.on_run_start(self)  # type: ignore[arg-type]
            except Exception:
                logger.debug('on_run_start hook failed (ignored)', exc_info=True)

            # Background human guidance processor (best-effort)
            guidance_task = None

            async def _guidance_processor():
                while True:
                    try:
                        status = await self.get_status()
                        if status in (AgentStatus.STOPPED, AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.MAX_STEPS_REACHED):
                            break
                        guidance = await self.get_human_guidance()
                        if guidance:
                            try:
                                self.message_manager.add_human_guidance(guidance)
                            except Exception:
                                logger.debug('add_human_guidance failed (ignored)', exc_info=True)
                            await self.set_current_goal(guidance)
                            await self.set_status(AgentStatus.RUNNING, force=True)
                        await asyncio.sleep(0.05)
                    except asyncio.CancelledError:
                        break
                    except Exception:
                        logger.debug('guidance_processor loop error', exc_info=True)
                        await asyncio.sleep(0.2)

            try:
                guidance_task = asyncio.create_task(_guidance_processor())
            except Exception:
                guidance_task = None

            # Classic loop (planner path falls back quickly if disabled)
            while True:
                status = await self.get_status()
                if status == AgentStatus.PAUSED:
                    await asyncio.sleep(0.2)
                    continue
                if status in TERMINAL_STATES:
                    break

                # Perceive
                try:
                    orchestrator_state = await self._perceive()
                except Exception as e:
                    logger.error(f'Perception failed: {e}')
                    await self.record_error(f'Perception failed: {e}', is_critical=True)
                    break

                # Step start hook
                try:
                    if getattr(self.settings, 'on_step_start', None):
                        await self.settings.on_step_start(self)  # type: ignore[arg-type]
                except Exception:
                    logger.debug('on_step_start hook failed (ignored)', exc_info=True)

                # Decide
                try:
                    agent_output = await self.decide(orchestrator_state)
                except Exception as e:
                    logger.error(f'Decision making failed: {e}')
                    await self.record_error(f'Decision making failed: {e}', is_critical=True)
                    break

                # Execute
                try:
                    async with ActuationLease():
                        orchestrator_state.messages_prepared = getattr(self, '_last_messages', None)
                        try:
                            await self._persist_conversation(orchestrator_state, agent_output)
                        except Exception:
                            logger.debug('pre-actuation conversation persistence failed (ignored)', exc_info=True)
                        result = await self.execute(agent_output, orchestrator_state)

                    # Optional pause on first click
                    try:
                        if (
                            bool(getattr(self.settings, 'pause_on_first_click', False)) and
                            not self._pause_triggered and
                            getattr(agent_output, 'action', None)
                        ):
                            if any('click' in type(a).__name__.lower() for a in agent_output.action):
                                await self.set_status(AgentStatus.PAUSED)
                                self._pause_triggered = True
                    except Exception:
                        logger.debug('pause_on_first_click handling failed (ignored)', exc_info=True)

                except Exception as e:
                    logger.error(f'Action execution failed: {e}')
                    await self.record_error(f'Action execution failed: {e}', is_critical=True)
                    break

                # Evaluate & reflect
                try:
                    outcome = await self.decide_and_apply_after_step(
                        result,
                        max_steps=self.settings.max_steps,
                        step_completed=True,
                        oscillation_score=orchestrator_state.oscillation_score,
                    )

                    try:
                        from .step_summary import log_step_summary
                        log_step_summary(result, self)
                    except Exception:
                        logger.debug('step_summary logging failed', exc_info=True)

                    if outcome.status in TERMINAL_STATES:
                        break

                    if getattr(outcome, 'reflection_intent', False):
                        try:
                            reflect_out = await self._reflect(orchestrator_state.browser_state, reason='engine_reflection')
                            try:
                                if hasattr(self._state, 'task_stack') and self._state.task_stack:
                                    self._state.task_stack.apply_reflection(reflect_out)
                            finally:
                                await self.mark_reflection_exit()
                        except Exception:
                            logger.debug('reflection pass failed (ignored)', exc_info=True)

                    try:
                        await self._assess_and_adapt(orchestrator_state)
                    except Exception as e:
                        logger.debug(f'_assess_and_adapt failed (ignored): {e}', exc_info=True)

                except Exception as e:
                    logger.error(f'Evaluation failed: {e}')
                    await self.record_error(f'Evaluation failed: {e}', is_critical=False)

                    try:
                        if getattr(self.settings, 'on_step_end', None):
                            await self.settings.on_step_end(self)  # type: ignore[arg-type]
                    except Exception:
                        logger.debug('on_step_end hook failed (ignored)', exc_info=True)

                    try:
                        every = int(getattr(self.settings, 'checkpoint_every_n', 0) or 0)
                    except Exception:
                        every = 0
                    if every > 0 and getattr(self, '_conv_run_dir', None) is not None:
                        try:
                            step_idx = self._state.n_steps
                            if step_idx % every == 0:
                                import json as _json
                                cp_dir = self._conv_run_dir / 'checkpoints'
                                await anyio.Path(str(cp_dir)).mkdir(parents=True, exist_ok=True)
                                cp = self._state.to_checkpoint()
                                cp_file = cp_dir / f'state-step-{step_idx:03d}.json'
                                async with await anyio.open_file(str(cp_file), mode='w', encoding='utf-8') as f:
                                    await f.write(_json.dumps(cp, ensure_ascii=False, indent=2))
                        except Exception:
                            logger.debug('checkpoint write failed (ignored)', exc_info=True)

        except Exception as e:
            logger.error(f'Unified loop failed: {e}')
            await self.record_error(f'Unified loop failed: {e}', is_critical=True)
        finally:
            await self.set_status(AgentStatus.STOPPED)
            try:
                lr = bool(getattr(self.settings, 'long_running_mode', False) or getattr(self.settings, 'enable_long_running_mode', False))
            except Exception:
                lr = False
            try:
                prof = getattr(self.browser_session, 'browser_profile', None)
                ka = bool(getattr(prof, 'keep_alive', False))
            except Exception:
                ka = False
            if not (lr or ka):
                try:
                    await self.browser_session.stop(_hint='(agent.orchestrator exit)')
                except Exception:
                    logger.debug('Error during browser_session.stop() in agent.orchestrator', exc_info=True)

            try:
                if 'guidance_task' in locals() and guidance_task is not None:
                    guidance_task.cancel()
                    await guidance_task
            except Exception:
                pass

            try:
                if getattr(self.settings, 'on_run_end', None):
                    from .views import AgentHistoryList as _AHL
                    await self.settings.on_run_end(_AHL(history=list(self._state.history)))  # type: ignore[arg-type]
            except Exception:
                logger.debug('on_run_end hook failed (ignored)', exc_info=True)

        return AgentHistoryList(history=list(self._state.history))

    async def _reflect(self, browser_state, reason: str | None = None) -> 'AgentOutput':
        """
        Generate a concise reflection to adjust task stack or next goal.

        Similar to DecisionMaker.reflect() but simplified for the unified architecture.
        """
        state = self._state
        try:
            summaries = getattr(state, 'summaries_compact', None)
            if callable(summaries):
                compact = summaries()
                if compact:
                    self.message_manager.add_local_note(f"Memory summaries: {compact}")
        except Exception:
            pass

        _hist = getattr(state, 'history', [])
        try:
            history_obj = _hist if hasattr(_hist, 'history') else type('H', (), {'history': list(_hist)})()
        except Exception:
            history_obj = type('H', (), {'history': []})()

        self.message_manager.update_history_representation(history_obj)

        # Nudge prompt with a local note about reflection intent
        try:
            self.message_manager.add_local_note(f"Planner reflection requested: {reason or 'no reason provided'}")
        except Exception:
            pass

        messages = await self.message_manager.prepare_messages(
            state_manager=self,  # Pass orchestrator as unified state manager
            browser_state=browser_state,
        )

        # Use simplified reflection with limited retries
        # Inline _setup_action_models for reflection
        cache_key = (
            getattr(self.settings, 'flash_mode', False),
            getattr(self.settings, 'use_thinking', False)
        )

        if not hasattr(self, '_action_model_cache') or self._action_model_cache.get('key') != cache_key:
            # Prepare action and output models based on configuration
            from browser_use.agent.views import AgentOutput

            # Setup action model
            ActionModel = self.settings.controller.registry.create_action_model()
            done_action_model = self.settings.controller.registry.create_action_model(include_actions=['done'])

            if getattr(self.settings, 'flash_mode', False):
                self.AgentOutput = AgentOutput.type_with_custom_actions_flash_mode(ActionModel)
                self.DoneAgentOutput = AgentOutput.type_with_custom_actions_flash_mode(done_action_model)
            elif getattr(self.settings, 'use_thinking', True):
                self.AgentOutput = AgentOutput.type_with_custom_actions(ActionModel)
                self.DoneAgentOutput = AgentOutput.type_with_custom_actions(done_action_model)
            else:
                self.AgentOutput = AgentOutput.type_with_custom_actions_no_thinking(ActionModel)
                self.DoneAgentOutput = AgentOutput.type_with_custom_actions_no_thinking(done_action_model)

            # Export ActionModel to instance for later use
            self.ActionModel = ActionModel

            # Cache the models
            self._action_model_cache = {
                'key': cache_key,
                'ActionModel': self.ActionModel,
                'AgentOutput': self.AgentOutput,
                'DoneAgentOutput': self.DoneAgentOutput
            }
        else:
            # Use cached models
            cache = self._action_model_cache
            self.ActionModel = cache['ActionModel']
            self.AgentOutput = cache['AgentOutput']
            self.DoneAgentOutput = cache['DoneAgentOutput']

        parsed = await self._invoke_llm_with_retry(messages, max_retries=1)

        # Ensure at most one action for reflection (like a planning directive)
        try:
            if hasattr(parsed, 'action') and isinstance(parsed.action, list) and len(parsed.action) > 1:
                parsed.action = parsed.action[:1]
        except Exception:
            pass

        return parsed

    async def _create_strategic_plan(self, goal: str, mission_state, available_tasks: list[str] | None = None) -> list:
        """Generate a strategic task plan using the CDAD planner and map to BaseTask instances.

        This is gated by settings.use_task_planner; when disabled, returns an empty list.
        """
        try:
            if not bool(getattr(self.settings, 'task_layer_enabled', False)):
                return []
            # Lazy import to avoid overhead when disabled
            from .tasks.planner import plan_tasks_with_cdad
            # Log that planner is being consulted
            try:
                from .state import agent_log
                agent_log(logging.INFO, self._state.agent_id, self._state.n_steps,
                          "Planner: Generating strategic plan via CDAD")
            except Exception:
                pass
            # Call planner directly (no timeout)
            tasks = await plan_tasks_with_cdad(self, goal, mission_state)
            try:
                from .state import agent_log
                agent_log(logging.INFO, self._state.agent_id, self._state.n_steps,
                          f"Planner: Created {len(tasks)} task(s)")
            except Exception:
                pass
            return tasks
        except Exception as e:
            logger.debug(f"_create_strategic_plan failed (ignored): {e}", exc_info=True)
            return []

    def _is_captcha_block(self, orchestrator_state) -> bool:
        """Heuristic detection of CAPTCHA/sorry pages to bias action selection.

        This uses lightweight signals only to avoid heavy DOM inspection.
        """
        try:
            bs = getattr(orchestrator_state, 'browser_state', None)
            url = (getattr(bs, 'url', None) or '').lower()
            title = (getattr(bs, 'title', None) or '').lower()
            if any(k in url for k in ('/sorry/', 'recaptcha', 'captcha', 'sorry/index')):
                return True
            if any(k in title for k in ('recaptcha', 'captcha', 'verification', 'security check')):
                return True
        except Exception:
            pass
        return False
