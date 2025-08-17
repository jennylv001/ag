from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass
from enum import Enum, IntFlag
from typing import TYPE_CHECKING, Any, Dict, Optional, List
import time

from pydantic import BaseModel, Field, ConfigDict

from browser_use.agent.concurrency import bulletproof_lock
from browser_use.agent.memory_budget import MemoryBudgetEnforcer
from browser_use.agent.views import AgentHistory, AgentHistoryList, AgentOutput

if TYPE_CHECKING:
    from browser_use.filesystem.file_system import FileSystem

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    PENDING = "PENDING"; RUNNING = "RUNNING"; PAUSED = "PAUSED"
    STOPPED = "STOPPED"; COMPLETED = "COMPLETED"; FAILED = "FAILED"; MAX_STEPS_REACHED = "MAX_STEPS_REACHED"


class LoadStatus(Enum):
    NORMAL = "NORMAL"; SHEDDING = "SHEDDING"


TERMINAL_STATES = {AgentStatus.STOPPED, AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.MAX_STEPS_REACHED}
STATE_PRIORITY = {
    AgentStatus.STOPPED: 5, AgentStatus.FAILED: 4, AgentStatus.PAUSED: 3,
    AgentStatus.RUNNING: 1, AgentStatus.COMPLETED: 0, AgentStatus.MAX_STEPS_REACHED: 0, AgentStatus.PENDING: -1,
}


def agent_log(level: int, agent_id: str, step: int, message: str, **kwargs):
    log_extras = {'agent_id': agent_id, 'step': step}
    logger.log(level, message, extra=log_extras, **kwargs)


@dataclass
class TaskContext:
    """Represents a single task in the task stack."""
    task_id: str
    description: str
    parent_task_id: Optional[str] = None
    created_at_step: int = 0
    completed: bool = False


class TaskStack(BaseModel):
    """
    Manages a stack of tasks for hierarchical task execution.
    Tasks can be nested and the stack maintains the execution context.
    """
    tasks: List[TaskContext] = Field(default_factory=list)
    current_task_id: str = "root"

    def push_task(self, task_id: str, description: str, current_step: int) -> None:
        """Push a new task onto the stack."""
        parent_id = self.current_task_id if self.current_task_id != "root" else None
        new_task = TaskContext(
            task_id=task_id,
            description=description,
            parent_task_id=parent_id,
            created_at_step=current_step
        )
        self.tasks.append(new_task)
        self.current_task_id = task_id

    def pop_task(self) -> Optional[TaskContext]:
        """Pop the current task and return to parent task."""
        if not self.tasks:
            return None

        # Find and remove current task
        current_task = None
        for i, task in enumerate(self.tasks):
            if task.task_id == self.current_task_id:
                current_task = self.tasks.pop(i)
                current_task.completed = True
                break

        if current_task and current_task.parent_task_id:
            self.current_task_id = current_task.parent_task_id
        else:
            self.current_task_id = "root"

        return current_task

    def get_current_task(self) -> Optional[TaskContext]:
        """Get the current active task context."""
        if self.current_task_id == "root":
            return None
        for task in self.tasks:
            if task.task_id == self.current_task_id:
                return task
        return None

    def get_task_hierarchy(self) -> List[str]:
        """Get the full task hierarchy from root to current task."""
        if self.current_task_id == "root":
            return ["root"]

        hierarchy = []
        current_id = self.current_task_id

        while current_id and current_id != "root":
            hierarchy.append(current_id)
            # Find parent task
            parent_id = None
            for task in self.tasks:
                if task.task_id == current_id:
                    parent_id = task.parent_task_id
                    break
            current_id = parent_id

        hierarchy.append("root")
        return list(reversed(hierarchy))

    def to_dict(self) -> Dict[str, Any]:
        """Convert TaskStack to dictionary for serialization."""
        return {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "description": task.description,
                    "parent_task_id": task.parent_task_id,
                    "created_at_step": task.created_at_step,
                    "completed": task.completed
                }
                for task in self.tasks
            ],
            "current_task_id": self.current_task_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskStack':
        """Create TaskStack from dictionary for deserialization."""
        tasks = [
            TaskContext(
                task_id=task_data["task_id"],
                description=task_data["description"],
                parent_task_id=task_data.get("parent_task_id"),
                created_at_step=task_data.get("created_at_step", 0),
                completed=task_data.get("completed", False)
            )
            for task_data in data.get("tasks", [])
        ]

        task_stack = cls(tasks=tasks, current_task_id=data.get("current_task_id", "root"))
        return task_stack

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentState(BaseModel):
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str
    current_goal: str = ""
    status: AgentStatus = AgentStatus.PENDING
    load_status: LoadStatus = LoadStatus.NORMAL # New: Track system load status
    task_stack: TaskStack = Field(default_factory=TaskStack)  # Task context management
    n_steps: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    accumulated_output: Optional[str] = None
    history: AgentHistoryList = Field(default_factory=AgentHistoryList)
    message_manager_state: Dict[str, Any] = Field(default_factory=dict)
    file_system_state: Optional[Dict[str, Any]] = None
    human_guidance_queue: asyncio.Queue[str] = Field(default_factory=asyncio.Queue, exclude=True)
    # --- New internal-only fields for Transition Engine (no API change) ---
    # Bitmask of AgentMode; keep as int for simple serialization
    modes: int = 0
    # Health counters
    missed_heartbeats: int = 0
    # Recent I/O timeout markers; sized by configurable window in future (defaults empty)
    io_timeouts_recent: deque[int] = Field(default_factory=deque)
    # Reflection control
    last_reflect_exit_ts: Optional[float] = None
    reflection_requested_this_epoch: bool = False
    # Monitoring counters
    cooldown_blocks: int = 0
    reflections_requested: int = 0
    reflections_suppressed_by_load: int = 0
    downshift_guard_prevented_count: int = 0
    model_config = ConfigDict(arbitrary_types_allowed=True)


class StateManager:
    """Manages the agent's state with a lock to ensure atomic, safe operations."""

    def __init__(
        self,
        initial_state: AgentState,
        file_system: Optional[FileSystem],
        max_failures: int,
        lock_timeout_seconds: float,
        use_planner: bool,
        reflect_on_error: bool,
        max_history_items: int,
    memory_budget_mb: float = 100.0,
    # Feature flags and thresholds (kept optional to avoid breaking callers)
    enable_modes: bool = False,
    reflect_cadence: int = 0,
    reflect_cooldown_seconds: float = 0.0,
    io_timeout_window: int = 10,
    max_steps_fallback: int = 1_000_000,
    ):
        self._state = initial_state
        self._lock = asyncio.Lock()
        self.lock_timeout_seconds = lock_timeout_seconds
        self._file_system = file_system
        self.max_failures = max_failures
        self.use_planner = use_planner
        self.reflect_on_error = reflect_on_error
        self.enable_modes = enable_modes
        self.reflect_cadence = reflect_cadence
        self.reflect_cooldown_seconds = reflect_cooldown_seconds
        self._io_timeout_window = max(1, int(io_timeout_window))
        self._max_steps_fallback = max_steps_fallback
        # Convert history to a bounded deque to prevent memory leaks
        initial_state.history.history = deque(initial_state.history.history, maxlen=max_history_items)
        self._state = initial_state

        # Initialize memory budget enforcer
        self._memory_enforcer = MemoryBudgetEnforcer(memory_budget_mb)

        if not self._state.current_goal:
            self._state.current_goal = self._state.task

    @property
    def state(self) -> AgentState:
        return self._state

    async def get_status(self) -> AgentStatus:
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            return self._state.status

    async def get_load_status(self) -> LoadStatus:
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            return self._state.load_status

    async def set_load_status(self, new_status: LoadStatus):
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            if self._state.load_status != new_status:
                self._state.load_status = new_status
                agent_log(logging.WARNING, self._state.agent_id, self._state.n_steps, f"System load status changed to: {new_status.value}")

    def _set_status_internal(self, new_status: AgentStatus, force: bool = False):
        """Internal, non-locking version of set_status. Must be called from within a held lock."""
        current_priority = STATE_PRIORITY.get(self._state.status, -1)
        new_priority = STATE_PRIORITY.get(new_status, -1)
        if force or (new_priority >= current_priority and self._state.status != new_status):
            agent_log(
                logging.DEBUG,
                self._state.agent_id,
                self._state.n_steps,
                f"State transition ACCEPTED: {self._state.status.value} (P:{current_priority}) -> {new_status.value} (P:{new_priority})",
            )
            self._state.status = new_status
        else:
            agent_log(
                logging.DEBUG,
                self._state.agent_id,
                self._state.n_steps,
                f"State transition REJECTED: {self._state.status.value} (P:{current_priority}) -> {new_status.value} (P:{new_priority})",
            )

    async def set_status(self, new_status: AgentStatus, force: bool = False):
        """Public method that acquires the lock before setting status."""
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            self._set_status_internal(new_status, force=force)

    async def add_history_item(self, item: AgentHistory):
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            self._state.history.history.append(item)  # deque handles maxlen

            # Enforce memory budget after adding item
            pruned_count = self._memory_enforcer.enforce_budget(self._state.history.history)
            if pruned_count > 0:
                agent_log(logging.INFO, self._state.agent_id, self._state.n_steps,
                         f"Memory budget enforced: pruned {pruned_count} old history items")

            if self._file_system:
                self._state.file_system_state = self._file_system.get_state()

    async def update_after_step(self, results: list[Any], max_steps: int, planner_interval: int):
        """Update state after a step is completed."""
        import traceback
        logger.debug(f"update_after_step called from: {traceback.format_stack()[-3:-1]}")

        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            if self._state.n_steps >= max_steps:
                next_status = AgentStatus.MAX_STEPS_REACHED
            elif any((getattr(r, 'success', False) is False) for r in results if hasattr(r, 'success')):
                self._state.consecutive_failures += 1
                if self._state.consecutive_failures >= self.max_failures:
                    next_status = AgentStatus.FAILED
                else:
                    next_status = AgentStatus.RUNNING
            else:
                next_status = AgentStatus.RUNNING

            self._set_status_internal(next_status)

    async def record_error(self, error_msg: str, is_critical: bool = False):
        next_status = None
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            self._state.last_error = error_msg
            if is_critical:
                self._state.consecutive_failures += 1
                if self._state.consecutive_failures >= self.max_failures:
                    next_status = AgentStatus.FAILED

            if next_status:
                self._set_status_internal(next_status, force=True)

    async def update_task(self, new_task: str):
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            self._state.task = new_task
            self._state.current_goal = new_task

    async def set_current_goal(self, new_goal: str):
        """Update only the current_goal without changing the root task.

        This is used for short-lived pivots such as human guidance so the
        next LLM prompt prioritizes the new intent immediately.
        """
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            self._state.current_goal = new_goal
            try:
                agent_log(
                    logging.INFO,
                    self._state.agent_id,
                    self._state.n_steps,
                    f"Current goal updated from human guidance: {new_goal}"
                )
            except Exception:
                pass

    async def update_last_history_with_reflection(self, memory_summary: str, next_goal: str, effective_strategy: Optional[str]):
        """
        Overwrites the memory and goal of the last history item with the superior
        output from the planner. This injects the reflection directly into the
        agent's historical context.
        """
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            if not self._state.history.history:
                return
            last_item = self._state.history.history[-1]

            if last_item.model_output:
                # Update fields on existing model output
                last_item.model_output.task_log = f"REVISED MEMORY AFTER REFLECTION: {memory_summary}"
                last_item.model_output.next_goal = next_goal
            else:
                # No model_output available for the last step; avoid creating an invalid placeholder.
                # Persist the next_goal at the state level to steer subsequent decisions.
                self._state.current_goal = next_goal
                agent_log(
                    logging.INFO,
                    self._state.agent_id,
                    self._state.n_steps,
                    "Planner reflection applied to state only (no model_output on last history item)",
                )

    async def clear_error_and_failures(self):
        """Resets the error and failure counters, typically after reflection."""
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            self._state.last_error = None
            self._state.consecutive_failures = 0

    async def mark_reflection_exit(self) -> None:
        """Record the time when a reflection completes to enforce cooldown accurately."""
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            self._state.last_reflect_exit_ts = time.monotonic()
            try:
                agent_log(
                    logging.INFO,
                    self._state.agent_id,
                    self._state.n_steps,
                    "Reflection completed; cooldown timestamp recorded",
                )
            except Exception:
                pass

    async def add_human_guidance(self, text: str):
        # Enqueue human guidance for supervisor pause handler to consume
        try:
            await self._state.human_guidance_queue.put(text)
            try:
                agent_log(logging.INFO, self._state.agent_id, self._state.n_steps, f"Human guidance enqueued")
            except Exception:
                pass
        except Exception:
            # Never raise from guidance enqueue
            try:
                agent_log(logging.WARNING, self._state.agent_id, self._state.n_steps, "Failed to enqueue human guidance")
            except Exception:
                pass

    async def get_human_guidance(self) -> Optional[str]:
        try:
            guidance = await asyncio.wait_for(self._state.human_guidance_queue.get(), timeout=0.5)
            self._state.human_guidance_queue.task_done()
            return guidance
        except asyncio.TimeoutError:
            return None

    # Task Context Management Methods

    async def push_task(self, task_id: str, description: str) -> None:
        """Push a new task onto the task stack."""
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            self._state.task_stack.push_task(task_id, description, self._state.n_steps)
            agent_log(
                logging.INFO,
                self._state.agent_id,
                self._state.n_steps,
                f"Task pushed: {task_id} - {description}"
            )

    async def pop_task(self) -> Optional[TaskContext]:
        """Pop the current task and return to parent task."""
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            completed_task = self._state.task_stack.pop_task()
            if completed_task:
                agent_log(
                    logging.INFO,
                    self._state.agent_id,
                    self._state.n_steps,
                    f"Task completed: {completed_task.task_id} - {completed_task.description}"
                )
            return completed_task

    async def get_current_task_id(self) -> str:
        """Get the current active task ID."""
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            return self._state.task_stack.current_task_id

    async def get_current_task_context(self) -> Optional[TaskContext]:
        """Get the current active task context."""
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            return self._state.task_stack.get_current_task()

    async def get_task_hierarchy(self) -> List[str]:
        """Get the full task hierarchy from root to current task."""
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            return self._state.task_stack.get_task_hierarchy()

    async def get_task_stack_summary(self) -> str:
        """Get a human-readable summary of the task stack for reflection prompts."""
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            if not self._state.task_stack.tasks:
                return "Task Context: root (main task)"

            hierarchy = self._state.task_stack.get_task_hierarchy()
            current_task = self._state.task_stack.get_current_task()

            summary = f"Task Hierarchy: {' -> '.join(hierarchy)}\n"
            if current_task:
                summary += f"Current Task: {current_task.task_id} - {current_task.description}\n"
                summary += f"Created at step: {current_task.created_at_step}"

            return summary

    # --- Health signal ingestion and decision application ---
    async def ingest_signal(self, signal_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Ingest health signals and immediately apply TransitionEngine decision.

        This does not mutate history and runs under the state lock.
        When enable_modes is False, returns early with no-ops.
        """
        if not getattr(self, 'enable_modes', False):
            return
        payload = payload or {}

        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            # Update counters based on signal
            if signal_type == 'heartbeat_miss':
                self._state.missed_heartbeats += 1
            elif signal_type == 'heartbeat_ok':
                self._state.missed_heartbeats = 0
            elif signal_type == 'io_timeout':
                self._state.io_timeouts_recent.append(1)
                # Bound the window size
                while len(self._state.io_timeouts_recent) > self._io_timeout_window:
                    try:
                        self._state.io_timeouts_recent.popleft()
                    except Exception:
                        break
            elif signal_type == 'io_ok':
                # Decay one recent timeout marker if present
                try:
                    if self._state.io_timeouts_recent:
                        self._state.io_timeouts_recent.popleft()
                except Exception:
                    pass
            elif signal_type == 'load_status':
                new_status = payload.get('status')
                if isinstance(new_status, LoadStatus) and new_status != self._state.load_status:
                    self._state.load_status = new_status
            else:
                # Unknown signal; ignore
                return

            # Compose engine inputs
            eng = _TransitionEngine()
            seconds_since_last_reflect: Optional[float] = None
            if self._state.last_reflect_exit_ts is not None:
                seconds_since_last_reflect = max(0.0, time.monotonic() - self._state.last_reflect_exit_ts)

            inputs = TransitionInputs(
                status=self._state.status,
                n_steps=self._state.n_steps,
                consecutive_failures=self._state.consecutive_failures,
                modes=self._state.modes,
                load_status=self._state.load_status,
                step_completed=False,
                had_failure=False,
                missed_heartbeats=self._state.missed_heartbeats,
                io_timeouts_recent_count=len(self._state.io_timeouts_recent),
                max_steps=self._max_steps_fallback,
                max_failures=self.max_failures,
                reflect_on_error=self.reflect_on_error,
                use_planner=self.use_planner,
                reflect_cadence=self.reflect_cadence,
                reflect_cooldown_seconds=self.reflect_cooldown_seconds,
                seconds_since_last_reflect=seconds_since_last_reflect,
                reflection_requested_this_epoch=self._state.reflection_requested_this_epoch,
            )

            decision = eng.decide(inputs)
            self._apply_decision(decision, source=f"signal:{signal_type}")

    @dataclass(frozen=True)
    class StepOutcome:
        status: AgentStatus
        modes: int
        reflection_intent: bool
        task_completed: bool

    async def decide_and_apply_after_step(self, result, max_steps: int, step_completed: bool = True) -> "StateManager.StepOutcome":
        """Unify post-step state transitions under a single lock.

        - Appends history
        - Updates counters and last_error
        - Calls pure engine and applies decision/modes
        - Increments n_steps
        """
        from browser_use.browser.views import BrowserStateHistory  # local import to avoid cycles
        from browser_use.agent.views import AgentHistory

        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            # Derive task completion and failure
            task_completed = False
            had_failure = False
            error_msg = None
            for r in result.action_results:
                if getattr(r, 'is_done', False) and getattr(r, 'success', None) is True:
                    task_completed = True
                if getattr(r, 'success', None) is False:
                    had_failure = True
                    if getattr(r, 'error', None):
                        error_msg = r.error

            # Build history item similar to Supervisor
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

            # Append history and enforce memory budget
            self._state.history.history.append(history_item)
            pruned_count = self._memory_enforcer.enforce_budget(self._state.history.history)
            if pruned_count > 0:
                logger.debug(f"Memory enforcer pruned {pruned_count} history items")

            # Update counters and last_error
            if had_failure:
                self._state.last_error = error_msg or self._state.last_error
                self._state.consecutive_failures += 1
            else:
                self._state.consecutive_failures = 0

            # Prepare engine inputs using pre-increment n_steps
            eng = _TransitionEngine()
            seconds_since_last_reflect: Optional[float] = None
            if self._state.last_reflect_exit_ts is not None:
                seconds_since_last_reflect = max(0.0, time.monotonic() - self._state.last_reflect_exit_ts)

            inputs = TransitionInputs(
                status=self._state.status,
                n_steps=self._state.n_steps,
                consecutive_failures=self._state.consecutive_failures,
                modes=self._state.modes,
                load_status=self._state.load_status,
                step_completed=step_completed,
                had_failure=had_failure,
                missed_heartbeats=self._state.missed_heartbeats,
                io_timeouts_recent_count=len(self._state.io_timeouts_recent),
                max_steps=max_steps,
                max_failures=self.max_failures,
                reflect_on_error=self.reflect_on_error,
                use_planner=self.use_planner,
                reflect_cadence=self.reflect_cadence,
                reflect_cooldown_seconds=self.reflect_cooldown_seconds,
                seconds_since_last_reflect=seconds_since_last_reflect,
                reflection_requested_this_epoch=self._state.reflection_requested_this_epoch,
            )

            decision = eng.decide(inputs)

            # If task completed, prefer COMPLETED status
            if task_completed:
                decision = TransitionDecision(
                    next_status=AgentStatus.COMPLETED,
                    set_modes=decision.set_modes,
                    clear_modes=decision.clear_modes,
                    reason=decision.reason + ",task_completed",
                    reflection_intent=decision.reflection_intent,
                )

            # Apply decision (status/modes/intent)
            self._apply_decision(decision, source="after_step")

            # Increment step counter last to preserve engine semantics
            self._state.n_steps += 1

            return StateManager.StepOutcome(
                status=self._state.status,
                modes=self._state.modes,
                reflection_intent=self._state.reflection_requested_this_epoch and decision.reflection_intent,
                task_completed=task_completed,
            )

    def _apply_decision(self, decision: "TransitionDecision", source: str) -> None:
        """Apply TransitionDecision under the assumption the lock is held.

        Priority rule: health-based decisions can upshift priority (to higher or equal priority)
        but cannot downshift within the same epoch unless counters are cleared.
        Epoch resets when missed_heartbeats == 0, len(io_timeouts_recent) == 0 and load_status is NORMAL.
        """
        # Modes update
        before_modes = self._state.modes
        after_modes = (before_modes | decision.set_modes) & (~decision.clear_modes)

        # Determine if epoch reset conditions are met
        counters_cleared = (
            self._state.missed_heartbeats == 0 and
            len(self._state.io_timeouts_recent) == 0 and
            self._state.load_status == LoadStatus.NORMAL
        )

        # Apply priority rule for status transitions
        current_pri = STATE_PRIORITY.get(self._state.status, -1)
        proposed_pri = STATE_PRIORITY.get(decision.next_status, -1)
        next_status = self._state.status
        downshift_prevented = False
        if decision.next_status != self._state.status:
            if proposed_pri >= current_pri:
                # Allow upshift or equal
                next_status = decision.next_status
            else:
                # Downshift only if counters cleared
                if counters_cleared:
                    next_status = decision.next_status
                else:
                    # Guard prevented a downshift
                    downshift_prevented = True

        # Reflection intent handling
        if decision.reflection_intent:
            self._state.reflection_requested_this_epoch = True
            try:
                self._state.reflections_requested += 1
            except Exception:
                pass
        elif counters_cleared:
            # Reset epoch when healthy again
            self._state.reflection_requested_this_epoch = False

        # Monitoring: track cooldown blocks
        if getattr(decision, 'cooldown_blocked', False):
            try:
                self._state.cooldown_blocks += 1
            except Exception:
                pass

        # Count reflection suppression due to load shedding
        try:
            if 'shed_suppresses_reflection' in getattr(decision, 'reason', ''):
                self._state.reflections_suppressed_by_load += 1
        except Exception:
            pass

        # Track downshift guard prevention
        if downshift_prevented:
            try:
                self._state.downshift_guard_prevented_count += 1
            except Exception:
                pass

        # Assign modes and status
        self._state.modes = after_modes
        if next_status != self._state.status:
            self._set_status_internal(next_status, force=False)

        # Structured decision log (encoded in message to avoid extra collision)
        try:
            msg = (
                f"transition_decision source={source} "
                f"reason={decision.reason} from={self._state.status.value} to={next_status.value} "
                f"modes={int(self._state.modes)} missed_hb={int(self._state.missed_heartbeats)} "
                f"io_timeouts={int(len(self._state.io_timeouts_recent))} load={self._state.load_status.value} "
                f"cooldown_blocks={self._state.cooldown_blocks} "
                f"reflections_requested={self._state.reflections_requested} "
                f"reflections_suppressed_by_load={self._state.reflections_suppressed_by_load} "
                f"downshift_guard_prevented={self._state.downshift_guard_prevented_count}"
            )
            agent_log(logging.INFO, self._state.agent_id, self._state.n_steps, msg)
        except Exception:
            # Never fail on logging
            pass

    # --- Debug/observability helpers ---
    async def get_health_snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of modes, counters, and health-related state for debugging."""
        async with bulletproof_lock(self._lock, self.lock_timeout_seconds):
            return {
                "status": self._state.status.value,
                "load_status": self._state.load_status.value,
                "n_steps": self._state.n_steps,
                "modes": int(self._state.modes),
                "missed_heartbeats": int(self._state.missed_heartbeats),
                "io_timeouts_recent": int(len(self._state.io_timeouts_recent)),
                "cooldown_blocks": int(self._state.cooldown_blocks),
                "reflections_requested": int(self._state.reflections_requested),
                "reflections_suppressed_by_load": int(self._state.reflections_suppressed_by_load),
                "downshift_guard_prevented": int(self._state.downshift_guard_prevented_count),
                "reflection_requested_this_epoch": bool(self._state.reflection_requested_this_epoch),
            }


# === Transition Engine (pure) ===
from dataclasses import dataclass


class AgentMode(IntFlag):
    """Health and flow-control overlays; do not expand public AgentStatus."""
    NONE = 0
    DEGRADED = 1 << 0
    STALLING = 1 << 1
    UNCERTAIN = 1 << 2
    HIGH_LOAD = 1 << 3
    DRAINING = 1 << 4


@dataclass(frozen=True)
class TransitionInputs:
    # State snapshot
    status: AgentStatus
    n_steps: int
    consecutive_failures: int
    modes: int
    load_status: LoadStatus
    # Step outcome
    step_completed: bool
    had_failure: bool
    # Health signals (summarized)
    missed_heartbeats: int
    io_timeouts_recent_count: int
    # Config
    max_steps: int
    max_failures: int
    reflect_on_error: bool
    use_planner: bool
    reflect_cadence: int
    reflect_cooldown_seconds: float
    # Reflection guards (pure-friendly)
    seconds_since_last_reflect: Optional[float]
    reflection_requested_this_epoch: bool


@dataclass(frozen=True)
class TransitionDecision:
    next_status: AgentStatus
    set_modes: int
    clear_modes: int
    reason: str
    reflection_intent: bool
    cooldown_blocked: bool = False


class _TransitionEngine:
    """Pure decision engine for next-state and mode overlays.

    No logging, no I/O. Deterministic and testable.
    """

    # Internal thresholds (can be made configurable later)
    HB_DEGRADED_THRESHOLD = 1
    HB_STALLING_THRESHOLD = 3
    IO_UNCERTAIN_THRESHOLD = 1
    IO_STALLING_THRESHOLD = 3

    def decide(self, inp: TransitionInputs) -> TransitionDecision:
        # 1) Compute desired modes from signals
        desired_modes = AgentMode.NONE

        # Heartbeat-derived modes
        if inp.missed_heartbeats >= self.HB_DEGRADED_THRESHOLD:
            desired_modes |= AgentMode.DEGRADED
        if inp.missed_heartbeats >= self.HB_STALLING_THRESHOLD:
            desired_modes |= AgentMode.STALLING

        # IO timeouts
        if inp.io_timeouts_recent_count >= self.IO_UNCERTAIN_THRESHOLD:
            desired_modes |= AgentMode.UNCERTAIN
        if inp.io_timeouts_recent_count >= self.IO_STALLING_THRESHOLD:
            desired_modes |= AgentMode.STALLING

        # High-load overlay
        if inp.load_status == LoadStatus.SHEDDING:
            desired_modes |= AgentMode.HIGH_LOAD

        current_modes = AgentMode(inp.modes)
        set_modes_mask = int(desired_modes & ~current_modes)
        clear_modes_mask = int(current_modes & ~desired_modes)
        active_modes = desired_modes | current_modes

        # 2) Next status based on step outcome and limits (no task-completed input here)
        next_status = inp.status
        reason_parts: List[str] = []

        if inp.step_completed:
            # Check max steps first (parity with existing logic: increment + compare)
            # Here we reason using n_steps + 1 when step completes
            next_step = inp.n_steps + 1
            if next_step >= inp.max_steps:
                next_status = AgentStatus.MAX_STEPS_REACHED
                reason_parts.append("max_steps_reached")
            else:
                if inp.had_failure:
                    if (inp.consecutive_failures + 1) >= inp.max_failures:
                        next_status = AgentStatus.FAILED
                        reason_parts.append("failures_threshold")
                    else:
                        next_status = AgentStatus.RUNNING
                        reason_parts.append("failure_but_under_threshold")
                else:
                    # Success path resets failures externally; remain RUNNING
                    next_status = AgentStatus.RUNNING
                    reason_parts.append("success_running")
        else:
            # No step finalization; maintain current status
            reason_parts.append("no_step_change")

        # 3) Reflection intent (pure intent only; planner wires later)
        reflection_intent = False
        cooldown_blocked = False
        if (
            inp.step_completed
            and inp.had_failure
            and inp.reflect_on_error
            and inp.use_planner
        ):
            # Anti-thrash: cooldown guard blocks only on the first failure after success
            # and only if not in UNCERTAIN/STALLING modes.
            if (
                inp.reflect_cooldown_seconds > 0
                and inp.seconds_since_last_reflect is not None
                and inp.seconds_since_last_reflect < inp.reflect_cooldown_seconds
                and inp.consecutive_failures <= 1
                and not (AgentMode.UNCERTAIN & active_modes or AgentMode.STALLING & active_modes)
            ):
                reason_parts.append("cooldown_guard")
                cooldown_blocked = True
            # Load shedding suppresses reflection
            elif inp.load_status == LoadStatus.SHEDDING:
                reason_parts.append("shed_suppresses_reflection")
            # Cadence gate (if >0, only allow on cadence steps)
            elif inp.reflect_cadence > 0 and ((inp.n_steps + 1) % inp.reflect_cadence != 0):
                reason_parts.append("cadence_gate")
            # Epoch guard (only one request per epoch until cleared externally)
            elif inp.reflection_requested_this_epoch:
                reason_parts.append("epoch_guard")
            else:
                reflection_intent = True
                reason_parts.append("reflect_on_error")
        else:
            # Health-based reflection when stalling or uncertain, even without a step failure
            if (
                (AgentMode.STALLING & active_modes or AgentMode.UNCERTAIN & active_modes)
                and inp.use_planner
            ):
                # Apply guards: cooldown is bypassed when modes indicate uncertainty/stall
                if inp.load_status == LoadStatus.SHEDDING:
                    reason_parts.append("shed_suppresses_reflection")
                elif inp.reflect_cadence > 0 and ((inp.n_steps + 1) % inp.reflect_cadence != 0):
                    reason_parts.append("cadence_gate")
                elif inp.reflection_requested_this_epoch:
                    reason_parts.append("epoch_guard")
                else:
                    reflection_intent = True
                    reason_parts.append("reflect_on_stall")

        reason = ",".join(reason_parts) if reason_parts else "noop"
        return TransitionDecision(
            next_status=next_status,
            set_modes=set_modes_mask,
            clear_modes=clear_modes_mask,
            reason=reason,
            reflection_intent=reflection_intent,
            cooldown_blocked=cooldown_blocked,
        )


    # End of pure engine













































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































