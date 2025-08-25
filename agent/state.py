from __future__ import annotations

"""
Task 2: State skeletons and interfaces (no behavior change).
"""
from dataclasses import dataclass, field
from enum import Enum, IntFlag
from typing import Optional, List, Iterable, Any, Dict, Deque, TYPE_CHECKING, Tuple
import asyncio
import logging
from collections import deque
import time

if TYPE_CHECKING:
    from browser_use.browser.views import BrowserStateSummary

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"


class LoadStatus(Enum):
    NORMAL = "NORMAL"
    SHEDDING = "SHEDDING"


TERMINAL_STATES = {AgentStatus.STOPPED, AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.MAX_STEPS_REACHED}


@dataclass
class OrchestratorState:
    """
    Centralized state object that holds all in-flight data for a single step.

    This consolidates state management for the unified orchestrator loop,
    containing browser state, health metrics, current protocol data, and
    other contextual information needed across perceive → decide → execute.
    """
    # Core browser and page state
    browser_state: Optional['BrowserStateSummary'] = None

    # Health and performance metrics (Phase 3: Grounding System Robustness)
    health_metrics: Optional[Dict[str, Any]] = None
    consecutive_failures: int = 0
    io_timeouts_this_step: int = 0
    last_step_duration: float = 0.0
    oscillation_score: float = 0.0
    no_progress_score: float = 0.0

    # Current step context
    step_number: int = 0
    step_start_time: Optional[float] = None

    # Protocol and messaging context (Phase 3: Dynamic Protocol Switching)
    current_protocol: Optional[str] = None
    messages_prepared: Optional[List[Any]] = None

    # Semantic and perception data
    semantic_snapshot: Optional[Dict[str, Any]] = None
    page_hash: Optional[str] = None

    # Human guidance and task context
    current_goal: Optional[str] = None
    human_guidance: Optional[str] = None
    task_stack_summary: Optional[str] = None

    # Decision context
    confidence_score: Optional[float] = None
    decision_rationale: Optional[str] = None

    # Error and retry state
    last_error: Optional[str] = None
    retry_count: int = 0
    backoff_until: Optional[float] = None


def agent_log(level: int, agent_id: str, step: int, message: str, **kwargs) -> None:
    try:
        logger.log(level, message, extra={"agent_id": agent_id, "step": step}, **kwargs)
    except Exception:
        logger.log(level, message, **kwargs)


@dataclass
class HistoryItem:
    note: str


@dataclass
class AgentState:
    task: str
    status: AgentStatus = AgentStatus.PENDING
    # Identifier for logging/telemetry (optional)
    agent_id: str = "agent"
    # Optional mission-scoped state for high-level goals and progress
    mission: Optional["MissionState"] = None
    # Optional persisted file system state for checkpoint/restore of filesystem
    file_system_state: Optional[Dict[str, Any]] = None
    # Message manager state (pydantic model) expected by Supervisor/MessageManager
    message_manager_state: Any = None
    # Execution counters and error tracking
    n_steps: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    # Current goal can diverge from root task during planning
    current_goal: str = ""
    # Rolling history with bounded memory
    history: Deque[HistoryItem] = field(default_factory=lambda: deque(maxlen=200))
    # Summaries maintained during pruning
    facts_learned: List[str] = field(default_factory=list)
    constraints_discovered: List[str] = field(default_factory=list)
    tried_and_failed: List[str] = field(default_factory=list)
    successful_patterns: List[str] = field(default_factory=list)
    # Optional task stack and navigation anchors for checkpoint/restore
    task_stack: Optional["TaskStack"] = None
    last_stable_url: Optional[str] = None
    semantic_anchors: List[str] = field(default_factory=list)
    # Phase 2: Semantic page state for comprehensive snapshots
    last_semantic_page_hash: Optional[str] = None
    last_semantic_flags: Optional[Dict[str, Any]] = None
    # Human guidance inbox
    human_guidance_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue, repr=False)
    # Health/mode related fields (kept lightweight; maintained by StateManager)
    missed_heartbeats: int = 0
    modes: int = 0
    # Recent I/O timeouts window; maintained as deque of markers
    io_timeouts_recent: Deque[int] = field(default_factory=lambda: deque(maxlen=10), repr=False)
    load_status: LoadStatus = LoadStatus.NORMAL
    action_failure_streak: int = 0
    reflection_requested_this_epoch: bool = False
    cooldown_blocks: int = 0
    reflections_requested: int = 0
    reflections_suppressed_by_load: int = 0
    downshift_guard_prevented_count: int = 0
    last_reflect_exit_ts: Optional[float] = None

    def add_history(self, note: str | HistoryItem, max_history_items: int = 40, memory_budget_mb: float = 100.0) -> None:
        item = note if isinstance(note, HistoryItem) else HistoryItem(note=str(note))
        self.history.append(item)
        self._prune_if_needed(max_history_items=max_history_items, memory_budget_mb=memory_budget_mb)

    def _estimate_memory_bytes(self) -> int:
        base = 0
        try:
            base = sum(len(getattr(h, 'note', '') or '') for h in self.history)
        except Exception:
            base = 0
        # Include summaries approximation
        for bucket in (self.facts_learned, self.constraints_discovered, self.tried_and_failed, self.successful_patterns):
            base += sum(len(x) for x in bucket)
        return base

    def _prune_if_needed(self, max_history_items: int, memory_budget_mb: float) -> None:
        # Enforce item cap
        while len(self.history) > max(1, int(max_history_items)):
            self._promote_to_summary(self.history.popleft())
        # Enforce memory cap (approximate)
        budget_bytes = max(1.0, float(memory_budget_mb)) * 1024 * 1024
        while self._estimate_memory_bytes() > budget_bytes and self.history:
            self._promote_to_summary(self.history.popleft())

    def _promote_to_summary(self, item: HistoryItem) -> None:
        text = (item.note or '').strip()
        if not text:
            return
        lower = text.lower()
        try:
            if 'constraint' in lower or 'must' in lower or 'forbid' in lower:
                self.constraints_discovered.append(text)
            elif 'fail' in lower or 'error' in lower or 'retry' in lower:
                self.tried_and_failed.append(text)
            elif 'success' in lower or 'worked' in lower or 'passed' in lower:
                self.successful_patterns.append(text)
            else:
                self.facts_learned.append(text)
        except Exception:
            self.facts_learned.append(text)

    def summaries_compact(self, max_each: int = 5) -> str:
        parts: list[str] = []
        def fmt(label: str, items: List[str]):
            if not items:
                return
            take = items[-max_each:]
            parts.append(f"{label}: " + "; ".join(take))
        fmt("Facts learned", self.facts_learned)
        fmt("Constraints", self.constraints_discovered)
        fmt("Tried & failed", self.tried_and_failed)
        fmt("Successful patterns", self.successful_patterns)
        return " | ".join(parts)

    # --- Minimal checkpoint/restore (Task 12) ---
    def to_checkpoint(self, include_conversation_trail: bool = False, max_conversation_steps: int = 10) -> Dict[str, Any]:
        """Serialize essential fields for minimal long-running resume.

        Includes summaries, last stable URL, semantic anchors, and a compact task stack.
        History items are not stored verbatim; only summaries are preserved to keep payload small.

        Args:
            include_conversation_trail: Whether to include compact conversation history
            max_conversation_steps: Maximum number of recent conversation steps to include
        """
        import copy
        stack = []
        if isinstance(self.task_stack, TaskStack):
            for t in self.task_stack._stack:
                stack.append({
                    "title": t.title,
                    "acceptance_criteria": list(t.acceptance_criteria),
                    "constraints": list(t.constraints),
                })

        checkpoint = {
            "task": self.task,
            "status": self.status.value,
            "facts_learned": list(self.facts_learned),
            "constraints_discovered": list(self.constraints_discovered),
            "tried_and_failed": list(self.tried_and_failed),
            "successful_patterns": list(self.successful_patterns),
            "task_stack": stack,
            "last_stable_url": self.last_stable_url,
            "semantic_anchors": list(self.semantic_anchors),
            # Phase 2: Semantic page state for enhanced perception
            "last_semantic_page_hash": self.last_semantic_page_hash,
            "last_semantic_flags": copy.deepcopy(self.last_semantic_flags) if self.last_semantic_flags else None,
        }

        # Phase 2: Add conversation trail if requested
        if include_conversation_trail:
            conversation_trail = []
            try:
                # Extract last N steps from history with compact representation
                recent_history = list(self.history)[-max_conversation_steps:] if self.history else []
                for item in recent_history:
                    step_data = {
                        "step_number": getattr(item, "step_number", None),
                        "url": None,
                        "actions": [],
                        "task_log_structured": None,
                    }

                    # Extract URL from state if available
                    if hasattr(item, "state") and item.state:
                        step_data["url"] = getattr(item.state, "url", None)

                    # Extract actions and structured task log from model_output
                    if hasattr(item, "model_output") and item.model_output:
                        # Extract action names and key parameters
                        actions = []
                        for action in getattr(item.model_output, "action", []):
                            try:
                                action_dump = action.model_dump(exclude_none=True)
                                if action_dump:
                                    action_name = next(iter(action_dump.keys()))
                                    action_params = action_dump[action_name]
                                    # Keep only essential parameters to minimize size
                                    essential_params = {}
                                    for key in ["url", "file_name", "query", "text", "xpath"]:
                                        if key in action_params:
                                            val = action_params[key]
                                            if isinstance(val, str) and len(val) > 100:
                                                essential_params[key] = val[:100] + "..."
                                            else:
                                                essential_params[key] = val
                                    actions.append({"name": action_name, "params": essential_params})
                            except Exception:
                                continue
                        step_data["actions"] = actions

                        # Extract structured task log
                        task_log_structured = getattr(item.model_output, "task_log_structured", None)
                        if task_log_structured:
                            try:
                                # Serialize structured task log with size limits
                                tls_data = task_log_structured.model_dump(exclude_none=True)
                                # Cap list sizes to prevent bloat
                                for list_field in ["objectives", "checklist", "risks", "blockers"]:
                                    if list_field in tls_data and isinstance(tls_data[list_field], list):
                                        tls_data[list_field] = tls_data[list_field][:10]  # Limit to 10 items
                                step_data["task_log_structured"] = tls_data
                            except Exception:
                                pass

                    conversation_trail.append(step_data)

                checkpoint["conversation_trail"] = conversation_trail
            except Exception:
                # Never fail checkpoint creation on conversation trail extraction
                checkpoint["conversation_trail"] = []

        return checkpoint

    @classmethod
    def from_checkpoint(cls, data: Dict[str, Any]) -> "AgentState":
        """Restore AgentState from checkpoint produced by to_checkpoint."""
        task = str(data.get("task", ""))
        status_val = data.get("status", AgentStatus.PENDING.value)
        try:
            status = AgentStatus(status_val)
        except Exception:
            status = AgentStatus.PENDING
        st = cls(task=task, status=status)
        st.facts_learned = list(data.get("facts_learned", []))
        st.constraints_discovered = list(data.get("constraints_discovered", []))
        st.tried_and_failed = list(data.get("tried_and_failed", []))
        st.successful_patterns = list(data.get("successful_patterns", []))
        st.last_stable_url = data.get("last_stable_url")
        st.semantic_anchors = list(data.get("semantic_anchors", []))
        # Phase 2: Restore semantic page state
        st.last_semantic_page_hash = data.get("last_semantic_page_hash")
        st.last_semantic_flags = data.get("last_semantic_flags")

        # Restore task stack
        try:
            items = data.get("task_stack", []) or []
            if items:
                ts = TaskStack()
                for it in items:
                    ts.push(title=str(it.get("title", "")),
                            acceptance_criteria=list(it.get("acceptance_criteria", []) or []),
                            constraints=list(it.get("constraints", []) or []))
                st.task_stack = ts
        except Exception:
            st.task_stack = None

        # Phase 2: Store conversation trail for potential reconstruction
        # Note: This doesn't restore full history but provides resumption context
        conversation_trail = data.get("conversation_trail", [])
        if conversation_trail:
            # Store as metadata for use by StateManager.import_full_snapshot
            setattr(st, "_checkpoint_conversation_trail", conversation_trail)

        return st


@dataclass
class MissionState:
    """High-level mission state attached to AgentState.

    This holds the mission description and simple progress flags. It does not
    introduce a new manager; it's carried within AgentState and threaded
    through settings or direct attributes as needed.
    """

    description: str = ""
    # Simple progress markers; expand later as needed without behavior changes
    started: bool = False
    completed: bool = False
    success: bool = False


@dataclass
class HealthSnapshot:
    """Lightweight self-awareness signals for the core loop.

    All counters are conservative and reset only on clear success signals.
    """
    failure_streak: int = 0
    no_progress_count: int = 0
    io_timeouts_recent: int = 0
    last_step_duration_seconds: float = 0.0
    load_level: str = "normal"  # one of: normal, shedding
    confidence_last_decision: float | None = None
    last_error: Optional[str] = None
    # Optional progress/oscillation tracking
    recent_page_hashes: deque[str] = field(default_factory=lambda: deque(maxlen=6))
    oscillation_score: float = 0.0

    def on_io_timeout(self) -> None:
        self.io_timeouts_recent = min(self.io_timeouts_recent + 1, 10)

    def on_io_ok(self) -> None:
        # Gradually decay timeouts
        if self.io_timeouts_recent > 0:
            self.io_timeouts_recent -= 1

    def on_failure(self) -> None:
        self.failure_streak += 1

    def on_success(self) -> None:
        self.failure_streak = 0
        self.no_progress_count = 0

    def on_no_progress(self) -> None:
        self.no_progress_count += 1

    def on_step_end(self, duration_seconds: float) -> None:
        self.last_step_duration_seconds = max(0.0, float(duration_seconds))

    # --- Progress helpers ---
    def record_page_hash(self, page_hash: Optional[str]) -> None:
        if not page_hash:
            return
        self.recent_page_hashes.append(str(page_hash))
        # Simple oscillation: detect ABAB in last 4
        try:
            if len(self.recent_page_hashes) >= 4:
                a, b, c, d = list(self.recent_page_hashes)[-4:]
                self.oscillation_score = 1.0 if a == c and b == d and a != b else max(self.oscillation_score * 0.9, 0.0)
        except Exception:
            # Conservative fallback
            self.oscillation_score = max(self.oscillation_score * 0.9, 0.0)

    def set_last_error(self, msg: Optional[str]) -> None:
        self.last_error = msg or None


@dataclass
class Task:
    title: str
    acceptance_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


class TaskStack:
    """Task-first execution stack with acceptance criteria and constraints."""

    def __init__(self) -> None:
        self._stack: list[Task] = []

    def push(self, title: str, acceptance_criteria: Optional[Iterable[str]] = None, constraints: Optional[Iterable[str]] = None) -> None:
        self._stack.append(Task(
            title=title,
            acceptance_criteria=list(acceptance_criteria or []),
            constraints=list(constraints or []),
        ))

    def pop(self) -> Optional[Task]:
        return self._stack.pop() if self._stack else None

    def current(self) -> Optional[Task]:
        return self._stack[-1] if self._stack else None

    def update_current(self, title: Optional[str] = None,
                       acceptance_criteria: Optional[Iterable[str]] = None,
                       constraints: Optional[Iterable[str]] = None) -> None:
        cur = self.current()
        if not cur:
            return
        if title is not None:
            cur.title = title
        if acceptance_criteria is not None:
            cur.acceptance_criteria = list(acceptance_criteria)
        if constraints is not None:
            cur.constraints = list(constraints)

    def compact_summary(self, max_items: int = 3) -> str:
        if not self._stack:
            return ""
        items = self._stack[-max_items:]
        parts = []
        for i, t in enumerate(items, 1):
            ac = "; ".join(t.acceptance_criteria) if t.acceptance_criteria else "-"
            cs = "; ".join(t.constraints) if t.constraints else "-"
            parts.append(f"{i}. {t.title} [AC: {ac}] [C: {cs}]")
        return " | ".join(parts)

    def apply_reflection(self, reflection_output: object) -> None:
        """Very small protocol for tests: supports attributes 'new_subtasks' (list[str]) and 'updated_task' (str)."""
        try:
            # Update current task title if provided
            updated = getattr(reflection_output, 'updated_task', None)
            if isinstance(updated, str) and updated.strip():
                self.update_current(title=updated.strip())

            # Push any new subtasks beneath the (updated) current task so that
            # a subsequent pop() leaves another task present (per unit tests)
            new_subtasks = getattr(reflection_output, 'new_subtasks', None)
            if isinstance(new_subtasks, (list, tuple)) and new_subtasks:
                # Insert new subtasks just below the current (top) task so that
                # current remains on top; popping once reveals the first subtask.
                subtasks_clean = [s.strip() for s in new_subtasks if isinstance(s, str) and s.strip()]
                if subtasks_clean:
                    # If stack empty, just push in order
                    if not self._stack:
                        for s in subtasks_clean:
                            self.push(title=s)
                    else:
                        insert_at = len(self._stack) - 1  # position just below current top
                        for s in reversed(subtasks_clean):
                            self._stack.insert(insert_at, Task(title=s))
        except Exception:
            pass


# ===== Minimal StateManager (new home) =====

STATE_PRIORITY = {
    AgentStatus.STOPPED: 5,
    AgentStatus.FAILED: 4,
    AgentStatus.PAUSED: 3,
    AgentStatus.RUNNING: 1,
    AgentStatus.COMPLETED: 0,
    AgentStatus.MAX_STEPS_REACHED: 0,
    AgentStatus.PENDING: -1,
}


@dataclass(frozen=True)
class StepOutcome:
    status: AgentStatus
    modes: int
    reflection_intent: bool
    task_completed: bool


class StateManager:
    def __init__(
        self,
        initial_state: AgentState,
        file_system: Optional[Any],
        max_failures: int,
        lock_timeout_seconds: float,
        use_planner: bool,
        reflect_on_error: bool,
        max_history_items: int,
        memory_budget_mb: float = 100.0,
        enable_modes: bool = False,
        reflect_cadence: int = 0,
        reflect_cooldown_seconds: float = 0.0,
        io_timeout_window: int = 10,
        max_steps_fallback: int = 1_000_000,
    ) -> None:
        self._state = initial_state
        self._lock = asyncio.Lock()
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self._file_system = file_system
        self.max_failures = int(max_failures)
        self.use_planner = bool(use_planner)
        self.reflect_on_error = bool(reflect_on_error)
        self.enable_modes = bool(enable_modes)
        self.reflect_cadence = int(reflect_cadence)
        self.reflect_cooldown_seconds = float(reflect_cooldown_seconds)
        self._io_timeout_window = max(1, int(io_timeout_window))
        self._max_steps_fallback = int(max_steps_fallback)
        # Bound history size
        try:
            self._state.history = deque(self._state.history, maxlen=max_history_items)  # type: ignore[arg-type]
        except Exception:
            pass
        if not self._state.task_stack:
            self._state.task_stack = TaskStack()
        # Initialize current_goal safely
        if not getattr(self._state, 'current_goal', None):
            try:
                self._state.current_goal = self._state.task
            except Exception:
                pass

        # Health/mode fields expected by engine
        self._state.missed_heartbeats = getattr(self._state, 'missed_heartbeats', 0)
        self._state.io_timeouts_recent = getattr(self._state, 'io_timeouts_recent', deque(maxlen=self._io_timeout_window))
        self._state.modes = getattr(self._state, 'modes', 0)
        self._state.load_status = getattr(self._state, 'load_status', LoadStatus.NORMAL)
        self._state.action_failure_streak = getattr(self._state, 'action_failure_streak', 0)
        self._state.reflection_requested_this_epoch = getattr(self._state, 'reflection_requested_this_epoch', False)
        self._state.cooldown_blocks = getattr(self._state, 'cooldown_blocks', 0)
        self._state.reflections_requested = getattr(self._state, 'reflections_requested', 0)
        self._state.reflections_suppressed_by_load = getattr(self._state, 'reflections_suppressed_by_load', 0)
        self._state.downshift_guard_prevented_count = getattr(self._state, 'downshift_guard_prevented_count', 0)

    @property
    def state(self) -> AgentState:
        return self._state

    async def get_status(self) -> AgentStatus:
        async with self._lock:
            return self._state.status

    async def set_status(self, new_status: AgentStatus, force: bool = False) -> None:
        async with self._lock:
            self._set_status_internal(new_status, force=force)

    def _set_status_internal(self, new_status: AgentStatus, force: bool = False) -> None:
        cur_pri = STATE_PRIORITY.get(self._state.status, -1)
        new_pri = STATE_PRIORITY.get(new_status, -1)
        if force or (new_pri >= cur_pri and self._state.status != new_status):
            self._state.status = new_status

    # History helpers
    async def add_history_item(self, item: Any) -> None:
        async with self._lock:
            try:
                self._state.history.append(item)  # type: ignore[arg-type]
            except Exception:
                pass

    async def record_error(self, error_msg: str, is_critical: bool = False) -> None:
        async with self._lock:
            self._state.last_error = error_msg
            if is_critical:
                self._state.consecutive_failures += 1
                if self._state.consecutive_failures >= self.max_failures:
                    self._set_status_internal(AgentStatus.FAILED, force=True)

    async def update_task(self, new_task: str) -> None:
        async with self._lock:
            self._state.task = new_task
            self._state.current_goal = new_task

    async def set_current_goal(self, new_goal: str) -> None:
        async with self._lock:
            self._state.current_goal = new_goal

    async def clear_error_and_failures(self) -> None:
        async with self._lock:
            self._state.last_error = None
            self._state.consecutive_failures = 0

    async def mark_reflection_exit(self) -> None:
        async with self._lock:
            self._state.last_reflect_exit_ts = time.monotonic()

    async def add_human_guidance(self, text: str) -> None:
        try:
            await self._state.human_guidance_queue.put(text)
        except Exception:
            pass

    async def get_human_guidance(self) -> Optional[str]:
        try:
            guidance = await asyncio.wait_for(self._state.human_guidance_queue.get(), timeout=0.5)
            self._state.human_guidance_queue.task_done()
            return guidance
        except asyncio.TimeoutError:
            return None

    # Task stack
    async def push_task(self, task_id: str, description: str) -> None:
        async with self._lock:
            self._state.task_stack.push(task_id, description, self._state.n_steps)  # type: ignore[arg-type]

    async def pop_task(self) -> Optional[Task]:
        async with self._lock:
            return self._state.task_stack.pop()  # type: ignore[return-value]

    async def get_current_task_id(self) -> str:
        async with self._lock:
            cur = self._state.task_stack.current() if hasattr(self._state.task_stack, 'current') else None  # type: ignore[attr-defined]
            return getattr(cur, 'title', 'root') if cur else 'root'

    async def get_task_stack_summary(self) -> str:
        async with self._lock:
            if not self._state.task_stack:
                return "Task Context: root (main task)"
            try:
                return self._state.task_stack.compact_summary()
            except Exception:
                return "Task Context: root (main task)"

    # Signals and decisions
    async def ingest_signal(self, signal_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if not self.enable_modes:
            return
        payload = payload or {}
        async with self._lock:
            if signal_type == 'heartbeat_miss':
                self._state.missed_heartbeats += 1
            elif signal_type == 'heartbeat_ok':
                self._state.missed_heartbeats = 0
            elif signal_type == 'io_timeout':
                self._state.io_timeouts_recent.append(1)
                while len(self._state.io_timeouts_recent) > self._io_timeout_window:
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

            # Phase 4: Simplified - no longer using _TransitionEngine internally
            # All complex transition logic now handled by AgentOrchestrator

    async def get_load_status(self) -> LoadStatus:
        async with self._lock:
            return getattr(self._state, 'load_status', LoadStatus.NORMAL)

    async def set_load_status(self, new_status: LoadStatus) -> None:
        async with self._lock:
            old = getattr(self._state, 'load_status', LoadStatus.NORMAL)
            if old != new_status:
                self._state.load_status = new_status
                try:
                    agent_log(logging.WARNING, getattr(self._state, 'agent_id', '-'), getattr(self._state, 'n_steps', 0), f"System load status changed to: {new_status.value}")
                except Exception:
                    pass
                # Phase 4: Simplified - complex transitions now handled by AgentOrchestrator

    # --- Semantic Page State Management ---
    async def update_semantic_page_state(self, page_hash: str, semantic_flags: Dict[str, Any]) -> None:
        """Update semantic page state for enhanced perception."""
        import copy
        async with self._lock:
            self._state.last_semantic_page_hash = page_hash
            self._state.last_semantic_flags = copy.deepcopy(semantic_flags)

    async def get_semantic_page_state(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get current semantic page state."""
        import copy
        async with self._lock:
            if self._state.last_semantic_page_hash and self._state.last_semantic_flags:
                return (self._state.last_semantic_page_hash, copy.deepcopy(self._state.last_semantic_flags))
            return None

    # Phase 4: Removed _build_transition_inputs and _apply_decision methods
    # All complex transition logic now handled by AgentOrchestrator

    async def decide_and_apply_after_step(
        self,
        result,
        max_steps: int,
        step_completed: bool = True,
        oscillation_score: float | None = None,
        no_progress_score: float | None = None,
    ) -> StepOutcome:
        from browser_use.browser.views import BrowserStateHistory  # local import to avoid cycles
        from browser_use.agent.views import AgentHistory
        async with self._lock:
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
                # Append with bound if it's a deque
                if hasattr(self._state.history, 'append'):
                    self._state.history.append(history_item)  # type: ignore[arg-type]
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

            last_step_duration = 0.0
            try:
                if result.step_metadata is not None:
                    last_step_duration = float(getattr(result.step_metadata, 'duration_seconds', 0.0) or 0.0)
            except Exception:
                last_step_duration = 0.0

            # Phase 4: Simplified - basic state updates only
            # Complex decision logic now handled by AgentOrchestrator._assess_and_adapt()
            if task_completed:
                self._set_status_internal(AgentStatus.COMPLETED, force=True)
            elif had_failure and not any_success:
                if self._state.consecutive_failures >= self.max_failures:
                    self._set_status_internal(AgentStatus.FAILED, force=True)

            self._state.n_steps += 1

            return StepOutcome(
                status=self._state.status,
                modes=int(getattr(self._state, 'modes', 0)),
                reflection_intent=False,  # Phase 4: Orchestrator now handles reflection decisions
                task_completed=task_completed,
            )

