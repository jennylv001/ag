from __future__ import annotations
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar, Union
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict
from pydantic import ValidationError

from browser_use.agent.views import AgentHistoryList
from browser_use.agent.state import AgentState
from browser_use.browser import BrowserProfile
from browser_use.browser.session import BrowserSession
from browser_use.browser.types import Browser, BrowserContext, Page
from browser_use.controller.service import Controller
from browser_use.llm.base import BaseChatModel
from browser_use.exceptions import AgentConfigurationError

from typing import TYPE_CHECKING, Literal

# Avoid runtime forward-ref to Agent to prevent Pydantic "class-not-fully-defined" error.
# Preserve precise typing for editors/type-checkers only.
if TYPE_CHECKING:
    from browser_use.agent.service import Agent  # noqa: F401 (type-checking only)
    AgentHookFunc = Callable[['Agent'], Awaitable[None]]
else:
    AgentHookFunc = Callable[[Any], Awaitable[None]]

AgentDoneHookFunc = Callable[['AgentHistoryList'], Awaitable[None]]

class AgentSettings(BaseModel):
    task: str
    llm: BaseChatModel
    controller: Controller = Field(default_factory=Controller)
    browser_session: Optional[BrowserSession] = None
    # Deprecation wall (Task 0)
    deprecation_mode: bool = Field(True, description="When true, legacy/deprecated modules are hard-disabled by default.")
    break_glass_allow: list[str] = Field(default_factory=list, description="Explicit allowlist for deprecated modules (requires env AGENT_BREAK_GLASS).")
    use_planner: bool = True
    # Scoped task planner (CDAD) feature flag
    use_task_planner: bool = Field(False, description="Enable CDAD task planner to suggest BaseTask sequences (off by default)")
    # Explicit Task Layer enablement: shows Task Catalog and allows task usage in normal operations
    # Independent of CDAD planner; if None, it falls back to use_task_planner for backward compatibility
    task_layer_enabled: Optional[bool] = Field(
        None,
        description="Enable task layer (catalog + task usage) even if planner is off; defaults to use_task_planner when unset."
    )
    # Reflection/Replanning rollout toggle
    use_replanning: bool = Field(
        False,
        description=(
            "Enable reflection/replanning triggers (on failure, stagnation, or cadence). "
            "Off by default to avoid behavior drift."
        ),
    )
    reflect_on_error: bool = True
    planner_interval: int = 0
    # Planner cadence/frequency control (default: every_step to preserve legacy behavior)
    planner_frequency: Literal["every_step", "startup", "on_change", "interval", "never"] = Field(
        "every_step",
        description=(
            "When to run the CDAD planner: 'every_step' | 'startup' (first step only) | 'on_change' (URL/title/page changes) | "
            "'interval' (respect planner_interval/planner_interval_seconds) | 'never' (disabled while keeping task catalog)."
        ),
    )
    # Reflection cadence gate used by TransitionEngine
    reflect_cadence: int = Field(0, description="If >0, require every N steps before allowing reflection/planner triggers.")
    planner_history_window: int = 5
    planner_interval_seconds: float = Field(0.0, description="If >0, time-based cadence for planner triggers.")
    max_steps: int = 100
    max_failures: int = 3
    max_actions_per_step: int = 7
    use_thinking: bool = True
    flash_mode: bool = False
    planner_llm: Optional[BaseChatModel] = None
    page_extraction_llm: Optional[BaseChatModel] = None
    injected_agent_state: Optional[AgentState] = None
    context: Optional[Any] = None
    sensitive_data: Optional[Dict[str, Union[str, Dict[str, str]]]] = None
    initial_actions: Optional[List[Dict[str, Any]]] = None
    available_file_paths: list[Union[str, Path]] = Field(default_factory=list)
    images_per_step: int = 1
    max_history_items: int = 40
    # Task 14: Guardrails and caps (defaults conservative)
    cap_elements_per_snapshot: int = Field(500, description="Hard cap for elements returned by perception.")
    cap_actions_per_step: int = Field(3, description="Hard cap for actions emitted per step.")
    cap_llm_tokens_prompt: int = Field(8000, description="Soft cap for prompt tokens; can clip context.")
    cap_retries_total: int = Field(6, description="Total retry budget across the run; excess attempts clipped.")
    # I/O timeout & retry controls
    default_action_timeout_seconds: float = Field(60.0, description="Default timeout for browser I/O actions.")
    action_timeout_overrides: Dict[str, float] = Field(default_factory=dict, description="Per-action-type timeout overrides in seconds. Keys are action class names or logical action types.")
    site_profile_overrides: Dict[str, float] = Field(default_factory=dict, description="Domain pattern -> timeout seconds overrides (e.g., {'upload.example.com': 180.0}).")
    max_attempts_per_action: int = Field(2, description="Max retry attempts per actuation batch on timeout/IO failure.")
    backoff_base_seconds: float = Field(1.0, description="Base backoff seconds for exponential retry.")
    backoff_jitter_seconds: float = Field(0.3, description="Jitter seconds added to retry backoff.")
    # Scoped timeout guard for explicit wait actions: effective timeout = requested seconds + guard
    wait_timeout_guard_seconds: float = Field(
        5.0,
        description=(
            "Additional guard seconds added to the wait action's I/O timeout. "
            "Prevents premature timeout by extending per-call timeout to (requested seconds + guard)."
        ),
    )
    # Reflection controls
    reflect_cooldown_seconds: float = Field(0.0, description="Minimum seconds to wait after a reflection before allowing another, unless failures persist or health modes force it.")
    include_attributes: list[str] = Field(default_factory=lambda: ["data-test-id", "data-testid", "aria-label", "placeholder", "title", "alt"])
    # Perception controls (Task 5)
    max_actionable: int = Field(150, description="Cap for actionable elements sampled from DOM.")
    max_landmarks: int = Field(20, description="Cap for landmark/heading elements included in semantic page.")
    use_accessibility_tree: bool = Field(True, description="Merge Playwright accessibility tree (interesting_only) into DOM sampling.")
    perception_min_quiet_ms: int = Field(150, description="Minimum quiet time (ms) to wait before capturing to avoid thrash.")
    perception_timeout_ms: int = Field(2000, description="Maximum time (ms) to spend capturing perception snapshot.")
    include_tool_call_examples: bool = False
    on_run_start: Optional[AgentHookFunc] = None
    on_step_start: Optional[AgentHookFunc] = None
    on_step_end: Optional[AgentHookFunc] = None
    on_run_end: Optional[AgentDoneHookFunc] = None
    generate_gif: Union[bool, str] = False
    save_conversation_path: Optional[str] = None
    file_system_path: Optional[str] = None
    page: Optional[Page] = None
    max_perception_staleness_seconds: float = 10.0
    lock_timeout_seconds: float = Field(30.0, description="Timeout in seconds for acquiring the state lock to prevent deadlocks.")
    memory_budget_mb: float = Field(100.0, description="Memory budget in MB for state/history. When exceeded, old items are summarized and pruned.")
    max_concurrent_io: int = Field(3, description="Maximum number of concurrent I/O operations (browser actions and LLM calls).")
    max_concurrent_tasks: int = Field(3, description="Maximum number of concurrent tasks to prevent tab-sprawl and maintain focus.")
    check_ui_stability: bool = True
    output_model: Optional[type[BaseModel]] = None
    browser: Optional[Union[Browser, BrowserSession]] = None
    browser_context: Optional[BrowserContext] = None
    browser_profile: Optional[BrowserProfile] = None
    # Search preferences (must be explicitly configured; no hard-coded default)
    default_search_engine: str | None = Field(None, description="Default search engine for search actions: one of 'duckduckgo' | 'google' | 'bing'.")
    file_system: Any = None # To be populated by Supervisor
    is_planner_reasoning: bool = Field(False, description="Controls if the planner prompt encourages verbose reasoning.")
    extend_planner_system_message: Optional[str] = Field(None, description="Additional text for the planner's system message.")
    # Planner vision controls
    use_vision_for_planner: bool = Field(False, description="Include screenshots in planner prompts.")
    planner_images_per_step: int = Field(1, description="Number of recent screenshots to include in planner prompts (if available).")
    planner_vision_detail: str = Field('auto', description="Vision detail level for planner images: 'auto' | 'low' | 'high'.")
    planner_use_latest_screenshot_only: bool = Field(False, description="When true, include only the most recent screenshot to minimize token usage.")
    calculate_cost: bool = Field(False, description="Whether to calculate and track token costs for LLM calls.")

    # LLM caller controls (Task 6)
    llm_timeout_seconds: float = Field(30.0, description="Default timeout for LLM calls (seconds).")
    llm_max_retries: int = Field(2, description="Max retries for LLM calls on timeout/LLMException.")
    llm_backoff_base_seconds: float = Field(1.0, description="Base seconds for exponential backoff for LLM retries.")
    llm_backoff_jitter_seconds: float = Field(0.3, description="Jitter seconds added to LLM retry backoff.")

    # Human-in-the-loop convenience
    pause_on_first_click: bool = Field(
        False,
        description="If true, automatically pause after the very first click action and prompt for human guidance (Enter=resume; type to inject guidance; Ctrl+C again to exit).",
    )
    long_running_monitoring_interval: float = Field(30.0, description="Interval in seconds for long-running mode health monitoring.")
    long_running_checkpoint_interval: float = Field(300.0, description="Interval in seconds for automatic state checkpointing.")
    long_running_checkpoint_dir: Optional[str] = Field(None, description="Directory for storing state checkpoints. Uses temp dir if None.")
    long_running_max_checkpoints: int = Field(50, description="Maximum number of checkpoints to retain.")
    long_running_cpu_threshold_warning: float = Field(80.0, description="CPU usage percentage that triggers warning mode.")
    long_running_cpu_threshold_critical: float = Field(95.0, description="CPU usage percentage that triggers critical mode.")
    long_running_memory_threshold_warning: float = Field(80.0, description="Memory usage percentage that triggers warning mode.")
    long_running_memory_threshold_critical: float = Field(95.0, description="Memory usage percentage that triggers critical mode.")
    long_running_circuit_breaker_failure_threshold: int = Field(5, description="Number of failures before opening circuit breaker.")
    long_running_circuit_breaker_recovery_timeout: float = Field(60.0, description="Seconds before attempting to close circuit breaker.")
    long_running_enable_auto_recovery: bool = Field(True, description="Enable automatic recovery from checkpoints on critical failures.")
    long_running_enable_autonomous_continuation: bool = Field(True, description="Enable autonomous task continuation after failure recovery.")
    long_running_max_consecutive_failures: int = Field(3, description="Maximum consecutive failures before requiring manual intervention.")
    long_running_failure_escalation_delay: float = Field(120.0, description="Seconds to wait before escalating repeated failures.")

    # Load shedding configuration
    cpu_shed_threshold: float = Field(98.0, description="CPU% at or above which to enter shedding mode.")
    cpu_normal_threshold: float = Field(96.0, description="CPU% at or below which to return to normal mode.")
    shed_policy: Dict[str, Any] = Field(default_factory=dict, description="Policy knobs when shedding: {perception_skip_n:int, cap_images:int, defer_planner:bool}.")

    # Cooperative shutdown
    shutdown_grace_seconds: float = Field(2.0, description="Grace period for components to wind down on shutdown.")
    # Perception flags (Task 1)
    enable_semantic_map: bool = Field(False, description="Enable experimental DOM+AX merged Semantic Map (read-only pathway).")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Single-flag cascade helpers ---
    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        # Backward compatibility: if task_layer_enabled is unset, mirror use_task_planner
        if getattr(self, 'task_layer_enabled', None) is None:
            try:
                self.task_layer_enabled = bool(getattr(self, 'use_task_planner', False))
            except Exception:
                self.task_layer_enabled = False

    @property
    def replanning_enabled(self) -> bool:
        """Cascade replanning to follow the single planner flag by default.

        Keeping the original field for backward compatibility, but when the single flag is on,
        we treat replanning as enabled regardless of the legacy toggle value.
        """
        if bool(getattr(self, 'use_task_planner', False)):
            return True
        return bool(getattr(self, 'use_replanning', False))

    def parse_initial_actions(self, action_model: Any) -> list[Any]:
        if not self.initial_actions:
            return []
        try:
            return [action_model.model_validate(a) for a in self.initial_actions]
        except ValidationError as e:
            raise AgentConfigurationError(f"Validation error for initial actions: {e}")
