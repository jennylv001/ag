## Agent configuration flags and settings

This document inventories the configuration surface of the agent module. It captures the knobs exposed via the Pydantic models used at runtime and notes any environment variables (none used by the agent module directly). It mirrors the Browser and Controller configuration docs to provide full project visibility.

Scope
- AgentSettings: primary runtime configuration for the Agent and orchestrator
- MessageManagerSettings: prompt construction controls consumed by the MessageManager
- Long-running mode knobs: thresholds and behavior under sustained operation

Key points
- The agent module does not read environment variables directly; all configuration is provided via model fields at construction time.
- Planner interval is disabled by default (planner_interval=0); assessment-driven planning uses cooldowns and hysteresis.
- Long-running mode is opt-in (enable_long_running_mode=false by default) and configurable via dedicated fields.

### AgentSettings (browser_use.agent.settings.AgentSettings)

Core
- task: str — Required task description provided to the agent.
- llm: BaseChatModel — Primary LLM for decision making and action selection.
- controller: Controller = Controller() — Action registry and execution controller.
- browser_session: BrowserSession | None — Optional, reuse an existing session.
- browser / browser_context / browser_profile: Optional references to lower-level browser components when wiring manually.
- use_planner: bool = True — Enable planner/reflection loop.
- reflect_on_error: bool = True — Trigger planner on errors.
- planner_interval: int = 0 — Plan every N steps; 0 disables interval planning (default).
- planner_history_window: int = 5 — Steps of history considered for planner context.
- planner_interval_seconds: float = 0.0 — Time-based cadence; 0 disables.
- max_steps: int = 100 — Hard step cap; reaching it sets MAX_STEPS_REACHED.
- max_failures: int = 3 — Consecutive failure limit per run.
- max_actions_per_step: int = 7 — Safety cap for actions per decision.
- use_thinking: bool = True — Use <thinking> blocks in prompts where supported.
- flash_mode: bool = False — Use flash-optimized system prompts if enabled.
- planner_llm: BaseChatModel | None — Alternate LLM for planning.
- page_extraction_llm: BaseChatModel | None — Optional LLM for structured page extraction.
- injected_agent_state: AgentState | None — Seed state for advanced orchestration.
- context: Any | None — Arbitrary user context passed through to components.
- sensitive_data: dict[str, str|dict[str,str]] | None — Values to obfuscate in logs/prompts.
- initial_actions: list[dict] | None — Seed actions executed before the first decision.
- output_model: type[BaseModel] | None — Optional schema for the agent’s final output.

I/O, retry, and timing
- default_action_timeout_seconds: float = 60.0 — Default per action I/O timeout.
- action_timeout_overrides: dict[str, float] = {} — Per action-type overrides.
- site_profile_overrides: dict[str, float] = {} — Domain-pattern -> timeout seconds.
- max_attempts_per_action: int = 2 — Retries for an actuation batch.
- backoff_base_seconds: float = 1.0 — Exponential backoff base.
- backoff_jitter_seconds: float = 0.3 — Jitter added to backoff.
- max_perception_staleness_seconds: float = 10.0 — Acceptable age of perception state.
- lock_timeout_seconds: float = 30.0 — State lock acquisition timeout.
- shutdown_grace_seconds: float = 2.0 — Graceful shutdown delay.

History, files, and media
- available_file_paths: list[str|Path] = [] — Read/upload whitelist exposed to the agent.
- images_per_step: int = 1 — Images to include per step in user prompts.
- max_history_items: int = 40 — Sliding window cap for stored history in memory.
- include_attributes: list[str] — Attributes included in DOM element text (default: data-test-id, data-testid, aria-label, placeholder, title, alt).
- save_conversation_path: str | None — Persist conversation to path.
- file_system_path: str | None — Root path for sandboxed FS when enabled.
- file_system: Any — Injected FS implementation (populated by Supervisor).
- page: Page | None — Attach an existing page (advanced usage).
- generate_gif: bool | str = False — Enable recording; if str, path prefix.

Planner and vision
- is_planner_reasoning: bool = False — Encourage verbose planner reasoning.
- extend_planner_system_message: str | None — Extra system instructions.
- use_vision_for_planner: bool = False — Attach screenshots to planner prompts.
- planner_images_per_step: int = 1 — Number of recent screenshots for planner.
- planner_vision_detail: str = 'auto' — Vision detail: 'auto' | 'low' | 'high'.
- planner_use_latest_screenshot_only: bool = False — Only include most recent screenshot.
- calculate_cost: bool = False — Track token cost estimates for LLM calls.

Scheduler and assessor
- scheduler_enabled: bool = True — Lightweight scheduler heartbeat.
- scheduler_interval_seconds: float = 2.0 — Heartbeat interval.
- assessor_enabled: bool = True — Enable fused signal assessor.
- assessor_interval_seconds: float = 1.0 — Signal update cadence.
- assessor_risk_trigger: float = 0.65 — Risk threshold for reactive planning.
- assessor_risk_clear: float = 0.45 — Risk threshold to clear reactive bias.
- assessor_confidence_trigger: float = 0.35 — Confidence threshold to trigger planning.
- assessor_confidence_clear: float = 0.55 — Confidence threshold to enable proactive planning.
- assessor_cooldown_steps: int = 2 — Legacy step-based cooldown.
- assessor_cooldown_seconds: float = 2.0 — Seconds-based cooldown between assessment triggers.
- assessor_dwell_seconds: float = 0.5 — Signal must persist this long before triggering.

Rollout, QoS, and finalization
- enable_unified_finalization: bool = False — Use StateManager.decide_and_apply_after_step for transitions.
- enable_modes: bool = False — Enable health-aware modes and overlays.
- enable_prompt_deltas: bool = False — Placeholder for prompt delta optimization.
- enable_control_work_split: bool = False — Split event buses for lower-latency finalization.

Long-running mode (opt-in)
- enable_long_running_mode: bool = False — Enable long-running mode integration.
- long_running_monitoring_interval: float = 30.0 — Health monitoring cadence.
- long_running_checkpoint_interval: float = 300.0 — Periodic checkpoint cadence.
- long_running_checkpoint_dir: str | None — Override checkpoint directory (defaults to temp dir per agent id).
- long_running_max_checkpoints: int = 50 — Keep at most this many checkpoints.
- long_running_cpu_threshold_warning: float = 80.0 — CPU% for warning mode.
- long_running_cpu_threshold_critical: float = 95.0 — CPU% for critical mode.
- long_running_memory_threshold_warning: float = 80.0 — Memory% for warning mode.
- long_running_memory_threshold_critical: float = 95.0 — Memory% for critical mode.
- long_running_circuit_breaker_failure_threshold: int = 5 — Open circuit after this many failures.
- long_running_circuit_breaker_recovery_timeout: float = 60.0 — Seconds before attempting recovery.
- long_running_enable_auto_recovery: bool = True — Auto-recover from checkpoints on critical failures.
- long_running_enable_autonomous_continuation: bool = True — Continue autonomously after recovery.
- long_running_max_consecutive_failures: int = 3 — Escalate after this many failures in a row.
- long_running_failure_escalation_delay: float = 120.0 — Delay before escalating repeated failures.
- cpu_shed_threshold: float = 98.0 — Enter load shedding at/above this CPU%.
- cpu_normal_threshold: float = 96.0 — Exit shedding when CPU% falls below this.
- shed_policy: dict[str, Any] = {} — Shedding policy knobs: {perception_skip_n, cap_images, defer_planner}.

Hooks
- on_run_start|on_step_start|on_step_end: Optional[Callable[[Agent], Awaitable[None]]] — Lifecycle hooks.
- on_run_end: Optional[Callable[[AgentHistoryList], Awaitable[None]]] — Finalization hook.

Other execution safety
- check_ui_stability: bool = True — Reject brittle DOM selections during actuation when unstable.

Notes
- AgentSettings fields are type-validated by Pydantic. Construction errors raise AgentConfigurationError where applicable (e.g., invalid initial_actions schema).
- No environment variables are consumed directly by the agent module.

### MessageManagerSettings (browser_use.agent.message_manager.views.MessageManagerSettings)

Prompt construction and truncation controls
- max_input_tokens: int = 128000 — Upper bound for prompt token budget.
- max_clickable_elements_length: int = 12000 — Max characters from clickable DOM text.
- include_attributes: list[str] = DEFAULT_INCLUDE_ATTRIBUTES — Attributes included when serializing clickable elements.
- message_context: str | None — Context appended to the system prompt.
- available_file_paths: list[str] = [] — Files made visible to the LLM via prompt.
- max_history_items: int | None = 10 — Cap history items included in the prompt.
- max_history_for_planner: int | None = 5 — History steps reserved for planner prompts.
- images_per_step: int = 1 — Number of screenshots per step in prompts.
- use_vision: bool = True — Include screenshots for the main user prompt.
- use_vision_for_planner: bool = False — Include screenshots in planner prompts.
- use_thinking: bool = True — Instruct LLM to use a <thinking> block.
- image_tokens: int = 850 — Estimated tokens per image (for budget calculations).
- recent_message_window_priority: int = 5 — Recent turns to prioritize in truncation.

### Environment variables

None used directly by the agent module. All configuration is passed via the models above. Browser- and stealth-related environment variables are documented separately in docs/BROWSER_FLAGS.md.

### Programmatic enumeration

You can enumerate agent flags directly from the Pydantic models:
- AgentSettings.model_fields
- MessageManagerSettings.model_fields

Each field provides name, type annotation, default, and description metadata.
