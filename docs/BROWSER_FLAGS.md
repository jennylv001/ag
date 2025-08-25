# Browser Module Flags and Env Vars

This document enumerates all configuration flags, environment variables, and Chrome arg groups in the `browser_use.browser` module, plus stealth behavior knobs and how to use them.

> Tip: You can also inspect these programmatically:
>
> - from browser_use.browser import summarize_flags, list_stealth_flags
> - summarize_flags() returns profile fields, session fields, chrome arg groups, and stealth categories.

## Configuration

Two primary flags control stealth behavior:

- BrowserProfile.stealth (bool): Use stealth mode with Patchright and human-like behaviors. Enables core actions (typing, scroll) and strict launch hygiene.
- BrowserProfile.advanced_stealth (bool): When true with stealth, turns on advanced behaviors: entropy timing, behavioral planning, page exploration, error simulation, and navigation.

Optional environment variables:
- STEALTH_RUN_SEED: Integer seed for deterministic variability across runs.
- PW_TEST_SCREENSHOT_NO_FONTS_READY=1: Improve screenshot stability for tests.
- BROWSERUSE_EVENT_LISTENER_TRACK: Track event listener injection (debug).

## BrowserProfile flags (selected)

All fields are available via `BrowserProfile.model_fields`. Commonly used:
- stealth (bool): Use stealth mode with Patchright and human-like behaviors.
- advanced_stealth (bool): One-line enablement of behavioral stealth defaults. When true, sensible STEALTH_* env defaults are applied unless already set by the user: entropy/planning/page exploration enabled; typing/scroll enabled; navigation disabled; error simulation disabled. Pairs best with `stealth=True`.
- disable_security (bool): Apply `CHROME_DISABLE_SECURITY_ARGS`.
- deterministic_rendering (bool): Apply `CHROME_DETERMINISTIC_RENDERING_ARGS` (not recommended; breaks sites).
- allowed_domains (list[str]): Limit navigation to these patterns.
- keep_alive (bool): Keep browser running after session stops.
- enable_default_extensions (bool): Load uBlock Origin, I-don't-care-about-cookies, ClearURLs.
- window_size, window_position: Headful window geometry.
- default_navigation_timeout, default_timeout: Timeouts (ms). Also `minimum_wait_page_load_time`, `wait_for_network_idle_page_load_time`, `maximum_wait_page_load_time`, `wait_between_actions`.
- include_dynamic_attributes, highlight_elements, viewport_expansion
- cookies_file (deprecated; use storage_state)
- user_data_dir: Persistent profile directory; increases stealth.
- channel/executable_path: Choose Chrome/Chromium binary; for stealth prefer Chrome stable.
- headless/no_viewport/viewport/screen/device_scale_factor: Display and rendering behavior.
- proxy/permissions/extra_http_headers/http_credentials/ignore_https_errors/java_script_enabled/base_url/service_workers
- user_agent/locale/geolocation/timezone_id/color_scheme/contrast/reduced_motion/forced_colors
- record_har_* and record_video_*: Recording options.
- traces_dir: Save Playwright traces on close.

Behavioral defaults when `stealth=True`:
- Locale defaults to en-US.
- Default automation extensions disabled (opt-in to re-enable).

Advanced behavioral set when `advanced_stealth=True` (no env required): entropy, behavioral planning, page exploration, error simulation, navigation are enabled.

Notes:
- The system no longer relies on per-feature STEALTH_* env variables. Behavior is derived from the two flags above.
- Recommended pairing: `stealth=True, advanced_stealth=True` for natural interactions.

See `browser/profile.py` for complete field list and descriptions.

## BrowserSession fields (selected)

Session metadata and runtime objects (Pydantic model):
- id
- browser_profile: The effective `BrowserProfile` (session overrides applied).
- wss_url, cdp_url, browser_pid: External connection targets.
- playwright, browser, browser_context: Runtime handles.
- initialized, agent_current_page, human_current_page

Private attrs track stealth counters and health; see `browser/session.py`.

## Chrome arg groups

- CHROME_DEFAULT_ARGS: Baseline arguments applied by default (with some overrides). Notable: disable background throttling, enable features, disable AutomationControlled, etc.
- CHROME_DOCKER_ARGS: Args for Docker/CI container stability.
- CHROME_HEADLESS_ARGS: Headless mode flags.
- CHROME_DISABLE_SECURITY_ARGS: Disable site isolation and SSL checks (for controlled environments only).
- CHROME_DETERMINISTIC_RENDERING_ARGS: Force deterministic rendering (can increase block rate).

Programmatic:
- from browser_use.browser import list_chrome_arg_groups

## Stealth behavior knobs

Environment toggles (see above) affect whether stealth engines engage. Additional knobs:

HumanProfile (dataclass, used by stealth engines):
- typing_speed_wpm, reaction_time_ms, motor_precision, impulsivity, tech_savviness
- deliberation_tendency, multitasking_ability, error_proneness
- movement_smoothness, overshoot_tendency, correction_speed

AgentBehavioralState tunables:
- max_history_length, confidence_adaptation_rate, stress_decay_rate

Engines and seeds:
- CognitiveTimingEngine / BiometricMotionEngine / HumanInteractionEngine support per-run seed (STEALTH_RUN_SEED) and per-action seeds internally; bounded entropy via STEALTH_ENTROPY.

Manager toggles:
- entropy_enabled: mirrors STEALTH_ENTROPY on initialization.

Feature usage
- Behavioral planning: enable STEALTH_BEHAVIORAL_PLANNING; optionally STEALTH_PAGE_EXPLORATION to include exploration steps; STEALTH_ERROR_SIMULATION to allow simulated mistakes.
- Typing/Navigation/Scroll: control with STEALTH_TYPE, STEALTH_NAVIGATION, STEALTH_SCROLL.

## Programmatic helpers

- list_profile_flags(): List of `BrowserProfile` field names.
- list_session_flags(): List of `BrowserSession` field names.
- list_chrome_arg_groups(): Dict of Chrome arg list groups.
- list_stealth_flags(): Dict of stealth categories (environment, human_profile_fields, behavioral_state_tunables, manager_toggles, engine_seed_controls).
- summarize_flags(): Combined snapshot including stealth.

## Quick reference

- Prefer Chrome stable and a persistent `user_data_dir` for best stealth.
- Avoid `deterministic_rendering=True` unless you need identical screenshots across OSes.
- Use `allowed_domains` to limit navigation scope.
- For headful stealth, set headless=False, no_viewport=True, window_size=screen.
- Set STEALTH_RUN_SEED to make stealth behavior reproducible, STEALTH_ENTROPY=true to add bounded variability.
