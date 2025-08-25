# System Model Manifest (Phase 1)

This document consolidates a full-system understanding of the project for downstream UI/UX and control-plane design. It includes an overview, component map, primary data models, and a definitive Master Configuration Table with names, types, locations, and purposes.

## Project Overview

- Language/stack: Python 3, Pydantic v2-style models, Playwright/Patchright (Chromium) for browser automation, Rich/Textual for TUI, MCP stdio server for tool integration.
- Key concerns: deterministic yet realistic browser control (stealth, behavior), robust session lifecycle (connect/launch/CDP/WSS), configuration via env + DB-style JSON, observability/logging.
- Entry points: CLI (`cli.py`) and MCP server (`mcp/server.py`) initialize `BrowserSession` with a `BrowserProfile`, LLM config, and agent config from `CONFIG`.
- Observability & logging: custom RESULT level, Windows-safe stream handler, optional debug decorators.

## Component Map

- browser/profile.py
  - Chrome arg bundles; Pydantic models for Launch/Connect/NewContext; `BrowserProfile` (global browser/profile config); arg composition and stealth filters; extension handling; display detection.
- browser/session.py
  - `BrowserSession`: lifecycle, connect/launch precedence, stealth feature computation, setup (viewports/listeners/tracing), ownership semantics.
- config.py
  - `CONFIG` singleton that merges env (FlatEnvConfig, OldConfig) with DB-style `config.json` (`DBStyleConfigJSON` with entries for browser profile/LLM/agent) + MCP overrides.
- cli.py
  - TUI/prompt flows, loads config, builds session and LLM, telemetry capture.
- mcp/server.py
  - MCP stdio server; tools for navigate/click/type/scroll/tabs; builds session objects and routes logging.
- logging_config.py
  - RESULT log level, `SafeStreamHandler`, convenience setup.
- observability.py
  - `observe_debug` decorator and optional tracing hooks.

## Data Models (Primary)

- Browser (profile and args)
  - BrowserContextArgs, BrowserConnectArgs, BrowserLaunchArgs, BrowserNewContextArgs, BrowserLaunchPersistentContextArgs
  - BrowserProfile (composes the above and adds project-specific fields)
- Session
  - BrowserSession (runtime container + orchestration)
- Config (DB-style + env)
  - DBStyleEntry, BrowserProfileEntry, LLMEntry, AgentEntry, DBStyleConfigJSON
  - FlatEnvConfig (env), OldConfig (compat), Config (dynamic facade)

## Master Configuration Table

Each row provides: Name, Type, Location (file:line), Inferred Purpose.

### BrowserProfile and inherited args (browser/profile.py)

- id — str — profile.py:561 — UUID identifier for profile instances.
- stealth — bool — profile.py:565 — Core stealth: strict CLI flag filtering, human-like defaults.
- advanced_stealth — bool — profile.py:566 — Behavioral stealth (planning/exploration/errors/navigation); implies `stealth` and auto-enables it.
- disable_security — bool — profile.py:575 — Relax site isolation/CSP/cert checks for dev/trusted usage.
- deterministic_rendering — bool — profile.py:576 — Force rendering determinism; increases block risk (warned).
- allowed_domains — list[str] | None — profile.py:577 — Navigation allowlist (wildcards/schemes allowed).
- keep_alive — bool | None — profile.py:581 — Keep the browser alive after agent run.
- enable_default_extensions — bool — profile.py:582 — Load uBlock, “I still don’t care about cookies”, ClearURLs (headful only).
- window_size — ViewportSize | None — profile.py:586 — Window size in headful.
- window_height — int | None — profile.py:590 — Deprecated; use window_size.height.
- window_width — int | None — profile.py:591 — Deprecated; use window_size.width.
- window_position — ViewportSize | None — profile.py:592 — Window X,Y position (headful).
- default_navigation_timeout — float | None — profile.py:598 — Default navigation timeout.
- default_timeout — float | None — profile.py:599 — Default Playwright call timeout.
- minimum_wait_page_load_time — float — profile.py:600 — Min wait before capturing page state.
- wait_for_network_idle_page_load_time — float — profile.py:601 — Idle wait time.
- maximum_wait_page_load_time — float — profile.py:602 — Max page load wait.
- wait_between_actions — float — profile.py:603 — Delay between actions.
- include_dynamic_attributes — bool — profile.py:606 — Include dynamic attributes in selectors.
- highlight_elements — bool — profile.py:608 — Draw on-DOM highlight overlays (off by default).
- overlay_highlights_on_screenshots — bool — profile.py:610 — Server-side screenshot overlays (no DOM mutation).
- overlay_max_items — int — profile.py:615 — Limit overlay clutter.
- viewport_expansion — int — profile.py:616 — LLM context viewport expansion.
- cookies_file — Path | None — profile.py:625 — Deprecated; prefer `storage_state`.

Inherited: BrowserLaunchArgs / BrowserLaunchPersistentContextArgs / BrowserNewContextArgs / BrowserConnectArgs

- env — dict — profile.py:409 — Extra env vars when launching.
- executable_path — str|Path|None — profile.py:413 — Custom Chromium/Chrome path for stealth.
- headless — bool|None — profile.py:418 — Headless/windowed preference (defaults via display detection).
- args — list[str] — profile.py:419 — Extra CLI args (merged last, highest precedence).
- ignore_default_args — list[str] | True — profile.py:422 — Exclude Playwright defaults.
- chromium_sandbox — bool — profile.py:432 — Enable sandbox (default True unless Docker).
- devtools — bool — profile.py:435 — Auto-open DevTools (headful only).
- slow_mo — float — profile.py:438 — Slowdown actions (ms).
- timeout — float — profile.py:439 — Connect timeout (ms).
- proxy — ProxySettings | None — profile.py:440 — Proxy configuration.
- downloads_path — str|Path|None — profile.py:441 — Download directory.
- traces_dir — str|Path|None — profile.py:446 — Playwright trace output dir.
- handle_sighup — bool — profile.py:451 — Signal handling for browser.
- handle_sigint — bool — profile.py:454 — Signal handling for browser.
- handle_sigterm — bool — profile.py:457 — Signal handling for browser.
- user_data_dir — str|Path|None — profile.py:531 — Persistent Chrome profile directory (or None for incognito temp).
- storage_state — str|Path|dict|None — profile.py (BrowserNewContextArgs) — Seed context storage (cookies/localStorage/etc.).
- headers — dict[str,str]|None — profile.py:385 — Connect headers.
- permissions — list[str] — profile.py:331 — Granted browser permissions.
- client_certificates — list[...] — profile.py:338 — Client TLS certificates for auth.
- extra_http_headers — dict[str,str] — profile.py:339 — Extra HTTP headers on context.
- http_credentials — HttpCredentials|None — profile.py (context args) — Basic auth.
- ignore_https_errors — bool — profile.py (context args) — Ignore HTTPS errors.
- java_script_enabled — bool — profile.py (context args) — Enable JS.
- base_url — UrlStr|None — profile.py (context args) — Base URL for relative navigation.
- service_workers — enum — profile.py (context args) — Allow/block service workers.
- user_agent — str|None — profile.py (context args) — Custom UA string.
- screen — ViewportSize|None — profile.py (context args) — Screen size (for viewport mode).
- viewport — ViewportSize|None — profile.py:349 — Viewport size.
- no_viewport — bool|None — profile.py (context args) — Disable viewport for full-window sizing.
- device_scale_factor — float|None — profile.py (context args) — Device pixel ratio (viewport mode only).
- is_mobile — bool — profile.py (context args) — Emulate mobile.
- has_touch — bool — profile.py (context args) — Enable touch capability.
- locale — str|None — profile.py (context args) — Locale (set by stealth defaults when None).
- geolocation — Geolocation|None — profile.py (context args) — Geolocation.
- timezone_id — str|None — profile.py (context args) — Time zone.
- color_scheme — enum — profile.py (context args) — Prefers-color-scheme.
- contrast — enum — profile.py (context args) — Prefers-contrast.
- reduced_motion — enum — profile.py (context args) — Prefers-reduced-motion.
- forced_colors — enum — profile.py (context args) — Forced-colors.
- record_har_content — enum — profile.py (context args) — HAR content handling.
- record_har_mode — enum — profile.py (context args) — HAR mode.
- record_har_omit_content — bool — profile.py (context args) — Omit content in HAR.
- record_har_path — str|Path|None — profile.py:366 — HAR file path.
- record_har_url_filter — str|Pattern|None — profile.py (context args) — HAR URL filter.
- record_video_dir — str|Path|None — profile.py:368 — Video directory.
- record_video_size — ViewportSize|None — profile.py (context args) — Video size.
- headers/slow_mo/timeout (connect/launch) — profile.py:385/438/439 — See above.

Validators and constraints (profile)
- validate_devtools_headless — profile.py — Asserts not (headless and devtools).
- ensure_advanced_stealth_implies_stealth — profile.py — Forces stealth=True if advanced_stealth=True.
- apply_stealth_defaults — profile.py — If stealth: locale default, headful preferred, no_viewport default True.
- warn_storage_state_user_data_dir_conflict — profile.py — Warn on conflicting persistence sources.
- warn_user_data_dir_non_default_version — profile.py — Avoid corrupting default dirs with non-default channels.
- warn_deterministic_rendering_weirdness — profile.py — Advises against deterministic rendering.
- detect_display_configuration — profile.py — Initializes screen/headless/window/viewport/dsf from environment; assert prevents headless+no_viewport.
- get_args — profile.py — CLI arg synthesis; strict stealth filtering (removes automation fingerprints, re-adds `--disable-blink-features=AutomationControlled`).

Chrome flag bundles (profile constants)
- CHROME_HEADLESS_ARGS — profile.py — Headless plumbing flags.
- CHROME_DOCKER_ARGS — profile.py — Docker-friendly flags (no sandbox/dev-shm etc.).
- CHROME_DISABLE_SECURITY_ARGS — profile.py — Weakens security (for dev/trust only).
- CHROME_DETERMINISTIC_RENDERING_ARGS — profile.py — Makes rendering deterministic.
- CHROME_DEFAULT_ARGS — profile.py — Baseline flags (includes disable-features bundle with components + our additions).

Key relationships (profile)
- headless ↔ viewport/no_viewport: headless ⇒ viewport; headful prefers window with no_viewport=True; assert prevents headless+no_viewport.
- security ↔ allowed_domains: disabling security increases risk; combine with allowlist cautiously.
- stealth ↔ advanced_stealth: advanced implies stealth; session derives features accordingly.
- extensions ↔ headless: default extensions skipped when headless.

### BrowserSession (browser/session.py)

- id — str — session.py:282 — Session identifier.
- browser_profile — BrowserProfile — session.py:285 — Base config; session applies extra kwargs onto a copy.
- wss_url — str|None — session.py:294 — Connect to Playwright browser server (WSS).
- cdp_url — str|None — session.py:298 — Connect to Chromium via CDP (http/ws).
- browser_pid — int|None — session.py:302 — Attach to local Chrome by PID.
- playwright — PlaywrightOrPatchright|None — session.py:307 — Runtime object (excluded).
- browser — Browser|None — session.py:312 — Runtime object (excluded).
- browser_context — BrowserContext|None — session.py:318 — Runtime object (excluded).
- initialized — bool — session.py:326 — Whether session setup finished.
- agent_current_page — Page|None — session.py:331 — Current page for agent (runtime; excluded).
- human_current_page — Page|None — session.py:337 — Current page for human (runtime; excluded).

Validators and behaviors (session)
- apply_session_overrides_to_profile — session.py — Merges extra kwargs into `browser_profile` copy.
- set_browser_ownership — session.py — Computes resource ownership from connect vs. launch.
- _compute_stealth_features — session.py — Feature map from `stealth` and `advanced_stealth`.
- start — session.py — Detects display, sets up playwright, connect/launch, and context setup.
- _connection_str — session.py — Describes connection backend; chooses patchright in stealth unless env overrides.

Constraints (session)
- Advanced features (entropy/planning/exploration/error_simulation/navigation) require both flags; base typing/scroll require `stealth`.
- Ownership becomes external if any of cdp_url, wss_url, browser, or browser_context is provided.

### CONFIG surfaces (config.py)

FlatEnvConfig (env)
- BROWSER_USE_LOGGING_LEVEL — str — config.py:199 — Logging level.
- ANONYMIZED_TELEMETRY — bool — config.py:200 — Telemetry on/off.
- BROWSER_USE_CLOUD_SYNC — bool|None — config.py:201 — Cloud sync toggle.
- BROWSER_USE_CLOUD_API_URL — str — config.py:202 — Cloud API base URL.
- BROWSER_USE_CLOUD_UI_URL — str — config.py:203 — Cloud UI URL.
- XDG_CACHE_HOME — str — config.py:206 — Cache base.
- XDG_CONFIG_HOME — str — config.py:207 — Config base.
- BROWSER_USE_CONFIG_DIR — str|None — config.py:208 — Override config dir.
- OPENAI_API_KEY — str — config.py:211 — LLM key.
- ANTHROPIC_API_KEY — str — config.py:212 — LLM key.
- GOOGLE_API_KEY — str — config.py:213 — LLM key.
- DEEPSEEK_API_KEY — str — config.py:214 — LLM key.
- GROK_API_KEY — str — config.py:215 — LLM key.
- NOVITA_API_KEY — str — config.py:216 — LLM key.
- AZURE_OPENAI_ENDPOINT — str — config.py:217 — Azure endpoint.
- AZURE_OPENAI_KEY — str — config.py:218 — Azure key.
- SKIP_LLM_API_KEY_VERIFICATION — bool — config.py:219 — Skip key verification.
- IN_DOCKER — bool|None — config.py:222 — Runtime hint for Docker.
- IS_IN_EVALS — bool — config.py:223 — Evaluation mode hint.
- WIN_FONT_DIR — str — config.py:224 — Windows font dir.
- BROWSER_USE_CONFIG_PATH — str|None — config.py:227 — Direct path to config.json.
- BROWSER_USE_HEADLESS — bool|None — config.py:228 — Env override for headless.
- BROWSER_USE_ALLOWED_DOMAINS — str|None — config.py:229 — CSV allowlist override.
- BROWSER_USE_LLM_MODEL — str|None — config.py:230 — LLM model override.

OldConfig (properties)
- BROWSER_USE_LOGGING_LEVEL — str — config.py:70 — Derived from env; lowercased.
- ANONYMIZED_TELEMETRY — bool — config.py:74 — Env-backed.
- BROWSER_USE_CLOUD_SYNC — bool — config.py:78 — Env-backed.
- BROWSER_USE_CLOUD_API_URL — str — config.py:82 — Validated URL.
- BROWSER_USE_CLOUD_UI_URL — str — config.py:88 — Optional URL (validated if set).
- XDG_CACHE_HOME — Path — config.py:97 — Path base.
- XDG_CONFIG_HOME — Path — config.py:101 — Path base.
- BROWSER_USE_CONFIG_DIR — Path — config.py:105 — Dir with autocreation.
- BROWSER_USE_CONFIG_FILE — Path — config.py:111 — Path to config.json.
- BROWSER_USE_PROFILES_DIR — Path — config.py:115 — Profiles dir.
- BROWSER_USE_DEFAULT_USER_DATA_DIR — Path — config.py:121 — Default profile dir.
- BROWSER_USE_EXTENSIONS_DIR — Path — config.py:125 — Extensions cache dir.
- OPENAI_API_KEY — str — config.py:143 — Env-backed.
- ANTHROPIC_API_KEY — str — config.py:147 — Env-backed.
- GOOGLE_API_KEY — str — config.py:151 — Env-backed.
- DEEPSEEK_API_KEY — str — config.py:155 — Env-backed.
- GROK_API_KEY — str — config.py:159 — Env-backed.
- NOVITA_API_KEY — str — config.py:163 — Env-backed.
- AZURE_OPENAI_ENDPOINT — str — config.py:167 — Env-backed.
- AZURE_OPENAI_KEY — str — config.py:171 — Env-backed.
- SKIP_LLM_API_KEY_VERIFICATION — bool — config.py:175 — Env-backed.
- IN_DOCKER — bool — config.py:180 — Env-backed + detection.
- IS_IN_EVALS — bool — config.py:184 — Env-backed.
- WIN_FONT_DIR — str — config.py:188 — Env-backed.

DB-style config schema
- BrowserProfileEntry — config.py:241 — Accepts any BrowserProfile fields; common: headless, user_data_dir, allowed_domains, downloads_path.
- LLMEntry — config.py:253 — api_key, model, temperature, max_tokens.
- AgentEntry — config.py:262 — max_steps, use_vision, system_prompt.
- DBStyleConfigJSON — config.py:270 — browser_profile, llm, agent maps keyed by UUIDs.

Loader behavior and overrides
- CONFIG.load_config() merges DB-style defaults and applies MCP env overrides:
  - BROWSER_USE_HEADLESS → browser_profile.headless
  - BROWSER_USE_ALLOWED_DOMAINS → browser_profile.allowed_domains (CSV parsed)
  - OPENAI_API_KEY → llm.api_key
  - BROWSER_USE_LLM_MODEL → llm.model

## Key Constraints and Relationships (Cross-cutting)

- Headless vs viewport/no_viewport: headless requires viewport; headful prefers window, no_viewport=True; assertion prevents invalid pairing.
- Security vs allowed_domains: disabling security broadens attack surface — pair with strict allowlists only in trusted scenarios.
- Stealth vs advanced_stealth: advanced implies stealth; session feature map depends on both.
- Extensions vs headless: default extensions disabled in headless to avoid instability.

---
End of Phase 1 (System Model Manifest).
