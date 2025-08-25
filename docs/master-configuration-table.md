# Master Configuration Table

This document enumerates the primary configuration surfaces used by browser-use, covering BrowserProfile, BrowserSession, and CONFIG (environment and DB-style config). It also captures key constraints and relationships (headless↔viewport, security↔allowed_domains, stealth↔advanced_stealth).

Note: Types shown are simplified; see source for exact unions and aliasing. Line numbers refer to current main branch files.

## BrowserProfile (browser/profile.py)

Key fields (name — type — default — file:line) and notes:
- id — str — uuid7str() — profile.py:561
- stealth — bool — False — profile.py:565
  - Notes: Enables strict CLI arg filtering; strips automation/test fingerprints.
- advanced_stealth — bool — False — profile.py:566
  - Notes: Implies stealth (validator ensures stealth=True if advanced_stealth=True). Enables behavioral features in session.
- disable_security — bool — False — profile.py:575
  - Notes: Adds insecure Chrome flags (CSP bypass, mixed content, cert ignores). Use only for trusted/dev.
- deterministic_rendering — bool — False — profile.py:576
  - Notes: Forces rendering determinism; warned as not recommended.
- allowed_domains — list[str] | None — None — profile.py:577
  - Notes: Restricts navigation; supports wildcards and schemes (incl. chrome-extension://).
- keep_alive — bool | None — None — profile.py:581
- enable_default_extensions — bool — True — profile.py:582
  - Notes: Downloads/loads uBlock Origin, I still don't care about cookies, ClearURLs (headful only).
- window_size — ViewportSize | None — None — profile.py:586
- window_height — int | None — deprecated — profile.py:590
- window_width — int | None — deprecated — profile.py:591
- window_position — ViewportSize | None — {width:0,height:0} — profile.py:592
- default_navigation_timeout — float | None — None — profile.py:598
- default_timeout — float | None — None — profile.py:599
- minimum_wait_page_load_time — float — 0.25 — profile.py:600
- wait_for_network_idle_page_load_time — float — 0.5 — profile.py:601
- maximum_wait_page_load_time — float — 5.0 — profile.py:602
- wait_between_actions — float — 0.5 — profile.py:603
- include_dynamic_attributes — bool — True — profile.py:606
- highlight_elements — bool — False — profile.py:608
- overlay_highlights_on_screenshots — bool — True — profile.py:610
- overlay_max_items — int — 60 — profile.py:615
- viewport_expansion — int — 500 — profile.py:616
- cookies_file — Path | None — None — profile.py:625 (deprecated; prefer storage_state)

Inherited and launch/connect/context fields (partial):
- env — dict — None — profile.py:409
- executable_path — str | Path | None — None — profile.py:413
- headless — bool | None — None — profile.py:418
- args — list[str] — [] — profile.py:419
- ignore_default_args — list[str] | True — [..] — profile.py:422
- channel — BrowserChannel | None — None — profile.py (BROWSERUSE_DEFAULT_CHANNEL default elsewhere)
- chromium_sandbox — bool — not CONFIG.IN_DOCKER — profile.py:432
- devtools — bool — False — profile.py:435
- slow_mo — float — 0 — profile.py:438
- timeout — float — 30000 — profile.py:439
- proxy — ProxySettings | None — None — profile.py:440
- downloads_path — str | Path | None — None — profile.py:441
- traces_dir — str | Path | None — None — profile.py:446
- handle_sighup — bool — True — profile.py:451
- handle_sigint — bool — False — profile.py:454
- handle_sigterm — bool — False — profile.py:457
- user_data_dir — str | Path | None — CONFIG.BROWSER_USE_DEFAULT_USER_DATA_DIR — profile.py:531
- storage_state — Any | None — None — profile.py (in BrowserNewContextArgs)

Validators and behaviors:
- validate_devtools_headless: prevents headless=True with devtools=True — profile.py (BrowserLaunchArgs)
- copy_old_config_names_to_new: maps window_width/height -> window_size — profile.py
- ensure_advanced_stealth_implies_stealth: auto-enables stealth when advanced_stealth=True — profile.py
- apply_stealth_defaults: when stealth=True, sets sensible defaults (locale=en-US if None, headless=False if None, no_viewport=True if None) — profile.py
- warn_storage_state_user_data_dir_conflict — profile.py
- warn_user_data_dir_non_default_version — profile.py
- warn_deterministic_rendering_weirdness — profile.py
- detect_display_configuration: populates screen/headless/window/viewport/device_scale_factor based on environment — profile.py
- get_args: composes Chrome CLI flags; stealth=True strips a strict set of flags and adds --disable-blink-features=AutomationControlled; also strips unsupported flags globally — profile.py

Key constraints/relationships:
- headless ↔ viewport/no_viewport: headless implies viewport enabled; headful sets window_size and typically no_viewport=True; validator asserts not (headless and no_viewport) — profile.py detect_display_configuration
- security ↔ allowed_domains: disable_security relaxes site isolation and security; allowed_domains restricts navigation (enforcement occurs in higher layers; use with caution together).
- stealth ↔ advanced_stealth: advanced implies stealth (validator). Session derives feature map from both flags.

## BrowserSession (browser/session.py)

Fields (name — type — default — file:line) and notes:
- id — str — uuid7str() — session.py:282
- browser_profile — BrowserProfile — DEFAULT_BROWSER_PROFILE — session.py:285
  - Notes: Extra kwargs passed to BrowserSession are applied onto a copy via apply_session_overrides_to_profile.
- wss_url — str | None — None — session.py:294
- cdp_url — str | None — None — session.py:298
- browser_pid — int | None — None — session.py:302
- playwright — PlaywrightOrPatchright | None — None — session.py:307 (runtime only; excluded)
- browser — Browser | None — None — session.py:312 (runtime only; excluded)
- browser_context — BrowserContext | None — None — session.py:318 (runtime only; excluded)
- initialized — bool — False — session.py:326
- agent_current_page — Page | None — None — session.py:331 (runtime; excluded)
- human_current_page — Page | None — None — session.py:337 (runtime; excluded)

Key validators/behaviors:
- apply_session_overrides_to_profile: merges extra kwargs into browser_profile copy — session.py
- set_browser_ownership: sets internal ownership flag based on connect vs. launch — session.py
- _compute_stealth_features: derives feature map from profile.stealth and profile.advanced_stealth — session.py
- start(): orchestrates connect/launch precedence; calls browser_profile.detect_display_configuration(); sets up viewports, listeners, tracing — session.py
- _connection_str: chooses driver name (patchright vs playwright) based on stealth and env; formats binary/channel — session.py

Constraints and relationships:
- Stealth features: base behaviors (type, scroll) require profile.stealth=True; advanced behaviors (entropy, planning, exploration, error simulation, navigation) require both stealth and advanced_stealth=True — session.py
- Ownership: If connecting via cdp_url/wss_url or provided browser/context, session doesn’t own resources — session.py

## CONFIG surfaces (config.py)

Environment and DB-style config fields that affect runtime:

FlatEnvConfig (env-driven):
- Logging/telemetry: BROWSER_USE_LOGGING_LEVEL (str, default info); ANONYMIZED_TELEMETRY (bool, default True); BROWSER_USE_CLOUD_SYNC (bool|None); BROWSER_USE_CLOUD_API_URL (str); BROWSER_USE_CLOUD_UI_URL (str)
- Paths: XDG_CACHE_HOME (str); XDG_CONFIG_HOME (str); BROWSER_USE_CONFIG_DIR (str|None)
- LLM keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY, GROK_API_KEY, NOVITA_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, SKIP_LLM_API_KEY_VERIFICATION (bool)
- Runtime hints: IN_DOCKER (bool|None); IS_IN_EVALS (bool); WIN_FONT_DIR (str)
- MCP overrides: BROWSER_USE_CONFIG_PATH (str|None); BROWSER_USE_HEADLESS (bool|None); BROWSER_USE_ALLOWED_DOMAINS (csv str|None); BROWSER_USE_LLM_MODEL (str|None)

OldConfig (computed/env-backed properties):
- Mirrors many of the above, plus derived paths and directory creators: XDG_CACHE_HOME, XDG_CONFIG_HOME, BROWSER_USE_CONFIG_DIR, BROWSER_USE_CONFIG_FILE, BROWSER_USE_PROFILES_DIR, BROWSER_USE_DEFAULT_USER_DATA_DIR, BROWSER_USE_EXTENSIONS_DIR; runtime hints (IN_DOCKER, IS_IN_EVALS), WIN_FONT_DIR.

DBStyleConfigJSON (config.json structure):
- browser_profile: dict[id -> BrowserProfileEntry]; common fields: headless, user_data_dir, allowed_domains, downloads_path
- llm: dict[id -> LLMEntry]; fields: api_key, model, temperature, max_tokens
- agent: dict[id -> AgentEntry]; fields: max_steps, use_vision, system_prompt

Loader/merger behavior:
- CONFIG.load_config(): returns merged dict {browser_profile, llm, agent} with MCP env overrides applied.
  - Overrides: BROWSER_USE_HEADLESS -> browser_profile.headless; BROWSER_USE_ALLOWED_DOMAINS -> browser_profile.allowed_domains; OPENAI_API_KEY -> llm.api_key; BROWSER_USE_LLM_MODEL -> llm.model.

Constraints and relationships:
- IN_DOCKER influences BrowserProfile.chromium_sandbox default (disabled in Docker).
- BROWSER_USE_CONFIG_PATH/DIR control where DB-style config is read/written; directories auto-created as needed.
- Headless override via env can interact with profile.detect_display_configuration logic; final truth resolved at session start.

## Cross-cutting constraints
- Headless vs Viewport: Not allowed to have headless=True with no_viewport=True (asserted in detect_display_configuration). Headless uses viewport; headful prefers window/no_viewport.
- Security vs Allowed Domains: Disabling security increases risk; combine with allowed_domains carefully and only in trusted environments.
- Stealth vs Advanced Stealth: advanced_stealth implies stealth; session uses both to compute feature map. Stealth strips specific Chrome flags and adds Blink AutomationControlled disable.
- Extensions vs Headless: Default extensions are skipped in headless mode to avoid stalls.

## Notes
- Exact CLI flags filtered under stealth include removal of “--enable-features=NetworkService,NetworkServiceInProcess” and any “--disable-features=” while re-adding “--disable-blink-features=AutomationControlled”.
- Deprecated fields: window_width/window_height, cookies_file (prefer storage_state).
