# Phase 3 — UI/UX Architecture & Wireframing

This document proposes the app’s information architecture, text-based wireframes, and a component inventory. Assumption: Fluent UI v9 (Fluent 2) for component naming and patterns.

## Information Architecture (Site Map)

- Dashboard
  - Live Job Tracker
  - Recent Sessions & Status
  - Quick Actions (Run Agent, Attach, New Profile)
  - Health & Alerts (environment checks)
- Sessions
  - Session Control Panel (connect/launch/CDP/WSS/PID)
  - Stealth Feature Matrix (read-only chips)
  - Active Pages & Contexts (open tabs, quick actions)
  - Artifacts (HAR, video, traces)
- Profiles
  - Profiles Catalog (list, duplicate, set default)
  - Profile Builder (tabs: Stealth, Security, Display/Viewport, Network/Proxy, Recording/Tracing, Persistence, Launch Args)
  - Chrome Flag Inspector (computed args, with explanations)
  - Extension Manager (default extensions state)
- Config
  - Config Sources Manager (DB-style vs Env vs Runtime Overrides with precedence)
  - LLM Config (api key, model, params)
  - Agent Config (max steps, vision, prompt)
- Observability
  - Logging Center (level, filters, RESULT)
  - Debug Trace Viewer (observe_debug traces)
  - Telemetry snapshot
- Tools (MCP)
  - Actions Panel (navigate, click, type, scroll, tabs)
  - stderr-only log stream

## Text-Based Wireframes

Note: These are schematic blueprints—not pixel layouts.

### Dashboard

```
[Header: App Title | Profile Selector | Run Agent (Primary Button) | Logging Level]
--------------------------------------------------------------
| Sidebar                       | Main                         |
| - Dashboard (active)          | [Row]
| - Sessions                    |  - Live Job Tracker (DataGrid):
| - Profiles                    |    • Job ID • Status • Profile • Started • Duration • Actions
| - Config                      |
| - Observability               | [Row]
| - Tools                       |  - Recent Sessions (DataGrid compact)
|                               |
|                               | [Row]
|                               |  - Quick Actions: Run Agent, New Profile, Attach to Browser
|                               |
|                               | [Row]
|                               |  - Health & Alerts: IN_DOCKER, Fonts, Extensions Cache
--------------------------------------------------------------
[Footer: Status bar (connected/disconnected) | Version]
```

### Sessions — Session Control Panel

```
[Header: Sessions | New Session]
--------------------------------------------------------------
| Filters: Status [Dropdown] | Text search | Clear
|
| Sessions List (DataGrid): ID | State | Ownership | Profile | Entry | Actions
|
| [Split Panel]
|  Left: Session Details
|   - Connection: [Tabs: Launch | Connect]
|     • Launch: channel/executable_path/headless/devtools/proxy/timeout
|     • Connect: wss_url/cdp_url/browser_pid/headers
|   - Controls: Start, Stop, Open DevTools (headful)
|   - Ownership Badge: Internal/External
|
|  Right: Runtime View
|   - Stealth Feature Matrix (Chips): typing | scroll | nav | exploration | errors
|   - Active Pages (List): URL | Title | Open in Browser | Screenshot
|   - Artifacts: HAR/Video/Trace (download links)
--------------------------------------------------------------
```

### Profiles — Catalog + Builder

```
[Header: Profiles | New Profile]
--------------------------------------------------------------
| Left: Profiles Catalog (DataGrid)
|  • Name/ID | Default | Last Modified | Actions (Edit, Duplicate, Delete)
|
| Right: Profile Builder (Tabs)
|  [Tabs]
|   - Stealth
|     • Switch: stealth
|     • Switch: advanced_stealth (auto-enables stealth)
|     • Info: feature matrix summary
|
|   - Security
|     • Switch: disable_security (Warning)
|     • Allowed Domains (Tag Input)
|     • Permissions (Multiselect)
|     • HTTP Auth (Username/Password)
|     • Client Certificates (Repeater: name + file)
|     • Extra Headers (Key/Value)
|
|   - Display / Viewport
|     • Switch: headless
|     • Window Size (SpinButtons w/h) | Position (SpinButtons x/y)
|     • Viewport Size (conditional) | Device Scale Factor (SpinButton)
|     • Locale/UA/Timezone (Inputs)
|
|   - Network / Proxy
|     • Proxy Editor (server, bypass, creds)
|
|   - Recording / Tracing
|     • HAR: mode/content/omit content (Dropdown + Switch)
|     • HAR Path (File Picker) | URL Filter (Text/Regex)
|     • Video Dir (Folder) | Size (SpinButtons w/h)
|     • Traces Dir (Folder)
|
|   - Persistence
|     • user_data_dir (Folder)
|     • storage_state (File/JSON Editor toggle)
|     • keep_alive (Switch)
|
|   - Launch Args
|     • env (Key/Value) | args (List Editor w/ validation)
|     • ignore_default_args (Multi-select)
|     • chromium_sandbox/devtools/slow_mo/timeout/handle_* (Switch/SpinButtons)
|
|  Side Panel: Chrome Flag Inspector
|    • Computed CLI args (monospace list)
|    • Tooltips: rationale per flag
|    • Copy to clipboard
--------------------------------------------------------------
```

### Config — Sources & Overrides

```
[Header: Config]
--------------------------------------------------------------
| Sources Precedence: [DB-style] > [Env] > [Runtime]
|
| [Accordion]
|  - Browser Profile Overrides (Env)
|    • BROWSER_USE_HEADLESS (Switch)
|    • BROWSER_USE_ALLOWED_DOMAINS (Tag/CSV)
|
|  - LLM
|    • OPENAI_API_KEY (Secret) | BROWSER_USE_LLM_MODEL (Text)
|    • temperature/max_tokens (SpinButtons)
|
|  - Agent
|    • max_steps (SpinButton) | use_vision (Switch) | system_prompt (Textarea)
|
| Paths
|  • XDG_CACHE_HOME / XDG_CONFIG_HOME (Folder Pickers)
|  • BROWSER_USE_CONFIG_DIR / ..._PATH (Pickers)
|
| Diagnostics
|  • IN_DOCKER (Badge) / IS_IN_EVALS (Switch)
--------------------------------------------------------------
```

### Observability — Logs & Traces

```
[Header: Observability]
--------------------------------------------------------------
| Toolbar: Level (Dropdown) | Filters (Text) | Pause | Clear | Export
|
| Log Stream (Virtualized DataGrid): time | level | source | message
|
| Traces Panel
|  • observe_debug events timeline
|  • Select run → details (duration, args, outputs)
--------------------------------------------------------------
```

### Tools — MCP Actions

```
[Header: Tools]
--------------------------------------------------------------
| Connection Status (Badge) | Connect/Disconnect
|
| Actions Panel
|  • Navigate (URL Input + Button)
|  • Click (Selector Input)
|  • Type (Selector + Text)
|  • Scroll (Selector + amount)
|  • Tabs (List + activate/close)
|
| stderr Log (read-only console)
--------------------------------------------------------------
```

## Component Library (Fluent UI v9 / Fluent 2)

Foundations
- AppLayout primitives: Header, Sidebar, Content, Footer (layout components)
- Tabs, Accordion, Toolbar, Breadcrumb, Card, Divider
- DataGrid (virtualized table), List, Tree
- Button (primary/secondary), IconButton, SplitButton
- Switch, Checkbox, RadioGroup, Slider
- Input (text), Textarea, Combobox/Dropdown, Tag (chips), Badge, Tooltip, Popover
- SpinButton (numeric input), DatePicker (if needed), ProgressBar, Spinner
- Dialog (modal), Drawer/Panel (side sheet), Toast (notifications)
- FilePicker/FolderPicker (implemented via Input[type=file]/directory picker wrapper)

Specialized Patterns
- KeyValueEditor (key/value grid using DataGrid + Inputs)
- ListEditor (string list with add/remove)
- FlagInspector (monospace list + tooltips + copy)
- StealthFeatureMatrix (Tag/Badge group)
- CodeBlock (monospace, selectable)

Per Screen Usage
- Dashboard: DataGrid, Card, Badge, Button, Toolbar
- Sessions: DataGrid, Tabs, Switch, Inputs, SpinButtons, Tag, Badge, Button, Dialog
- Profiles: Tabs, Switch, SpinButtons, Dropdown, Combobox, KeyValueEditor, ListEditor, File/Folder pickers, FlagInspector, Tooltip
- Config: Accordion, Inputs, Switch, SpinButtons, SecretInput (Input with mask), File/Folder pickers, Badge
- Observability: DataGrid (virtualized), Toolbar, Dropdown, Button, Spinner, CodeBlock
- Tools: Inputs, Buttons, List, Badge, CodeBlock

## Interaction & Validation Rules Embedded in UI
- Headless vs viewport/no_viewport: enforce valid combinations; disable conflicting controls with inline hints.
- Advanced stealth implies stealth: auto-check stealth when advanced_stealth is on; disallow unchecking stealth while advanced is on.
- Extensions vs headless: disable extension manager in headless with rationale.
- Security disablement: warning badge and confirm; recommend pairing with allowed_domains.
- storage_state vs user_data_dir: show conflict warning; suggest ephemeral context helper.
- Deterministic rendering: danger tooltip; confirm before enabling.

## Notes
- All computed values (e.g., CLI flags) should be reproducible via a single “Copy profile JSON” and “Copy launch flags” action.
- Tables should support column chooser, resize, and CSV export for support workflows.
