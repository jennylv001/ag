# Controller Module Flags and Config

This document summarizes the configuration knobs in the `browser_use.controller` module. Unlike the browser module, the controller does not use environment variables; configuration is passed via function parameters and model fields.

> Programmatic note: there is no dedicated helper yet to enumerate these. The types and defaults are stable and discoverable from the code in `controller/service.py` and `controller/registry/*`.

## Environment variables

- None. The controller layer has no `os.environ` usage.

## Controller runtime parameters

- Controller.__init__(exclude_actions: list[str] = [], output_model: type[T] | None = None, display_files_in_done_text: bool = True)
  - exclude_actions: Prevent selected actions from being registered (by function name).
  - output_model: Use structured final output (tool returns JSON from this model).
  - display_files_in_done_text: Whether to inline file contents into the final done message.

- Controller.act(...)
  - action: ActionModel
  - browser_session: BrowserSession (required for browser actions)
  - page_extraction_llm: BaseChatModel | None (used by content extraction tool)
  - sensitive_data: dict[str, str | dict[str, str]] | None
    - Two formats:
      - Legacy: {key: value}
      - Domain-scoped: {domain_pattern: {key: value}}
  - available_file_paths: list[str] | None (allowlist for file operations)
  - file_system: FileSystem | None
  - context: Any | None (user-provided context passed to actions)

- Controller.multi_act(...)
  - actions: list[ActionModel]
  - browser_session: BrowserSession
  - check_ui_stability: bool = True
    - Halts batch if the DOM element targeted by index changed or new elements appear between actions.
  - page_extraction_llm, sensitive_data, available_file_paths, file_system, context: same as in `act()`

## Action registration knobs

- Registry.action(description: str, param_model: type[BaseModel] | None = None, domains: list[str] | None = None, allowed_domains: list[str] | None = None, page_filter: Callable[[Any], bool] | None = None)
  - description: Natural language purpose shown to the LLM.
  - param_model: Explicit pydantic model for inputs (or inferred from function signature).
  - domains | allowed_domains (aliases): Glob patterns (e.g., `*.google.com`) to restrict action availability by URL.
  - page_filter: Callable(Page) -> bool to allow/deny action on a specific page.

Notes:
- `exclude_actions` prunes registrations by function name at startup.
- Param models are normalized and validated; special injected parameters (see below) are reserved.

## Special injected parameters (available to actions)

Defined in `controller/registry/views.py: SpecialActionParameters` and injected when actions request them:
- context: Any | None
- browser_session: BrowserSession | None (also legacy aliases: browser, browser_context)
- page: Page | None (playwright page shortcut when requested)
- page_extraction_llm: BaseChatModel | None
- file_system: FileSystem | None
- available_file_paths: list[str] | None
- has_sensitive_data: bool (auto true for `input_text` when sensitive data provided)

Reserved names: actions cannot re-use these as custom parameters.

## Sensitive data placeholders

- Placeholders in action params like `<secret>API_KEY</secret>` are replaced at runtime.
- Domain-scoped secrets apply only when the current URL matches the provided domain pattern.
- Logs used placeholders and warns on missing/empty ones.

## Result semantics and safety guards

- ActionResult.success defaults to False unless explicitly set by the action.
- `multi_act` prevents premature `done` if the previous action did not confirm success.
- `upload_file` enforces path allowlist and verifies UI acknowledgment to avoid false positives.

## Quick reference

- No env vars here; configure via Controller constructor, action registration filters, and call-time parameters.
- Use `exclude_actions` to hide actions globally.
- Use `domains`/`page_filter` to gate actions per page.
- Pass `sensitive_data` in domain-scoped form when possible (e.g., `{ "https://*.example.com": { "API_KEY": "..." } }`).
- Set `check_ui_stability=True` (default) to avoid acting on stale DOM after navigation or dynamic content loads.
