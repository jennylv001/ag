import asyncio
import enum
import json
import logging
import os
import re
import time
from typing import Generic, TypeVar, cast, Optional

try:
    from lmnr import Laminar  # type: ignore
except ImportError:
    Laminar = None  # type: ignore
# Lightweight retry decorator to avoid external dependency on bubus.helpers.retry
from functools import wraps
def retry(wait: float = 0.5, retries: int = 3, timeout: float | None = None):
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(retries):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    if wait:
                        await asyncio.sleep(wait)
            raise last_err
        return wrapper
    return decorator
from pydantic import BaseModel, Field

from browser_use.agent.views import ActionModel, ActionResult
from browser_use.browser import BrowserSession
from browser_use.browser.types import Page
from browser_use.browser.views import BrowserError
from browser_use.controller.registry.service import Registry
from browser_use.controller.views import (
    ClickElementAction,
    CloseTabAction,
    DoneAction,
    ExtractToMemoryAction,
    GoToUrlAction,
    InputTextAction,
    NoParamsAction,
    ScrollAction,
    FillFromMemoryAction,
    SearchGoogleAction,
    SendKeysAction,
    StructuredOutputAction,
    SwitchTabAction,
    UploadFileAction,
)
from browser_use.filesystem.file_system import FileSystem
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import UserMessage
from browser_use.observability import observe_debug
from browser_use.utils import time_execution_sync

logger = logging.getLogger(__name__)


Context = TypeVar('Context')

T = TypeVar('T', bound=BaseModel)


# Module-level resilient frame.evaluate helper for iframe/dropdown interactions
async def _safe_frame_evaluate(_frame, _script: str, _arg=None, retries: int = 3):
    attempt = 0
    last_exc: Exception | None = None
    while attempt < retries:
        try:
            if _arg is None:
                return await _frame.evaluate(_script)
            return await _frame.evaluate(_script, _arg)
        except Exception as e:
            msg = str(e).lower()
            last_exc = e
            # Common transient errors during navigation/context loss or frame detachments
            transient = (
                'execution context was destroyed' in msg
                or 'frame was detached' in msg
                or 'navigation' in msg
                or 'cannot find context with specified id' in msg
            )
            if transient:
                try:
                    await asyncio.sleep(0.15 * (attempt + 1))
                except Exception:
                    pass
                attempt += 1
                continue
            # Hard error – bubble up immediately
            raise
    raise last_exc if last_exc else RuntimeError('frame evaluate failed')


class Controller(Generic[Context]):
    def __init__(
        self,
        exclude_actions: list[str] = [],
        output_model: type[T] | None = None,
        display_files_in_done_text: bool = True,
    ):
        self.registry = Registry[Context](exclude_actions)
        self.display_files_in_done_text = display_files_in_done_text

        """Register all default browser actions"""

        self._register_done_action(output_model)

        # Basic Navigation Actions
        # Unified search action respecting default_search_engine if present
        @self.registry.action(
            'Search the query on the web using the configured search engine.',
            param_model=SearchGoogleAction,
        )
        async def search_google(
            params: SearchGoogleAction,
            browser_session: BrowserSession,
        ):
            # Determine target engine (must be provided via settings.controller.default_search_engine)
            engine = getattr(self, 'default_search_engine', None)
            if not engine:
                raise RuntimeError('No default_search_engine configured. Set settings.controller.default_search_engine to one of: google | bing | duckduckgo')
            engine = engine.lower()
            query = params.query
            if engine == 'google':
                search_url = f'https://www.google.com/search?q={query}&udm=14'
                engine_name = 'Google'
            elif engine == 'bing':
                search_url = f'https://www.bing.com/search?q={query}'
                engine_name = 'Bing'
            else:
                # Explicit duckduckgo, only if configured
                if engine != 'duckduckgo':
                    raise RuntimeError(f"Unsupported search engine: {engine}")
                search_url = f'https://duckduckgo.com/?q={query}'
                engine_name = 'DuckDuckGo'

            # Navigate according to caller context; keep current-tab first, fallback to new tab if needed
            nav_note = 'current-tab'
            try:
                await browser_session.navigate_to(search_url)
            except Exception:
                page = await browser_session.create_new_tab(search_url)
                try:
                    tab_idx = browser_session.tabs.index(page)
                    nav_note = f'new-tab #{tab_idx} (fallback)'
                except Exception:
                    nav_note = 'new-tab (fallback)'

            # No embedded CAPTCHA handling here; solver is a separate action invoked by the LLM when needed.

            msg = f'🔍  Searched for "{params.query}" on {engine_name} [{nav_note}]'
            logger.info(msg)
            return ActionResult(
                extracted_content=msg,
                include_in_memory=True,
                long_term_memory=f"Searched {engine_name} for '{params.query}'",
            )

        # NOTE: We keep the action registered for backward compatibility but hide it from
        # LLM tool schemas/prompts by attaching a page_filter that always returns False.
        # This effectively disables the action path so that SolveCaptchaTask is used directly.
        @self.registry.action(
            'Attempt to solve a visible CAPTCHA by iteratively clicking suggested indices. Provide indices explicitly.',
            param_model=NoParamsAction,
            page_filter=lambda _page: False,
        )
        async def solve_captcha(
            _: NoParamsAction,
            browser_session: BrowserSession,
            page_extraction_llm: Optional[BaseChatModel] = None,
            context: Context | None = None,
        ):
            try:
                # Default page_extraction_llm to the main LLM if not provided (same logic as act/multi_act)
                if page_extraction_llm is None:
                    try:
                        settings = getattr(context, 'settings', None) or getattr(self, 'settings', None)
                        candidate = getattr(settings, 'llm', None)
                        if candidate is not None:
                            page_extraction_llm = candidate
                    except Exception:
                        pass

                if page_extraction_llm is None:
                    return ActionResult(success=False, error='Action solve_captcha requires page_extraction_llm but none provided and no default LLM found in settings.')

                # Lazy import to avoid cycles
                # Use the new task-based shim to keep public API stable
                from browser_use.agent.tasks.solve_captcha_tool import tool_solve_captcha, SolveCaptchaAction
                return await tool_solve_captcha(
                    controller=self,
                    params=SolveCaptchaAction(),
                    browser=browser_session,
                    page_extraction_llm=page_extraction_llm,
                )
            except Exception as e:
                logger.error(f'solve_captcha action failed: {e}', exc_info=True)
                return ActionResult(success=False, error=str(e))

        @self.registry.action(
            'Navigate to URL, set new_tab=True to open in new tab, False to navigate in current tab', param_model=GoToUrlAction
        )
        async def go_to_url(params: GoToUrlAction, browser_session: BrowserSession):
            try:
                # Avoid creating an extra tab if the only current tab is a new-tab/about:blank
                try:
                    current_page = await browser_session.get_current_page()
                    current_is_blank = bool(current_page and (current_page.url == 'about:blank' or current_page.url.startswith('chrome://new-tab-page')))
                except Exception:
                    current_is_blank = False

                if params.new_tab and not current_is_blank:
                    # Open in new tab (logic from open_tab function)
                    page = await browser_session.create_new_tab(params.url)
                    tab_idx = browser_session.tabs.index(page)
                    memory = f'Opened new tab with URL {params.url}'
                    msg = f'🔗  Opened new tab #{tab_idx} with url {params.url}'
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=memory)
                else:
                    # Navigate in current tab (original logic)
                    # SECURITY FIX: Use browser_session.navigate_to() instead of direct page.goto()
                    # This ensures URL validation against allowed_domains is performed
                    await browser_session.navigate_to(params.url)
                    memory = f'Navigated to {params.url}'
                    msg = f'🔗 {memory}'
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=memory)
            except Exception as e:
                error_msg = str(e)
                # Check for network-related errors
                if any(
                    err in error_msg
                    for err in [
                        'ERR_NAME_NOT_RESOLVED',
                        'ERR_INTERNET_DISCONNECTED',
                        'ERR_CONNECTION_REFUSED',
                        'ERR_TIMED_OUT',
                        'net::',
                    ]
                ):
                    site_unavailable_msg = f'Site unavailable: {params.url} - {error_msg}'
                    logger.warning(site_unavailable_msg)
                    raise BrowserError(site_unavailable_msg)
                else:
                    # Re-raise non-network errors (including URLNotAllowedError for unauthorized domains)
                    raise

        @self.registry.action('Go back', param_model=NoParamsAction)
        async def go_back(_: NoParamsAction, browser_session: BrowserSession):
            await browser_session.go_back()
            msg = '🔙  Navigated back'
            logger.info(msg)
            return ActionResult(extracted_content=msg)

        # Unified, accurate wait semantics: honor requested seconds up to a safe cap (300s)
        class WaitActionParams(BaseModel):
            seconds: int = Field(default=3, ge=0, le=300)

        @self.registry.action(
            'Wait for x seconds (default 3, max 300). Use this to pause until the page settles or a timer elapses.',
            param_model=WaitActionParams,
        )
        async def wait(params: WaitActionParams):
            # Bound the request defensively and sleep exactly that amount
            seconds = max(0, min(int(params.seconds), 300))
            start = time.monotonic()
            await asyncio.sleep(seconds)
            elapsed = time.monotonic() - start
            msg = f'🕒  Waited for {seconds} seconds'
            # Include actual elapsed in logs for observability; keep response concise
            logger.info(f"{msg} (actual ~{elapsed:.2f}s)")
            return ActionResult(extracted_content=msg)

        # Element Interaction Actions

        @self.registry.action('Click element by index', param_model=ClickElementAction)
        async def click_element_by_index(params: ClickElementAction, browser_session: BrowserSession):
            # Optional UX: pause on first click for human guidance / resume
            try:
                ctrl_settings = getattr(self, 'settings', None)
                pause_on_first_click = bool(getattr(ctrl_settings, 'pause_on_first_click', False))
                if pause_on_first_click and not getattr(self, '_first_click_pause_done', False):
                    handler = getattr(ctrl_settings, 'signal_handler', None)
                    if handler is not None:
                        try:
                            asyncio.create_task(handler._async_wait_for_resume())
                        except Exception:
                            pass
                    setattr(self, '_first_click_pause_done', True)
            except Exception:
                pass

            element_node = await browser_session.get_dom_element_by_index(params.index)
            if element_node is None:
                raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

            initial_pages = len(browser_session.tabs)

            # Do not click file inputs directly (use upload_file action)
            if browser_session.is_file_input(element_node):
                msg = (
                    f'Index {params.index} - has an element which opens file upload dialog. '
                    'To upload files please use a specific function to upload files '
                )
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True, success=False, long_term_memory=msg)

            try:
                download_path, stealth_used = await browser_session._click_element_node(element_node)
                # Small settle wait for UI updates
                try:
                    await asyncio.sleep(0.18)
                except Exception:
                    pass

                if download_path:
                    emoji = '💾'
                    msg = f'Downloaded file to {download_path}'
                else:
                    emoji = '🖱️'
                    msg = (
                        f'Clicked button with index {params.index}: '
                        f"{element_node.get_all_text_till_next_clickable_element(max_depth=2)}"
                    )

                logger.info(f'{emoji} {msg}')
                logger.debug(f'Element xpath: {element_node.xpath}')

                # Switch to new tab if one opened
                if len(browser_session.tabs) > initial_pages:
                    new_tab_msg = 'New tab opened - switching to it'
                    msg += f' - {new_tab_msg}'
                    logger.info(f'🔗 {new_tab_msg}')
                    await browser_session.switch_to_tab(-1)

                # Minimal debug only; avoid heuristic-heavy signals
                debug_info = {
                    'stealth_used': stealth_used,
                    'index': int(params.index),
                    'executed_xpath': getattr(element_node, 'xpath', None),
                }

                return ActionResult(
                    extracted_content=msg,
                    include_in_memory=True,
                    long_term_memory=msg,
                    debug=debug_info,
                )
            except Exception as e:
                raise BrowserError(str(e))

        @self.registry.action(
            'Click and input text into a input interactive element',
            param_model=InputTextAction,
        )
        async def input_text(params: InputTextAction, browser_session: BrowserSession, has_sensitive_data: bool = False):
            element_node = await browser_session.get_dom_element_by_index(params.index)
            if element_node is None:
                raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

            try:
                await browser_session._input_text_element_node(element_node, params.text)
            except Exception:
                msg = f'Failed to input text into element {params.index}.'
                raise BrowserError(msg)

            if not has_sensitive_data:
                msg = f'⌨️  Input {params.text} into index {params.index}'
            else:
                msg = f'⌨️  Input sensitive data into index {params.index}'
            logger.info(msg)
            logger.debug(f'Element xpath: {element_node.xpath}')
            return ActionResult(
                extracted_content=msg,
                include_in_memory=True,
                long_term_memory=f"Input '{params.text}' into element {params.index}.",
            )

        @self.registry.action('Upload a file to a file input element at index', param_model=UploadFileAction)
        async def upload_file(
            params: UploadFileAction,
            browser_session: BrowserSession,
            available_file_paths: list[str],
        ):
            # Normalize and validate allowlist (accept exact files or any path under allowed directories)
            def _norm(p: str) -> str:
                return os.path.normcase(os.path.abspath(os.path.normpath(p)))

            target_path_norm = _norm(params.path)

            if available_file_paths:
                allowed = False
                normalized_allowlist: list[str] = []
                for ap in available_file_paths or []:
                    try:
                        ap_norm = _norm(str(ap))
                        normalized_allowlist.append(ap_norm)
                        # Exact file match
                        if target_path_norm == ap_norm:
                            allowed = True
                            break
                        # Directory prefix match
                        if os.path.isdir(ap_norm) and target_path_norm.startswith(ap_norm + os.sep):
                            allowed = True
                            break
                    except Exception:
                        continue
                if not allowed:
                    joined = "\n - ".join(available_file_paths or [])
                    raise BrowserError(
                        (
                            f"File path is not allowed: {params.path}.\n"
                            f"Whitelist the exact file or its containing directory in AgentSettings.available_file_paths.\n"
                            f"Current allowlist:\n - {joined}" if joined else "No allowed file paths configured."
                        )
                    )

            if not os.path.exists(target_path_norm):
                raise BrowserError(f"File does not exist: {params.path}")

            # Try to find the closest file input near the provided index (start narrow)
            file_upload_dom_el = await browser_session.find_file_upload_element_by_index(
                params.index, max_height=3, max_descendant_depth=3
            )

            if file_upload_dom_el is None:
                msg = f"No file upload element found at index {params.index}"
                logger.info(msg)
                raise BrowserError(msg)

            file_upload_el = await browser_session.get_locate_element(file_upload_dom_el)
            if file_upload_el is None:
                msg = f"No file upload element found at index {params.index}"
                logger.info(msg)
                raise BrowserError(msg)

            try:
                # Attempt upload on the located input[type=file]
                await file_upload_el.set_input_files(target_path_norm)

                # Verify selection actually registered on the input element
                async def _verify_selected(handle) -> tuple[int, list[str]]:
                    try:
                        info = await handle.evaluate(
                            "(el) => ({ cnt: el && el.files ? el.files.length : 0, names: el && el.files ? Array.from(el.files).map(f => f.name) : [] })"
                        )
                        cnt = int((info or {}).get('cnt', 0))
                        names = list((info or {}).get('names', []) or [])
                        return cnt, names
                    except Exception:
                        return 0, []

                cnt, names = await _verify_selected(file_upload_el)

                # If verification failed, widen the search and retry once
                if cnt == 0:
                    logger.debug(
                        f"Upload verification found 0 files on initial input. Retrying with wider search (index={params.index})."
                    )
                    widened_dom_el = await browser_session.find_file_upload_element_by_index(
                        params.index, max_height=10, max_descendant_depth=6
                    )
                    if widened_dom_el is not None:
                        widened_handle = await browser_session.get_locate_element(widened_dom_el)
                        if widened_handle is not None:
                            try:
                                await widened_handle.set_input_files(target_path_norm)
                                cnt, names = await _verify_selected(widened_handle)
                            except Exception as e:
                                logger.debug(f"Retry upload on widened handle failed: {type(e).__name__}: {e}")

                # If still not selected, report a hard failure to prevent false success
                if cnt == 0:
                    try:
                        accept_attr = await file_upload_el.evaluate("(el) => el.getAttribute('accept') || ''")
                    except Exception:
                        accept_attr = ''
                    file_ext = os.path.splitext(target_path_norm)[1].lower()
                    guidance = (
                        "Selection did not register on the input element. This may happen if the element isn't a real file input, "
                        "the page uses a different hidden input, or the input rejects the file type."
                    )
                    diag = f"accept='{accept_attr}', file_ext='{file_ext}'"
                    error_msg = f"File selection failed: no files present in input after upload attempt (index={params.index}; {diag}). {guidance}"
                    logger.error(error_msg)
                    raise BrowserError(error_msg)

                # Try to confirm UI acknowledgment beyond input.files (avoid overclaiming)
                chosen = names[0] if names else os.path.basename(target_path_norm)

                async def _confirm_ui_acknowledged(handle, filename: str) -> tuple[bool, str]:
                    try:
                        info = await handle.evaluate(
                            "(el, name) => {\n"
                            "  const lower = (name || '').toLowerCase();\n"
                            "  // Only check if the filename appears in related UI elements, not just input.value\n"
                            "  // input.value for file inputs can be misleading\n"
                            "  try {\n"
                            "    const id = el && el.id;\n"
                            "    if (id) {\n"
                            "      const labels = document.getElementsByTagName('label');\n"
                            "      for (let i = 0; i < labels.length; i++) {\n"
                            "        const lbl = labels[i];\n"
                            "        if ((lbl.htmlFor === id) && lbl.textContent && lbl.textContent.toLowerCase().includes(lower)) {\n"
                            "          return { ok: true, how: 'label[for]' };\n"
                            "        }\n"
                            "      }\n"
                            "    }\n"
                            "  } catch(e) {}\n"
                            "  try {\n"
                            "    const desc = el && el.getAttribute('aria-describedby');\n"
                            "    if (desc) {\n"
                            "      const d = document.getElementById(desc);\n"
                            "      if (d && d.textContent && d.textContent.toLowerCase().includes(lower)) return { ok: true, how: 'aria-describedby' };\n"
                            "    }\n"
                            "  } catch(e) {}\n"
                            "  try {\n"
                            "    let p = el && el.parentElement; let depth = 0;\n"
                            "    while (p && depth < 3) {\n"
                            "      const t = (p.textContent || '').toLowerCase();\n"
                            "      if (t.includes(lower)) return { ok: true, how: 'ancestor-text-depth-' + depth };\n"
                            "      p = p.parentElement; depth++;\n"
                            "    }\n"
                            "  } catch(e) {}\n"
                            "  return { ok: false, how: '' };\n"
                            "}",
                            chosen,
                        )
                        return bool((info or {}).get('ok', False)), str((info or {}).get('how', ''))
                    except Exception:
                        return False, ''

                confirmed, how = await _confirm_ui_acknowledged(file_upload_el, chosen)

                if confirmed:
                    msg = f"📁 Uploaded '{chosen}' to index {params.index} (confirmed: {how})"
                    logger.info(msg)
                    return ActionResult(
                        extracted_content=msg,
                        include_in_memory=True,
                        long_term_memory=f"Uploaded file {params.path} to element {params.index} (confirmed)",
                        # success intentionally omitted for non-done actions
                    )
                else:
                    # CRITICAL: Do not claim success when upload is not confirmed
                    # File selection ≠ upload completion
                    warning_msg = f"⚠️  File '{chosen}' selected at index {params.index} but upload NOT confirmed - may need additional steps"
                    logger.warning(warning_msg)
                    return ActionResult(
                        extracted_content=f"File selected but upload uncertain: {warning_msg}",
                        include_in_memory=True,
                        long_term_memory=f"File {params.path} selected at element {params.index} - UPLOAD NOT CONFIRMED, may need submit/validation",
                        success=False,  # Do not claim success without confirmation
                    )
            except Exception as e:
                msg = f"Failed to upload file to index {params.index}: {str(e)}"
                logger.info(msg)
                raise BrowserError(msg)

        # Tab Management Actions

        @self.registry.action('Switch tab', param_model=SwitchTabAction)
        async def switch_tab(params: SwitchTabAction, browser_session: BrowserSession):
            await browser_session.switch_to_tab(params.page_id)
            page = await browser_session.get_current_page()
            try:
                await page.wait_for_load_state(state='domcontentloaded', timeout=5_000)
                # page was already loaded when we first navigated, this is additional to wait for onfocus/onblur animations/ajax to settle
            except Exception as e:
                pass
            msg = f'🔄  Switched to tab #{params.page_id} with url {page.url}'
            logger.info(msg)
            return ActionResult(
                extracted_content=msg, include_in_memory=True, long_term_memory=f'Switched to tab {params.page_id}'
            )

        @self.registry.action('Close an existing tab', param_model=CloseTabAction)
        async def close_tab(params: CloseTabAction, browser_session: BrowserSession):
            await browser_session.switch_to_tab(params.page_id)
            page = await browser_session.get_current_page()
            url = page.url
            await page.close()
            new_page = await browser_session.get_current_page()
            new_page_idx = browser_session.tabs.index(new_page)
            msg = f'❌  Closed tab #{params.page_id} with {url}, now focused on tab #{new_page_idx} with url {new_page.url}'
            logger.info(msg)
            return ActionResult(
                extracted_content=msg,
                include_in_memory=True,
                long_term_memory=f'Closed tab {params.page_id} with url {url}, now focused on tab {new_page_idx} with url {new_page.url}.',
            )

        # Content Actions

        @self.registry.action(
            """Extract structured, semantic data (e.g. product description, price, all information about XYZ) from the current webpage based on a textual query.
This tool takes the entire markdown of the page and extracts the query from it.
Set extract_links=True ONLY if your query requires extracting links/URLs from the page.
Only use this for specific queries for information retrieval from the page. Don't use this to get interactive elements - the tool does not see HTML elements, only the markdown.
""",
        )
        async def extract_structured_data(
            query: str,
            extract_links: bool,
            page: Page,
            page_extraction_llm: BaseChatModel,
            file_system: FileSystem,
        ):
            from functools import partial

            import markdownify

            strip = []

            if not extract_links:
                strip = ['a', 'img']

            # Run markdownify in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()

            # Aggressive timeout for page content
            try:
                page_html_result = await asyncio.wait_for(page.content(), timeout=10.0)  # 5 second aggressive timeout
            except TimeoutError:
                raise RuntimeError('Page content extraction timed out after 5 seconds')
            except Exception as e:
                raise RuntimeError(f"Couldn't extract page content: {e}")

            page_html = page_html_result

            markdownify_func = partial(markdownify.markdownify, strip=strip)

            try:
                content = await asyncio.wait_for(
                    loop.run_in_executor(None, markdownify_func, page_html), timeout=5.0
                )  # 5 second aggressive timeout
            except Exception as e:
                logger.warning(f'Markdownify failed: {type(e).__name__}')
                raise RuntimeError(f'Could not convert html to markdown: {type(e).__name__}')

            # manually append iframe text into the content so it's readable by the LLM (includes cross-origin iframes)
            for iframe in page.frames:
                try:
                    await iframe.wait_for_load_state(timeout=1000)  # 1 second aggressive timeout for iframe load
                except Exception:
                    pass

                if iframe.url != page.url and not iframe.url.startswith('data:') and not iframe.url.startswith('about:'):
                    content += f'\n\nIFRAME {iframe.url}:\n'
                    # Run markdownify in a thread pool for iframe content as well
                    try:
                        # Aggressive timeouts for iframe content
                        iframe_html = await asyncio.wait_for(iframe.content(), timeout=2.0)  # 2 second aggressive timeout
                        iframe_markdown = await asyncio.wait_for(
                            loop.run_in_executor(None, markdownify_func, iframe_html),
                            timeout=2.0,  # 2 second aggressive timeout for iframe markdownify
                        )
                    except Exception:
                        iframe_markdown = ''  # Skip failed iframes
                    content += iframe_markdown
            # replace multiple sequential \n with a single \n
            content = re.sub(r'\n+', '\n', content)

            # limit to 30000 characters - remove text in the middle (≈15000 tokens)
            max_chars = 30000
            if len(content) > max_chars:
                logger.info(f'Content is too long, removing middle {len(content) - max_chars} characters')
                content = (
                    content[: max_chars // 2]
                    + '\n... left out the middle because it was too long ...\n'
                    + content[-max_chars // 2 :]
                )

            prompt = """You convert websites into structured information. Extract information from this webpage based on the query. Focus only on content relevant to the query. If
1. The query is vague
2. Does not make sense for the page
3. Some/all of the information is not available

Explain the content of the page and that the requested information is not available in the page. Respond in JSON format.\nQuery: {query}\n Website:\n{page}"""
            try:
                formatted_prompt = prompt.format(query=query, page=content)
                # Aggressive timeout for LLM call
                response = await asyncio.wait_for(
                    page_extraction_llm.ainvoke([UserMessage(content=formatted_prompt)]),
                    timeout=120.0,  # 120 second aggressive timeout for LLM call
                )

                extracted_content = f'Page Link: {page.url}\nQuery: {query}\nExtracted Content:\n{response.completion}'

                # if content is small include it to memory
                MAX_MEMORY_SIZE = 600
                if len(extracted_content) < MAX_MEMORY_SIZE:
                    memory = extracted_content
                    include_extracted_content_only_once = False
                else:
                    # find lines until MAX_MEMORY_SIZE
                    lines = extracted_content.splitlines()
                    display = ''
                    display_lines_count = 0
                    for line in lines:
                        if len(display) + len(line) < MAX_MEMORY_SIZE:
                            display += line + '\n'
                            display_lines_count += 1
                        else:
                            break
                    save_result = await file_system.save_extracted_content(extracted_content)
                    memory = f'Extracted content from {page.url}\n<query>{query}\n</query>\n<extracted_content>\n{display}{len(lines) - display_lines_count} more lines...\n</extracted_content>\n<file_system>{save_result}</file_system>'
                    include_extracted_content_only_once = True
                logger.info(f'📄 {memory}')
                return ActionResult(
                    extracted_content=extracted_content,
                    include_extracted_content_only_once=include_extracted_content_only_once,
                    long_term_memory=memory,
                )
            except TimeoutError:
                error_msg = f'LLM call timed out for query: {query}'
                logger.warning(error_msg)
                raise RuntimeError(error_msg)
            except Exception as e:
                logger.debug(f'Error extracting content: {e}')
                msg = f'📄  Extracted from page\n: {content}\n'
                logger.info(msg)
                raise RuntimeError(str(e))

        @self.registry.action(
            'Scroll the page by specified number of pages (set down=True to scroll down, down=False to scroll up, num_pages=number of pages to scroll like 0.5 for half page, 1.0 for one page, etc.). Optional index parameter to scroll within a specific element or its scroll container (works well for dropdowns and custom UI components).',
            param_model=ScrollAction,
        )
        async def scroll(params: ScrollAction, browser_session: BrowserSession):
            """
            (a) If index is provided, find scrollable containers in the element hierarchy and scroll directly.
            (b) If no index or no container found, use browser._scroll_container for container-aware scrolling.
            (c) If that JavaScript throws, fall back to window.scrollBy().
            """
            page = await browser_session.get_current_page()

            # Helper function to get window height with retry decorator
            @retry(wait=1, retries=3, timeout=5)
            async def get_window_height():
                return await page.evaluate('() => window.innerHeight')

            # Get window height with retries
            try:
                window_height = await get_window_height()
            except Exception as e:
                raise RuntimeError(f'Scroll failed due to an error: {e}')
            window_height = window_height or 0

            # Determine scroll amount based on num_pages
            scroll_amount = int(window_height * params.num_pages)
            pages_scrolled = params.num_pages

            # Set direction based on down parameter
            dy = scroll_amount if params.down else -scroll_amount

            # Initialize result message components
            direction = 'down' if params.down else 'up'
            scroll_target = 'the page'

            # Element-specific scrolling if index is provided
            if params.index is not None:
                try:
                    element_node = await browser_session.get_dom_element_by_index(params.index)
                    if element_node is None:
                        raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

                    # Try direct container scrolling (no events that might close dropdowns)
                    container_scroll_js = """
                    (params) => {
                        const { dy, elementXPath } = params;

                        // Get the target element by XPath
                        const targetElement = document.evaluate(elementXPath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                        if (!targetElement) {
                            return { success: false, reason: 'Element not found by XPath' };
                        }

                        console.log('[SCROLL DEBUG] Starting direct container scroll for element:', targetElement.tagName);

                        // Try to find scrollable containers in the hierarchy (starting from element itself)
                        let currentElement = targetElement;
                        let scrollSuccess = false;
                        let scrolledElement = null;
                        let scrollDelta = 0;
                        let attempts = 0;

                        // Check up to 10 elements in hierarchy (including the target element itself)
                        while (currentElement && attempts < 10) {
                            const computedStyle = window.getComputedStyle(currentElement);
                            const hasScrollableY = /(auto|scroll|overlay)/.test(computedStyle.overflowY);
                            const canScrollVertically = currentElement.scrollHeight > currentElement.clientHeight;

                            console.log('[SCROLL DEBUG] Checking element:', currentElement.tagName,
                                'hasScrollableY:', hasScrollableY,
                                'canScrollVertically:', canScrollVertically,
                                'scrollHeight:', currentElement.scrollHeight,
                                'clientHeight:', currentElement.clientHeight);

                            if (hasScrollableY && canScrollVertically) {
                                const beforeScroll = currentElement.scrollTop;
                                const maxScroll = currentElement.scrollHeight - currentElement.clientHeight;

                                // Calculate scroll amount (1/3 of provided dy for gentler scrolling)
                                let scrollAmount = dy / 3;

                                // Ensure we don't scroll beyond bounds
                                if (scrollAmount > 0) {
                                    scrollAmount = Math.min(scrollAmount, maxScroll - beforeScroll);
                                } else {
                                    scrollAmount = Math.max(scrollAmount, -beforeScroll);
                                }

                                // Try direct scrollTop manipulation (most reliable)
                                currentElement.scrollTop = beforeScroll + scrollAmount;

                                const afterScroll = currentElement.scrollTop;
                                const actualScrollDelta = afterScroll - beforeScroll;

                                console.log('[SCROLL DEBUG] Scroll attempt:', currentElement.tagName,
                                    'before:', beforeScroll, 'after:', afterScroll, 'delta:', actualScrollDelta);

                                if (Math.abs(actualScrollDelta) > 0.5) {
                                    scrollSuccess = true;
                                    scrolledElement = currentElement;
                                    scrollDelta = actualScrollDelta;
                                    console.log('[SCROLL DEBUG] Successfully scrolled container:', currentElement.tagName, 'delta:', actualScrollDelta);
                                    break;
                                }
                            }

                            // Move to parent (but don't go beyond body for dropdown case)
                            if (currentElement === document.body || currentElement === document.documentElement) {
                                break;
                            }
                            currentElement = currentElement.parentElement;
                            attempts++;
                        }

                        if (scrollSuccess) {
                            // Successfully scrolled a container
                            return {
                                success: true,
                                method: 'direct_container_scroll',
                                containerType: 'element',
                                containerTag: scrolledElement.tagName.toLowerCase(),
                                containerClass: scrolledElement.className || '',
                                containerId: scrolledElement.id || '',
                                scrollDelta: scrollDelta
                            };
                        } else {
                            // No container found or could scroll
                            console.log('[SCROLL DEBUG] No scrollable container found for element');
                            return {
                                success: false,
                                reason: 'No scrollable container found',
                                needsPageScroll: true
                            };
                        }
                    }
                    """

                    # Pass parameters as a single object
                    scroll_params = {'dy': dy, 'elementXPath': element_node.xpath}
                    result = await page.evaluate(container_scroll_js, scroll_params)

                    if result['success']:
                        if result['containerType'] == 'element':
                            container_info = f'{result["containerTag"]}'
                            if result['containerId']:
                                container_info += f'#{result["containerId"]}'
                            elif result['containerClass']:
                                container_info += f'.{result["containerClass"].split()[0]}'
                            scroll_target = f"element {params.index}'s scroll container ({container_info})"
                            # Don't do additional page scrolling since we successfully scrolled the container
                        else:
                            scroll_target = f'the page (fallback from element {params.index})'
                    else:
                        # Container scroll failed, need page-level scrolling
                        logger.debug(f'Container scroll failed for element {params.index}: {result.get("reason", "Unknown")}')
                        scroll_target = f'the page (no container found for element {params.index})'
                        # This will trigger page-level scrolling below

                except Exception as e:
                    logger.debug(f'Element-specific scrolling failed for index {params.index}: {e}')
                    scroll_target = f'the page (fallback from element {params.index})'
                    # Fall through to page-level scrolling

            # Page-level scrolling (default or fallback)
            if (
                scroll_target == 'the page'
                or 'fallback' in scroll_target
                or 'no container found' in scroll_target
                or 'mouse wheel failed' in scroll_target
            ):
                logger.debug(f'🔄 Performing page-level scrolling. Reason: {scroll_target}')
                try:
                    await browser_session._scroll_container(cast(int, dy))
                except Exception as e:
                    # Hard fallback: always works on root scroller
                    await page.evaluate('(y) => window.scrollBy(0, y)', dy)
                    logger.debug('Smart scroll failed; used window.scrollBy fallback', exc_info=e)

            # Create descriptive message
            if pages_scrolled == 1.0:
                long_term_memory = f'Scrolled {direction} {scroll_target} by one page'
            else:
                long_term_memory = f'Scrolled {direction} {scroll_target} by {pages_scrolled} pages'

            msg = f'🔍 {long_term_memory}'

            logger.info(msg)
            return ActionResult(
                extracted_content=msg,
                include_in_memory=True,
                long_term_memory=long_term_memory,
            )

        @self.registry.action(
            'Send strings of special keys to use Playwright page.keyboard.press - examples include Escape, Backspace, Insert, PageDown, Delete, Enter, or Shortcuts such as `Control+o`, `Control+Shift+T`',
            param_model=SendKeysAction,
        )
        async def send_keys(params: SendKeysAction, page: Page):
            try:
                await page.keyboard.press(params.keys)
            except Exception as e:
                if 'Unknown key' in str(e):
                    # loop over the keys and try to send each one
                    for key in params.keys:
                        try:
                            await page.keyboard.press(key)
                        except Exception as e:
                            logger.debug(f'Error sending key {key}: {str(e)}')
                            raise e
                else:
                    raise e
            msg = f'⌨️  Sent keys: {params.keys}'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=f'Sent keys: {params.keys}')

        @self.registry.action(
            description='Scroll to a text in the current page',
        )
        async def scroll_to_text(text: str, page: Page):  # type: ignore
            try:
                # Try different locator strategies
                locators = [
                    page.get_by_text(text, exact=False),
                    page.locator(f'text={text}'),
                    page.locator(f"//*[contains(text(), '{text}')]"),
                ]

                for locator in locators:
                    try:
                        if await locator.count() == 0:
                            continue

                        element = locator.first
                        is_visible = await element.is_visible()
                        bbox = await element.bounding_box()

                        if is_visible and bbox is not None and bbox['width'] > 0 and bbox['height'] > 0:
                            await element.scroll_into_view_if_needed()
                            await asyncio.sleep(0.5)  # Wait for scroll to complete
                            msg = f'🔍  Scrolled to text: {text}'
                            logger.info(msg)
                            return ActionResult(
                                extracted_content=msg, include_in_memory=True, long_term_memory=f'Scrolled to text: {text}'
                            )

                    except Exception as e:
                        logger.debug(f'Locator attempt failed: {str(e)}')
                        continue

                msg = f"Text '{text}' not found or not visible on page"
                logger.info(msg)
                return ActionResult(
                    extracted_content=msg,
                    include_in_memory=True,
                    long_term_memory=f"Tried scrolling to text '{text}' but it was not found",
                )

            except Exception as e:
                msg = f"Failed to scroll to text '{text}': {str(e)}"
                logger.error(msg)
                raise BrowserError(msg)

        # File System Actions
        @self.registry.action(
            'Write or append content to file_name in file system. Allowed extensions are .md, .txt, .json, .csv, .pdf. For .pdf files, write the content in markdown format and it will automatically be converted to a properly formatted PDF document.'
        )
        async def write_file(
            file_name: str,
            content: str,
            file_system: FileSystem,
            append: bool = False,
            trailing_newline: bool = True,
            leading_newline: bool = False,
        ):
            if trailing_newline:
                content += '\n'
            if leading_newline:
                content = '\n' + content
            if append:
                result = await file_system.append_file(file_name, content)
            else:
                result = await file_system.write_file(file_name, content)
            logger.info(f'💾 {result}')
            return ActionResult(extracted_content=result, include_in_memory=True, long_term_memory=result)

        @self.registry.action(
            'Replace old_str with new_str in file_name. old_str must exactly match the string to replace in original text. Recommended tool to mark completed items in todo.md or change specific contents in a file.'
        )
        async def replace_file_str(file_name: str, old_str: str, new_str: str, file_system: FileSystem):
            result = await file_system.replace_file_str(file_name, old_str, new_str)
            logger.info(f'💾 {result}')
            return ActionResult(extracted_content=result, include_in_memory=True, long_term_memory=result)

        # Memory primitives -------------------------------------------------

        @self.registry.action(
            description=(
                "Extract a value from the current page into long-term memory under a key. "
                "Source can be 'url', 'title', 'selection', or 'css:<selector>'."
            ),
            param_model=ExtractToMemoryAction,
        )
        async def extract_to_memory(
            params: ExtractToMemoryAction,
            browser_session: BrowserSession,
            file_system: FileSystem,
        ) -> ActionResult:
            page = await browser_session.get_current_page()

            async def _norm_url(u: str) -> str:
                try:
                    from urllib.parse import urlparse, urlunparse

                    p = urlparse(u)
                    # lowercase scheme and netloc; drop fragment
                    netloc = p.netloc.lower()
                    scheme = (p.scheme or 'https').lower()
                    # leave path/query as-is; test only checks host lowercase and no fragment
                    return urlunparse((scheme, netloc, p.path, p.params, p.query, ''))
                except Exception:
                    return u

            value: str = ''
            src = params.source.strip()
            try:
                if src == 'url':
                    value = await _norm_url(page.url)
                elif src == 'title':
                    value = await page.title()
                elif src == 'selection':
                    # Best-effort selection extraction
                    value = await page.evaluate(
                        '() => (window.getSelection ? window.getSelection().toString() : "") || ""'
                    )
                elif src.startswith('css:'):
                    selector = src.split(':', 1)[1]
                    value = await page.evaluate(
                        '(sel) => { const el = document.querySelector(sel); return el ? (el.textContent || "").trim() : ""; }',
                        selector,
                    )
                else:
                    raise ValueError("Unsupported source. Use 'url', 'title', 'selection', or 'css:<selector>'.")
            except Exception as e:
                msg = f"Failed to extract from source '{src}': {e}"
                logger.info(msg)
                return ActionResult(extracted_content=msg, success=False)

            # Load existing memory.json (if any), update, and write back
            try:
                raw = file_system.display_file('memory.json')
                data = json.loads(raw) if raw else {}
            except Exception:
                data = {}
            data[params.key] = value
            await file_system.write_file('memory.json', json.dumps(data))

            msg = f"Extracted '{params.key}' from {src}"
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=msg)

        @self.registry.action(
            description=(
                "Fill a page field from memory. Provide field_selector as 'css:<selector>' or 'xpath:<expr>' and a key."
            ),
            param_model=FillFromMemoryAction,
        )
        async def fill_from_memory(
            params: FillFromMemoryAction,
            browser_session: BrowserSession,
            file_system: FileSystem,
        ) -> ActionResult:
            page = await browser_session.get_current_page()

            # Read memory.json
            raw = file_system.display_file('memory.json')
            if not raw:
                msg = 'No memory.json found to read from.'
                logger.info(msg)
                return ActionResult(extracted_content=msg, success=False)
            try:
                data = json.loads(raw)
            except Exception:
                data = {}

            if params.key not in data:
                msg = f"Key '{params.key}' not found in memory."
                logger.info(msg)
                return ActionResult(extracted_content=msg, success=False)

            value = data.get(params.key, '')

            sel = params.field_selector.strip()
            try:
                if sel.startswith('css:'):
                    css = sel.split(':', 1)[1]
                    # Use page.fill directly so tests can assert selector string
                    await page.fill(css, value)
                elif sel.startswith('xpath:'):
                    xpath = sel.split(':', 1)[1]
                    # Use locator for xpath
                    await page.locator(xpath).fill(value)
                else:
                    raise ValueError("Unsupported field_selector. Use 'css:<selector>' or 'xpath:<expr>'.")
            except Exception as e:
                msg = f"Failed to fill field '{sel}': {e}"
                logger.info(msg)
                return ActionResult(extracted_content=msg, success=False)

            msg = f"Filled field from memory key '{params.key}'"
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=msg)

        @self.registry.action('Read file_name from file system')
        async def read_file(file_name: str, available_file_paths: list[str], file_system: FileSystem):
            if available_file_paths and file_name in available_file_paths:
                result = await file_system.read_file(file_name, external_file=True)
            else:
                result = await file_system.read_file(file_name)

            MAX_MEMORY_SIZE = 1000
            if len(result) > MAX_MEMORY_SIZE:
                lines = result.splitlines()
                display = ''
                lines_count = 0
                for line in lines:
                    if len(display) + len(line) < MAX_MEMORY_SIZE:
                        display += line + '\n'
                        lines_count += 1
                    else:
                        break
                remaining_lines = len(lines) - lines_count
                memory = f'{display}{remaining_lines} more lines...' if remaining_lines > 0 else display
            else:
                memory = result
            logger.info(f'💾 {memory}')
            return ActionResult(
                extracted_content=result,
                include_in_memory=True,
                long_term_memory=memory,
                include_extracted_content_only_once=True,
            )

        @self.registry.action(
            description='Get all options from a native dropdown or ARIA menu',
        )
        async def get_dropdown_options(index: int, browser_session: BrowserSession) -> ActionResult:
            """Get all options from a native dropdown or ARIA menu"""
            page = await browser_session.get_current_page()
            dom_element = await browser_session.get_dom_element_by_index(index)
            if dom_element is None:
                raise Exception(f'Element index {index} does not exist - retry or use alternative actions')

            try:
                # Frame-aware approach since we know it works
                all_options = []
                frame_index = 0

                for frame in page.frames:
                    try:
                        # First check if it's a native select element
                        options = await _safe_frame_evaluate(
                            frame,
                            """
                            (xpath) => {
                                const element = document.evaluate(xpath, document, null,
                                    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                if (!element) return null;

                                // Check if it's a native select element
                                if (element.tagName.toLowerCase() === 'select') {
                                    return {
                                        type: 'select',
                                        options: Array.from(element.options).map(opt => ({
                                            text: opt.text, //do not trim, because we are doing exact match in select_dropdown_option
                                            value: opt.value,
                                            index: opt.index
                                        })),
                                        id: element.id,
                                        name: element.name
                                    };
                                }

                                // Check if it's an ARIA menu
                                if (element.getAttribute('role') === 'menu' ||
                                    element.getAttribute('role') === 'listbox' ||
                                    element.getAttribute('role') === 'combobox') {
                                    // Find all menu items
                                    const menuItems = element.querySelectorAll('[role="menuitem"], [role="option"]');
                                    const options = [];

                                    menuItems.forEach((item, idx) => {
                                        // Get the text content of the menu item
                                        const text = item.textContent.trim();
                                        if (text) {
                                            options.push({
                                                text: text,
                                                value: text, // For ARIA menus, use text as value
                                                index: idx
                                            });
                                        }
                                    });

                                    return {
                                        type: 'aria',
                                        options: options,
                                        id: element.id || '',
                                        name: element.getAttribute('aria-label') || ''
                                    };
                                }

                                return null;
                            }
                        """,
                            dom_element.xpath,
                        )

                        if options:
                            logger.debug(f'Found {options["type"]} dropdown in frame {frame_index}')
                            logger.debug(f'Element ID: {options["id"]}, Name: {options["name"]}')

                            formatted_options = []
                            for opt in options['options']:
                                # encoding ensures AI uses the exact string in select_dropdown_option
                                encoded_text = json.dumps(opt['text'])
                                formatted_options.append(f'{opt["index"]}: text={encoded_text}')

                            all_options.extend(formatted_options)

                    except Exception as frame_e:
                        logger.debug(f'Frame {frame_index} evaluation failed: {str(frame_e)}')

                    frame_index += 1

                if all_options:
                    msg = '\n'.join(all_options)
                    msg += '\nUse the exact text string in select_dropdown_option'
                    logger.info(msg)
                    return ActionResult(
                        extracted_content=msg,
                        include_in_memory=True,
                        long_term_memory=f'Found dropdown options for index {index}.',
                        include_extracted_content_only_once=True,
                    )
                else:
                    msg = 'No options found in any frame for dropdown'
                    logger.info(msg)
                    return ActionResult(
                        extracted_content=msg, include_in_memory=True, long_term_memory='No dropdown options found'
                    )

            except Exception as e:
                logger.error(f'Failed to get dropdown options: {str(e)}')
                msg = f'Error getting options: {str(e)}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            description='Select dropdown option or ARIA menu item for interactive element index by the text of the option you want to select',
        )
        async def select_dropdown_option(
            index: int,
            text: str,
            browser_session: BrowserSession,
        ) -> ActionResult:
            """Select dropdown option or ARIA menu item by the text of the option you want to select"""
            page = await browser_session.get_current_page()
            dom_element = await browser_session.get_dom_element_by_index(index)
            if dom_element is None:
                raise Exception(f'Element index {index} does not exist - retry or use alternative actions')

            logger.debug(f"Attempting to select '{text}' using xpath: {dom_element.xpath}")
            logger.debug(f'Element attributes: {dom_element.attributes}')
            logger.debug(f'Element tag: {dom_element.tag_name}')

            xpath = '//' + dom_element.xpath

            try:
                frame_index = 0
                for frame in page.frames:
                    try:
                        logger.debug(f'Trying frame {frame_index} URL: {frame.url}')

                        # First check what type of element we're dealing with
                        element_info_js = """
                            (xpath) => {
                                try {
                                    const element = document.evaluate(xpath, document, null,
                                        XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                    if (!element) return null;

                                    const tagName = element.tagName.toLowerCase();
                                    const role = element.getAttribute('role');

                                    // Check if it's a native select
                                    if (tagName === 'select') {
                                        return {
                                            type: 'select',
                                            found: true,
                                            id: element.id,
                                            name: element.name,
                                            tagName: element.tagName,
                                            optionCount: element.options.length,
                                            currentValue: element.value,
                                            availableOptions: Array.from(element.options).map(o => o.text.trim())
                                        };
                                    }

                                    // Check if it's an ARIA menu or similar
                                    if (role === 'menu' || role === 'listbox' || role === 'combobox') {
                                        const menuItems = element.querySelectorAll('[role="menuitem"], [role="option"]');
                                        return {
                                            type: 'aria',
                                            found: true,
                                            id: element.id || '',
                                            role: role,
                                            tagName: element.tagName,
                                            itemCount: menuItems.length,
                                            availableOptions: Array.from(menuItems).map(item => item.textContent.trim())
                                        };
                                    }

                                    return {
                                        error: `Element is neither a select nor an ARIA menu (tag: ${tagName}, role: ${role})`,
                                        found: false
                                    };
                                } catch (e) {
                                    return {error: e.toString(), found: false};
                                }
                            }
                        """

                        element_info = await _safe_frame_evaluate(frame, element_info_js, dom_element.xpath)

                        if element_info and element_info.get('found'):
                            logger.debug(f'Found {element_info.get("type")} element in frame {frame_index}: {element_info}')

                            if element_info.get('type') == 'select':
                                # Handle native select element: no keyboard path; select + verify; fallback to overlay click if hidden or verification fails
                                stealth_mgr = getattr(browser_session, '_stealth_manager', None)
                                stealth_enabled = bool(getattr(browser_session.browser_profile, 'stealth', False))
                                selected_option_values = None
                                overlay_clicked = False

                                # Focus the select (best-effort)
                                handle = None
                                try:
                                    handle = await frame.locator('//' + dom_element.xpath).nth(0).element_handle()
                                    if handle is not None:
                                        try:
                                            if stealth_enabled and stealth_mgr is not None:
                                                bbox = await handle.bounding_box()
                                                if bbox:
                                                    cx = bbox['x'] + bbox['width'] / 2
                                                    cy = bbox['y'] + bbox['height'] / 2
                                                    try:
                                                        await stealth_mgr.execute_human_like_click(page, (cx, cy))
                                                    except Exception:
                                                        await handle.click(timeout=1_500)
                                            else:
                                                await handle.click(timeout=1_500)
                                        except Exception:
                                            pass
                                except Exception:
                                    pass

                                # Attempt select_option by label
                                try:
                                    selected_option_values = (
                                        await frame.locator('//' + dom_element.xpath).nth(0).select_option(label=text, timeout=2_000)
                                    )
                                except Exception:
                                    selected_option_values = None

                                # Fallback: resolve option value by exact text (trimmed, case-insensitive) and select by value
                                if not selected_option_values:
                                    try:
                                        match_value = await _safe_frame_evaluate(
                                            frame,
                                            "(params)=>{\n"
                                            "  const el = document.evaluate(params.xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;\n"
                                            "  if(!el || el.tagName.toLowerCase()!=='select') return null;\n"
                                            "  const target = String(params.target||'');\n"
                                            "  const norm = s=>String(s||'').trim().toLowerCase();\n"
                                            "  for(const opt of Array.from(el.options)){\n"
                                            "    if(norm(opt.text)===norm(target)) return opt.value;\n"
                                            "  }\n"
                                            "  return null;\n"
                                            "}",
                                            { 'xpath': dom_element.xpath, 'target': text }
                                        )
                                        if match_value:
                                            try:
                                                selected_option_values = (
                                                    await frame.locator('//' + dom_element.xpath).nth(0).select_option(value=match_value, timeout=2_000)
                                                )
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass

                                # Helper: verify and detect visibility/hidden
                                verify_info = await _safe_frame_evaluate(
                                    frame,
                                    "(params) => {\n"
                                    "  const el = document.evaluate(params.xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;\n"
                                    "  if (!el) return { ok:false, text:'', value:'', hidden:true, reason:'not_found' };\n"
                                    "  const tagOk = el.tagName && el.tagName.toLowerCase() === 'select';\n"
                                    "  const rect = el.getBoundingClientRect();\n"
                                    "  const style = window.getComputedStyle(el);\n"
                                    "  const hidden = (rect.width === 0 || rect.height === 0 || style.visibility === 'hidden' || style.display === 'none');\n"
                                    "  let selText = '', selValue = '';\n"
                                    "  if (tagOk) {\n"
                                    "    selText = (el.options[el.selectedIndex]?.text || '').trim();\n"
                                    "    selValue = el.value;\n"
                                    "  }\n"
                                    "  const ok = (selText === params.target) || (selValue === params.target);\n"
                                    "  return { ok, text: selText, value: selValue, hidden };\n"
                                    "}",
                                    { 'xpath': dom_element.xpath, 'target': text }
                                )

                                # If not ok or the select is hidden (custom widget), try overlay menu click by text
                                if not (verify_info or {}).get('ok', False) or (verify_info or {}).get('hidden', False):
                                    try:
                                        # Open the dropdown if possible
                                        if handle is not None:
                                            try:
                                                await handle.click(timeout=1_500)
                                            except Exception:
                                                pass
                                        # Find a visible option in any overlay/listbox matching the text (exact)
                                        click_res = await _safe_frame_evaluate(
                                            frame,
                                            '''(targetText) => {
                                                function isVisible(el){
                                                  const r = el.getBoundingClientRect();
                                                  const cs = getComputedStyle(el);
                                                  return r.width>0 && r.height>0 && cs.visibility!=='hidden' && cs.display!=='none';
                                                }
                                                const candidates = [];
                                                // ARIA options anywhere in document
                                                document.querySelectorAll('[role="option"]').forEach(el=>{ if(isVisible(el)) candidates.push(el); });
                                                // Common widget classes (PrimeFaces/other)
                                                document.querySelectorAll('.ui-selectonemenu-item, .select2-results__option, li[role="option"]').forEach(el=>{ if(isVisible(el)) candidates.push(el); });
                                                // De-dup
                                                const uniq = Array.from(new Set(candidates));
                                                for(const el of uniq){
                                                  const txt = (el.textContent||'').trim();
                                                  if(txt === targetText){
                                                    el.scrollIntoView({block:'nearest'});
                                                    el.click();
                                                    try { el.dispatchEvent(new MouseEvent('click', {bubbles:true})); } catch(e) {}
                                                    return {clicked:true};
                                                  }
                                                }
                                                return {clicked:false};
                                            }''',
                                            text
                                        )
                                        try:
                                            overlay_clicked = bool(click_res and click_res.get('clicked'))
                                        except Exception:
                                            overlay_clicked = False
                                        # Re-verify after overlay click
                                        verify_info = await _safe_frame_evaluate(
                                            frame,
                                            "(params) => {\n"
                                            "  const el = document.evaluate(params.xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;\n"
                                            "  if (!el) return { ok:false, text:'', value:'', reason:'not_found' };\n"
                                            "  const selText = (el.options[el.selectedIndex]?.text || '').trim();\n"
                                            "  const selValue = el.value;\n"
                                            "  const ok = (selText === params.target) || (selValue === params.target);\n"
                                            "  return { ok, text: selText, value: selValue };\n"
                                            "}",
                                            { 'xpath': dom_element.xpath, 'target': text }
                                        )
                                    except Exception:
                                        pass

                                final_text = (verify_info or {}).get('text', '')
                                final_value = (verify_info or {}).get('value', '')
                                ok = bool((verify_info or {}).get('ok', False))
                                # Treat a successful select_option or overlay click as success
                                try:
                                    if selected_option_values and isinstance(selected_option_values, list) and len(selected_option_values) > 0:
                                        ok = True
                                except Exception:
                                    pass
                                if overlay_clicked:
                                    ok = True

                                msg = f"selected option {text} with value {final_value or (selected_option_values if selected_option_values else '[unknown]')}"
                                logger.info(msg + f' in frame {frame_index}')

                                return ActionResult(
                                    extracted_content=msg,
                                    include_in_memory=True,
                                    long_term_memory=f"Selected option '{text}'",
                                    # success omitted on success; State will mark failures via success=False
                                    **({} if ok else {"success": False})
                                )

                            elif element_info.get('type') == 'aria':
                                # Handle ARIA menu
                                stealth_mgr = getattr(browser_session, '_stealth_manager', None)
                                stealth_enabled = bool(getattr(browser_session.browser_profile, 'stealth', False))
                                used_stealth_path = False

                                try:
                                    # Ensure menu is open: click the element itself
                                    trigger_handle = await frame.locator('//' + dom_element.xpath).nth(0).element_handle()
                                    if trigger_handle is not None:
                                        bbox = await trigger_handle.bounding_box()
                                        if bbox:
                                            cx = bbox['x'] + bbox['width'] / 2
                                            cy = bbox['y'] + bbox['height'] / 2
                                            try:
                                                if stealth_enabled and stealth_mgr is not None:
                                                    await stealth_mgr.execute_human_like_click(page, (cx, cy))
                                                    used_stealth_path = True
                                                else:
                                                    await trigger_handle.click(timeout=1_500)
                                            except Exception:
                                                # Fallback to simple click if stealth or primary click fails
                                                await trigger_handle.click(timeout=1_500)

                                    # Find target menu item center coordinates
                                    target_rect = await _safe_frame_evaluate(
                                        frame,
                                        '''(params) => {
                                            const { xpath, targetText } = params;
                                            const el = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                            if (!el) return null;
                                            const items = el.querySelectorAll('[role="menuitem"], [role="option"]');
                                            for (const item of items) {
                                                const txt = (item.textContent || '').trim();
                                                if (txt === targetText) {
                                                    item.scrollIntoView({block: 'nearest'});
                                                    const r = item.getBoundingClientRect();
                                                    return { x: r.left + r.width/2, y: r.top + r.height/2 };
                                                }
                                            }
                                            return null;
                                        }''',
                                        { 'xpath': dom_element.xpath, 'targetText': text }
                                    )

                                    if target_rect and isinstance(target_rect, dict) and 'x' in target_rect and 'y' in target_rect:
                                        try:
                                            if stealth_enabled and stealth_mgr is not None:
                                                await stealth_mgr.execute_human_like_click(page, (target_rect['x'], target_rect['y']))
                                                used_stealth_path = True
                                            else:
                                                await page.mouse.click(target_rect['x'], target_rect['y'])
                                                used_stealth_path = True
                                        except Exception:
                                            used_stealth_path = False
                                except Exception:
                                    used_stealth_path = False

                                if not used_stealth_path:
                                    # Fallback: original JS click
                                    click_aria_item_js = """
                                        (params) => {
                                            const { xpath, targetText } = params;
                                            try {
                                                const element = document.evaluate(xpath, document, null,
                                                    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                                if (!element) return {success: false, error: 'Element not found'};

                                                // Find all menu items
                                                const menuItems = element.querySelectorAll('[role="menuitem"], [role="option"]');

                                                for (const item of menuItems) {
                                                    const itemText = item.textContent.trim();
                                                    if (itemText === targetText) {
                                                        // Simulate click on the menu item
                                                        item.click();

                                                        // Also try dispatching a click event in case the click handler needs it
                                                        const clickEvent = new MouseEvent('click', {
                                                            view: window,
                                                            bubbles: true,
                                                            cancelable: true
                                                        });
                                                        item.dispatchEvent(clickEvent);

                                                        return {
                                                            success: true,
                                                            message: `Clicked menu item: ${targetText}`
                                                        };
                                                    }
                                                }

                                                return {
                                                    success: false,
                                                    error: `Menu item with text '${targetText}' not found`
                                                };
                                            } catch (e) {
                                                return {success: false, error: e.toString()};
                                            }
                                        }
                                    """

                                    result = await _safe_frame_evaluate(
                                        frame, click_aria_item_js, {'xpath': dom_element.xpath, 'targetText': text}
                                    )

                                    if result.get('success'):
                                        msg = result.get('message', f'Selected ARIA menu item: {text}')
                                        logger.info(msg + f' in frame {frame_index}')
                                        return ActionResult(
                                            extracted_content=msg,
                                            include_in_memory=True,
                                            long_term_memory=f"Selected menu item '{text}'",
                                        )
                                    else:
                                        logger.error(f'Failed to select ARIA menu item: {result.get("error")}')
                                        continue

                        elif element_info:
                            logger.error(f'Frame {frame_index} error: {element_info.get("error")}')
                            continue

                    except Exception as frame_e:
                        logger.error(f'Frame {frame_index} attempt failed: {str(frame_e)}')
                        logger.error(f'Frame type: {type(frame)}')
                        logger.error(f'Frame URL: {frame.url}')

                    frame_index += 1

                msg = f"Could not select option '{text}' in any frame"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=msg, success=False)

            except Exception as e:
                msg = f'Selection failed: {str(e)}'
                logger.error(msg)
                raise BrowserError(msg)

        @self.registry.action('Google Sheets: Get the contents of the entire sheet', domains=['https://docs.google.com'])
        async def read_sheet_contents(page: Page):
            # Ensure grid focus, then Select All and Copy
            await page.keyboard.press('Enter')
            await page.keyboard.press('Escape')
            await page.keyboard.press('ControlOrMeta+A')
            await page.keyboard.press('ControlOrMeta+C')

            # Clipboard can lag; poll briefly
            extracted_tsv = ''
            for _ in range(15):
                try:
                    extracted_tsv = await page.evaluate('() => navigator.clipboard.readText()')
                except Exception:
                    extracted_tsv = ''
                if extracted_tsv:
                    break
                await asyncio.sleep(0.1)

            return ActionResult(
                extracted_content=extracted_tsv,
                include_in_memory=True,
                long_term_memory='Retrieved sheet contents',
                include_extracted_content_only_once=True,
            )

        @self.registry.action('Google Sheets: Get the contents of a cell or range of cells', domains=['https://docs.google.com'])
        async def read_cell_contents(cell_or_range: str, browser_session: BrowserSession):
            page = await browser_session.get_current_page()

            await select_cell_or_range(cell_or_range=cell_or_range, page=page)

            # Copy and read clipboard with small retry window
            await page.keyboard.press('ControlOrMeta+C')
            extracted_tsv = ''
            for _ in range(15):
                try:
                    extracted_tsv = await page.evaluate('() => navigator.clipboard.readText()')
                except Exception:
                    extracted_tsv = ''
                if extracted_tsv != '':
                    break
                await asyncio.sleep(0.1)

            return ActionResult(
                extracted_content=extracted_tsv,
                include_in_memory=True,
                long_term_memory=f'Retrieved contents from {cell_or_range}',
                include_extracted_content_only_once=True,
            )

        @self.registry.action(
            'Google Sheets: Update the content of a cell or range of cells', domains=['https://docs.google.com']
        )
        async def update_cell_contents(cell_or_range: str, new_contents_tsv: str, browser_session: BrowserSession):
            page = await browser_session.get_current_page()

            await select_cell_or_range(cell_or_range=cell_or_range, page=page)

            # Write to clipboard and do a real paste via keyboard to let Sheets handle TSV parsing.
            def _normalize_tsv(s: str) -> str:
                return (s or '').replace('\r\n', '\n').replace('\r', '\n').rstrip('\n')

            desired = _normalize_tsv(new_contents_tsv)
            last_clip = ''
            success = False
            for attempt in range(3):
                try:
                    await page.evaluate(
                        """
                        async (text) => {
                            await navigator.clipboard.writeText(text);
                            return true;
                        }
                        """,
                        new_contents_tsv,
                    )
                except Exception as e:
                    logger.warning(f'Clipboard write failed (attempt {attempt+1}): {e}')
                    await asyncio.sleep(0.2)
                    continue

                await page.keyboard.press('ControlOrMeta+V')
                await asyncio.sleep(0.3)

                # Verify by copying back and comparing
                await page.keyboard.press('ControlOrMeta+C')
                copied = ''
                for _ in range(10):
                    try:
                        copied = await page.evaluate('() => navigator.clipboard.readText()')
                    except Exception:
                        copied = ''
                    if copied:
                        break
                    await asyncio.sleep(0.1)

                last_clip = _normalize_tsv(copied)
                if last_clip == desired:
                    success = True
                    break

                logger.info(
                    f'Google Sheets paste verification mismatch on attempt {attempt+1}. Will retry. '\
                    f'Expected (normalized) length={len(desired)}, got length={len(last_clip)}'
                )
                await asyncio.sleep(0.3)

            if not success:
                msg = (
                    f"Failed to update cells {cell_or_range}: pasted content didn't match after retries. "
                    f"Expected={desired[:200]!r}... vs Got={last_clip[:200]!r}..."
                )
                return ActionResult(
                    extracted_content=msg,
                    include_in_memory=True,
                    long_term_memory=msg,
                    success=False,
                )

            return ActionResult(
                extracted_content=f'Updated cells: {cell_or_range} = {new_contents_tsv}',
                include_in_memory=False,
                long_term_memory=f'Updated cells {cell_or_range} with {new_contents_tsv}',
            )

        @self.registry.action('Google Sheets: Clear whatever cells are currently selected', domains=['https://docs.google.com'])
        async def clear_cell_contents(cell_or_range: str, browser_session: BrowserSession):
            page = await browser_session.get_current_page()

            await select_cell_or_range(cell_or_range=cell_or_range, page=page)

            await page.keyboard.press('Backspace')
            await asyncio.sleep(0.2)

            # Verify cleared by copying and checking no non-whitespace, non-tab content remains
            await page.keyboard.press('ControlOrMeta+C')
            content = ''
            for _ in range(10):
                try:
                    content = await page.evaluate('() => navigator.clipboard.readText()')
                except Exception:
                    content = ''
                if content is not None:
                    break
                await asyncio.sleep(0.1)

            normalized = (content or '').replace('\r\n', '\n').replace('\r', '\n').replace('\t', '').strip()
            if normalized != '':
                msg = f"Attempted to clear {cell_or_range}, but cells still contain data (len={len(content or '')})."
                return ActionResult(
                    extracted_content=msg,
                    include_in_memory=True,
                    long_term_memory=msg,
                    success=False,
                )

            return ActionResult(
                extracted_content=f'Cleared cells: {cell_or_range}',
                include_in_memory=False,
                long_term_memory=f'Cleared cells {cell_or_range}',
            )

        @self.registry.action('Google Sheets: Select a specific cell or range of cells', domains=['https://docs.google.com'])
        async def select_cell_or_range(cell_or_range: str, page: Page):
            # Ensure we are not in edit mode and the grid has focus
            await page.keyboard.press('Enter')
            await page.keyboard.press('Escape')
            await asyncio.sleep(0.1)

            # Open the "Go to range" dialog (platform-aware) and type the range
            opened = False
            for chord in ('ControlOrMeta+G', 'F5'):
                try:
                    await page.keyboard.press(chord)
                    await asyncio.sleep(0.25)
                    await page.keyboard.type(cell_or_range, delay=0.05)
                    await asyncio.sleep(0.2)
                    await page.keyboard.press('Enter')
                    opened = True
                    break
                except Exception:
                    continue

            await asyncio.sleep(0.3)
            await page.keyboard.press('Escape')  # attempt to close the popup if still open

            # Light verification: try copying and ensure clipboard read succeeds (implies selection exists)
            await page.keyboard.press('ControlOrMeta+C')
            ok = False
            for _ in range(10):
                try:
                    clip = await page.evaluate('() => navigator.clipboard.readText()')
                except Exception:
                    clip = None
                if clip is not None:
                    ok = True
                    break
                await asyncio.sleep(0.1)

            msg = f'Selected cells: {cell_or_range}' if opened and ok else f'Attempted to select cells: {cell_or_range}'
            return ActionResult(
                extracted_content=msg,
                include_in_memory=False,
                long_term_memory=msg,
                success=opened and ok,
            )

        @self.registry.action(
            'Google Sheets: Fallback method to type text into (only one) currently selected cell',
            domains=['https://docs.google.com'],
        )
        async def fallback_input_into_single_selected_cell(text: str, page: Page):
            await page.keyboard.type(text, delay=0.1)
            await page.keyboard.press('Enter')  # make sure to commit the input so it doesn't get overwritten by the next action
            await page.keyboard.press('ArrowUp')
            return ActionResult(
                extracted_content=f'Inputted text {text}',
                include_in_memory=False,
                long_term_memory=f"Inputted text '{text}' into cell",
            )

        # Task Integration as Action: Gather Structured Data (Reservoir -> Source -> Sink)
        class GatherStructuredDataParams(BaseModel):
            sheet_url: str | None = Field(
                default=None,
                description='Optional Google Sheet URL for Tab C (Sink). If omitted, sink is skipped.',
            )
            targets: list[str] = Field(..., description='List of target URLs (strings).')
            titles: list[str] | None = Field(
                default=None, description='Optional list of titles (same length as targets).'
            )

        @self.registry.action(
            'Gather structured data: Reservoir -> Source -> Sink (optional). Provide targets, optional titles, and optional sheet_url.',
            param_model=GatherStructuredDataParams,
        )
        async def gather_structured_data(params: GatherStructuredDataParams, browser_session: BrowserSession):
            try:
                from browser_use.agent.tasks.gather_links_task import (
                    GatherStructuredDataTask as _GatherTask,
                    ReservoirSeedInput as _SeedIn,
                    seed_reservoir as _seed,
                )

                # Build task with current controller + browser
                task = _GatherTask(controller=self, browser=browser_session)

                # Apply sink configuration if provided
                if params.sheet_url:
                    try:
                        # Let pydantic validate later; we keep as str here
                        task.ctx.sink.sheet_url = params.sheet_url  # type: ignore[assignment]
                    except Exception:
                        task.ctx.sink.sheet_url = None  # type: ignore[assignment]

                # Seed reservoir from user input (pydantic will validate HttpUrls)
                seed = _SeedIn(targets=params.targets, titles=params.titles)
                task.ctx = _seed(task.ctx, seed)

                # Execute orchestrated flow
                tr = await task.run()

                # Summarize outcome for LLM
                written = [k for k in task.ctx.cache if k.startswith('sink:') and not k.startswith('sink:error:')]
                errors = {k: task.ctx.cache[k] for k in task.ctx.cache if k.startswith('sink:error:')}
                src_count = len([k for k in task.ctx.cache if k.startswith('source:')])
                msg = (
                    f"Gathered {src_count} items; wrote {len(written)} rows"
                    + (f"; {len(errors)} sink errors" if errors else "")
                )

                # Include a compact manifest pointer (kept in ctx cache)
                return ActionResult(
                    extracted_content=msg,
                    include_in_memory=True,
                    long_term_memory=msg,
                    success=bool(tr and tr.success),
                )
            except Exception as e:
                err = f"gather_structured_data failed: {e}"
                logger.error(err, exc_info=True)
                return ActionResult(extracted_content=err, include_in_memory=True, success=False)

    # Custom done action for structured output
    def _register_done_action(self, output_model: type[T] | None, display_files_in_done_text: bool = True):
        if output_model is not None:
            self.display_files_in_done_text = display_files_in_done_text

            @self.registry.action(
                'Complete task - with return text and if the task is finished (success=True) or not yet completely finished (success=False), because last step is reached',
                param_model=StructuredOutputAction[output_model],
            )
            async def done(params: StructuredOutputAction):
                # Exclude success from the output JSON since it's an internal parameter
                output_dict = params.data.model_dump()

                # Enums are not serializable, convert to string
                for key, value in output_dict.items():
                    if isinstance(value, enum.Enum):
                        output_dict[key] = value.value

                return ActionResult(
                    is_done=True,
                    success=params.success,
                    extracted_content=json.dumps(output_dict),
                    long_term_memory=f'Task completed. Success Status: {params.success}',
                )

        else:

            @self.registry.action(
                'Complete task - provide a summary of results for the user. Set success=True if task completed successfully, false otherwise. Text should be your response to the user summarizing results. Include files you would like to display to the user in files_to_display.',
                param_model=DoneAction,
            )
            async def done(params: DoneAction, file_system: FileSystem):
                user_message = params.text

                len_text = len(params.text)
                len_max_memory = 100
                memory = f'Task completed: {params.success} - {params.text[:len_max_memory]}'
                if len_text > len_max_memory:
                    memory += f' - {len_text - len_max_memory} more characters'

                attachments = []
                if params.files_to_display:
                    if self.display_files_in_done_text:
                        file_msg = ''
                        for file_name in params.files_to_display:
                            if file_name == 'todo.md':
                                continue
                            file_content = file_system.display_file(file_name)
                            if file_content:
                                file_msg += f'\n\n{file_name}:\n{file_content}'
                                attachments.append(file_name)
                        if file_msg:
                            user_message += '\n\nAttachments:'
                            user_message += file_msg
                        else:
                            logger.warning('Agent wanted to display files but none were found')
                    else:
                        for file_name in params.files_to_display:
                            if file_name == 'todo.md':
                                continue
                            file_content = file_system.display_file(file_name)
                            if file_content:
                                attachments.append(file_name)

                attachments = [str(file_system.get_dir() / file_name) for file_name in attachments]

                return ActionResult(
                    is_done=True,
                    success=params.success,
                    extracted_content=user_message,
                    long_term_memory=memory,
                    attachments=attachments,
                )

    def use_structured_output_action(self, output_model: type[T]):
        self._register_done_action(output_model)

    # Register ---------------------------------------------------------------

    def action(self, description: str, **kwargs):
        """Decorator for registering custom actions

        @param description: Describe the LLM what the function does (better description == better function calling)
        """
        return self.registry.action(description, **kwargs)

    # Act --------------------------------------------------------------------
    @observe_debug(ignore_input=True, ignore_output=True, name='act')
    @time_execution_sync('--act')
    async def act(
        self,
        action: ActionModel,
        browser_session: BrowserSession,
        #
        page_extraction_llm: BaseChatModel | None = None,
        sensitive_data: dict[str, str | dict[str, str]] | None = None,
        available_file_paths: list[str] | None = None,
        file_system: FileSystem | None = None,
        #
        context: Context | None = None,
    ) -> ActionResult:
        """Execute an action"""

        # Default page_extraction_llm to the main LLM if not provided
        if page_extraction_llm is None:
            try:
                settings = getattr(context, 'settings', None) or getattr(self, 'settings', None)
                candidate = getattr(settings, 'llm', None)
                if candidate is not None:
                    page_extraction_llm = candidate
            except Exception:
                pass

        for action_name, params in action.model_dump(exclude_unset=True).items():
            if params is not None:
                # Use Laminar span if available, otherwise use no-op context manager
                if Laminar is not None:
                    span_context = Laminar.start_as_current_span(
                        name=action_name,
                        input={
                            'action': action_name,
                            'params': params,
                        },
                        span_type='TOOL',
                    )
                else:
                    # No-op context manager when lmnr is not available
                    from contextlib import nullcontext

                    span_context = nullcontext()

                with span_context:
                    try:
                        result = await self.registry.execute_action(
                            action_name=action_name,
                            params=params,
                            browser_session=browser_session,
                            page_extraction_llm=page_extraction_llm,
                            file_system=file_system,
                            sensitive_data=sensitive_data,
                            available_file_paths=available_file_paths,
                            context=context,
                        )
                    except Exception as e:
                        result = ActionResult(success=False, error=str(e))

                    if Laminar is not None:
                        Laminar.set_span_output(result)

                if isinstance(result, str):
                    return ActionResult(extracted_content=result, success=False)
                elif isinstance(result, ActionResult):
                    # Preserve success=None for non-terminal actions; the StateManager will interpret
                    # None as neutral (neither success nor failure). Only explicit False indicates failure,
                    # and True is allowed only when is_done=True (enforced by the model validator).
                    return result
                elif result is None:
                    return ActionResult(success=False)
                else:
                    raise ValueError(f'Invalid action result type: {type(result)} of {result}')
        return ActionResult()

    @observe_debug(name='multi_act')
    @time_execution_sync('--multi_act')
    async def multi_act(
        self,
        actions: list[ActionModel],
        browser_session: BrowserSession,
        check_ui_stability: bool = True,  # Make this feature configurable
    page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[dict[str, str | dict[str, str]]] = None,
        available_file_paths: Optional[list[str]] = None,
        file_system: Optional[FileSystem] = None,
        context: Optional[Context] = None,
    ) -> list[ActionResult]:
        """
        Executes a sequence of actions, stopping on failure or 'done'.
        Includes an optional UI stability check to prevent acting on a stale DOM.
        """
        results: list[ActionResult] = []

        # Default page_extraction_llm to the main LLM if not provided
        if page_extraction_llm is None:
            try:
                settings = getattr(context, 'settings', None) or getattr(self, 'settings', None)
                candidate = getattr(settings, 'llm', None)
                if candidate is not None:
                    page_extraction_llm = candidate
            except Exception:
                pass

        # Get the initial state of the page before any actions in this batch are taken.
        cached_selector_map = await browser_session.get_selector_map()
        cached_path_hashes = {e.hash.branch_path_hash for e in cached_selector_map.values()}

        for i, action_model_instance in enumerate(actions):
            action_name = next(iter(action_model_instance.model_dump(exclude_unset=True)), 'unknown')

            try:
                # --- UI Stability Check ---
                # For every action after the first, if it targets an element by index,
                # we can check if the page has changed unexpectedly.
                action_params = action_model_instance.model_dump(exclude_unset=True).get(action_name, {})
                target_index = action_params.get('index')

                if i > 0 and check_ui_stability and target_index is not None:
                    # Get the current state of the page to compare against our cached initial state.
                    new_browser_state_summary = await browser_session.get_state_summary(cache_clickable_elements_hashes=False)
                    new_selector_map = new_browser_state_summary.selector_map

                    # Check 1: Has the specific element we are targeting changed?
                    orig_target = cached_selector_map.get(target_index)
                    new_target = new_selector_map.get(target_index)
                    if (orig_target and new_target and orig_target.hash.branch_path_hash != new_target.hash.branch_path_hash) or (not orig_target and new_target):
                        msg = f"Halting execution: Element at index {target_index} has changed since the start of the step. The agent should re-evaluate the page."
                        logger.info(msg)
                        results.append(ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=msg))
                        break  # Stop processing further actions in this batch

                    # Check 2: Have any new, unexpected elements appeared on the page?
                    new_path_hashes = {e.hash.branch_path_hash for e in new_selector_map.values()}
                    if not new_path_hashes.issubset(cached_path_hashes):
                        msg = "Halting execution: New elements have appeared on the page since the start of the step. The agent should re-evaluate the page."
                        logger.info(msg)
                        results.append(ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=msg))
                        break  # Stop processing further actions in this batch

                # --- Execute Action ---
                action_result = await self.act(
                    action=action_model_instance,
                    browser_session=browser_session,
                    page_extraction_llm=page_extraction_llm,
                    sensitive_data=sensitive_data,
                    available_file_paths=available_file_paths,
                    file_system=file_system,
                    context=context,
                )
                # Prevent premature 'done' if prior actions in this step did not verify success
                if action_name == 'done' and results:
                    last = results[-1]
                    if getattr(last, 'success', False) is False and not getattr(last, 'is_done', False):
                        caut_msg = (
                            "Halting: Previous action did not confirm success. Refusing to mark task as done."
                        )
                        logger.info(caut_msg)
                        results.append(
                            ActionResult(
                                extracted_content=caut_msg,
                                include_in_memory=True,
                                long_term_memory=caut_msg,
                                success=False,
                            )
                        )
                        break

                results.append(action_result)

                if action_result.error is not None or action_result.is_done:
                    reason = 'failed' if action_result.error is not None else "signaled 'done'"
                    logger.info(f"Action '{action_name}' {reason}. Halting further actions in this step.")
                    break

                # --- Wait Between Actions ---
                if i < len(actions) - 1 and browser_session.browser_profile.wait_between_actions > 0:
                    await asyncio.sleep(browser_session.browser_profile.wait_between_actions)

            except Exception as e:
                error_msg = f"Controller-level error executing action '{action_name}': {e}"
                logger.error(error_msg, exc_info=True)
                results.append(ActionResult(action=action_model_instance, success=False, error=error_msg))
                break

        return results
