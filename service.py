# filename: browser_use/controller/service.py
import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from browser_use.llm.base import BaseChatModel
from pydantic import BaseModel, Field as PydanticField

from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.registry.service import Registry
from browser_use.controller.registry.views import ActionModel

from .views import (
    ClickElementAction,
    CloseTabAction,
    DoneAction,
    DragDropAction,
    GoToUrlAction,
    InputTextAction,
    NoParamsAction,
    OpenTabAction,
    Position as DragDropPositionParam,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
    FileParameters,
    ReadFileParameters,
    ExcelParameters,
    LaunchNotepadParameters,
)

# Use the new task-based shim for captcha
from browser_use.agent.tasks.solve_captcha_tool import (
    SolveCaptchaAction,
    tool_solve_captcha,
)

from browser_use.local_pc_tools import (
    ToolError as LocalPCToolError,
    FileNotFoundInAgentDir,
    InvalidPathError,
    FileOperationError as LocalPCFileOperationError,
    ToolConfigurationError as LocalPCToolConfigurationError,
    list_agent_files as pc_tool_list_agent_files,
    read_agent_text_file as pc_tool_read_agent_text_file,
    write_agent_text_file as pc_tool_write_agent_text_file,
    read_excel_file as pc_tool_read_excel_file,
    write_excel_file as pc_tool_write_excel_file,
    launch_notepad as pc_tool_launch_notepad,
)
from browser_use.utils import time_execution_async

logger = logging.getLogger(__name__)
Context = TypeVar("Context")


class ExtractContentActionParams(BaseModel):
    goal: str
    should_strip_link_urls: bool = PydanticField(default=True)


class ScrollActionWithDirection(BaseModel):
    direction: str = PydanticField(
        description="Direction to scroll: 'up', 'down', 'top', or 'bottom'."
    )
    amount: Optional[int] = PydanticField(
        None,
        description="Pixel amount for 'up'/'down'. If None, scrolls one viewport height.",
    )


class ScrollToTextActionParams(BaseModel):
    text: str


class GetDropdownOptionsActionParams(BaseModel):
    index: int


class SelectDropdownOptionActionParams(BaseModel):
    index: int
    text: str


class WaitActionParams(BaseModel):
    seconds: int = PydanticField(default=3, ge=0, le=300)


SavePdfActionParams = NoParamsAction


class Controller(Generic[Context]):
    def _create_action_result(
        self,
        action_name: str,
        parameters: Optional[Dict[str, Any]],
        success: Optional[bool],
        is_done: bool = False,
        extracted_data: Optional[Any] = None,
        include_in_memory: bool = True,
        error_message: Optional[str] = None,
    ) -> ActionResult:
        extracted_content_str: Optional[str] = None
        raw_tool_outputs_dict: Optional[Dict[str, Any]] = None

        if success and extracted_data is not None:
            if isinstance(extracted_data, str):
                extracted_content_str = extracted_data
            elif isinstance(extracted_data, (list, tuple)):
                try:
                    extracted_content_str = json.dumps(
                        extracted_data, ensure_ascii=False
                    )
                    raw_tool_outputs_dict = {"list_data": extracted_data}
                except TypeError:
                    extracted_content_str = str(extracted_data)
                    raw_tool_outputs_dict = {
                        "stringified_list_data": extracted_content_str
                    }
            elif isinstance(extracted_data, dict):
                raw_tool_outputs_dict = extracted_data
                if "message" in extracted_data and isinstance(
                    extracted_data["message"], str
                ):
                    extracted_content_str = extracted_data["message"]
                elif "summary" in extracted_data and isinstance(
                    extracted_data["summary"], str
                ):
                    extracted_content_str = extracted_data["summary"]
                elif "content" in extracted_data and isinstance(
                    extracted_data["content"], str
                ):
                    extracted_content_str = extracted_data["content"]
                elif "filename" in extracted_data and isinstance(
                    extracted_data["filename"], str
                ):
                    extracted_content_str = (
                        f"Operation resulted in file: {extracted_data['filename']}"
                    )
                if extracted_content_str is None:
                    try:
                        extracted_content_str = f"Tool output (JSON): {json.dumps(extracted_data, ensure_ascii=False)}"
                    except TypeError:
                        extracted_content_str = (
                            f"Tool output (stringified): {str(extracted_data)}"
                        )
            else:
                extracted_content_str = str(extracted_data)

        if error_message is not None and not isinstance(error_message, str):
            error_message = str(error_message)

        serializable_parameters = {}
        if parameters:
            for k, v in parameters.items():
                if hasattr(v, "model_dump"):
                    serializable_parameters[k] = v.model_dump()
                else:
                    try:
                        json.dumps({k: v})
                        serializable_parameters[k] = v
                    except TypeError:
                        serializable_parameters[k] = str(v)

        # Enforce ActionResult invariant: success=True only allowed when is_done=True
        # For non-done successful actions, leave success unset (None)
        normalized_success: Optional[bool]
        if success is True and not is_done:
            normalized_success = None
        else:
            normalized_success = success

        return ActionResult(
            action_name=action_name,
            parameters=serializable_parameters if serializable_parameters else parameters,
            success=normalized_success,
            is_done=is_done,
            extracted_content=extracted_content_str,
            error=error_message,
            include_in_memory=include_in_memory,
            raw_tool_outputs=raw_tool_outputs_dict,
        )

    def __init__(
        self,
        exclude_actions: List[str] = [],
    ):
        self.registry = Registry[Context](exclude_actions)
        self._register_actions()

    def _register_actions(self):
        @self.registry.action(
            "Signals that the entire task is complete. Provide a final answer or summary, and indicate if the task was successful.",
            param_model=DoneAction,
        )
        async def done(params: DoneAction, browser: BrowserContext) -> ActionResult:
            return self._create_action_result(
                action_name="done",
                parameters=params.model_dump(),
                success=True,
                is_done=True,
                extracted_data={
                    "message": params.text or "Task marked as done.",
                    "overall_task_success": params.success,
                },
                include_in_memory=False,
            )

        @self.registry.action(
            'Performs an EXTERNAL Google search in the current tab using the provided query. Use this ONLY for general web searches, not for interacting with search bars or buttons on the current webpage.',
            param_model=SearchGoogleAction,
        )
        async def search_google(
            params: SearchGoogleAction, browser: BrowserContext
        ) -> ActionResult:
            action_name = "search_google"
            try:
                page = await browser.get_current_page()
                await page.goto(f"https://www.google.com/search?q={params.query}&udm=14")
                await page.wait_for_load_state("domcontentloaded", timeout=30000)
                msg = f'Searched externally on Google for "{params.query}".'
                logger.info(msg)
                return self._create_action_result(
                    action_name, params.model_dump(), success=True, extracted_data={"message": msg}
                )
            except Exception as e:
                logger.error(
                    f"Error during external search_google for '{params.query}': {e}",
                    exc_info=True,
                )
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            "Navigate to the specified URL in the current tab.", param_model=GoToUrlAction
        )
        async def go_to_url(
            params: GoToUrlAction, browser: BrowserContext
        ) -> ActionResult:
            action_name = "go_to_url"
            try:
                page = await browser.get_current_page()
                await page.goto(
                    params.url, wait_until="domcontentloaded", timeout=60000
                )
                msg = f"Navigated to {params.url}"
                logger.info(msg)
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=True,
                    extracted_data={"message": msg, "current_url": page.url},
                )
            except Exception as e:
                logger.error(
                    f"Error navigating to URL '{params.url}': {e}", exc_info=True
                )
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            "Navigate back to the previous page in the current tab history.",
            param_model=NoParamsAction,
        )
        async def go_back(_: NoParamsAction, browser: BrowserContext) -> ActionResult:
            action_name = "go_back"
            try:
                await browser.go_back()
                current_page = await browser.get_current_page()
                await current_page.wait_for_load_state(
                    "domcontentloaded", timeout=30000
                )
                msg = f"Navigated back. Current URL: {current_page.url}"
                logger.info(msg)
                return self._create_action_result(
                    action_name,
                    {},
                    success=True,
                    extracted_data={"message": msg, "current_url": current_page.url},
                )
            except Exception as e:
                logger.error(f"Error navigating back: {e}", exc_info=True)
                return self._create_action_result(
                    action_name, {}, success=False, error_message=str(e)
                )

        @self.registry.action(
            "Wait for a specified number of seconds (default is 3, max 300).",
            param_model=WaitActionParams,
        )
        async def wait(params: WaitActionParams, browser: BrowserContext) -> ActionResult:
            action_name = "wait"
            try:
                msg = f"Waiting for {params.seconds} seconds."
                logger.info(msg)
                await asyncio.sleep(params.seconds)
                return self._create_action_result(
                    action_name, params.model_dump(), success=True, extracted_data={"message": msg}
                )
            except Exception as e:
                logger.error(f"Error during wait: {e}", exc_info=True)
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            'Clicks an interactive element (button, link, etc.) on the CURRENT PAGE, identified by its `index`. Use this for on-page buttons, including those labeled "Search", "Submit", "Login", etc.',
            param_model=ClickElementAction,
        )
        async def click_element_by_index(
            params: ClickElementAction, browser: BrowserContext
        ) -> ActionResult:
            action_name = "click_element_by_index"
            element_change_details = None
            try:
                selector_map = await browser.get_selector_map()
                if params.index not in selector_map:
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=False,
                        error_message=f"Element with index {params.index} not found in prior observation map.",
                    )

                element_node = await browser.get_dom_element_by_index(params.index)
                if not element_node:
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=False,
                        error_message=f"Could not retrieve DOMElementNode for index {params.index}.",
                    )

                element_handle = await browser.get_locate_element(element_node)
                if not element_handle:
                    error_msg = f"Element at index {params.index} (Observed: {element_node.tag_name} '{element_node.get_all_text_till_next_clickable_element(max_depth=0)[:30].strip()}') could not be re-located on the current page for interaction."
                    logger.warning(error_msg)
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=False,
                        error_message=error_msg,
                        include_in_memory=True,
                    )

                live_tag_name = await element_handle.evaluate(
                    "el => el.tagName.toLowerCase()"
                )
                live_id = await element_handle.get_attribute("id")
                live_name = await element_handle.get_attribute("name")
                live_role = await element_handle.get_attribute("role")

                observed_tag_name = element_node.tag_name.lower()
                observed_id = element_node.attributes.get("id")
                observed_name = element_node.attributes.get("name")
                observed_role = element_node.attributes.get("role")

                significant_change_detected = False
                change_description = "No significant change detected by heuristic."

                if live_tag_name != observed_tag_name:
                    significant_change_detected = True
                    change_description = f"tag name changed (was '{observed_tag_name}', now '{live_tag_name}')"
                elif observed_id is not None and live_id != observed_id:
                    significant_change_detected = True
                    change_description = (
                        f"id changed (was '{observed_id}', now '{live_id or 'None'}')"
                    )
                elif observed_name is not None and live_name != observed_name:
                    significant_change_detected = True
                    change_description = f"name changed (was '{observed_name}', now '{live_name or 'None'}')"
                elif observed_role is not None and live_role != observed_role:
                    significant_change_detected = True
                    change_description = f"role changed (was '{observed_role}', now '{live_role or 'None'}')"

                if significant_change_detected:
                    change_warning_msg = f"Heuristic check indicates element at index {params.index} ({observed_tag_name}) may have changed since observation. Change: {change_description}. Original ID: '{observed_id}', Name: '{observed_name}', Role: '{observed_role}'. Current Live ID: '{live_id}', Name: '{live_name}', Role: '{live_role}'. Proceeding with action attempt, but agent should be aware of potential instability."
                    logger.warning(change_warning_msg)
                    element_change_details = {
                        "status": "element_potentially_changed_heuristic",
                        "change_description": change_description,
                        "original_attributes": {
                            "id": observed_id,
                            "name": observed_name,
                            "role": observed_role,
                            "tag": observed_tag_name,
                        },
                        "current_attributes": {
                            "id": live_id,
                            "name": live_name,
                            "role": live_role,
                            "tag": live_tag_name,
                        },
                        "message": change_warning_msg,
                    }

                page = await browser.get_current_page()
                initial_pages_count = len(page.context.pages)

                download_path = await browser._click_element_node(element_node)

                element_text_preview = (
                    element_node.get_all_text_till_next_clickable_element(max_depth=2)
                )

                msg_main = (
                    f'Clicked element with index {params.index} ("{element_text_preview}").'
                )
                full_message = msg_main
                if download_path:
                    full_message += f" A file download was triggered: {download_path}"

                logger.info(full_message)

                final_data = {"message": full_message}
                if download_path:
                    final_data["downloaded_file"] = download_path
                if element_change_details:
                    final_data["element_change_details"] = element_change_details

                if len(page.context.pages) > initial_pages_count:
                    new_tab_msg = (
                        "A new tab was opened and focus has been switched to it."
                    )
                    final_data["message"] += f" {new_tab_msg}"
                    logger.info(new_tab_msg)

                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=True,
                    extracted_data=final_data,
                    include_in_memory=True,
                )
            except Exception as e:
                element_repr = (
                    repr(element_node)
                    if "element_node" in locals() and element_node
                    else f"index {params.index}"
                )
                error_message_for_action = str(e)
                if element_change_details:
                    error_message_for_action = f"Element change detected ({element_change_details.get('change_description', 'unknown change')}) then click failed: {str(e)}"

                logger.warning(
                    f"Error clicking element {element_repr}: {error_message_for_action}",
                    exc_info=True,
                )
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=False,
                    error_message=error_message_for_action,
                    extracted_data=element_change_details,
                    include_in_memory=True,
                )

        @self.registry.action(
            "Input text into an interactive element (e.g., text field) specified by its index.",
            param_model=InputTextAction,
        )
        async def input_text(
            params: InputTextAction,
            browser: BrowserContext,
            sensitive_data: Optional[Dict[str, str]] = None,
        ) -> ActionResult:
            action_name = "input_text"
            log_text_param = params.text
            if sensitive_data and "<secret>" in params.text:
                log_text_display = "[SENSITIVE DATA]"
            else:
                log_text_display = f'"{log_text_param}"'

            try:
                selector_map = await browser.get_selector_map()
                if params.index not in selector_map:
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=False,
                        error_message=f"Element with index {params.index} not found.",
                    )

                element_node = await browser.get_dom_element_by_index(params.index)
                await browser._input_text_element_node(element_node, params.text)

                msg = f"Input text {log_text_display} into element at index {params.index} ({element_node.tag_name})."
                logger.info(msg)
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=True,
                    extracted_data={"message": msg},
                )
            except Exception as e:
                logger.error(
                    f"Error inputting text {log_text_display} into element at index {params.index}: {e}",
                    exc_info=True,
                )
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            "Save the current page as a PDF file. Filename is auto-generated.",
            param_model=SavePdfActionParams,
        )
        async def save_pdf(
            params: SavePdfActionParams, browser: BrowserContext
        ) -> ActionResult:
            action_name = "save_pdf"
            try:
                page = await browser.get_current_page()
                page_title = await page.title() or "untitled"
                slug_title = (
                    re.sub(r"[^\w\s-]", "", page_title.lower())
                    .strip()
                    .replace(" ", "-")[:50]
                )
                timestamp = int(asyncio.get_event_loop().time())
                pdf_filename = f"{slug_title}_{timestamp}.pdf"

                await page.emulate_media(media="screen")
                pdf_bytes = await page.pdf(format="A4", print_background=True)

                with open(pdf_filename, "wb") as f:
                    f.write(pdf_bytes)

                msg = f'Saved current page ({page.url}) as PDF: "{pdf_filename}" (in current directory).'
                logger.info(msg)
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=True,
                    extracted_data={"message": msg, "filename": pdf_filename},
                )
            except Exception as e:
                logger.error(f"Error saving page as PDF: {e}", exc_info=True)
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            "Switch focus to an existing tab using its page_id.",
            param_model=SwitchTabAction,
        )
        async def switch_tab(
            params: SwitchTabAction, browser: BrowserContext
        ) -> ActionResult:
            action_name = "switch_tab"
            try:
                await browser.switch_to_tab(params.page_id)
                page = await browser.get_current_page()
                await page.wait_for_load_state("domcontentloaded", timeout=30000)
                msg = f"Switched to tab {params.page_id} (URL: {page.url})."
                logger.info(msg)
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=True,
                    extracted_data={
                        "message": msg,
                        "current_url": page.url,
                        "current_page_id": params.page_id,
                    },
                )
            except Exception as e:
                logger.error(f"Error switching to tab {params.page_id}: {e}", exc_info=True)
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            "Open a URL in a new tab and switch focus to it.", param_model=OpenTabAction
        )
        async def open_tab(params: OpenTabAction, browser: BrowserContext) -> ActionResult:
            action_name = "open_tab"
            try:
                await browser.create_new_tab(params.url)
                page = await browser.get_current_page()
                await page.wait_for_load_state("domcontentloaded", timeout=60000)
                tabs_info = await browser.get_tabs_info()
                new_tab_info = next(
                    (t for t in tabs_info if t.url == page.url), None
                )
                new_page_id = new_tab_info.page_id if new_tab_info else -1

                msg = (
                    f"Opened new tab for {params.url} (ID: {new_page_id}). Focus switched."
                )
                logger.info(msg)
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=True,
                    extracted_data={
                        "message": msg,
                        "new_page_id": new_page_id,
                        "url": page.url,
                    },
                )
            except Exception as e:
                logger.error(
                    f"Error opening new tab for URL '{params.url}': {e}", exc_info=True
                )
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            "Close an existing tab using its page_id.", param_model=CloseTabAction
        )
        async def close_tab(
            params: CloseTabAction, browser: BrowserContext
        ) -> ActionResult:
            action_name = "close_tab"
            try:
                all_pages = browser.session.context.pages if browser.session else []
                if not (0 <= params.page_id < len(all_pages)):
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=False,
                        error_message=f"Invalid page_id {params.page_id} or browser session not found.",
                    )

                if len(all_pages) <= 1:
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=False,
                        error_message="Cannot close the last remaining tab.",
                    )

                page_to_close = all_pages[params.page_id]
                url_closed = page_to_close.url

                is_active_tab = browser.active_tab == page_to_close

                await page_to_close.close()

                if is_active_tab:
                    browser.active_tab = None
                    if browser.session and browser.session.context.pages:
                        await browser.switch_to_tab(0)

                msg = f"Closed tab page_id {params.page_id} (URL: {url_closed})."
                logger.info(msg)
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=True,
                    extracted_data={"message": msg, "closed_page_id": params.page_id},
                )
            except Exception as e:
                logger.error(f"Error closing tab {params.page_id}: {e}", exc_info=True)
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            "Scrolls the current page. Direction can be 'up', 'down', 'top', or 'bottom'. 'amount' is in pixels for 'up'/'down' (optional, defaults to viewport height).",
            param_model=ScrollActionWithDirection,
        )
        async def scroll_action(
            params: ScrollActionWithDirection, browser: BrowserContext
        ) -> ActionResult:
            action_name = "scroll_action"
            page = await browser.get_current_page()
            direction = params.direction.lower()
            amount_desc = ""
            try:
                if direction == "down":
                    scroll_value = (
                        params.amount if params.amount is not None else "window.innerHeight"
                    )
                    await page.evaluate(f"window.scrollBy(0, {scroll_value});")
                    amount_desc = (
                        f"by {params.amount} pixels"
                        if params.amount is not None
                        else "by one page height"
                    )
                elif direction == "up":
                    scroll_value = (
                        params.amount if params.amount is not None else "window.innerHeight"
                    )
                    await page.evaluate(f"window.scrollBy(0, -{scroll_value});")
                    amount_desc = (
                        f"by {params.amount} pixels"
                        if params.amount is not None
                        else "by one page height"
                    )
                elif direction == "top":
                    await page.evaluate("window.scrollTo(0, 0);")
                    amount_desc = "to top"
                elif direction == "bottom":
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                    amount_desc = "to bottom"
                else:
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=False,
                        error_message=f"Invalid scroll direction: '{params.direction}'.",
                    )

                await asyncio.sleep(0.5)
                msg = f"Scrolled page {direction} {amount_desc}".strip() + "."
                logger.info(msg)
                return self._create_action_result(
                    action_name, params.model_dump(), success=True, extracted_data={"message": msg}
                )
            except Exception as e:
                logger.error(f"Error scrolling {direction}: {e}", exc_info=True)
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            "Extracts textual content from the current page, guided by a specific goal or question. If the goal is vague, it summarizes the page.",
            param_model=ExtractContentActionParams,
        )
        async def extract_content(
            params: ExtractContentActionParams,
            browser: BrowserContext,
            page_extraction_llm: Optional[BaseChatModel],
        ) -> ActionResult:
            action_name = "extract_content"
            if not page_extraction_llm:
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=False,
                    error_message="Page extraction LLM not available.",
                )
            try:
                page = await browser.get_current_page()
                import markdownify

                strip_tags_list = ["script", "style", "nav", "footer", "aside"]
                if params.should_strip_link_urls:
                    strip_tags_list.extend(["a", "img"])

                html_content = await page.content()
                text_content = markdownify.markdownify(
                    html_content, strip=strip_tags_list, heading_style="atx"
                )

                for frame in page.frames:
                    if frame != page.main_frame and not frame.url.startswith("data:"):
                        try:
                            text_content += f"\n\n--- IFRAME: {frame.url} ---\n" + markdownify.markdownify(
                                await frame.content(),
                                strip=strip_tags_list,
                                heading_style="atx",
                            )
                        except Exception:
                            pass

                max_len = 15000
                text_content_for_llm = text_content[:max_len] + (
                    "... (truncated)" if len(text_content) > max_len else ""
                )

                prompt_str = PromptTemplate(
                    input_variables=["goal", "page_content"],
                    template="Goal: {goal}\n\nPage Content:\n{page_content}\n\nRelevant Extracted Information/Summary (respond directly with the information):",
                ).format(goal=params.goal, page_content=text_content_for_llm)

                # Using internal message types; legacy code path preserved for now

                llm_input_messages = [HumanMessage(content=prompt_str)]

                response = await page_extraction_llm.ainvoke(llm_input_messages)
                extracted = (
                    response.content if hasattr(response, "content") else str(response)
                )

                msg = f'Content extracted for goal: "{params.goal}". Preview: {extracted[:100]}...'
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=True,
                    extracted_data={"content": extracted, "message": msg},
                )
            except Exception as e:
                logger.error(f"Error in extract_content: {e}", exc_info=True)
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            'Sends special key presses or key combinations (e.g., "Enter", "Escape", "Control+A") to the page.',
            param_model=SendKeysAction,
        )
        async def send_keys(
            params: SendKeysAction, browser: BrowserContext
        ) -> ActionResult:
            action_name = "send_keys"
            try:
                page = await browser.get_current_page()
                await page.keyboard.press(params.keys)
                msg = f'Sent keys: "{params.keys}".'
                logger.info(msg)
                return self._create_action_result(
                    action_name, params.model_dump(), success=True, extracted_data={"message": msg}
                )
            except Exception as e:
                logger.error(f"Error sending keys '{params.keys}': {e}", exc_info=True)
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            "Scrolls the page to make the first visible occurrence of the specified text appear in the viewport.",
            param_model=ScrollToTextActionParams,
        )
        async def scroll_to_text(
            params: ScrollToTextActionParams, browser: BrowserContext
        ) -> ActionResult:
            action_name = "scroll_to_text"
            try:
                page = await browser.get_current_page()
                locator = page.get_by_text(params.text, exact=False)
                if await locator.count() > 0:
                    await locator.first.scroll_into_view_if_needed(timeout=10000)
                    await asyncio.sleep(0.5)
                    msg = f'Scrolled to text: "{params.text}".'
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=True,
                        extracted_data={"message": msg, "found_text": True},
                    )
                else:
                    msg = f'Text "{params.text}" not found on page for scrolling.'
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=True,
                        extracted_data={"message": msg, "found_text": False},
                    )
            except Exception as e:
                logger.error(
                    f"Error scrolling to text '{params.text}': {e}", exc_info=True
                )
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            "Retrieves all options (text and value) from a native HTML dropdown element specified by its index.",
            param_model=GetDropdownOptionsActionParams,
        )
        async def get_dropdown_options(
            params: GetDropdownOptionsActionParams, browser: BrowserContext
        ) -> ActionResult:
            action_name = "get_dropdown_options"
            try:
                selector_map = await browser.get_selector_map()
                if params.index not in selector_map:
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=False,
                        error_message=f"Element with index {params.index} not found.",
                    )

                dom_element = selector_map[params.index]
                if dom_element.tag_name.lower() != "select":
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=False,
                        error_message=f"Element at index {params.index} is not a 'select' element.",
                    )

                element_handle = await browser.get_locate_element(dom_element)
                if not element_handle:
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=False,
                        error_message=f"Could not locate select element for index {params.index}.",
                    )

                options_data = await element_handle.evaluate(
                    "(el) => Array.from(el.options).map(opt => ({text: opt.text, value: opt.value, index: opt.index, selected: opt.selected}))"
                )

                formatted_options = [
                    f"Text: '{opt['text']}' (Value: '{opt['value']}') {'[Selected]' if opt['selected'] else ''}"
                    for opt in options_data
                ]
                result_str = (
                    "\n".join(formatted_options) if formatted_options else "No options found."
                )
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=True,
                    extracted_content=result_str,
                    extracted_data={
                        "message": f"Found {len(options_data)} options.",
                        "options_list": options_data,
                    },
                )
            except Exception as e:
                logger.error(
                    f"Error getting dropdown options for index {params.index}: {e}",
                    exc_info=True,
                )
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            "Selects an option within a dropdown (specified by element index) by its visible text.",
            param_model=SelectDropdownOptionActionParams,
        )
        async def select_dropdown_option(
            params: SelectDropdownOptionActionParams, browser: BrowserContext
        ) -> ActionResult:
            action_name = "select_dropdown_option"
            try:
                selector_map = await browser.get_selector_map()
                if params.index not in selector_map:
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=False,
                        error_message=f"Element with index {params.index} not found.",
                    )

                dom_element = selector_map[params.index]
                if dom_element.tag_name.lower() != "select":
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=False,
                        error_message=f"Element at index {params.index} is not a 'select' element.",
                    )

                element_handle = await browser.get_locate_element(dom_element)
                if not element_handle:
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=False,
                        error_message=f"Could not locate select element for index {params.index}.",
                    )

                await element_handle.select_option(label=params.text, timeout=10000)
                selected_value = await element_handle.evaluate("el => el.value")
                msg = f"Selected option with text '{params.text}' (value: '{selected_value}') in dropdown index {params.index}."
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=True,
                    extracted_data={"message": msg},
                )
            except Exception as e:
                logger.error(
                    f"Error selecting option '{params.text}' in dropdown index {params.index}: {e}",
                    exc_info=True,
                )
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            "Performs a drag and drop operation using element selectors or coordinates.",
            param_model=DragDropAction,
        )
        async def drag_drop(
            params: DragDropAction, browser: BrowserContext
        ) -> ActionResult:
            action_name = "drag_drop"
            page = await browser.get_current_page()
            try:
                source_pos = (
                    params.element_source_offset.model_dump()
                    if params.element_source_offset
                    else None
                )
                target_pos_offset = (
                    params.element_target_offset.model_dump()
                    if params.element_target_offset
                    else None
                )

                if params.element_source and params.element_target:
                    await page.locator(params.element_source).drag_to(
                        page.locator(params.element_target),
                        source_position=source_pos,
                        target_position=target_pos_offset,
                        timeout=15000,
                    )
                    msg = f"Dragged '{params.element_source}' to '{params.element_target}'."
                elif (
                    params.coord_source_x is not None and params.coord_target_x is not None
                ):
                    await page.mouse.move(
                        float(params.coord_source_x), float(params.coord_source_y)
                    )
                    await page.mouse.down()
                    await asyncio.sleep(0.1)
                    await page.mouse.move(
                        float(params.coord_target_x),
                        float(params.coord_target_y),
                        steps=params.steps or 5,
                    )
                    await page.mouse.up()
                    msg = f"Dragged from ({params.coord_source_x},{params.coord_source_y}) to ({params.coord_target_x},{params.coord_target_y})."
                else:
                    return self._create_action_result(
                        action_name,
                        params.model_dump(),
                        success=False,
                        error_message="Invalid drag_drop parameters.",
                    )

                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=True,
                    extracted_data={"message": msg},
                )
            except Exception as e:
                logger.error(f"Error during drag_drop: {e}", exc_info=True)
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )

        @self.registry.action(
            "Lists files and directories in the agent's workspace (Agent_Dir).",
            param_model=NoParamsAction,
        )
        async def list_agent_files(_: NoParamsAction) -> ActionResult:
            action_name = "list_agent_files"
            try:
                items = await pc_tool_list_agent_files()
                return self._create_action_result(
                    action_name,
                    {},
                    success=True,
                    extracted_data=items or "Agent_Dir is empty.",
                )
            except LocalPCToolError as e:
                return self._create_action_result(
                    action_name, {}, success=False, error_message=str(e)
                )
            except Exception as e:
                return self._create_action_result(
                    action_name, {}, success=False, error_message=f"Unexpected error: {str(e)}"
                )

        @self.registry.action(
            "Reads text content from a file in Agent_Dir.",
            param_model=ReadFileParameters,
        )
        async def read_agent_text_file(params: ReadFileParameters) -> ActionResult:
            action_name = "read_agent_text_file"
            try:
                content = await pc_tool_read_agent_text_file(params.filename)
                return self._create_action_result(
                    action_name, params.model_dump(), success=True, extracted_data=content
                )
            except (
                FileNotFoundInAgentDir,
                InvalidPathError,
                LocalPCFileOperationError,
                LocalPCToolError,
            ) as e:
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )
            except Exception as e:
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=False,
                    error_message=f"Unexpected error: {str(e)}",
                )

        @self.registry.action(
            "Writes or appends text to a file in Agent_Dir.",
            param_model=FileParameters,
        )
        async def write_agent_text_file(params: FileParameters) -> ActionResult:
            action_name = "write_agent_text_file"
            try:
                msg = await pc_tool_write_agent_text_file(
                    params.filename, params.content, params.append
                )
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=True,
                    extracted_data={"message": msg},
                )
            except (
                InvalidPathError,
                LocalPCFileOperationError,
                LocalPCToolError,
            ) as e:
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )
            except Exception as e:
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=False,
                    error_message=f"Unexpected error: {str(e)}",
                )

        @self.registry.action(
            "Reads data from an Excel file in Agent_Dir.", param_model=ExcelParameters
        )
        async def read_excel_file(params: ExcelParameters) -> ActionResult:
            action_name = "read_excel_file"
            read_params_for_log = {
                "filename": params.filename,
                "sheet_name": params.sheet_name,
            }
            try:
                data_list = await pc_tool_read_excel_file(
                    filename=params.filename, sheet_name=params.sheet_name
                )
                preview_data = data_list[:5]
                summary_msg = f"Read {len(data_list)} rows from '{params.filename}'. Preview of first {len(preview_data)} rows shown."
                return self._create_action_result(
                    action_name,
                    read_params_for_log,
                    success=True,
                    extracted_content=summary_msg,
                    extracted_data=data_list,
                )
            except (
                FileNotFoundInAgentDir,
                InvalidPathError,
                LocalPCFileOperationError,
                LocalPCToolConfigurationError,
                LocalPCToolError,
            ) as e:
                return self._create_action_result(
                    action_name, read_params_for_log, success=False, error_message=str(e)
                )
            except Exception as e:
                return self._create_action_result(
                    action_name,
                    read_params_for_log,
                    success=False,
                    error_message=f"Unexpected error: {str(e)}",
                )

        @self.registry.action(
            "Writes a list of dictionaries to an Excel file in Agent_Dir.",
            param_model=ExcelParameters,
        )
        async def write_excel_file(params: ExcelParameters) -> ActionResult:
            action_name = "write_excel_file"
            if params.data is None:
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=False,
                    error_message="Parameter 'data' is required for writing to Excel.",
                )
            try:
                msg = await pc_tool_write_excel_file(
                    params.filename, params.data, params.sheet_name or "Sheet1"
                )
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=True,
                    extracted_data={"message": msg},
                )
            except (
                InvalidPathError,
                LocalPCFileOperationError,
                LocalPCToolConfigurationError,
                LocalPCToolError,
            ) as e:
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )
            except Exception as e:
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=False,
                    error_message=f"Unexpected error: {str(e)}",
                )

        # Disable the action form of solve_captcha for LLM tool schemas by attaching a page_filter
        # that always returns False. The SolveCaptchaTask path will be used instead.
        @self.registry.action(
            "Attempts to solve a CAPTCHA visible on the current page using LLM analysis of a screenshot.",
            param_model=SolveCaptchaAction,
            page_filter=lambda _page: False,
        )
        async def solve_captcha(
            params: SolveCaptchaAction,
            browser: BrowserContext,
            page_extraction_llm: BaseChatModel,
        ) -> ActionResult:
            action_name = "solve_captcha"
            try:
                return await tool_solve_captcha(self, params, browser, page_extraction_llm)
            except Exception as e:
                logger.error(
                    f"Error during solve_captcha controller action: {e}", exc_info=True
                )
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=False,
                    error_message=f"Unexpected error in solve_captcha: {str(e)}",
                )

        @self.registry.action(
            "Launches Notepad (Windows only), optionally opening a file from Agent_Dir.",
            param_model=LaunchNotepadParameters,
        )
        async def launch_notepad(params: LaunchNotepadParameters) -> ActionResult:
            action_name = "launch_notepad"
            try:
                msg = await pc_tool_launch_notepad(params.filename)
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=True,
                    extracted_data={"message": msg},
                    include_in_memory=False,
                )
            except (
                InvalidPathError,
                LocalPCToolConfigurationError,
                LocalPCToolError,
            ) as e:
                return self._create_action_result(
                    action_name, params.model_dump(), success=False, error_message=str(e)
                )
            except Exception as e:
                return self._create_action_result(
                    action_name,
                    params.model_dump(),
                    success=False,
                    error_message=f"Unexpected error: {str(e)}",
                )

    def action(self, description: str, **kwargs):
        return self.registry.action(description, **kwargs)

    @time_execution_async("--multi_act (controller)")
    async def multi_act(
        self,
        actions_from_llm_model_instances: List[ActionModel],
        browser_context: BrowserContext,
        page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        available_file_paths: Optional[List[str]] = None,
        context: Context | None = None,
    ) -> List[ActionResult]:
        results: List[ActionResult] = []

        for i, action_model_instance in enumerate(actions_from_llm_model_instances):
            action_data = action_model_instance.model_dump(exclude_unset=True)

            current_action_name = "unknown_action_in_sequence"
            current_params_dict: Dict[str, Any] = {}

            if not action_data or len(action_data) != 1:
                logger.warning(
                    f"Malformed ActionModel instance in sequence: {action_model_instance}. Skipping."
                )
                error_result = self._create_action_result(
                    action_name="malformed_action_model",
                    parameters={"malformed_action_data": str(action_model_instance)},
                    success=False,
                    error_message="Malformed action model instance received.",
                )
                results.append(error_result)
                continue

            try:
                current_action_name = next(iter(action_data.keys()))
                current_params_dict = action_data[current_action_name]
                if current_params_dict is None:
                    current_params_dict = {}

                log_params_display = current_params_dict
                if (
                    sensitive_data
                    and current_action_name == "input_text"
                    and "text" in current_params_dict
                    and "<secret>" in current_params_dict["text"]
                ):
                    log_params_display = {
                        **current_params_dict,
                        "text": "[SENSITIVE_DATA_PLACEHOLDER]",
                    }

                logger.info(
                    f"Executing action {i+1}/{len(actions_from_llm_model_instances)}: {current_action_name} with params: {log_params_display}"
                )

                action_result_obj: ActionResult = await self.registry.execute_action(
                    action_name=current_action_name,
                    params=current_params_dict,
                    browser=browser_context,
                    page_extraction_llm=page_extraction_llm,
                    sensitive_data=sensitive_data,
                    available_file_paths=available_file_paths,
                    context=context,
                )
                results.append(action_result_obj)

                # Halt criteria: explicit error or explicit done.
                # Do NOT treat success=None (neutral) as failure.
                if action_result_obj.is_done or (action_result_obj.error is not None):
                    if action_result_obj.is_done:
                        log_msg = f"Action '{current_action_name}' marked task as done. Halting further actions in this step."
                    else:
                        log_msg = (
                            f"Action '{current_action_name}' failed (Error: {action_result_obj.error}). "
                            "Halting further actions in this step."
                        )
                    logger.info(log_msg)
                    break

            except Exception as e:
                error_msg = f"Controller-level error executing action '{current_action_name}' in sequence: {str(e)}"
                logger.error(error_msg, exc_info=True)
                error_action_result = self._create_action_result(
                    action_name=current_action_name,
                    parameters=current_params_dict,
                    success=False,
                    error_message=error_msg,
                )
                results.append(error_action_result)
                break

            if (
                browser_context.config.wait_between_actions > 0
                and i < len(actions_from_llm_model_instances) - 1
            ):
                # Only wait if last action neither finished the task nor produced an error.
                if results:
                    last = results[-1]
                    if not (last.is_done or (last.error is not None)):
                        logger.debug(
                            f"Waiting for {browser_context.config.wait_between_actions}s before next action."
                        )
                        await asyncio.sleep(browser_context.config.wait_between_actions)

        return results

    async def act(
        self,
        action_model_instance: ActionModel,
        browser_context: BrowserContext,
        page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        available_file_paths: Optional[List[str]] = None,
        context: Context | None = None,
    ) -> ActionResult:
        results = await self.multi_act(
            actions_from_llm_model_instances=[action_model_instance],
            browser_context=browser_context,
            page_extraction_llm=page_extraction_llm,
            sensitive_data=sensitive_data,
            available_file_paths=available_file_paths,
            context=context,
        )
        return (
            results[0]
            if results
            else self._create_action_result(
                "empty_single_act_result",
                {},
                False,
                error_message="No result from single action execution.",
            )
        )
