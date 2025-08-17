from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any
from itertools import zip_longest

from browser_use.agent.prompts import AgentMessagePrompt, PlannerPrompt
from browser_use.agent.views import AgentHistory, AgentHistoryList
from browser_use.llm.messages import BaseMessage, SystemMessage
from browser_use.utils import (
    get_tool_examples,
    match_url_with_domain_pattern,
    redact_sensitive_data,
)

from .views import HistoryItem, MessageManagerSettings, MessageManagerState

if TYPE_CHECKING:
    from browser_use.browser.views import BrowserStateSummary
    from browser_use.filesystem.file_system import FileSystem

logger = logging.getLogger(__name__)

class MessageManager:
    """
    Constructs context-aware prompts by managing a prompt-optimized representation
    of the agent's history.
    """

    state: MessageManagerState
    settings: MessageManagerSettings
    task: str
    system_message: SystemMessage
    file_system: FileSystem | None

    def __init__(
        self,
        task: str,
        system_message: SystemMessage | None,
        settings: MessageManagerSettings,
        state: MessageManagerState | None = None,
        file_system: FileSystem | None = None,
        sensitive_data: dict[str, Any] | None = None,
    ):
        """Initializes the MessageManager."""
        self.task = task
        self.system_message = system_message
        self.settings = settings
        self.state = state or MessageManagerState()
        self.file_system = file_system
        self.sensitive_data = sensitive_data or {}
        self.last_input_messages = []
        if len(self.state.history.get_messages()) == 0:
            if self.system_message:
                self._add_message_with_type(self.system_message, 'system')
            # PATCH: Ensure user task/goal is present in the agent history at startup
            if self.task and not any('<user_request>' in item.system_message for item in self.state.agent_history_items):
                self.state.agent_history_items.append(
                    HistoryItem(step_number=0, system_message=f"<user_request>{self.task}</user_request>")
                )

    def _add_message_with_type(self, message, message_type):
        if message_type == 'system':
            self.state.history.system_message = message
        elif message_type == 'state':
            self.state.history.state_message = message
        elif message_type == 'consistent':
            self.state.history.consistent_messages.append(message)
        else:
            raise ValueError(f'Invalid message type: {message_type}')

    def add_new_task(self, new_task: str) -> None:
        self.task = new_task
        task_update_item = HistoryItem(system_message=f'User updated <user_request> to: {new_task}')
        self.state.local_system_notes.append(task_update_item)

    def add_human_guidance(self, guidance: str) -> None:
        """Adds a human guidance message to the history."""
        guidance_item = HistoryItem(system_message=f"Human guidance received: {guidance}")
        self.state.local_system_notes.append(guidance_item)

    def add_local_note(self, text: str) -> None:
        """Add a transient system note to steer the next LLM prompt."""
        try:
            note_item = HistoryItem(system_message=text)
            self.state.local_system_notes.append(note_item)
        except Exception:
            pass

    def update_history_representation(self, agent_history: AgentHistoryList) -> None:
        """
        Updates the manager's internal list of `HistoryItem` objects based on the
        agent's canonical history. This is the primary method for keeping the
        prompt context in sync with the agent's state.
        """
        self.state.agent_history_items.clear()
        self.state.agent_history_items.append(HistoryItem(step_number=0, system_message='Agent initialized'))

        self.state.agent_history_items.extend(self.state.local_system_notes)

        for step in agent_history.history:
            self._add_history_item(step)

    def _add_history_item(self, history_entry: AgentHistory) -> None:
        """Processes a single AgentHistory entry and adds it to the internal representation."""
        action_results_str = ""
        result_parts = []

        actions = history_entry.model_output.action if history_entry.model_output else []
        results = history_entry.result

        # Pair actions with their results. Handle cases where lengths might mismatch.
        for action, res in zip_longest(actions, results):
            if not res:
                continue

            action_name = "unknown_action"
            if action:
                # Extract action name from the Pydantic model's first key
                action_dump = action.model_dump(exclude_unset=True)
                if action_dump:
                    action_name = next(iter(action_dump))

            memory = res.long_term_memory
            if not memory:
                if res.error:
                    memory = f"Action '{action_name}' failed with error: {res.error[:150]}"
                else:
                    memory = f"Action '{action_name}' completed successfully."
            result_parts.append(memory)

        if result_parts:
            action_results_str = "\n".join(result_parts)

        history_item = HistoryItem(
            step_number=history_entry.metadata.step_number if history_entry.metadata else -1,
            prior_action_assessment=history_entry.model_output.current_state.prior_action_assessment if history_entry.model_output else "N/A",
            task_log=history_entry.model_output.current_state.task_log if history_entry.model_output else "N/A",
            next_goal=history_entry.model_output.current_state.next_goal if history_entry.model_output else "N/A",
            action_results=action_results_str
        )
        self.state.agent_history_items.append(history_item)

    @property
    def agent_history_description(self) -> str:
        """Builds the agent history string, respecting truncation settings."""
        items = self.state.agent_history_items
        limit = self.settings.max_history_items

        if not limit or len(items) <= limit:
            return '\n'.join(item.to_string() for item in items)

        omitted_count = len(items) - limit
        first_item = items[0].to_string()
        recent_items = [item.to_string() for item in items[-limit + 1:]]

        return "\n".join([
            first_item,
            f"<sys>... {omitted_count} older steps omitted ...</sys>",
            *recent_items
        ])

    def prepare_messages_for_llm(
        self,
        browser_state: BrowserStateSummary,
        current_goal: str | None,
        last_error: str | None,
        page_filtered_actions: str | None = None,
        agent_history_list: AgentHistoryList | None = None,
        current_task_id: str = "root",
        task_context: str | None = None,
    ) -> list[BaseMessage]:
        logger.debug(f"--- PREPARING MESSAGES FOR LLM ---")

        screenshots = []
        if agent_history_list and self.settings.images_per_step > 1:
            raw_screenshots = agent_history_list.screenshots(n_last=self.settings.images_per_step - 1, return_none_if_not_screenshot=False)
            screenshots = [s for s in raw_screenshots if s is not None]
        if browser_state.screenshot:
            screenshots.append(browser_state.screenshot)

        sensitive_data_description = self._get_sensitive_data_description(browser_state.url)

        logger.debug(f"--- CREATING AGENT MESSAGE PROMPT ---")
        agent_history_description = self.agent_history_description

        task = current_goal if current_goal is not None else self.task

        state_message_prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=self.file_system,
            agent_history_description=agent_history_description,
            task=task,
            current_task_id=current_task_id,
            task_context=task_context,
            step_info=None,
            include_attributes=self.settings.include_attributes,
            page_filtered_actions=page_filtered_actions,
            sensitive_data=sensitive_data_description,
            available_file_paths=self.settings.available_file_paths,
            screenshots=screenshots if self.settings.use_vision else [],
            max_clickable_elements_length=self.settings.max_clickable_elements_length,
            # If your AgentMessagePrompt supports it, add last_error=last_error,
        )
        logger.debug(f"--- GETTING USER MESSAGE FROM PROMPT ---")
        state_message = state_message_prompt.get_user_message(use_vision=self.settings.use_vision)

        logger.debug(f"--- ADDING MESSAGE TO HISTORY ---")
        self._add_message_with_type(state_message, 'state')


        messages = self.state.history.get_messages()
        logger.debug(f"--- FINAL MESSAGES TO LLM ---\n{messages}")
        return [self._filter_sensitive_data(msg) for msg in messages]


    def prepare_messages_for_planner(
        self,
    ) -> list[BaseMessage]:
        logger.warning("`prepare_messages_for_planner` is deprecated and should not be called.")
        return []

    def _get_sensitive_data_description(self, current_page_url: str | None) -> str:
        """Creates a prompt snippet describing available sensitive data placeholders."""
        if not self.sensitive_data or not current_page_url:
            return ""

        placeholders: set[str] = set()
        for key, value in self.sensitive_data.items():
            if isinstance(value, dict) and match_url_with_domain_pattern(current_page_url, key):
                placeholders.update(value.keys())
            else:
                placeholders.add(key)

        if not placeholders:
            return ""

        return f"Available sensitive data placeholders: {sorted(list(placeholders))}. Use by writing <secret>placeholder_name</secret>."

    def _filter_sensitive_data(self, message: BaseMessage) -> BaseMessage:
        """Filters sensitive data from a message's content."""
        if not self.sensitive_data or not message.content or not isinstance(message.content, str):
            return message

        message.content = redact_sensitive_data(message.content, self.sensitive_data)
        return message
