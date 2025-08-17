from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from browser_use.dom.views import DEFAULT_INCLUDE_ATTRIBUTES

from browser_use.llm.messages import (
    BaseMessage,
)

if TYPE_CHECKING:
    pass


class HistoryItem(BaseModel):
    """Represents a single agent history item with its data and string representation"""

    step_number: int | None = None
    prior_action_assessment: str | None = None
    task_log: str | None = None
    next_goal: str | None = None
    action_results: str | None = None
    error: str | None = None
    system_message: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        """Validate that error and system_message are not both provided"""
        if self.error is not None and self.system_message is not None:
            raise ValueError('Cannot have both error and system_message at the same time')

    def to_string(self) -> str:
        """Get string representation of the history item"""
        step_str = f'step_{self.step_number}' if self.step_number is not None else 'step_unknown'

        if self.error:
            return f"""<{step_str}>
{self.error}
</{step_str}>"""
        elif self.system_message:
            return f"""<sys>
{self.system_message}
</sys>"""
        else:
            content_parts = [
                f'Prior Action Assessment: {self.prior_action_assessment}',
                f'Task Log: {self.task_log}',
                f'Next Goal: {self.next_goal}',
            ]

            if self.action_results:
                content_parts.append(self.action_results)

            content = '\n'.join(content_parts)

            return f"""<{step_str}>
{content}
</{step_str}>"""


class MessageHistory(BaseModel):
    """History of messages"""

    system_message: BaseMessage | None = None
    state_message: BaseMessage | None = None
    consistent_messages: list[BaseMessage] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_messages(self) -> list[BaseMessage]:
        """Get all messages"""
        messages = []
        if self.system_message:
            messages.append(self.system_message)
        if self.state_message:
            messages.append(self.state_message)
        messages.extend(self.consistent_messages)

        return messages

class MessageManagerSettings(BaseModel):
    """Configuration settings for the MessageManager."""
    max_input_tokens: int = Field(
        default=128000,
        description="The maximum number of tokens to be sent to the LLM."
    )
    max_clickable_elements_length: int = Field(
        default=12000,
        description="Max characters of clickable elements text to include in prompts (truncated to reduce latency)."
    )
    include_attributes: list[str] = Field(
        default=DEFAULT_INCLUDE_ATTRIBUTES,
        description="HTML attributes to include in the browser state representation."
    )
    message_context: Optional[str] = Field(
        default=None,
        description="Additional context to be included in the system prompt."
    )
    available_file_paths: List[str] = Field(
        default_factory=list,
        description="A list of file paths available to the agent."
    )
    max_history_items: Optional[int] = Field(
        default=10,
        description="The maximum number of history items to include in the prompt."
    )
    max_history_for_planner: Optional[int] = Field(
        default=5,
        description="Max history steps to include in planner prompts."
    )
    images_per_step: int = Field(
        default=1,
        description="The number of screenshots to include in the prompt for each step."
    )
    use_vision: bool = Field(
        default=True,
        description="Whether to include visual information (screenshots) in the prompt."
    )
    use_vision_for_planner: bool = Field(
        default=False,
        description="Enable vision for the planner LLM."
    )
    use_thinking: bool = Field(
        default=True,
        description="Whether to instruct the LLM to use a <thinking> process block."
    )
    image_tokens: int = Field(
        default=850,
        description="Estimated token cost for including an image in the prompt."
    )
    recent_message_window_priority: int = Field(
        default=5,
        description="Number of recent turns to prioritize during truncation."
    )


class MessageManagerState(BaseModel):
    """Holds the state for MessageManager"""

    history: MessageHistory = Field(default_factory=MessageHistory)
    tool_id: int = 1
    agent_history_items: list[HistoryItem] = Field(
        default_factory=lambda: [HistoryItem(step_number=0, system_message='Agent initialized')]
    )
    read_state_description: str = ''

    model_config = ConfigDict(arbitrary_types_allowed=True)
    local_system_notes: list[HistoryItem] = Field(default_factory=list)
