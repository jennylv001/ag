from __future__ import annotations

import enum
import json
import logging
from collections import deque
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Type, Union
from browser_use.exceptions import RateLimitError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    model_validator,
)
from typing_extensions import Literal, TypeVar

from browser_use.agent.message_manager.views import MessageManagerState
from browser_use.browser.views import BrowserStateHistory, BrowserStateSummary
from browser_use.controller.registry.views import ActionModel
from browser_use.dom.history_tree_processor.service import (
    DOMElementNode,
    DOMHistoryElement,
    HistoryTreeProcessor,
)
from browser_use.dom.views import SelectorMap
from browser_use.filesystem.file_system import FileSystemState
from browser_use.llm.base import BaseChatModel
from browser_use.tokens.views import UsageSummary

logger = logging.getLogger(__name__)

ToolCallingMethod = Literal['function_calling', 'json_mode', 'raw', 'auto']


class AgentStepInfo(BaseModel):
    """Information about the current step, passed to methods."""
    step_number: int
    max_steps: int

    def is_last_step(self) -> bool:
        return self.step_number >= self.max_steps - 1


class ActionResult(BaseModel):
    """The result of a single executed action, compatible with the existing system."""
    is_done: bool = False
    success: Optional[bool] = None
    error: Optional[str] = None
    attachments: Optional[List[str]] = None
    long_term_memory: Optional[str] = None
    extracted_content: Optional[str] = None
    include_extracted_content_only_once: bool = False
    include_in_memory: bool = False
    action: Optional[ActionModel] = None  # Add the action field

    @model_validator(mode='after')
    def validate_success(self):
        # If success is not explicitly set but there's an error, mark as failed
        if self.success is None and self.error is not None:
            self.success = False
        # If success is not explicitly set and no error, mark as successful
        elif self.success is None and self.error is None:
            self.success = True
        return self


class StepMetadata(BaseModel):
    """Metadata associated with a single agent step."""
    step_start_time: float
    step_end_time: float
    step_number: int

    @property
    def duration_seconds(self) -> float:
        return self.step_end_time - self.step_start_time


class AgentBrain(BaseModel):
    """Brain snapshot for downstream components using the new schema names."""
    thinking: Optional[str] = None
    prior_action_assessment: str
    task_log: str
    next_goal: str


class AgentOutput(BaseModel):
    """
    The "flat" structured output from the LLM, maintaining compatibility
    with the existing system while providing a structured 'brain' property.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    thinking: Optional[str] = None
    # New schema fields (preferred)
    prior_action_assessment: str
    task_log: str
    next_goal: str
    action: List[ActionModel] = Field(..., min_length=1)

    # No legacy parsing; inputs must provide the new schema keys.

    @property
    def current_state(self) -> AgentBrain:
        """For backward compatibility with components expecting a nested 'AgentBrain'."""
        return AgentBrain(
            thinking=self.thinking,
            prior_action_assessment=self.prior_action_assessment,
            task_log=self.task_log,
            next_goal=self.next_goal,
        )

    @staticmethod
    def type_with_custom_actions(custom_actions: Type[ActionModel]) -> Type[AgentOutput]:
        """Creates a dynamic AgentOutput type with a specific set of actions."""
        return create_model(
            'AgentOutput',
            __base__=AgentOutput,
            action=(List[custom_actions], Field(..., min_length=1)), # type: ignore
        )

    @staticmethod
    def type_with_custom_actions_no_thinking(custom_actions: Type[ActionModel]) -> Type[AgentOutput]:
        """Creates a dynamic AgentOutput type that omits the 'thinking' field from its schema."""
        class AgentOutputNoThinking(AgentOutput):
            @classmethod
            def model_json_schema(cls, **kwargs):
                schema = super().model_json_schema(**kwargs)
                if 'thinking' in schema.get('properties', {}):
                    del schema['properties']['thinking']
                return schema

        return create_model(
            'AgentOutputNoThinking',
            __base__=AgentOutputNoThinking,
            action=(List[custom_actions], Field(..., min_length=1)), # type: ignore
        )


class AgentHistory(BaseModel):
    """A record of a single, complete step in the agent's run."""
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    model_output: Optional[AgentOutput]
    result: list[ActionResult]
    state: Optional[BrowserStateHistory]
    metadata: Optional[StepMetadata] = None

    @staticmethod
    def get_interacted_element(model_output: AgentOutput, selector_map: SelectorMap) -> list[DOMHistoryElement | None]:
        elements = []
        for action in model_output.action:
            index = action.get_index()
            if index is not None and index in selector_map:
                el: DOMElementNode = selector_map[index]
                elements.append(HistoryTreeProcessor.convert_dom_element_to_history_element(el))
            else:
                elements.append(None)
        return elements

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom serialization handling circular references"""

        # Handle action serialization
        model_output_dump = None
        if self.model_output:
            action_dump = [action.model_dump(exclude_none=True) for action in self.model_output.action]
            model_output_dump = {
                'prior_action_assessment': self.model_output.prior_action_assessment,
                'task_log': self.model_output.task_log,
                'next_goal': self.model_output.next_goal,
                'action': action_dump,  # This preserves the actual action data
            }
            # Only include thinking if it's present
            if self.model_output.thinking is not None:
                model_output_dump['thinking'] = self.model_output.thinking

        return {
            'model_output': model_output_dump,
            'result': [r.model_dump(exclude_none=True) for r in self.result],
            'state': self.state.model_dump(mode='json') if self.state else None,
            'metadata': self.metadata.model_dump() if self.metadata else None,
        }


AgentStructuredOutput = TypeVar('AgentStructuredOutput', bound=BaseModel)

class ReflectionPlannerOutput(BaseModel):
    """The structured output from the planner/reflection LLM."""
    model_config = ConfigDict(extra='forbid')
    memory_summary: str
    next_goal: str
    effective_strategy: Optional[str] = None

class AgentHistoryList(BaseModel, Generic[AgentStructuredOutput]):
    """List of AgentHistory messages, i.e. the history of the agent's actions and thoughts."""

    history: Union[list[AgentHistory], deque[AgentHistory]] = Field(default_factory=list)
    usage: UsageSummary | None = None

    _output_model_schema: type[AgentStructuredOutput] | None = None

    def total_duration_seconds(self) -> float:
        """Get total duration of all steps in seconds"""
        total = 0.0
        for h in self.history:
            if h.metadata:
                total += h.metadata.duration_seconds
        return total

    def __len__(self) -> int:
        """Return the number of history items"""
        return len(self.history)

    def __str__(self) -> str:
        """Representation of the AgentHistoryList object"""
        return f'AgentHistoryList(all_results={self.action_results()}, all_model_outputs={self.model_actions()})'

    def __repr__(self) -> str:
        """Representation of the AgentHistoryList object"""
        return self.__str__()

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Saves the agent history to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use a custom encoder to handle complex types if necessary, though model_dump should handle most cases
        with path.open('w', encoding='utf-8') as f:
            # Manually construct the dictionary to ensure proper serialization via model_dump
            dump_data = {
                'history': [h.model_dump(mode='json') for h in self.history],
                'usage': self.usage.model_dump(mode='json') if self.usage else None
            }
            json.dump(dump_data, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path], output_model: Type[AgentOutput]) -> AgentHistoryList:
        """Loads agent history from a JSON file."""
        with Path(filepath).open('r', encoding='utf-8') as f:
            data = json.load(f)

        for h_data in data.get('history', []):
            if h_data.get('model_output'):
                h_data['model_output'] = output_model.model_validate(h_data['model_output'])

        return cls.model_validate(data)

    # def save_as_playwright_script(
    #   self,
    #   output_path: str | Path,
    #   sensitive_data_keys: list[str] | None = None,
    #   browser_config: BrowserConfig | None = None,
    #   context_config: BrowserContextConfig | None = None,
    # ) -> None:
    #   """
    #   Generates a Playwright script based on the agent's history and saves it to a file.
    #   Args:
    #       output_path: The path where the generated Python script will be saved.
    #       sensitive_data_keys: A list of keys used as placeholders for sensitive data
    #                            (e.g., ['username_placeholder', 'password_placeholder']).
    #                            These will be loaded from environment variables in the
    #                            generated script.
    #       browser_config: Configuration of the original Browser instance.
    #       context_config: Configuration of the original BrowserContext instance.
    #   """
    #   from browser_use.agent.playwright_script_generator import PlaywrightScriptGenerator

    #   try:
    #       serialized_history = self.model_dump()['history']
    #       generator = PlaywrightScriptGenerator(serialized_history, sensitive_data_keys, browser_config, context_config)

    #       script_content = generator.generate_script_content()
    #       path_obj = Path(output_path)
    #       path_obj.parent.mkdir(parents=True, exist_ok=True)
    #       with open(path_obj, 'w', encoding='utf-8') as f:
    #           f.write(script_content)
    #   except Exception as e:
    #       raise e

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom serialization that properly uses AgentHistory's model_dump"""
        return {
            'history': [h.model_dump(**kwargs) for h in self.history],
        }

    def last_action(self) -> None | dict:
        """Last action in history"""
        if self.history and self.history[-1].model_output:
            return self.history[-1].model_output.action[-1].model_dump(exclude_none=True)
        return None

    def errors(self) -> list[str | None]:
        """Get all errors from history, with None for steps without errors"""
        errors = []
        for h in self.history:
            step_errors = [r.error for r in h.result if r.error]

            # each step can have only one error
            errors.append(step_errors[0] if step_errors else None)
        return errors

    def final_result(self) -> None | str:
        """Final result from history"""
        if self.history and self.history[-1].result[-1].extracted_content:
            return self.history[-1].result[-1].extracted_content
        return None

    def is_done(self) -> bool:
        """Check if the agent is done"""
        if self.history and len(self.history[-1].result) > 0:
            last_result = self.history[-1].result[-1]
            return last_result.is_done is True
        return False

    def is_successful(self) -> bool | None:
        """Check if the agent completed successfully - the agent decides in the last step if it was successful or not. None if not done yet."""
        if self.history and len(self.history[-1].result) > 0:
            last_result = self.history[-1].result[-1]
            if last_result.is_done is True:
                return last_result.success
        return None

    def has_errors(self) -> bool:
        """Check if the agent has any non-None errors"""
        return any(error is not None for error in self.errors())

    def urls(self) -> list[str | None]:
        """Get all unique URLs from history"""
        return [h.state.url if h.state.url is not None else None for h in self.history]

    def screenshots(self, n_last: int | None = None, return_none_if_not_screenshot: bool = True) -> list[str | None]:
        """Get all screenshots from history"""
        if n_last == 0:
            return []
        if n_last is None:
            if return_none_if_not_screenshot:
                return [h.state.screenshot if h.state.screenshot is not None else None for h in self.history]
            else:
                return [h.state.screenshot for h in self.history if h.state.screenshot is not None]
        else:
            if return_none_if_not_screenshot:
                return [h.state.screenshot if h.state.screenshot is not None else None for h in self.history[-n_last:]]
            else:
                return [h.state.screenshot for h in self.history[-n_last:] if h.state.screenshot is not None]

    def action_names(self) -> list[str]:
        """Get all action names from history"""
        action_names = []
        for action in self.model_actions():
            actions = list(action.keys())
            if actions:
                action_names.append(actions[0])
        return action_names

    def model_thoughts(self) -> list[AgentBrain]:
        """Get all thoughts from history"""
        return [h.model_output.current_state for h in self.history if h.model_output]

    def model_outputs(self) -> list[AgentOutput]:
        """Get all model outputs from history"""
        return [h.model_output for h in self.history if h.model_output]

    # get all actions with params
    def model_actions(self) -> list[dict]:
        """Get all actions from history"""
        outputs = []

        for h in self.history:
            if h.model_output:
                for action, interacted_element in zip(h.model_output.action, h.state.interacted_element):
                    output = action.model_dump(exclude_none=True)
                    output['interacted_element'] = interacted_element
                    outputs.append(output)
        return outputs

    def action_results(self) -> list[ActionResult]:
        """Get all results from history"""
        results = []
        for h in self.history:
            results.extend([r for r in h.result if r])
        return results

    def extracted_content(self) -> list[str]:
        """Get all extracted content from history"""
        content = []
        for h in self.history:
            content.extend([r.extracted_content for r in h.result if r.extracted_content])
        return content

    def model_actions_filtered(self, include: list[str] | None = None) -> list[dict]:
        """Get all model actions from history as JSON"""
        if include is None:
            include = []
        outputs = self.model_actions()
        result = []
        for o in outputs:
            for i in include:
                if i == list(o.keys())[0]:
                    result.append(o)
        return result

    def number_of_steps(self) -> int:
        """Get the number of steps in the history"""
        return len(self.history)

    @property
    def structured_output(self) -> AgentStructuredOutput | None:
        """Get the structured output from the history

        Returns:
            The structured output if both final_result and _output_model_schema are available,
            otherwise None
        """
        final_result = self.final_result()
        if final_result is not None and self._output_model_schema is not None:
            return self._output_model_schema.model_validate_json(final_result)

        return None


class AgentError:
    """Container for agent error handling"""

    VALIDATION_ERROR = 'Invalid model output format. Please follow the correct schema.'
    RATE_LIMIT_ERROR = 'Rate limit reached. Waiting before retry.'
    NO_VALID_ACTION = 'No valid action found'

    @staticmethod
    def format_error(error: Exception, include_trace: bool = False) -> str:
        """Format error message based on error type and optionally include trace"""
        message = ''
        if isinstance(error, ValidationError):
            return f'{AgentError.VALIDATION_ERROR}\nDetails: {str(error)}'
        if isinstance(error, RateLimitError):
            return AgentError.RATE_LIMIT_ERROR
        if include_trace:
            return f'{str(error)}\nStacktrace:\n{traceback.format_exc()}'
        return f'{str(error)}'


class PerceptionOutput(BaseModel):
    """Output from the perception module containing browser state and file information."""
    browser_state: BrowserStateSummary
    new_downloaded_files: Optional[list[str]] = None
