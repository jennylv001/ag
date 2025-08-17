from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Any

if TYPE_CHECKING:
    from browser_use.agent.views import ActionResult, AgentOutput, BrowserStateSummary, StepMetadata
    from browser_use.llm.messages import BaseMessage


# New Event Schema
@dataclass
class Event:
    """Base event class for the agent's event system."""
    step_token: int
    task_id: str = "root"  # Task context identifier for nested task tracking
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class PerceptionSnapshot(Event):
    """Event emitted when perception captures a browser state snapshot."""
    browser_state: BrowserStateSummary = field(default=None)
    new_downloaded_files: Optional[list[str]] = field(default=None)


@dataclass
class DecisionPlan(Event):
    """Event emitted when a decision has been made."""
    messages_to_llm: list[BaseMessage] = field(default_factory=list)
    llm_output: Optional[AgentOutput] = field(default=None)
    decision_type: str = field(default="action")  # "action", "reflection", "planning"


@dataclass
class LLMRequest(Event):
    """Event emitted when an LLM call is requested."""
    messages: list[BaseMessage] = field(default_factory=list)
    output_schema: Any = field(default=None)  # The expected output format/schema
    request_id: str = field(default="")  # Unique identifier to match response
    max_retries: int = field(default=2)
    request_type: str = field(default="action")  # "action", "reflection", "planning"


@dataclass
class LLMResponse(Event):
    """Event emitted when an LLM call is completed."""
    request_id: str = field(default="")  # Matches the LLMRequest
    success: bool = field(default=False)
    response: Optional[Any] = field(default=None)  # The actual LLM response
    error: Optional[str] = field(default=None)
    attempts: int = field(default=1)


@dataclass
class ActionExecuted(Event):
    """Event emitted when an action has been executed."""
    action_results: list[Any] = field(default_factory=list)  # Using Any to avoid circular imports
    success: bool = field(default=False)
    error: Optional[str] = field(default=None)


@dataclass
class StepFinalized(Event):
    """Event emitted when a complete step has been finalized."""
    step_number: Optional[int] = field(default=None)
    action_results: Optional[list[Any]] = field(default=None)
    browser_state: Optional[BrowserStateSummary] = field(default=None)
    step_metadata: Optional[StepMetadata] = field(default=None)


@dataclass
class ErrorEvent(Event):
    """Event emitted when an error occurs."""
    error_message: str = field(default="")
    error_type: str = field(default="")
    is_critical: bool = field(default=False)
    stack_trace: Optional[str] = field(default=None)


@dataclass
class Heartbeat(Event):
    """Heartbeat event published by long-running components to indicate health."""
    component_name: str = field(default="")
    # timestamp inherited from Event base class


@dataclass
class AssessmentUpdate(Event):
    """Assessor publishes fused, smoothed system signals for planning."""
    risk: float = 0.0          # 0..1
    opportunity: float = 0.0   # 0..1
    confidence: float = 0.0    # 0..1
    # Additional continuous signals
    stagnation: float = 0.0     # 0..1, high means little/no progress
    looping: float = 0.0        # 0..1, high means repetitive behavior
    # Provenance and interpretation aids
    contributors: list[str] = field(default_factory=list)  # short why-strings
    trend_window: int = 5
    screenshot_refs: list[str] = field(default_factory=list)
    visual_summary: str = ""
    change_map_ref: Optional[str] = None


# Legacy classes (preserved for compatibility)
@dataclass
class PerceptionOutput:
    """Data produced by the Perception component."""
    browser_state: BrowserStateSummary
    new_downloaded_files: Optional[list[str]] = None
    step_start_time: float = field(default_factory=time.monotonic)


@dataclass
class Decision:
    """Data produced by the DecisionMaker component."""
    messages_to_llm: list[BaseMessage]
    llm_output: Optional[AgentOutput] = None
    action_results: list[ActionResult] = field(default_factory=list)
    step_metadata: Optional[StepMetadata] = None
    browser_state: Optional[BrowserStateSummary] = None


@dataclass
class ActuationResult:
    """Data produced by the Actuator component."""
    action_results: list[ActionResult]
    llm_output: Optional[AgentOutput]
    browser_state: Optional[BrowserStateSummary]
    step_metadata: StepMetadata


# Export list for clean imports
__all__ = [
    # New Event Schema
    "Event",
    "PerceptionSnapshot",
    "DecisionPlan",
    "LLMRequest",
    "LLMResponse",
    "ActionExecuted",
    "StepFinalized",
    "ErrorEvent",
    "Heartbeat",
    "AssessmentUpdate",
    # Legacy classes
    "PerceptionOutput",
    "Decision",
    "ActuationResult"
]
