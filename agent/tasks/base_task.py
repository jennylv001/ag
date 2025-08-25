from __future__ import annotations

"""
Core task abstractions for the agent.

Provides a minimal BaseTask interface and a lightweight TaskResult container.
Behavior is intentionally unspecified here; this module defines contracts only.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class TaskResult:
    """Generic result for a task step.

    Attributes:
        success: Whether this step (or the overall task) is successful.
        message: Optional human-readable note.
        data: Optional arbitrary payload for downstream consumers.
    """

    success: bool
    message: Optional[str] = None
    data: dict[str, Any] | None = field(default=None)


class BaseTask(Protocol):
    """Minimal task interface for incremental, step-based execution.

    Implementations should be idempotent per step where possible and manage
    their own internal state. The runner will poll `is_done()` and
    `succeeded()` to decide when to stop.
    """

    def step(self) -> Optional[TaskResult]:
        """Advance the task by a small unit of work.

        Returns an optional TaskResult. Returning None indicates no externally
        relevant result for this step (e.g., waiting, probing, or no-op).
        """

        ...

    def is_done(self) -> bool:
        """Return True when the task has reached a terminal condition."""

        ...

    def succeeded(self) -> bool:
        """Return True if the task finished successfully, False if it failed.

        If `is_done()` is False, the value is advisory and may be ignored.
        """

        ...
