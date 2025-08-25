from __future__ import annotations

"""
Task-facing Pydantic models and views.

Central place to define schemas used by tasks (e.g., SolveCaptcha).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CaptchaInitialPlan(BaseModel):
    """The initial analysis and plan for attacking a captcha."""

    analysis: str = Field(
        ..., description="A brief analysis of the captcha image, identifying the target objects and instructions."
    )
    initial_click_indexes: List[int] = Field(
        ..., description="A list of element indexes to click for the first attempt."
    )
    verify_button_index: Optional[int] = Field(
        None,
        description="The index of the 'verify', 'submit', or 'next' button to be clicked after solving the puzzle.",
    )
    next_button_index: Optional[int] = Field(
        None,
        description="If present, the index of a 'Next' button used by rotating image challenges. Prefer verify when both exist.",
    )


class CaptchaFollowUpPlan(BaseModel):
    """The follow-up plan after an initial attempt, assessing the new state."""

    status: str = Field(
        ..., description="The current status of the captcha. Can be 'IN_PROGRESS', 'SUCCESS_CRITERIA_MET', or 'FAILED'."
    )
    analysis: str = Field(
        ..., description="A brief analysis of the new captcha state and the result of the previous clicks."
    )
    next_click_indexes: Optional[List[int]] = Field(
        None, description="If status is 'IN_PROGRESS', the list of new element indexes to click."
    )
    verify_button_index: Optional[int] = Field(
        None,
        description="Optional: if a 'verify/submit' button is visible/enabled now, return its index for immediate click.",
    )
    next_button_index: Optional[int] = Field(
        None,
        description="Optional: if a 'Next' button is visible for rotating challenges, return its index.",
    )
