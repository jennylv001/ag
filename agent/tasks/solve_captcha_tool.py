from __future__ import annotations

"""
Shim exposing the legacy SolveCaptchaAction and tool function, delegating to the
state-driven SolveCaptchaTask. Keeps the action name and public API stable.
"""
"""
Controller-facing shim for SolveCaptcha as a tool.

Keeps the action name and controller API stable while SolveCaptchaTask
contains the actual implementation. This separation allows the planner to
instantiate the task directly, and the Controller to continue exposing a
tool endpoint without coupling to task internals.
"""

from typing import Optional
import logging
from pydantic import BaseModel, Field

from browser_use.agent.tasks.solve_captcha_task import SolveCaptchaTask
from browser_use.llm.base import BaseChatModel

logger = logging.getLogger(__name__)

class SolveCaptchaAction(BaseModel):
    """No parameters; triggers the SolveCaptchaTask."""

    description: str = Field(
        default="Solve the currently visible CAPTCHA using LLM-guided clicks.",
        description="Initiates a specialized loop to analyze the CAPTCHA screenshot and click targets until solved.",
    )


async def tool_solve_captcha(
    controller,
    params: Optional[SolveCaptchaAction] = None,
    browser=None,
    page_extraction_llm: Optional[BaseChatModel] = None,
):
    """Delegate to SolveCaptchaTask; signature compatible with controller.registry expectations."""
    llm = page_extraction_llm or getattr(controller, 'page_extraction_llm', None)
    task = SolveCaptchaTask(controller=controller, browser=browser, page_extraction_llm=llm)
    try:
        logger.info("task_event task_invoked {'task_name': 'SolveCaptchaTask', 'via': 'tool_solve_captcha'}")
    except Exception:
        pass
    result = await task.run()
    try:
        status = 'success' if bool(getattr(task, 'succeeded') and task.succeeded()) else 'failed'
        logger.info("task_event task_completed {'task_name': 'SolveCaptchaTask', 'status': '%s'}", status)
    except Exception:
        pass
    return result
