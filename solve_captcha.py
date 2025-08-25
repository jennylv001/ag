"""
DEPRECATED MODULE: browser_use.solve_captcha

This legacy module has been replaced by the new task-based implementation.
Please import from:
  - browser_use.agent.tasks.solve_captcha_tool (tool and action)
    - browser_use.agent.tasks.views (schemas)

This shim re-exports the public symbols to maintain compatibility for any
straggling imports while guiding callers to the new locations.
"""

from __future__ import annotations

import warnings as _warnings

from browser_use.agent.tasks.solve_captcha_tool import (  # noqa: F401
    SolveCaptchaAction,
    tool_solve_captcha,
)
from browser_use.agent.tasks.views import (  # noqa: F401
    CaptchaInitialPlan,
    CaptchaFollowUpPlan,
)

__all__ = [
    "SolveCaptchaAction",
    "tool_solve_captcha",
    "CaptchaInitialPlan",
    "CaptchaFollowUpPlan",
]

_warnings.warn(
    "browser_use.solve_captcha is deprecated; use browser_use.agent.tasks.solve_captcha_tool "
    "and browser_use.agent.tasks.views instead.",
    DeprecationWarning,
    stacklevel=2,
)
