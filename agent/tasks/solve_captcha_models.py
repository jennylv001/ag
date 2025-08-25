from __future__ import annotations

"""
Deprecated shim: use browser_use.agent.tasks.views instead.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from browser_use.agent.tasks.views import CaptchaInitialPlan, CaptchaFollowUpPlan  # noqa: F401

# At runtime, import and re-export to maintain backward compatibility
try:
    from browser_use.agent.tasks.views import CaptchaInitialPlan, CaptchaFollowUpPlan  # type: ignore
except Exception:  # pragma: no cover - if import fails, leave module with no symbols
    pass
