from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from browser_use.agent.events import AssessmentUpdate, ErrorEvent


@dataclass
class RouterConfig:
    # Thresholds (enter); exits use hysteresis_delta
    risk_reactive: float = 0.80
    risk_hysteresis_delta: float = 0.10
    conf_reactive: float = 0.30
    conf_clear: float = 0.40
    opp_proactive: float = 0.70
    conf_proactive_min: float = 0.55
    stagnation_proactive: float = 0.60
    looping_confirm: float = 0.60
    # Cooldowns (seconds)
    proactive_cooldown_s: float = 3.0
    reactive_cooldown_s: float = 2.0


class ModeRouter:
    """Maps AssessmentUpdate (+ optional ErrorEvent) to planner mode 'reactive' | 'proactive' or None.

    Applies simple hysteresis and cooldown in seconds.
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        self.cfg = config or RouterConfig()
        self._last_mode: Optional[str] = None
        self._last_decision_ts: Optional[float] = None

    def decide(self, assess: AssessmentUpdate, error: Optional[ErrorEvent] = None) -> Optional[str]:
        now = time.monotonic()

        # Error short-circuit
        if error is not None:
            if self._in_cooldown('reactive', now):
                return None
            self._record('reactive', now)
            return 'reactive'

        # Hysteresis thresholds
        risk_enter = self.cfg.risk_reactive
        risk_exit = max(0.0, self.cfg.risk_reactive - self.cfg.risk_hysteresis_delta)

        # Reactive candidates
        reactive = (
            assess.risk >= risk_enter
            or assess.confidence <= self.cfg.conf_reactive
            or assess.looping >= self.cfg.looping_confirm
        )

        if reactive:
            if self._in_cooldown('reactive', now):
                return None
            self._record('reactive', now)
            return 'reactive'

        # Proactive candidates
        proactive = (
            (assess.opportunity >= self.cfg.opp_proactive and assess.confidence >= self.cfg.conf_proactive_min)
            or (assess.stagnation >= self.cfg.stagnation_proactive)
            or (assess.risk > risk_exit and self._last_mode == 'reactive')  # lingering risk allows a measured proactive follow-up
        )
        if proactive:
            if self._in_cooldown('proactive', now):
                return None
            self._record('proactive', now)
            return 'proactive'

        return None

    def _in_cooldown(self, mode: str, now: float) -> bool:
        if self._last_mode != mode or self._last_decision_ts is None:
            return False
        cd = self.cfg.reactive_cooldown_s if mode == 'reactive' else self.cfg.proactive_cooldown_s
        return (now - self._last_decision_ts) < cd

    def _record(self, mode: str, now: float) -> None:
        self._last_mode = mode
        self._last_decision_ts = now
