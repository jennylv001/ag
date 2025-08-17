from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional
import time

from browser_use.agent.events import AssessmentUpdate
from browser_use.agent.state_manager import StateManager

logger = logging.getLogger(__name__)


@dataclass
class EWMA:
    value: float = 0.0
    initialized: bool = False

    def update(self, x: float, alpha: float = 0.3) -> float:
        x = max(0.0, min(1.0, float(x)))
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = alpha * x + (1 - alpha) * self.value
        return self.value


class Assessor:
    """Computes fused system signals (risk, opportunity, confidence) and publishes AssessmentUpdate."""

    def __init__(self, state_manager: StateManager, agent_bus: asyncio.Queue, interval_seconds: float = 1.0):
        self.state_manager = state_manager
        self.agent_bus = agent_bus
        self.interval = max(0.2, float(interval_seconds))
        self._running = False
        # Smoothed metrics
        self._risk = EWMA()
        self._opp = EWMA()
        self._conf = EWMA()
        # Progress tracking
        self._last_hist_len: int | None = None
        self._last_ts: float | None = None
        # Tiny cache: map screenshot_path->base64 to avoid disk re-reads per tick
        self._shot_cache: dict[str, str] = {}
        self._shot_cache_cap: int = 8

    async def run(self, shutdown_event: Optional[asyncio.Event] = None):
        self._running = True
        logger.info("Assessor started")
        while self._running and (shutdown_event is None or not shutdown_event.is_set()):
            try:
                sav = await self._compute_signals()
                await self.agent_bus.put(sav)
            except Exception as e:
                logger.warning(f"Assessor iteration failed: {e}")
            await asyncio.sleep(self.interval)
        logger.info("Assessor stopped")

    async def _compute_signals(self) -> AssessmentUpdate:
        # Pull minimal counters from StateManager (no heavy locks needed for read-only snapshot fields)
        st = self.state_manager.state

        # Basic proxies; these can be extended with richer telemetry without changing the event contract
        actions = max(1, len(st.history.history))
        failures = int(st.consecutive_failures)
        io_timeouts = sum(st.io_timeouts_recent) if hasattr(st, 'io_timeouts_recent') else 0
        load = 1.0 if st.load_status.name == 'SHEDDING' else 0.0

        # Normalize
        fail_rate = min(1.0, failures / max(1, actions))
        timeout_rate = min(1.0, io_timeouts / 10.0)  # assume window ~10
        # Progress rate: history items per minute (capped)
        now = time.monotonic()
        hist_len = len(st.history.history)
        prog_rate = 0.0
        if self._last_ts is not None and self._last_hist_len is not None:
            dt = max(1e-3, now - self._last_ts)
            dsteps = max(0, hist_len - self._last_hist_len)
            per_min = (dsteps / dt) * 60.0
            prog_rate = min(1.0, per_min / 6.0)  # 6 steps/min saturates
        self._last_ts = now
        self._last_hist_len = hist_len

        # Fuse risk (weights can be made configurable later)
        risk_raw = 0.4 * timeout_rate + 0.4 * fail_rate + 0.2 * load
        risk = self._risk.update(risk_raw)

        # Opportunity proxy: combine low failure/load with progress rate
        opportunity_raw = max(0.0, min(1.0, 0.2 + 0.5 * prog_rate + 0.3 * (1.0 - fail_rate) - 0.2 * load))
        opportunity = self._opp.update(opportunity_raw)

        # Confidence proxy: inverse of failure + timeout, boosted by recent progress
        conf_raw = max(0.0, min(1.0, 0.1 + 0.5 * (1.0 - fail_rate) + 0.2 * (1.0 - timeout_rate) + 0.2 * prog_rate))
        confidence = self._conf.update(conf_raw)

        # Stagnation & looping proxies (lightweight placeholders; can be replaced by real metrics)
        # Stagnation increases when progress rate is low and failures/timeouts accumulate
        stagnation = max(0.0, min(1.0, 0.6 * (1.0 - prog_rate) + 0.2 * fail_rate + 0.2 * timeout_rate))
        # Looping proxy: elevate when failures repeat without progress
        looping = max(0.0, min(1.0, 0.5 * (1.0 - prog_rate) + 0.5 * (1.0 if failures >= 2 else 0.0)))

        contributors = []
        if timeout_rate > 0.2:
            contributors.append("timeouts")
        if fail_rate > 0.1:
            contributors.append("failures")
        if load > 0.0:
            contributors.append("load")
        if stagnation > 0.6:
            contributors.append("stagnation↑")
        if looping > 0.6:
            contributors.append("looping↑")

        # Visual provenance: pull the most recent 1-2 screenshots from history with robust fallback
        screenshot_refs: list[str] = []
        try:
            # Iterate from latest to oldest; collect up to 2 screenshots
            for item in reversed(list(st.history.history)):
                if len(screenshot_refs) >= 1:  # latest-only for latency
                    break
                try:
                    state_hist = getattr(item, 'state', None)
                    if not state_hist:
                        continue
                    # Preferred: if history stores file path, use loader to get base64
                    shot_b64 = None
                    try:
                        # Use cache if screenshot_path available
                        sref = getattr(state_hist, 'screenshot_path', None)
                        if isinstance(sref, str) and sref:
                            if sref in self._shot_cache:
                                shot_b64 = self._shot_cache[sref]
                            elif hasattr(state_hist, 'get_screenshot') and callable(state_hist.get_screenshot):
                                shot_b64 = state_hist.get_screenshot()
                                if shot_b64:
                                    # Maintain small cache with simple eviction
                                    if len(self._shot_cache) >= self._shot_cache_cap:
                                        try:
                                            self._shot_cache.pop(next(iter(self._shot_cache)))
                                        except Exception:
                                            self._shot_cache.clear()
                                    self._shot_cache[sref] = shot_b64
                        else:
                            if hasattr(state_hist, 'get_screenshot') and callable(state_hist.get_screenshot):
                                shot_b64 = state_hist.get_screenshot()
                    except Exception:
                        shot_b64 = None
                    # Fallback: some code paths store raw base64 in screenshot_path
                    if not shot_b64:
                        sref = getattr(state_hist, 'screenshot_path', None)
                        # Heuristic: treat as base64 if it looks long and not like a filesystem path
                        if isinstance(sref, str) and len(sref) > 100 and (':\\' not in sref[:5]) and ('\\' not in sref[:10]) and ('/' not in sref[:10]):
                            shot_b64 = sref
                    if shot_b64:
                        screenshot_refs.append(shot_b64)
                except Exception:
                    # Never fail signal computation due to screenshot issues
                    continue
        except Exception:
            screenshot_refs = []

        # Lightweight visual summary using last known state
        try:
            visual_summary = ""
            if st.history.history:
                last_state = getattr(st.history.history[-1], 'state', None)
                if last_state:
                    url = getattr(last_state, 'url', None) or 'about:blank'
                    title = getattr(last_state, 'title', None) or 'Untitled'
                    tabs = getattr(last_state, 'tabs', None)
                    n_tabs = len(tabs) if isinstance(tabs, list) else 0
                    visual_summary = f"{title} @ {url} | tabs={n_tabs} | screenshots_in_window={len(screenshot_refs)}"
        except Exception:
            visual_summary = ""

        change_map_ref = None

        return AssessmentUpdate(
            step_token=st.n_steps,
            risk=risk,
            opportunity=opportunity,
            confidence=confidence,
            stagnation=stagnation,
            looping=looping,
            contributors=contributors,
            trend_window=5,
            screenshot_refs=screenshot_refs,
            visual_summary=visual_summary,
            change_map_ref=change_map_ref,
        )
