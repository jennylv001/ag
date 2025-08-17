from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from browser_use.agent.events import Heartbeat

if TYPE_CHECKING:
    from browser_use.agent.state_manager import StateManager

logger = logging.getLogger(__name__)


class Scheduler:
    """
    Lightweight, non-blocking scheduler scaffold.
    Emits heartbeats and can later publish planning/maintenance events.
    """

    def __init__(self, state_manager: StateManager, agent_bus: asyncio.Queue, heartbeat_bus: asyncio.Queue, interval_seconds: float = 15.0):
        self.state_manager = state_manager
        self.agent_bus = agent_bus
        self.heartbeat_bus = heartbeat_bus
        self.interval_seconds = interval_seconds
        self._running = False

    async def run(self):
        logger.debug("Scheduler component started.")
        self._running = True
        try:
            # Emit an immediate heartbeat on startup to avoid early timeouts
            await self._emit_heartbeat()
            while self._running:
                await asyncio.sleep(self.interval_seconds)

                # Only run while agent is active
                from browser_use.agent.state_manager import TERMINAL_STATES
                status = await self.state_manager.get_status()
                if status in TERMINAL_STATES:
                    break

                # For now, just send a heartbeat so ReactorVitals can monitor us
                await self._emit_heartbeat()
        except asyncio.CancelledError:
            logger.debug("Scheduler run() cancelled")
        finally:
            logger.info("Scheduler component stopped.")

    def stop(self):
        self._running = False

    async def _emit_heartbeat(self):
        heartbeat = Heartbeat(
            step_token=self.state_manager.state.n_steps,
            component_name="scheduler"
        )
        try:
            self.heartbeat_bus.put_nowait(heartbeat)
        except asyncio.QueueFull:
            logger.warning("Scheduler: heartbeat queue full; dropped heartbeat")
