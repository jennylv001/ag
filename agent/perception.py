from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from browser_use.agent.events import PerceptionOutput, PerceptionSnapshot, StepFinalized, Heartbeat
from browser_use.agent.state_manager import AgentStatus, LoadStatus, agent_log, TERMINAL_STATES
from browser_use.browser.views import BrowserStateSummary

if TYPE_CHECKING:
    from browser_use.agent.settings import AgentSettings
    from browser_use.agent.state_manager import StateManager
    from browser_use.browser import BrowserSession

logger = logging.getLogger(__name__)

class Perception:
    """ "Perception ≠ Paralysis" & "The Canary Protocol" """

    def __init__(
        self,
        browser_session: BrowserSession,
        state_manager: StateManager,
        settings: AgentSettings,
        agent_bus: asyncio.Queue,
        step_barrier: asyncio.Event,
        heartbeat_bus: asyncio.Queue,
    ):
        self.browser_session = browser_session
        self.state_manager = state_manager
        self.settings = settings
        self.agent_bus = agent_bus
        self.step_barrier = step_barrier
        self.heartbeat_bus = heartbeat_bus
        self.has_downloads_path = self.browser_session.browser_profile.downloads_path is not None
        if self.has_downloads_path:
            self._last_known_downloads = []

    async def run(self):
        """Main perception loop that processes StepFinalized events and emits PerceptionSnapshot events."""
        logger.debug("Perception component started.")

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                    "Triggering initial perception snapshot to kickstart the system")

            # Create an initial snapshot regardless of step_barrier state
            new_files = await self._check_and_update_downloads()
            browser_state = await self._get_browser_state_with_recovery()

            # Publish initial PerceptionSnapshot event directly
            perception_snapshot = PerceptionSnapshot(
                step_token=0,
                browser_state=browser_state,
                new_downloaded_files=new_files
            )

            try:
                # Use put_nowait to avoid any chance of deadlock
                self.agent_bus.put_nowait(perception_snapshot)
                agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                         "Initial PerceptionSnapshot published successfully")
            except asyncio.QueueFull:
                agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                         "Agent bus full, could not publish initial PerceptionSnapshot")

            # Now start the normal event processing loop
            status_check_counter = 0
            while True:
                try:
                    # Check if agent is in terminal state every 10 iterations to reduce lock contention
                    if status_check_counter % 10 == 0:
                        if await self.state_manager.get_status() in TERMINAL_STATES:
                            agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                                    "Agent in terminal state, stopping perception component")
                            break
                    status_check_counter += 1

                    # Process events with a short timeout to avoid blocking the event loop
                    try:
                        event = await asyncio.wait_for(self.agent_bus.get(), timeout=0.5)

                        # Process StepFinalized events
                        if isinstance(event, StepFinalized):
                            agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                                    f"Received StepFinalized event (step_token={event.step_token})")
                            await self._handle_step_finalized(event)
                            self.agent_bus.task_done()
                        else:
                            # Not our event. Put it back on the bus and yield control
                            try:
                                self.agent_bus.put_nowait(event)
                            except asyncio.QueueFull:
                                agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                                        "Agent bus full, cannot requeue event. Event may be lost.")
                            finally:
                                self.agent_bus.task_done()
                                # CRITICAL: Yield control to the event loop
                                await asyncio.sleep(0)

                    except asyncio.TimeoutError:
                        # No events to process, continue the loop
                        continue

                except Exception as e:
                    error_msg = f"Error in perception event loop: {e}"
                    agent_log(logging.ERROR, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                             error_msg, exc_info=True)
                    await asyncio.sleep(1)  # Avoid tight error loop

        finally:
            # Cancel heartbeat task when shutting down
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        logger.info("Perception component stopped.")

    async def _handle_step_finalized(self, event: StepFinalized):
        """Handle StepFinalized event by capturing perception snapshot."""
        # Filter out stale events
        current_step = self.state_manager.state.n_steps
        if event.step_token < current_step:
            agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                     f"Perception: Ignoring stale StepFinalized event: step_token={event.step_token} < current_step={current_step}")
            return

        status = await self.state_manager.get_status()
        if status != AgentStatus.RUNNING:
            return

        # Dynamic Load Shedding Check
        load_status = await self.state_manager.get_load_status()
        if load_status == LoadStatus.SHEDDING:
            agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                      "System under high load, throttling perception.")
            await asyncio.sleep(2.0) # Throttle by waiting
            return

        try:
            # FIXED: Wait for step_barrier without holding bus task
            # This prevents deadlock by ensuring we release the task before waiting
            agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                     f"Waiting for step_barrier (step_token={event.step_token})")

            # Wait for step barrier to be set
            await self.step_barrier.wait()

            # Clear for next cycle immediately
            self.step_barrier.clear()

            agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                     f"Starting perception capture (step_token={event.step_token})")

            # Capture browser state and check downloads
            new_files = await self._check_and_update_downloads()
            browser_state = await self._get_browser_state_with_recovery()

            agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                     f"Publishing PerceptionSnapshot to agent_bus (step_token={event.step_token})")

            # Publish PerceptionSnapshot event with step_token from StepFinalized
            perception_snapshot = PerceptionSnapshot(
                step_token=event.step_token,
                browser_state=browser_state,
                new_downloaded_files=new_files
            )

            try:
                # Use put_nowait to avoid any chance of deadlock
                self.agent_bus.put_nowait(perception_snapshot)
                agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                         f"PerceptionSnapshot published successfully (step_token={event.step_token})")
            except asyncio.QueueFull:
                agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                         "Agent bus full, dropping perception snapshot to prevent blocking")

        except Exception as e:
            error_msg = f"Perception snapshot capture failed: {e}"
            agent_log(logging.CRITICAL, self.state_manager.state.agent_id, self.state_manager.state.n_steps, error_msg, exc_info=True)
            await self.state_manager.record_error(error_msg, is_critical=True)

    async def _heartbeat_loop(self):
        """Send periodic heartbeat events every 2 seconds."""
        while True:
            try:
                await asyncio.sleep(2.0)  # Send heartbeat every 2 seconds

                # Create and emit heartbeat event
                heartbeat = Heartbeat(
                    step_token=self.state_manager.state.n_steps,
                    component_name="perception"
                )

                try:
                    self.heartbeat_bus.put_nowait(heartbeat)
                    logger.debug("Perception heartbeat sent")
                except asyncio.QueueFull:
                    logger.warning("Agent bus full, dropped perception heartbeat")

            except asyncio.CancelledError:
                logger.debug("Perception heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in perception heartbeat loop: {e}")
                await asyncio.sleep(2.0)  # Continue after error

    async def _get_browser_state_with_recovery(self) -> BrowserStateSummary:
        """Get browser state with fallback for recovery."""
        try:
            return await self.browser_session.get_state_summary(cache_clickable_elements_hashes=True)
        except Exception as e:
            agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps, f"Full state retrieval failed: {e}")
            return await self.browser_session.get_minimal_state_summary()

    async def _check_and_update_downloads(self):
        """Check for new downloads and update the last known list."""
        if not self.has_downloads_path: return None
        try:
            current_downloads = self.browser_session.downloaded_files
            if current_downloads != self._last_known_downloads:
                new_files = list(set(current_downloads) - set(self._last_known_downloads))
                if new_files:
                    self._last_known_downloads = current_downloads
                    return new_files
        except Exception as e:
            agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps, f"Failed to check downloads: {e}")
        return None
