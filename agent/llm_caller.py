from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from browser_use.agent.events import LLMRequest, LLMResponse, ErrorEvent, Heartbeat
from browser_use.agent.state_manager import agent_log, TERMINAL_STATES
from browser_use.agent.concurrency import with_io_semaphore
from browser_use.exceptions import LLMException

if TYPE_CHECKING:
    from browser_use.agent.state_manager import StateManager
    from browser_use.llm.base import BaseChatModel

logger = logging.getLogger(__name__)


class LLMCaller:
    """
    Isolated LLM calling component that handles all LLM invocations.
    Subscribes to LLMRequest events and publishes LLMResponse events.
    Contains all API, timeout, and retry logic.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        planner_llm: BaseChatModel,
        state_manager: StateManager,
        agent_bus: asyncio.Queue,
        heartbeat_bus: asyncio.Queue,
    ):
        self.llm = llm
        self.planner_llm = planner_llm
        self.state_manager = state_manager
        self.agent_bus = agent_bus
        self.heartbeat_bus = heartbeat_bus

    async def run(self):
        """Event-driven LLM calling loop that subscribes to LLMRequest events."""
        logger.debug("LLMCaller component started.")

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        async def main_loop():
            """Main event processing loop."""
            status_check_counter = 0  # Check status less frequently
            while True:
                try:
                    # Use shorter timeout to prevent blocking heartbeats
                    event = await asyncio.wait_for(self.agent_bus.get(), timeout=0.5)  # Very short timeout to avoid blocking heartbeat

                    # Only process LLMRequest events
                    if isinstance(event, LLMRequest):
                        await self._handle_llm_request(event)
                        self.agent_bus.task_done()
                    else:
                        # Not our event. Put it back on the bus and yield control
                        # to give the correct consumer a chance to pick it up.
                        try:
                            self.agent_bus.put_nowait(event)
                        except asyncio.QueueFull:
                            agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                                     "Agent bus full, cannot requeue event. Event may be lost.")
                        finally:
                            self.agent_bus.task_done()
                            # CRITICAL: Yield control to the event loop
                            await asyncio.sleep(0)
                        continue

                except asyncio.TimeoutError:
                    # Check status only every 10 iterations to reduce lock contention
                    status_check_counter += 1
                    if status_check_counter >= 10:  # Check status every 5 seconds (10 * 0.5s)
                        if await self.state_manager.get_status() not in TERMINAL_STATES:
                            status_check_counter = 0
                            continue
                        else:
                            break
                    else:
                        continue
                except Exception as e:
                    error_msg = f"LLMCaller event loop failed: {e}"
                    agent_log(logging.CRITICAL, self.state_manager.state.agent_id, self.state_manager.state.n_steps, error_msg, exc_info=True)
                    await self.state_manager.record_error(error_msg, is_critical=True)
                    await asyncio.sleep(1)

        try:
            # Run main loop and heartbeat concurrently
            await asyncio.gather(main_loop(), heartbeat_task, return_exceptions=True)
        finally:
            # Cancel heartbeat task when shutting down
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        logger.info("LLMCaller component stopped.")

    async def _handle_llm_request(self, event: LLMRequest):
        """Handle LLMRequest event by calling the appropriate LLM and publishing LLMResponse."""
        try:
            # Route planning requests to planner_llm when available
            # Use planner LLM for planning requests when available
            if event.request_type == "planning" and self.planner_llm is not None:
                llm = self.planner_llm
            else:
                llm = self.llm

            if not llm:
                error_msg = f"No LLM available for request type: {event.request_type}"
                await self._publish_error_response(event.request_id, error_msg, event.step_token)
                return

            # Invoke LLM with retry logic
            response, attempts = await self._invoke_llm_with_retry(
                llm, event.messages, event.output_schema, event.max_retries, event.step_token
            )

            # Publish successful response
            llm_response = LLMResponse(
                step_token=event.step_token,
                request_id=event.request_id,
                success=True,
                response=response,
                attempts=attempts
            )

            try:
                self.agent_bus.put_nowait(llm_response)
            except asyncio.QueueFull:
                agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                         "Agent bus full, dropping LLM response")

        except Exception as e:
            error_msg = f"LLM call failed: {e}"
            agent_log(logging.ERROR, self.state_manager.state.agent_id, self.state_manager.state.n_steps, error_msg, exc_info=True)
            await self._publish_error_response(event.request_id, error_msg, event.step_token)

    async def _invoke_llm_with_retry(self, llm: BaseChatModel, messages: list, output_schema: Any, max_retries: int, step_token: int) -> tuple[Any, int]:
        """
        Invokes the LLM with retry logic, handling timeouts and API errors.
        Returns (response, attempts_made).
        """
        for attempt in range(max_retries + 1):
            agent_log(
                logging.INFO,
                self.state_manager.state.agent_id,
                self.state_manager.state.n_steps,
                f"LLM call attempt {attempt + 1}/{max_retries + 1}"
            )
            try:
                # Use I/O semaphore to limit concurrent LLM calls
                async with await with_io_semaphore():
                    response = await llm.ainvoke(messages, output_format=output_schema)
                    parsed = response.completion

                agent_log(
                    logging.INFO,
                    self.state_manager.state.agent_id,
                    self.state_manager.state.n_steps,
                    f"Raw LLM response type: {type(response)}"
                )
                agent_log(
                    logging.INFO,
                    self.state_manager.state.agent_id,
                    self.state_manager.state.n_steps,
                    f"Parsed completion type: {type(parsed)}"
                )

                # Basic validation - check if response has expected structure
                if self._validate_response(parsed):
                    # Successful call; emit io_ok to decay any previous IO timeouts
                    try:
                        await self.state_manager.ingest_signal('io_ok')
                    except Exception:
                        logger.debug("Failed to emit io_ok signal from LLMCaller", exc_info=True)
                    return parsed, attempt + 1
                else:
                    # Invalid response, retry if attempts remain
                    if attempt < max_retries:
                        agent_log(
                            logging.WARNING,
                            self.state_manager.state.agent_id,
                            self.state_manager.state.n_steps,
                            "LLM returned invalid response. Retrying..."
                        )
                        await asyncio.sleep(1.0 * (2 ** attempt))
                        continue
                    else:
                        # Return the response anyway after max retries
                        return parsed, attempt + 1

            except (asyncio.TimeoutError, LLMException) as e:
                agent_log(
                    logging.WARNING,
                    self.state_manager.state.agent_id,
                    self.state_manager.state.n_steps,
                    f"LLM call attempt {attempt+1} failed: {type(e).__name__}: {e}",
                    exc_info=True
                )
                # Emit io_timeout to signal engine about I/O instability
                try:
                    await self.state_manager.ingest_signal('io_timeout')
                except Exception:
                    logger.debug("Failed to emit io_timeout signal from LLMCaller", exc_info=True)
                if attempt >= max_retries:
                    raise LLMException("LLM call failed after all retries.") from e
                await asyncio.sleep(1.0 * (2 ** attempt))

        raise LLMException("LLM call failed.")

    def _validate_response(self, response: Any) -> bool:
        """Basic validation of LLM response structure."""
        # This is a simplified validation - can be extended based on needs
        return response is not None

    async def _publish_error_response(self, request_id: str, error_msg: str, step_token: int):
        """Publish an error response for a failed LLM request."""
        llm_response = LLMResponse(
            step_token=step_token,
            request_id=request_id,
            success=False,
            error=error_msg,
            attempts=0
        )

        try:
            self.agent_bus.put_nowait(llm_response)
        except asyncio.QueueFull:
            agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                     "Agent bus full, dropping LLM error response")

        # Also publish an ErrorEvent for critical failures
        error_event = ErrorEvent(
            step_token=step_token,
            error_message=error_msg,
            error_type="llm_failure",
            is_critical=True
        )

        try:
            self.agent_bus.put_nowait(error_event)
        except asyncio.QueueFull:
            agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                     "Agent bus full, dropping ErrorEvent for LLM failure")

    async def _heartbeat_loop(self):
        """Send periodic heartbeat events every 2 seconds."""
        while True:
            try:
                await asyncio.sleep(2.0)  # Send heartbeat every 2 seconds

                # Create and emit heartbeat event
                heartbeat = Heartbeat(
                    step_token=self.state_manager.state.n_steps,
                    component_name="llm_caller"
                )

                try:
                    self.heartbeat_bus.put_nowait(heartbeat)
                    logger.debug("LLMCaller heartbeat sent")
                except asyncio.QueueFull:
                    logger.warning("Agent bus full, dropped LLMCaller heartbeat")

            except asyncio.CancelledError:
                logger.debug("LLMCaller heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in LLMCaller heartbeat loop: {e}")
                await asyncio.sleep(2.0)  # Continue after error
