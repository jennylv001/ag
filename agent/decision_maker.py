from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from typing import TYPE_CHECKING, List, Optional

from browser_use.agent.events import PerceptionSnapshot, DecisionPlan, LLMRequest, LLMResponse, Heartbeat, ActionExecuted, ErrorEvent
from browser_use.agent.prompts import SystemPrompt
from browser_use.agent.state_manager import AgentStatus, agent_log
from browser_use.agent.views import AgentOutput, PerceptionOutput
from browser_use.exceptions import LLMException

if TYPE_CHECKING:
    from browser_use.agent.message_manager.service import MessageManager
    from browser_use.agent.settings import AgentSettings
    from browser_use.agent.state_manager import StateManager
    from browser_use.llm.base import BaseChatModel
    from browser_use.llm.messages import BaseMessage

logger = logging.getLogger(__name__)


class DecisionMaker:
    """
    "The Sacred/Profane Split"
    This component is the "sacred" synchronous core. It receives perception data,
    constructs prompts, invokes the LLM, and makes decisions. It does NOT perform
    any direct I/O (like browser actions).
    """

    def __init__(
        self,
        settings: AgentSettings,
        state_manager: StateManager,
        message_manager: MessageManager,
        agent_bus: asyncio.Queue,
        heartbeat_bus: asyncio.Queue,
    ):
        self.settings = settings
        self.state_manager = state_manager
        self.message_manager = message_manager
        self.agent_bus = agent_bus
        self.heartbeat_bus = heartbeat_bus
        self.pending_requests = {}  # Track pending LLM requests
        self._handled_snapshot_tokens: set[int] = set()
        self._pending_step_requests: set[int] = set()
        # Semaphore to prevent "tab-sprawl" by limiting concurrent tasks
        self.task_semaphore = asyncio.Semaphore(settings.max_concurrent_tasks)
        # Simple backoff flag when recent I/O timeouts occurred
        self._io_backoff_until_step: int | None = None

        # Ensure a reference to the LLM exists for internal retry helper (if used)
        # Note: normal flow uses LLMCaller component; this is a safe fallback.
        try:
            self.llm = getattr(self.settings, 'llm', None)
        except Exception:
            self.llm = None

        self._setup_action_models()

    def _setup_action_models(self):
        # This logic is from the original Agent, needed for creating the correct output schema.
        # It's assumed Controller is already configured.
        ActionModel = self.settings.controller.registry.create_action_model()
        done_action_model = self.settings.controller.registry.create_action_model(include_actions=['done'])
        if self.settings.flash_mode:
            self.AgentOutput = AgentOutput.type_with_custom_actions_flash_mode(ActionModel)
            self.DoneAgentOutput = AgentOutput.type_with_custom_actions_flash_mode(done_action_model)
        elif self.settings.use_thinking:
            self.AgentOutput = AgentOutput.type_with_custom_actions(ActionModel)
            self.DoneAgentOutput = AgentOutput.type_with_custom_actions(done_action_model)
        else:
            self.AgentOutput = AgentOutput.type_with_custom_actions_no_thinking(ActionModel)
            self.DoneAgentOutput = AgentOutput.type_with_custom_actions_no_thinking(done_action_model)
        # Export ActionModel to instance for later (noop creation etc)
        self.ActionModel = ActionModel

    async def run(self):
        """Event-driven decision loop that subscribes to PerceptionSnapshot and LLMResponse events."""
        from browser_use.agent.state_manager import TERMINAL_STATES
        logger.debug("DecisionMaker component started.")

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        async def main_loop():
            """Main event processing loop."""
            status_check_counter = 0  # Check status less frequently
            while True:
                try:
                    # Use shorter timeout to prevent blocking heartbeats
                    event = await asyncio.wait_for(self.agent_bus.get(), timeout=0.5)  # Very short timeout to avoid blocking heartbeat

                    # Only process PerceptionSnapshot and LLMResponse events
                    if isinstance(event, PerceptionSnapshot):
                        await self._handle_perception_snapshot(event)
                        self.agent_bus.task_done()
                    elif isinstance(event, LLMResponse):
                        await self._handle_llm_response(event)
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
                            # Give supervisor priority window for ActionExecuted/ErrorEvent
                            if isinstance(event, (ActionExecuted, ErrorEvent)):
                                await asyncio.sleep(0.01)  # 10ms priority window for supervisor
                            else:
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
                    error_msg = f"DecisionMaker event loop failed: {e}"
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
        logger.info("DecisionMaker component stopped.")

    async def _handle_perception_snapshot(self, event: PerceptionSnapshot):
        """Handle PerceptionSnapshot by preparing messages and emitting a single LLMRequest."""
        from browser_use.agent.state_manager import AgentStatus

        # Filter out stale events
        current_step = self.state_manager.state.n_steps
        if event.step_token < current_step:
            logger.warning(f"Ignoring stale PerceptionSnapshot event: step_token={event.step_token} < current_step={current_step}")
            return

        status = await self.state_manager.get_status()
        # Only RUNNING is valid now; ignore others
        if status not in (AgentStatus.RUNNING,):
            return

        # Prevent duplicate processing for the same step
        if event.step_token in self._handled_snapshot_tokens:
            return
        self._handled_snapshot_tokens.add(event.step_token)

        # Build perception output
        perception_output = PerceptionOutput(
            browser_state=event.browser_state,
            new_downloaded_files=event.new_downloaded_files,
        )

        # Ensure only one LLM request is in-flight per step
        if event.step_token in self._pending_step_requests:
            return
        self._pending_step_requests.add(event.step_token)

        # Always prepare action request
        await self._prepare_action_request(perception_output, event.step_token)

    async def _prepare_action_request(self, perception: PerceptionOutput, step_token: int):
        """Prepare and emit LLMRequest for action decision."""
        # Use semaphore to limit concurrent tasks and prevent tab-sprawl
        async with self.task_semaphore:
            try:
                state = self.state_manager.state
                self.message_manager.update_history_representation(state.history)

                # Get current task context for enhanced prompting
                current_task_id = await self.state_manager.get_current_task_id()
                task_stack_summary = await self.state_manager.get_task_stack_summary()

                # Prepare messages using existing logic with task context
                messages = self.message_manager.prepare_messages_for_llm(
                    browser_state=perception.browser_state,
                    current_goal=state.current_goal,
                    last_error=state.last_error,
                    agent_history_list=state.history,
                    current_task_id=current_task_id,
                    task_context=task_stack_summary
                )

                # Generate unique request ID and store context
                request_id = str(uuid.uuid4())
                self.pending_requests[request_id] = {
                    'step_token': step_token,
                    'request_type': 'action',
                    'messages': messages,
                }

                # Determine output schema
                output_schema = self.DoneAgentOutput if (state.n_steps + 1) >= self.settings.max_steps else self.AgentOutput

                # Heuristic: if we observed recent I/O timeouts, nudge the prompt to request fewer actions once
                try:
                    recent_timeouts = int(len(getattr(self.state_manager.state, 'io_timeouts_recent', [])))
                    if recent_timeouts > 0:
                        # Set a one-step backoff window
                        self._io_backoff_until_step = step_token
                        # Ask the model to propose a single, high-confidence action
                        self.message_manager.add_local_note(
                            "I/O timeouts detected recently. Propose at most one, highest-confidence action to progress safely."
                        )
                except Exception:
                    pass

                # Create and emit LLMRequest with task context
                llm_request = LLMRequest(
                    step_token=step_token,
                    task_id=current_task_id,
                    messages=messages,
                    output_schema=output_schema,
                    request_id=request_id,
                    max_retries=2,
                    request_type="action"
                )

                try:
                    self.agent_bus.put_nowait(llm_request)
                except asyncio.QueueFull:
                    agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                             "Agent bus full, dropping LLM request")

            except Exception as e:
                error_msg = f"Action request preparation failed: {e}"
                agent_log(logging.CRITICAL, self.state_manager.state.agent_id, self.state_manager.state.n_steps, error_msg, exc_info=True)
                await self.state_manager.record_error(error_msg, is_critical=True)

    async def _handle_llm_response(self, event: LLMResponse):
        """Handle LLMResponse event by processing the response and emitting DecisionPlan."""
        try:
            self._pending_step_requests.discard(event.step_token)
        except Exception:
            pass
        try:
            # CRITICAL FIX: Filter out stale events
            current_step = self.state_manager.state.n_steps
            if event.step_token < current_step:
                logger.warning(f"🧠 DecisionMaker: Ignoring stale LLMResponse event: step_token={event.step_token} < current_step={current_step}")
                return

            # Look up the request context (and clean up)
            request_context = self.pending_requests.pop(event.request_id, None)
            if not request_context:
                # Unknown response: likely for planner or other components; requeue and yield
                try:
                    self.agent_bus.put_nowait(event)
                except asyncio.QueueFull:
                    agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                              "Agent bus full, cannot requeue unmatched LLMResponse")
                await asyncio.sleep(0)
                return

            # Verify step token matches
            if request_context['step_token'] != event.step_token:
                agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                         f"Step token mismatch in LLM response: expected {request_context['step_token']}, got {event.step_token}")
                return

            # Handle successful response
            if event.success and event.response:
                request_type = request_context['request_type']

                if request_type == 'action':
                    await self._handle_action_response(event, request_context)
                else:
                    agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                             f"Unknown request type: {request_type}")
            else:
                # Handle failed LLM response
                error_msg = f"LLM request failed: {event.error}"
                agent_log(logging.ERROR, self.state_manager.state.agent_id, self.state_manager.state.n_steps, error_msg)
                await self.state_manager.record_error(error_msg)

        except Exception as e:
            error_msg = f"LLM response handling failed: {e}"
            agent_log(logging.CRITICAL, self.state_manager.state.agent_id, self.state_manager.state.n_steps, error_msg, exc_info=True)
            await self.state_manager.record_error(error_msg, is_critical=True)

    async def _handle_action_response(self, event: LLMResponse, request_context: dict):
        """Handle successful action LLM response."""
        try:
            # Log the agent's thinking process using standardized method
            agent_output: AgentOutput = event.response
            self._log_llm_output(agent_output)

            # Create DecisionPlan for action
            decision_plan = DecisionPlan(
                step_token=event.step_token,
                messages_to_llm=request_context['messages'],
                llm_output=event.response,
                decision_type="action"
            )

            # Emit DecisionPlan
            try:
                self.agent_bus.put_nowait(decision_plan)
                agent_log(logging.INFO, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                         f"Action decision published: {event.response.action}")
            except asyncio.QueueFull:
                agent_log(logging.WARNING, self.state_manager.state.agent_id, self.state_manager.state.n_steps,
                         "Agent bus full, dropping action decision plan")

        except Exception as e:
            error_msg = f"Action response handling failed: {e}"
            agent_log(logging.ERROR, self.state_manager.state.agent_id, self.state_manager.state.n_steps, error_msg, exc_info=True)
            await self.state_manager.record_error(error_msg)



    async def _invoke_llm_with_retry(self, messages: list["BaseMessage"], max_retries: int = 2) -> AgentOutput:
        """
        Invokes the LLM and post-processes its output to match Agent.get_model_output exactly.
        Handles retries, action list trimming, logging, and fallback insertion for empty output.
        """
        # Always reset action models before every LLM call to ensure up-to-date schema
        self._setup_action_models()
        output_schema = self.DoneAgentOutput if (self.state_manager.state.n_steps + 1) >= self.settings.max_steps else self.AgentOutput

        for attempt in range(max_retries + 1):
            agent_log(
                logging.INFO,
                self.state_manager.state.agent_id,
                self.state_manager.state.n_steps,
                f"LLM call attempt {attempt + 1}/{max_retries + 1}"
            )
            try:
                response = await self.llm.ainvoke(messages, output_format=output_schema)
                parsed = response.completion

                # Trim action list to max_actions_per_step if necessary
                if hasattr(parsed, "action") and isinstance(parsed.action, list):
                    if len(parsed.action) > self.settings.max_actions_per_step:
                        agent_log(
                            logging.WARNING,
                            self.state_manager.state.agent_id,
                            self.state_manager.state.n_steps,
                            f"Trimming actions from {len(parsed.action)} to {self.settings.max_actions_per_step}"
                        )
                        parsed.action = parsed.action[:self.settings.max_actions_per_step]

                # Log LLM output using standardized method
                self._log_llm_output(parsed)

                # Validate action output for empty/malformed responses
                has_action_attr = hasattr(parsed, "action")
                is_action_list = isinstance(getattr(parsed, "action", None), list) if has_action_attr else False
                actions = getattr(parsed, "action", []) if has_action_attr else []
                empty_actions = all(getattr(a, "model_dump", lambda: {})() == {} for a in actions) if is_action_list else True

                if (
                    not has_action_attr
                    or not is_action_list
                    or empty_actions
                ):
                    # Retry logic for empty/malformed output
                    if attempt < max_retries:
                        agent_log(
                            logging.WARNING,
                            self.state_manager.state.agent_id,
                            self.state_manager.state.n_steps,
                            "Model returned empty action. Retrying..."
                        )
                        await asyncio.sleep(1.0 * (2 ** attempt))
                        continue
                    # Insert noop action as fallback
                    agent_log(
                        logging.WARNING,
                        self.state_manager.state.agent_id,
                        self.state_manager.state.n_steps,
                        "Model still returned empty after retry. Inserting safe noop action."
                    )
                    action_instance = self.ActionModel()
                    setattr(
                        action_instance,
                        "done",
                        {
                            "success": False,
                            "text": "No next action returned by LLM!",
                        },
                    )
                    parsed.action = [action_instance]

                return parsed

            except (asyncio.TimeoutError, LLMException) as e:
                agent_log(
                    logging.WARNING,
                    self.state_manager.state.agent_id,
                    self.state_manager.state.n_steps,
                    f"LLM call attempt {attempt+1} failed: {type(e).__name__}: {e}",
                    exc_info=True
                )
                if attempt >= max_retries:
                    raise LLMException("LLM call failed after all retries.") from e
                await asyncio.sleep(1.0 * (2 ** attempt))

        raise LLMException("LLM call failed.")

    def _log_llm_output(self, parsed):
        """
        Logs LLM output fields using agent_log for consistency with rest of system.
        """
        state = self.state_manager.state

        # Log thinking if present
        thinking = getattr(parsed, "thinking", None)
        if thinking and thinking.strip() and thinking.strip() != "N/A":
            agent_log(logging.INFO, state.agent_id, state.n_steps, f"💡 Thinking:\n{thinking}")

        # Log prior action assessment with appropriate emoji
        eval_goal = getattr(parsed, "prior_action_assessment", None)
        if eval_goal and eval_goal.strip() and eval_goal.strip() != "N/A":
            if 'success' in eval_goal.lower():
                emoji = '👍'
            elif 'failure' in eval_goal.lower():
                emoji = '⚠️'
            else:
                emoji = '❔'
            agent_log(logging.INFO, state.agent_id, state.n_steps, f"{emoji} Eval: {eval_goal}")

        # Log task_log if present
        task_log = getattr(parsed, "task_log", None)
        if task_log and task_log.strip() and task_log.strip() != "N/A":
            agent_log(logging.INFO, state.agent_id, state.n_steps, f"🧠 Task Log: {task_log}")

        # Log next goal if present
        next_goal = getattr(parsed, "next_goal", None)
        if next_goal and next_goal.strip() and next_goal.strip() != "N/A":
            agent_log(logging.INFO, state.agent_id, state.n_steps, f"🎯 Next goal: {next_goal}")

        # Add spacing for readability
        agent_log(logging.INFO, state.agent_id, state.n_steps, "")

    async def _heartbeat_loop(self):
        """Send periodic heartbeat events every 2 seconds."""
        while True:
            try:
                await asyncio.sleep(2.0)  # Send heartbeat every 2 seconds

                # Create and emit heartbeat event
                heartbeat = Heartbeat(
                    step_token=self.state_manager.state.n_steps,
                    component_name="decision_maker"
                )

                try:
                    self.heartbeat_bus.put_nowait(heartbeat)
                    logger.debug("DecisionMaker heartbeat sent")
                except asyncio.QueueFull:
                    logger.warning("Agent bus full, dropped DecisionMaker heartbeat")

            except asyncio.CancelledError:
                logger.debug("DecisionMaker heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in DecisionMaker heartbeat loop: {e}")
                await asyncio.sleep(2.0)  # Continue after error
