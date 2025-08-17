from __future__ import annotations

import asyncio
import logging
import uuid
import time
from typing import TYPE_CHECKING, Optional

from browser_use.agent.events import ErrorEvent, LLMRequest, LLMResponse, StepFinalized, Heartbeat, AssessmentUpdate
from browser_use.agent.mode_router import ModeRouter, RouterConfig
from browser_use.agent.state_manager import agent_log
from browser_use.agent.views import AgentHistoryList, ReflectionPlannerOutput
from browser_use.agent.prompts import PlannerPrompt

if TYPE_CHECKING:
    from browser_use.agent.settings import AgentSettings
    from browser_use.agent.state_manager import StateManager

logger = logging.getLogger(__name__)


class Planner:
    """
    Standalone planning component.
    Listens to StepFinalized and ErrorEvent, triggers planning via LLM,
    and applies the results into the agent state/history.
    """

    def __init__(
        self,
        settings: AgentSettings,
        state_manager: StateManager,
        agent_bus: asyncio.Queue,
        heartbeat_bus: asyncio.Queue,
    ):
        self.settings = settings
        self.state_manager = state_manager
        self.agent_bus = agent_bus
        self.heartbeat_bus = heartbeat_bus
        self.pending_requests: dict[str, dict] = {}
        self._last_planned_step: int | None = None
        # Assessment-driven trigger state
        self._last_mode_hint: str | None = None
        self._last_assessment_plan_ts: float | None = None
        self._assessment_cooldown_seconds: float = max(
            0.1, float(getattr(self.settings, 'assessor_cooldown_seconds', 2.0))
        )
        self._dwell_enter_ts: float | None = None
        self._dwell_mode: str | None = None
        # Time-based cadence
        self._last_plan_ts: float | None = None
        # Mode router and last assessment snapshot
        self._router = ModeRouter(RouterConfig())
        self._last_assessment: Optional[AssessmentUpdate] = None
        self._last_error_event: Optional[ErrorEvent] = None
        # Planner screenshot per-step cache to avoid repeated captures within the same step
        self._planner_shot_cache_step: Optional[int] = None
        self._planner_shot_cache_b64: Optional[str] = None

    async def run(self):
        logger.debug("Planner component started.")
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        async def main_loop():
            from browser_use.agent.state_manager import TERMINAL_STATES
            status_check_counter = 0
            while True:
                try:
                    event = await asyncio.wait_for(self.agent_bus.get(), timeout=0.5)

                    if isinstance(event, StepFinalized):
                        await self._maybe_plan_on_interval(event)
                        self.agent_bus.task_done()
                    elif isinstance(event, ErrorEvent):
                        self._last_error_event = event
                        await self._maybe_plan_on_error(event)
                        self.agent_bus.task_done()
                    elif isinstance(event, LLMResponse):
                        await self._handle_llm_response(event)
                        self.agent_bus.task_done()
                    elif isinstance(event, AssessmentUpdate):
                        self._last_assessment = event
                        await self._maybe_plan_on_assessment(event)
                        self.agent_bus.task_done()
                    elif isinstance(event, dict) and event.get('type') == 'autonomous_planning_request':
                        # Long-running mode can request autonomous planning
                        # Use current step if not provided
                        step_token = event.get('context', {}).get('current_step', self.state_manager.state.n_steps)
                        await self._emit_planning_request(step_token, source="autonomous")
                        self.agent_bus.task_done()
                    else:
                        # Not for us; requeue and yield
                        try:
                            self.agent_bus.put_nowait(event)
                        except asyncio.QueueFull:
                            agent_log(
                                logging.WARNING,
                                self.state_manager.state.agent_id,
                                self.state_manager.state.n_steps,
                                "Agent bus full, cannot requeue event. Event may be lost.",
                            )
                        finally:
                            self.agent_bus.task_done()
                            await asyncio.sleep(0)
                        continue

                except asyncio.TimeoutError:
                    # Time-based planner cadence
                    try:
                        interval_s = float(getattr(self.settings, 'planner_interval_seconds', 0.0) or 0.0)
                    except Exception:
                        interval_s = 0.0
                    if interval_s > 0:
                        now = time.monotonic()
                        if self._last_plan_ts is None or (now - self._last_plan_ts) >= interval_s:
                            step_token = self.state_manager.state.n_steps
                            asyncio.create_task(self._emit_planning_request(step_token))
                            self._last_plan_ts = now
                    status_check_counter += 1
                    if status_check_counter >= 10:
                        status_check_counter = 0
                        status = await self.state_manager.get_status()
                        if status not in TERMINAL_STATES:
                            continue
                        else:
                            break
                    else:
                        continue
                except Exception as e:
                    error_msg = f"Planner event loop failed: {e}"
                    agent_log(
                        logging.CRITICAL,
                        self.state_manager.state.agent_id,
                        self.state_manager.state.n_steps,
                        error_msg,
                        exc_info=True,
                    )
                    await self.state_manager.record_error(error_msg, is_critical=True)
                    await asyncio.sleep(1)

        try:
            await asyncio.gather(main_loop(), heartbeat_task, return_exceptions=True)
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        logger.info("Planner component stopped.")

    async def _maybe_plan_on_interval(self, event: StepFinalized):
        # Only plan if enabled
        if not self.settings.use_planner:
            return
        # Don't plan on step 0 to avoid duplication with initial context
        if event.step_token is None or event.step_token <= 0:
            return
        # Health-driven reflection override: if the engine requested reflection this epoch,
        # run planner immediately (debounced per step) regardless of interval/cadence.
        try:
            if getattr(self.state_manager.state, 'reflection_requested_this_epoch', False):
                step_num = int(event.step_token)
                if self._last_planned_step != step_num:
                    agent_log(
                        logging.INFO,
                        self.state_manager.state.agent_id,
                        self.state_manager.state.n_steps,
                        f"Planner triggered by reflection intent at step={step_num}",
                    )
                    asyncio.create_task(self._emit_planning_request(step_num, source="reflection"))
                    return
        except Exception:
            # Never fail planner trigger on inspection
            pass

        if self.settings.planner_interval <= 0:
            return

        # Enforce interval
        step_num = event.step_token
        if step_num % self.settings.planner_interval != 0:
            return

        # Debounce if we already planned on this step
        if self._last_planned_step == step_num:
            return

        agent_log(
            logging.INFO,
            self.state_manager.state.agent_id,
            self.state_manager.state.n_steps,
            f"Planner triggered on interval: step={step_num}, every={self.settings.planner_interval} steps",
        )
        asyncio.create_task(self._emit_planning_request(step_num, source="interval"))

    async def _maybe_plan_on_error(self, event: ErrorEvent):
        if not (self.settings.use_planner and self.settings.reflect_on_error):
            return
        step_num = event.step_token
        # Debounce per step
        if self._last_planned_step == step_num:
            return
        # Log why planner is running
        last_err = getattr(self.state_manager.state, "last_error", None)
        agent_log(
            logging.INFO,
            self.state_manager.state.agent_id,
            self.state_manager.state.n_steps,
            f"Planner triggered due to error at step={step_num}: {last_err}",
        )
        asyncio.create_task(self._emit_planning_request(step_num, source="error"))

    async def _maybe_plan_on_assessment(self, event: AssessmentUpdate):
        # Only if planner is enabled
        if not self.settings.use_planner:
            return

        now = time.monotonic()

        # First, ask the ModeRouter for a decision (includes hysteresis/cooldowns)
        mode_hint = self._router.decide(event)

        # Backward-compatible fallback if router returns None: simple thresholds + hysteresis from settings
        if mode_hint is None:
            risk_hi = float(getattr(self.settings, 'assessor_risk_trigger', 0.65))
            risk_lo = float(getattr(self.settings, 'assessor_risk_clear', 0.45))
            conf_lo = float(getattr(self.settings, 'assessor_confidence_trigger', 0.35))
            conf_hi = float(getattr(self.settings, 'assessor_confidence_clear', 0.55))
            if event.risk >= risk_hi or event.confidence <= conf_lo:
                mode_hint = 'reactive'
            elif event.risk <= risk_lo and event.confidence >= conf_hi and event.opportunity >= 0.5:
                mode_hint = 'proactive'
            else:
                # No decisive signal
                self._dwell_enter_ts = None
                self._dwell_mode = None
                return

        dwell = float(getattr(self.settings, 'assessor_dwell_seconds', 0.5))

        # Dwell requirement: maintain same mode over dwell seconds before triggering
        if self._dwell_mode != mode_hint:
            self._dwell_mode = mode_hint
            self._dwell_enter_ts = now
            return
        if self._dwell_enter_ts is not None and (now - self._dwell_enter_ts) < dwell:
            return

        self._last_mode_hint = mode_hint
        # Cooldown in seconds to avoid oscillation
        if self._last_assessment_plan_ts is not None and (now - self._last_assessment_plan_ts) < self._assessment_cooldown_seconds:
            return
        await self._emit_planning_request(event.step_token, source=f"assessment:{mode_hint}")
        self._last_assessment_plan_ts = now

    async def _emit_planning_request(self, step_token: int, source: str = "manual"):
        try:
            state = self.state_manager.state

            # Prepare history window
            history_for_planner = AgentHistoryList(
                history=list(state.history.history)[-self.settings.planner_history_window :]
            )

            # Get task context
            current_task_id = await self.state_manager.get_current_task_id()
            task_stack_summary = await self.state_manager.get_task_stack_summary()

            # Collect screenshots for planner if enabled
            screenshots: list[str] = []
            if getattr(self.settings, 'use_vision_for_planner', False):
                # Enforce: one current screenshot only. Use per-step cache to avoid repeated capture.
                latest_only = bool(getattr(self.settings, 'planner_use_latest_screenshot_only', False))
                browser_session = getattr(self.state_manager, 'browser_session', None)

                current_screenshot: Optional[str] = None
                # Reuse cache if already captured for this step
                if self._planner_shot_cache_step == step_token and self._planner_shot_cache_b64:
                    current_screenshot = self._planner_shot_cache_b64
                else:
                    if browser_session is not None:
                        try:
                            current_screenshot = await browser_session.take_screenshot()
                            self._planner_shot_cache_step = step_token
                            self._planner_shot_cache_b64 = current_screenshot
                        except Exception as fresh_e:
                            logger.debug("Fresh screenshot failed: %s", type(fresh_e).__name__)
                            # Ensure we don't reuse a stale cache if capture failed
                            self._planner_shot_cache_step = step_token
                            self._planner_shot_cache_b64 = None

                if current_screenshot:
                    screenshots = [current_screenshot]
                    logger.debug("🖼️ Planner using current screenshot (cached=%s)", str(self._planner_shot_cache_b64 is not None))
                else:
                    logger.debug("⚠️ No current screenshot available for planner")

                # If user demands latest-only, skip all history fallbacks to avoid stale/extra images
                if not screenshots and not latest_only:
                    n = max(0, int(getattr(self.settings, 'planner_images_per_step', 1))) or 0
                    if n > 0 and state.history:
                        try:
                            recent = state.history.screenshots(n_last=n, return_none_if_not_screenshot=False)
                            screenshots.extend([s for s in recent if s is not None])
                        except Exception:
                            pass
                    # Final fallback to last known browser screenshot if not present in history
                    try:
                        bs = state.history.history[-1].state.screenshot if state.history and state.history.history else None
                        if bs:
                            screenshots.append(bs)
                    except Exception:
                        pass

                # Hard-cap to 1 image for planner to minimize latency/cost
                if screenshots:
                    screenshots = screenshots[:1]

            # Determine doctrine from last mode hint and error presence
            doctrine = 'medic' if (self._last_mode_hint == 'reactive' or self._last_error_event is not None) else 'scout'

            # Build FailureContext if we have recent error info
            failure_ctx = None
            if doctrine == 'medic':
                try:
                    last_action = None
                    try:
                        last_action = history_for_planner.last_action()
                    except Exception:
                        last_action = None
                    if self._last_error_event is not None:
                        failure_ctx = {
                            "error_type": self._last_error_event.error_type or None,
                            "error_message": self._last_error_event.error_message or None,
                            "last_action": last_action,
                        }
                    else:
                        failure_ctx = {
                            "error_type": None,
                            "error_message": getattr(state, 'last_error', None),
                            "last_action": last_action,
                        }
                except Exception:
                    failure_ctx = {
                        "error_type": None,
                        "error_message": getattr(state, 'last_error', None),
                        "last_action": None,
                    }

            # Build browser summary for environment state in templates
            browser_summary = {}
            try:
                if history_for_planner.history and history_for_planner.history[-1].state:
                    last_state = history_for_planner.history[-1].state
                    browser_summary = {
                        'url': last_state.url,
                        'clickable_elements': 'indexed clickable elements available',
                        'new_downloads': [],
                    }
            except Exception:
                browser_summary = {}

            planner_prompt = PlannerPrompt(
                task=state.task,
                history=history_for_planner,
                last_error=state.last_error,
                current_task_id=current_task_id,
                task_context=task_stack_summary,
                screenshots=screenshots,
                vision_detail_level=getattr(self.settings, 'planner_vision_detail', 'auto'),
                use_vision=getattr(self.settings, 'use_vision_for_planner', False),
                doctrine=doctrine,
                failure_context=failure_ctx,
                current_intent=state.current_goal or state.task,
                browser_summary=browser_summary,
            )
            # Pass a lightweight mode hint for the planner based on latest assessment
            try:
                hint_lines = []
                if self._last_mode_hint == 'reactive':
                    hint_lines.append("Mode: Reactive (Medic) — diagnose recent failures/timeouts; propose stabilizing next steps.")
                elif self._last_mode_hint == 'proactive':
                    hint_lines.append("Mode: Proactive (Scout) — explore promising leads or accelerate progress.")
                # Budget awareness
                assess = self._last_assessment
                if assess and assess.risk >= 0.70:
                    hint_lines.append("Budget hint: high risk — keep I/O and token usage low; prefer 1–2 bounded actions.")
                elif assess and assess.opportunity >= 0.75 and assess.confidence >= 0.60:
                    hint_lines.append("Budget hint: solid opportunity — up to 3 short, bounded actions allowed.")
                planner_prompt.extra_context_hint = "\n".join(hint_lines) if hint_lines else None
            except Exception:
                pass

            # Attach assessment context for explainability (numbers + contributors + visual summary)
            try:
                if self._last_assessment is not None:
                    a = self._last_assessment
                    planner_prompt.assessment_context = {
                        "risk": round(float(a.risk), 3),
                        "opportunity": round(float(a.opportunity), 3),
                        "confidence": round(float(a.confidence), 3),
                        "stagnation": round(float(getattr(a, 'stagnation', 0.0)), 3),
                        "looping": round(float(getattr(a, 'looping', 0.0)), 3),
                        "contributors": list(a.contributors or []),
                        "visual_summary": getattr(a, 'visual_summary', "") or "",
                        "screenshot_refs": list(getattr(a, 'screenshot_refs', []) or []),
                        "trend_window": int(getattr(a, 'trend_window', 5) or 5),
                    }
            except Exception:
                pass

            # Log planner image usage
            try:
                if getattr(self.settings, 'use_vision_for_planner', False):
                    agent_log(
                        logging.INFO,
                        self.state_manager.state.agent_id,
                        self.state_manager.state.n_steps,
                        f"Planner including images: count={len(screenshots)}, detail={getattr(self.settings, 'planner_vision_detail', 'auto')}, latest_only={getattr(self.settings, 'planner_use_latest_screenshot_only', False)}",
                    )
            except Exception:
                pass
            messages = planner_prompt.get_messages()

            request_id = str(uuid.uuid4())
            self.pending_requests[request_id] = {
                "step_token": step_token,
                "messages": messages,
                "request_type": "planning",
            }

            # Debug: log outbound LLM request details (safe summary)
            try:
                msg_kinds = [getattr(m, "type", getattr(m, "role", type(m).__name__)) for m in messages]
                sample = None
                if messages:
                    # Attempt to extract a short preview of the last content
                    content = getattr(messages[-1], "content", None)
                    if isinstance(content, str):
                        sample = (content[:500] + "…") if len(content) > 500 else content
                agent_log(
                    logging.DEBUG,
                    self.state_manager.state.agent_id,
                    self.state_manager.state.n_steps,
                    f"Planner LLMRequest prepared (request_id={request_id}): messages={len(messages)}, kinds={msg_kinds}, sample_tail={bool(sample)}",
                )
                if sample:
                    agent_log(
                        logging.DEBUG,
                        self.state_manager.state.agent_id,
                        self.state_manager.state.n_steps,
                        f"Planner LLMRequest sample content (tail message, truncated):\n{sample}",
                    )
            except Exception:
                # Never fail planning due to logging issues
                pass

            llm_request = LLMRequest(
                step_token=step_token,
                task_id=current_task_id,
                messages=messages,
                output_schema=ReflectionPlannerOutput,
                request_id=request_id,
                max_retries=2,
                request_type="planning",
            )
            try:
                self.agent_bus.put_nowait(llm_request)
                self._last_planned_step = step_token
                agent_log(
                    logging.INFO,
                    self.state_manager.state.agent_id,
                    self.state_manager.state.n_steps,
                    f"Planner request published for step {step_token} (request_id={request_id}, doctrine={doctrine}, source={source})",
                )
            except asyncio.QueueFull:
                agent_log(
                    logging.WARNING,
                    self.state_manager.state.agent_id,
                    self.state_manager.state.n_steps,
                    "Agent bus full, dropping Planner LLM request",
                )
        except Exception as e:
            error_msg = f"Planning request preparation failed: {e}"
            agent_log(
                logging.CRITICAL,
                self.state_manager.state.agent_id,
                self.state_manager.state.n_steps,
                error_msg,
                exc_info=True,
            )
            await self.state_manager.record_error(error_msg, is_critical=True)

    async def _handle_llm_response(self, event: LLMResponse):
        # Only handle responses that correspond to our requests
        try:
            context = self.pending_requests.pop(event.request_id, None)
            if not context:
                # Not ours — requeue for other consumers and yield
                try:
                    self.agent_bus.put_nowait(event)
                except asyncio.QueueFull:
                    agent_log(
                        logging.WARNING,
                        self.state_manager.state.agent_id,
                        self.state_manager.state.n_steps,
                        "Planner: queue full, cannot requeue unmatched LLMResponse",
                    )
                await asyncio.sleep(0)
                return

            # Step token check
            if context.get("step_token") != event.step_token:
                agent_log(
                    logging.WARNING,
                    self.state_manager.state.agent_id,
                    self.state_manager.state.n_steps,
                    f"Planner LLMResponse step token mismatch: expected {context.get('step_token')}, got {event.step_token}",
                )
                return

            if not event.success or not event.response:
                agent_log(
                    logging.WARNING,
                    self.state_manager.state.agent_id,
                    self.state_manager.state.n_steps,
                    "Planner LLMResponse unsuccessful or empty",
                )
                return

            try:
                if isinstance(event.response, ReflectionPlannerOutput):
                    # Debug: raw planner response summary
                    try:
                        ms_len = len(event.response.memory_summary or "")
                        tl_preview = (event.response.task_log or "")[:200]
                        ng_preview = (event.response.next_goal or "")[:200]
                        strat_preview = (event.response.effective_strategy or "")[:200]
                        agent_log(
                            logging.DEBUG,
                            self.state_manager.state.agent_id,
                            self.state_manager.state.n_steps,
                            "Planner LLMResponse received (request_id=%s): memory_summary_len=%d, task_log='%s', next_goal='%s', strategy='%s'"
                            % (event.request_id, ms_len, tl_preview, ng_preview, strat_preview),
                        )
                    except Exception:
                        pass

                    await self.state_manager.update_last_history_with_reflection(
                        event.response.memory_summary,
                        event.response.next_goal,
                        event.response.effective_strategy,
                    )
                    await self.state_manager.clear_error_and_failures()
                    # Mark reflection completion for cooldown timing
                    try:
                        await self.state_manager.mark_reflection_exit()
                    except Exception:
                        pass
                    # Info: succinct planner output insight
                    insight_next = (event.response.next_goal or "").strip()
                    insight_strat = (event.response.effective_strategy or "").strip()
                    if len(insight_strat) > 160:
                        insight_strat = insight_strat[:160] + "…"
                    agent_log(
                        logging.INFO,
                        self.state_manager.state.agent_id,
                        self.state_manager.state.n_steps,
                        f"Planner results persisted: next_goal='{insight_next}', strategy='{insight_strat}'",
                    )
                else:
                    agent_log(
                        logging.WARNING,
                        self.state_manager.state.agent_id,
                        self.state_manager.state.n_steps,
                        f"Unexpected planner response type: {type(event.response)}",
                    )
            except Exception as e:
                agent_log(
                    logging.WARNING,
                    self.state_manager.state.agent_id,
                    self.state_manager.state.n_steps,
                    f"Failed to persist planner output: {e}",
                )
        except Exception as e:
            error_msg = f"Planner LLM response handling failed: {e}"
            agent_log(
                logging.ERROR,
                self.state_manager.state.agent_id,
                self.state_manager.state.n_steps,
                error_msg,
                exc_info=True,
            )
            await self.state_manager.record_error(error_msg)

    async def _heartbeat_loop(self):
        while True:
            try:
                await asyncio.sleep(2.0)
                hb = Heartbeat(step_token=self.state_manager.state.n_steps, component_name="planner")
                try:
                    self.heartbeat_bus.put_nowait(hb)
                except asyncio.QueueFull:
                    logger.warning("Planner heartbeat dropped (queue full)")
            except asyncio.CancelledError:
                logger.debug("Planner heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in Planner heartbeat loop: {e}")
                await asyncio.sleep(2.0)
