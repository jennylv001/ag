from __future__ import annotations

"""
State-driven SolveCaptchaTask.

Implements the existing iterative CAPTCHA-solving flow using the new BaseTask
contract without changing behavior. The prompts and loop are largely translated
from the legacy implementation.
"""

import asyncio
import base64
import json
import logging
from typing import Any, Dict, List, Optional

from browser_use.agent.tasks.base_task import BaseTask, TaskResult
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import (
    UserMessage,
    ContentPartTextParam,
    ContentPartImageParam,
    ImageURL,
)
from browser_use.agent.tasks.views import (
    CaptchaInitialPlan as _LegacyInitialModel,
    CaptchaFollowUpPlan as _LegacyFollowModel,
)

logger = logging.getLogger(__name__)


# --- Hardened Prompts ---
CAPTCHA_INITIAL_PROMPT = """
Role: CAPTCHA Solver — Initial Planning (One-Sequence Strategy)
Objective: Analyze the screenshot and prescribe a single, complete clicking sequence that selects ALL required targets and then submits the solution.

Inputs you receive:
- Screenshot: The CAPTCHA as an image.
- Indexed Elements JSON: All clickable elements with indexes and any visible labels.

Your job:
1) Read the instruction in the CAPTCHA (e.g., "Select all images with cars"). For reCAPTCHA/rotating tasks, assume: keep selecting matching tiles until none remain.
2) Identify ALL image tiles that satisfy the instruction now. Do not include controls like Verify/Next in the tile list.
3) Find the submission control:
    - Prefer a 'Verify'/'Submit' button. If present now, set `verify_button_index`.
    - If a rotating challenge uses 'Next', set `next_button_index` when present.
    - If both appear, prefer Verify.
4) Produce a plan that allows the agent to perform a ONE multi-action step:
    - Click every target tile exactly once.
    - Then click Verify if visible; otherwise, click Next if that is used by this challenge.
    - If neither Verify nor Next is visible at this moment, leave both indexes null (the agent will wait and re-check).

Output Schema:
- Return a JSON object matching CaptchaInitialPlan: { analysis, initial_click_indexes, verify_button_index?, next_button_index? }

Strict rules:
- initial_click_indexes must be unique integers with only target tiles.
- Prefer verify_button_index over next_button_index when both exist.
- Return ONLY the JSON object. No extra text.
"""

CAPTCHA_FOLLOW_UP_PROMPT = """
Role: CAPTCHA Solver — Follow-up Planning (Finish in One Sequence)
Objective: After the first sequence, analyze the new state and prescribe the next minimal one-shot sequence to complete or advance the challenge.

Inputs:
- New Screenshot and Indexed Elements JSON
- Previous Plan JSON

Your job:
1) Compare states. If the instruction persists, select ALL remaining tiles that match.
2) Decide status:
    - SUCCESS_CRITERIA_MET: All required tiles are selected; ready to submit.
    - IN_PROGRESS: More tiles must be selected or a rotating set requires Next.
    - FAILED: An error/reset happened (e.g., explicit failure message).
3) Next sequence guidance:
    - If IN_PROGRESS: set next_click_indexes with ALL new tiles to select now. Also include verify_button_index or next_button_index when visible now (prefer Verify).
    - If SUCCESS_CRITERIA_MET: include verify_button_index if visible; else include next_button_index only if the challenge uses it.

Output Schema: CaptchaFollowUpPlan with fields { status, analysis, next_click_indexes?, verify_button_index?, next_button_index? }

Strict rules:
- Provide only JSON; no extra text.
- Prefer Verify over Next when both exist.
"""


# Note: Reuse legacy Pydantic models for structured output to avoid drift


class SolveCaptchaTask(BaseTask):
    """State-driven task to solve a visible CAPTCHA using LLM guidance and controller clicks."""

    def __init__(
        self,
        controller,
        browser,
        page_extraction_llm: BaseChatModel,
        max_follow_ups: int = 5,
    ) -> None:
        self.controller = controller
        self.browser = browser
        self.llm = page_extraction_llm
        self.max_follow_ups = int(max_follow_ups)
        self._done: bool = False
        self._success: bool = False
        self._last_plan: Dict[str, Any] | None = None
        self._verify_index: Optional[int] = None

    # --- BaseTask interface ---
    def is_done(self) -> bool:
        return self._done

    def succeeded(self) -> bool:
        return self._success

    def step(self) -> Optional[TaskResult]:  # Delegates to async runner
        # This task runs as a single async routine; step() is not used directly.
        return None

    # --- Public entry point ---
    async def run(self) -> "ActionResult":
        """Execute the full captcha solving sequence, returning an ActionResult."""
        from browser_use.agent.views import ActionResult
        try:
            logger.info("task_event task_started {'task_name': 'SolveCaptchaTask'}")
        except Exception:
            pass
        try:
            initial_plan = await self._initial_plan()
        except Exception as e:
            logger.error(f"Error getting initial captcha plan from LLM: {e}", exc_info=True)
            self._done = True
            self._success = False
            return ActionResult(
                is_done=False,
                success=False,
                error=f"Failed to generate initial captcha plan from LLM: {e}",
                extracted_content="Captcha solver failed to generate initial plan.",
                include_in_memory=True,
            )

        if not initial_plan.get("initial_click_indexes"):
            self._done = True
            self._success = False
            return ActionResult(
                is_done=False,
                success=False,
                error="LLM failed to generate an initial click plan for the captcha.",
                extracted_content="Captcha solver did not propose any initial clicks.",
                include_in_memory=True,
            )

        # Execute initial sequence: all target tiles, then verify/next if provided
        click_order = self._dedup_order(initial_plan.get("initial_click_indexes", []))
        self._verify_index = initial_plan.get("verify_button_index")
        self._next_index = initial_plan.get("next_button_index")
        if self._verify_index is not None:
            click_order = click_order + [self._verify_index]
        elif getattr(self, "_next_index", None) is not None:
            click_order = click_order + [self._next_index]  # type: ignore[attr-defined]

        await self._click_indexes(click_order)
        self._last_plan = initial_plan

        # Iterative attempts
        for i in range(self.max_follow_ups):
            logger.info(f"Captcha Loop Attempt {i+1}/{self.max_follow_ups}: Re-assessing battlefield...")
            await asyncio.sleep(2.5)

            follow = await self._follow_up(self._last_plan)
            status = follow.get("status")

            # Capture any newly provided verify/next buttons from LLM
            v_from_llm = follow.get("verify_button_index")
            n_from_llm = follow.get("next_button_index")
            if v_from_llm is not None:
                self._verify_index = v_from_llm
            if n_from_llm is not None:
                self._next_index = n_from_llm

            if status == "SUCCESS_CRITERIA_MET":
                logger.info("Captcha criteria met. Attempting to submit (Verify preferred, Next if applicable).")
                submission_clicks: List[int] = []
                if self._verify_index is None:
                    # Try to discover verify from current state
                    try:
                        current = await self.browser.get_state_summary(cache_clickable_elements_hashes=False, include_screenshot=False)
                        self._verify_index = self._find_button_index(current.selector_map, ["verify", "submit"]) or None
                    except Exception:
                        pass
                if self._verify_index is not None:
                    submission_clicks.append(self._verify_index)
                elif getattr(self, "_next_index", None) is not None:
                    submission_clicks.append(self._next_index)  # type: ignore[attr-defined]

                if submission_clicks:
                    await self._click_indexes(submission_clicks)
                self._done = True
                self._success = True
                return ActionResult(
                    is_done=False,
                    extracted_content="Captcha solved successfully.",
                    include_in_memory=True,
                )
            elif status == "IN_PROGRESS":
                next_clicks = self._dedup_order(follow.get("next_click_indexes", []) or [])
                if not next_clicks:
                    logger.warning("LLM reported IN_PROGRESS but provided no new clicks. Ending attempts.")
                    break
                # Append submission control if LLM surfaced it this round
                if self._verify_index is not None:
                    next_clicks = next_clicks + [self._verify_index]
                elif getattr(self, "_next_index", None) is not None:
                    next_clicks = next_clicks + [self._next_index]  # type: ignore[attr-defined]
                await self._click_indexes(next_clicks)
                self._last_plan = follow
            else:  # FAILED or unknown
                logger.error(f"LLM reported captcha attempt failed or invalid status: {status}. Aborting loop.")
                break

        self._done = True
        self._success = False
        return ActionResult(
            is_done=False,
            success=False,
            error="Failed to solve captcha after multiple iterative attempts.",
            extracted_content="Captcha solving attempts exhausted without success.",
            include_in_memory=True,
        )

    # --- Helpers ---
    async def _initial_plan(self) -> Dict[str, Any]:
        state = await self.browser.get_state_summary(cache_clickable_elements_hashes=False, include_screenshot=True)
        screenshot_b64 = state.screenshot or ""
        if not screenshot_b64:
            try:
                maybe_ss = await self.browser.take_screenshot(full_page=True)
                screenshot_b64 = maybe_ss or ""
            except Exception:
                screenshot_b64 = ""

        selector_map = state.selector_map
        if screenshot_b64:
            screenshot_b64 = await self._clip_screenshot_if_possible(screenshot_b64, selector_map)

        indexed_elements = [self._deep_serialize(node.__json__()) for node in selector_map.values()]

        prompt_text = f"{CAPTCHA_INITIAL_PROMPT}\n\nIndexed Elements JSON:\n{json.dumps(indexed_elements, indent=2)}"
        message = UserMessage(
            content=[
                ContentPartTextParam(text=prompt_text),
                ContentPartImageParam(image_url=ImageURL(url=f"data:image/png;base64,{screenshot_b64}")),
            ]
        )
        resp = await self.llm.ainvoke([message], output_format=_LegacyInitialModel)
        # Above: keep dynamic to avoid tight coupling; rely on model_dump if present.
        try:
            return resp.completion.model_dump()
        except Exception:
            # Fallback to plain dict
            return json.loads(getattr(resp, 'completion', '{}'))

    async def _follow_up(self, previous_plan: Dict[str, Any]) -> Dict[str, Any]:
        new_state = await self.browser.get_state_summary(cache_clickable_elements_hashes=False, include_screenshot=True)
        new_screenshot_b64 = new_state.screenshot or ""
        if not new_screenshot_b64:
            try:
                maybe_ss = await self.browser.take_screenshot(full_page=True)
                new_screenshot_b64 = maybe_ss or ""
            except Exception:
                new_screenshot_b64 = ""
        if new_screenshot_b64:
            new_screenshot_b64 = await self._clip_screenshot_if_possible(new_screenshot_b64, new_state.selector_map)

        new_indexed_elements = [self._deep_serialize(node.__json__()) for node in new_state.selector_map.values()]

        prompt_text = (
            f"{CAPTCHA_FOLLOW_UP_PROMPT}\n\nPrevious Plan:\n{json.dumps(previous_plan, indent=2)}\n\n"
            f"New Indexed Elements JSON:\n{json.dumps(new_indexed_elements, indent=2)}"
        )
        message = UserMessage(
            content=[
                ContentPartTextParam(text=prompt_text),
                ContentPartImageParam(image_url=ImageURL(url=f"data:image/png;base64,{new_screenshot_b64}")),
            ]
        )
        resp = await self.llm.ainvoke([message], output_format=_LegacyFollowModel)
        try:
            return resp.completion.model_dump()
        except Exception:
            return json.loads(getattr(resp, 'completion', '{}'))

    async def _click_indexes(self, indexes: List[int]) -> None:
        from browser_use.controller.views import ClickElementAction
        ActionModel = self.controller.registry.create_action_model()
        actions = [ActionModel(click_element_by_index=ClickElementAction(index=i)) for i in indexes]
        await self.controller.multi_act(actions, self.browser, page_extraction_llm=self.llm)

    def _dedup_order(self, indexes: List[int]) -> List[int]:
        seen = set()
        ordered: List[int] = []
        for i in indexes:
            if isinstance(i, int) and i not in seen:
                seen.add(i)
                ordered.append(i)
        return ordered

    def _find_button_index(self, selector_map: Dict[int, Any], phrases: List[str]) -> Optional[int]:
        try:
            lowered = [p.lower() for p in phrases]
            for idx, node in selector_map.items():
                try:
                    data = node.__json__() if hasattr(node, "__json__") else getattr(node, "model_dump", lambda: {})()
                except Exception:
                    data = {}
                text_candidates: List[str] = []
                for key in ("text", "inner_text", "innerText", "aria_label", "ariaLabel", "name", "label", "accessible_name", "role"):
                    val = None
                    try:
                        # deep search in dict
                        def walk(v):
                            if isinstance(v, dict):
                                for kk, vv in v.items():
                                    if kk == key and isinstance(vv, str):
                                        text_candidates.append(vv)
                                    else:
                                        walk(vv)
                            elif isinstance(v, list):
                                for it in v:
                                    walk(it)
                        walk(data)
                    except Exception:
                        pass
                blob = " ".join([t for t in text_candidates if isinstance(t, str)]).lower()
                if any(p in blob for p in lowered):
                    return int(idx)
        except Exception:
            return None
        return None

    async def _clip_screenshot_if_possible(self, screenshot_b64_in: str, elements: Dict[int, Any]) -> str:
        try:
            from PIL import Image  # type: ignore
            import io

            raw = base64.b64decode(screenshot_b64_in)
            im = Image.open(io.BytesIO(raw))
            img_w, img_h = im.width, im.height
            if img_w <= 0 or img_h <= 0:
                return screenshot_b64_in

            boxes: List[Dict[str, int]] = []
            for n in elements.values():
                vc = getattr(n, 'viewport_coordinates', None)
                if vc and hasattr(vc, 'model_dump'):
                    d = vc.model_dump()
                    x = int(d['top_left']['x']); y = int(d['top_left']['y'])
                    w = int(d['width']); h = int(d['height'])
                    if w <= 0 or h <= 0:
                        continue
                    area = w * h
                    if area <= 4 or area >= img_w * img_h * 0.9:
                        continue
                    boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area})

            if not boxes:
                return screenshot_b64_in

            pad = 8
            gap = max(10, int(0.01 * max(img_w, img_h)))

            def rect_distance(a: Dict[str, int], b: Dict[str, int]) -> int:
                ax0, ay0, ax1, ay1 = a['x'], a['y'], a['x'] + a['w'], a['y'] + a['h']
                bx0, by0, bx1, by1 = b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']
                dx = 0
                if ax1 < bx0:
                    dx = bx0 - ax1
                elif bx1 < ax0:
                    dx = ax0 - bx1
                dy = 0
                if ay1 < by0:
                    dy = by0 - ay1
                elif by1 < ay0:
                    dy = ay0 - by1
                return max(dx, dy)

            n = len(boxes)
            adj: List[List[int]] = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if rect_distance(boxes[i], boxes[j]) <= gap:
                        adj[i].append(j)
                        adj[j].append(i)

            seen = [False] * n
            clusters: List[List[int]] = []
            for i in range(n):
                if seen[i]:
                    continue
                stack = [i]
                seen[i] = True
                comp: List[int] = []
                while stack:
                    k = stack.pop()
                    comp.append(k)
                    for nb in adj[k]:
                        if not seen[nb]:
                            seen[nb] = True
                            stack.append(nb)
                clusters.append(comp)

            def cluster_bounds(comp: List[int]) -> Dict[str, int]:
                xs0 = [boxes[i]['x'] for i in comp]
                ys0 = [boxes[i]['y'] for i in comp]
                xs1 = [boxes[i]['x'] + boxes[i]['w'] for i in comp]
                ys1 = [boxes[i]['y'] + boxes[i]['h'] for i in comp]
                return {'x0': min(xs0), 'y0': min(ys0), 'x1': max(xs1), 'y1': max(ys1)}

            scored: List[tuple[float, int, Dict[str, int], List[int]]] = []
            for comp in clusters:
                b = cluster_bounds(comp)
                w = max(1, b['x1'] - b['x0'])
                h = max(1, b['y1'] - b['y0'])
                perim_padded = 2 * (max(1, w + 2 * pad) + max(1, h + 2 * pad))
                sum_area = sum(boxes[i]['area'] for i in comp)
                density = sum_area / max(1.0, float(perim_padded))
                scored.append((-density, -len(comp), b, comp))

            scored.sort()
            _, _, best_b, _ = scored[0]

            x0 = max(0, best_b['x0'] - pad)
            y0 = max(0, best_b['y0'] - pad)
            x1 = min(img_w, best_b['x1'] + pad)
            y1 = min(img_h, best_b['y1'] + pad)
            if x1 <= x0 or y1 <= y0:
                return screenshot_b64_in

            from io import BytesIO
            cropped = im.crop((x0, y0, x1, y1))
            buf = BytesIO()
            cropped.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode('ascii')
        except Exception:
            return screenshot_b64_in

    def _deep_serialize(self, element_dict):
        if isinstance(element_dict, dict):
            return {k: self._deep_serialize(v) for k, v in element_dict.items()}
        elif isinstance(element_dict, list):
            return [self._deep_serialize(i) for i in element_dict]
        elif hasattr(element_dict, 'model_dump'):
            return element_dict.model_dump()
        elif hasattr(element_dict, '__dict__'):
            return {k: self._deep_serialize(v) for k, v in element_dict.__dict__.items()}
        else:
            return element_dict
