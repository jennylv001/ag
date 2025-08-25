from __future__ import annotations

import json
import logging
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    # Define ActuationResult for type hints (no longer importing from events)
    class ActuationResult:
        action_results: Any
        llm_output: Any
        browser_state: Any
        step_metadata: Any

    # Import only for typing to avoid runtime import issues
    from browser_use.agent.state import StateManager

logger = logging.getLogger(__name__)


def _safe(obj: Any, default: Any = None) -> Any:
    try:
        return obj
    except Exception:
        return default


def build_step_summary(result: 'ActuationResult', state_manager: 'StateManager') -> Dict[str, Any]:
    """Build a compact, structured step summary for operators.

    Includes:
    - step number, timings
    - url/title
    - llm thinking/task_log/next_goal (when present)
    - actions (names only) and outcomes
    - transition signals and status after decide
    """
    try:
        state = state_manager.state
        step_num = _safe(result.step_metadata.step_number if result.step_metadata else state.n_steps)
        url = _safe(getattr(result.browser_state, 'url', None))
        title = _safe(getattr(result.browser_state, 'title', None))
        timing = None
        if result.step_metadata:
            try:
                timing = {
                    'start': result.step_metadata.step_start_time,
                    'end': result.step_metadata.step_end_time,
                    'duration_seconds': result.step_metadata.duration_seconds,
                }
            except Exception:
                pass

        thinking = None
        task_log = None
        next_goal = None
        if result.llm_output:
            thinking = getattr(result.llm_output, 'thinking', None)
            task_log = getattr(result.llm_output, 'task_log', None)
            next_goal = getattr(result.llm_output, 'next_goal', None)

        actions = []
        try:
            for a in (result.llm_output.action if result.llm_output and result.llm_output.action else []):
                try:
                    actions.append(type(a).__name__)
                except Exception:
                    actions.append('UnknownAction')
        except Exception:
            pass

        outcomes = []
        for r in (result.action_results or []):
            try:
                outcomes.append({
                    'is_done': bool(getattr(r, 'is_done', False)),
                    'success': getattr(r, 'success', None),
                    'error': getattr(r, 'error', None),
                })
            except Exception:
                outcomes.append({'error': 'unserializable_result'})

        # Health/modes snapshot
        health = {
            'status': state.status.value,
            'modes': int(getattr(state, 'modes', 0)),
            'missed_heartbeats': int(getattr(state, 'missed_heartbeats', 0)),
            'io_timeouts_recent': int(len(getattr(state, 'io_timeouts_recent', []))),
            'load_status': getattr(state, 'load_status', None).value if getattr(state, 'load_status', None) else None,
        }

        return {
            'step': int(step_num),
            'url': url,
            'title': title,
            'timing': timing,
            'thinking': thinking,
            'task_log': task_log,
            'next_goal': next_goal,
            'actions': actions,
            'outcomes': outcomes,
            'health': health,
        }
    except Exception as e:
        logger.debug("Failed to build step summary", exc_info=True)
        return {'error': f'failed_to_build_summary: {e!s}'}


def log_step_summary(result: 'ActuationResult', state_manager: 'StateManager') -> None:
    summary = build_step_summary(result, state_manager)
    # Log the summary directly; logging will safely stringify complex objects
    logger.debug("STEP_SUMMARY %s", summary)
