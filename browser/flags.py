"""
Helpers to list configurable flags in the browser module.
- Profile flags (BrowserProfile fields)
- Session runtime state/params (BrowserSession fields)
- Chrome argument groups (defaults, docker, headless, disable security, deterministic rendering)
"""
from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from typing import Any, Dict, List, Tuple, Optional

try:
    from .profile import BrowserProfile, CHROME_DEFAULT_ARGS, CHROME_DOCKER_ARGS, CHROME_HEADLESS_ARGS, CHROME_DISABLE_SECURITY_ARGS, CHROME_DETERMINISTIC_RENDERING_ARGS
except Exception:
    # Fallbacks if imports fail in lightweight envs
    BrowserProfile = object  # type: ignore
    CHROME_DEFAULT_ARGS = []  # type: ignore
    CHROME_DOCKER_ARGS = []  # type: ignore
    CHROME_HEADLESS_ARGS = []  # type: ignore
    CHROME_DISABLE_SECURITY_ARGS = []  # type: ignore
    CHROME_DETERMINISTIC_RENDERING_ARGS = []  # type: ignore

@dataclass
class FlagsSummary:
    profile_fields: List[str]
    session_fields: List[str]
    chrome_arg_groups: Dict[str, List[str]]
    # Optional stealth category breakdown
    stealth: Optional[Dict[str, List[str]]] = None


def list_profile_flags() -> List[str]:
    """Return BrowserProfile field names that act as configuration flags."""
    try:
        return list(BrowserProfile.model_fields.keys())  # type: ignore[attr-defined]
    except Exception:
        # Best-effort fallback if Pydantic is unavailable in this context
        return [
            'stealth','disable_security','deterministic_rendering','allowed_domains','keep_alive','enable_default_extensions',
            'window_size','window_position','default_navigation_timeout','default_timeout','minimum_wait_page_load_time',
            'wait_for_network_idle_page_load_time','maximum_wait_page_load_time','wait_between_actions','include_dynamic_attributes',
            'highlight_elements','viewport_expansion','cookies_file','id','env','executable_path','headless','args',
            'ignore_default_args','channel','chromium_sandbox','devtools','slow_mo','timeout','proxy','downloads_path','traces_dir',
            'handle_sighup','handle_sigint','handle_sigterm','accept_downloads','offline','strict_selectors','proxy','permissions',
            'bypass_csp','client_certificates','extra_http_headers','http_credentials','ignore_https_errors','java_script_enabled',
            'base_url','service_workers','user_agent','screen','viewport','no_viewport','device_scale_factor','is_mobile','has_touch',
            'locale','geolocation','timezone_id','color_scheme','contrast','reduced_motion','forced_colors','record_har_content',
            'record_har_mode','record_har_omit_content','record_har_path','record_har_url_filter','record_video_dir','record_video_size',
            'storage_state','user_data_dir'
        ]


def list_session_flags() -> List[str]:
    """Return BrowserSession configurable fields and runtime toggles."""
    try:
        from .session import BrowserSession
        # Use pydantic model metadata
        return list(BrowserSession.model_fields.keys())  # type: ignore[attr-defined]
    except Exception:
        return [
            'id','browser_profile','wss_url','cdp_url','browser_pid','playwright','browser','browser_context','initialized',
            'agent_current_page','human_current_page'
        ]


def list_chrome_arg_groups() -> Dict[str, List[str]]:
    """Return dictionary of Chrome args grouped by purpose."""
    return {
        'default': list(CHROME_DEFAULT_ARGS),
        'docker': list(CHROME_DOCKER_ARGS),
        'headless': list(CHROME_HEADLESS_ARGS),
        'disable_security': list(CHROME_DISABLE_SECURITY_ARGS),
        'deterministic_rendering': list(CHROME_DETERMINISTIC_RENDERING_ARGS),
    }


def list_stealth_flags() -> Dict[str, List[str]]:
    """Return stealth-related flags and tunables from the stealth module.

    Categories returned:
    - environment: environment variable toggles for stealth features
    - human_profile_fields: fields on HumanProfile that shape behavior
    - behavioral_state_tunables: adaptation params on AgentBehavioralState
    - manager_toggles: simple runtime toggles exposed on StealthManager
    - engine_seed_controls: seed-related controls supported by engines
    """
    result: Dict[str, List[str]] = {
        'environment': [
            'STEALTH_ENTROPY',
            'STEALTH_RUN_SEED',
            'STEALTH_BEHAVIORAL_PLANNING',
            'STEALTH_ERROR_SIMULATION',
            'STEALTH_PAGE_EXPLORATION',
        ],
        'human_profile_fields': [],
        'behavioral_state_tunables': [
            'max_history_length',
            'confidence_adaptation_rate',
            'stress_decay_rate',
        ],
        'manager_toggles': [
            'entropy_enabled',
        ],
        'engine_seed_controls': [
            'run_seed',
            'action_seed',
        ],
    }

    # Try to obtain HumanProfile fields programmatically
    try:
        from .stealth import HumanProfile  # type: ignore
        result['human_profile_fields'] = [f.name for f in dc_fields(HumanProfile)]  # type: ignore[arg-type]
    except Exception:
        # Fallback: hard-coded common fields
        result['human_profile_fields'] = [
            'typing_speed_wpm',
            'reaction_time_ms',
            'motor_precision',
            'impulsivity',
            'tech_savviness',
            'deliberation_tendency',
            'multitasking_ability',
            'error_proneness',
            'movement_smoothness',
            'overshoot_tendency',
            'correction_speed',
        ]

    return result


def summarize_flags() -> FlagsSummary:
    """Return a combined summary of profile, session, and Chrome args."""
    return FlagsSummary(
        profile_fields=list_profile_flags(),
        session_fields=list_session_flags(),
    chrome_arg_groups=list_chrome_arg_groups(),
    stealth=list_stealth_flags(),
    )
