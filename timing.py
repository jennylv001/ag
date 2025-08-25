"""
Centralized timing utilities for consistent, accurate time across the system.

- Uses monotonic/perf_counter for durations (not affected by system clock changes)
- Exposes process uptime and UTC timestamp helpers for logging and telemetry
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

_PROCESS_START_MONOTONIC = time.monotonic()
_PROCESS_START_WALL = time.time()


def monotonic_seconds() -> float:
    """Current monotonic time in seconds."""
    return time.monotonic()


def perf_seconds() -> float:
    """High-resolution performance counter in seconds."""
    return time.perf_counter()


def uptime_seconds() -> float:
    """Seconds since process start based on monotonic clock."""
    return time.monotonic() - _PROCESS_START_MONOTONIC


def now_utc_timestamp() -> float:
    """Current UNIX timestamp (UTC, wall clock)."""
    return time.time()


def now_utc_iso(ms: bool = True) -> str:
    """ISO-8601 UTC timestamp string suitable for logs (e.g., 2025-08-25T12:34:56.789Z)."""
    dt = datetime.now(timezone.utc)
    if ms:
        return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return dt.isoformat().replace("+00:00", "Z")


def process_start_utc_iso() -> str:
    """UTC ISO for process start time (approx; uses wall clock at import)."""
    return datetime.fromtimestamp(_PROCESS_START_WALL, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
