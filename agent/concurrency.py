"""
Concurrency utilities for the browser_use agent.
Provides lightweight primitives for I/O and actuation coordination.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Type
from types import TracebackType
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

from ..exceptions import LockTimeoutError


## NOTE: bulletproof_lock removed — it had no external usages. Prefer direct asyncio primitives.


# ResourceSemaphore removed - use asyncio.Semaphore directly

# Global semaphore for I/O operations
_io_semaphore: Optional[asyncio.Semaphore] = None
_actuation_semaphore: Optional[asyncio.Semaphore] = None


def set_global_io_semaphore(max_io: int):
    """Initialize the global I/O semaphore."""
    global _io_semaphore
    _io_semaphore = asyncio.Semaphore(max_io)
    logger.info(f"Global I/O semaphore initialized with {max_io} permits")


def set_single_actuation_semaphore():
    """Initialize a single-permit semaphore for actuation."""
    global _actuation_semaphore
    if _actuation_semaphore is None:
        _actuation_semaphore = asyncio.Semaphore(1)
        logger.info("Single actuation semaphore initialized (1 permit)")


def get_global_io_semaphore() -> asyncio.Semaphore:
    """Get the global I/O semaphore (direct access)."""
    if _io_semaphore is None:
        raise RuntimeError("Global I/O semaphore not initialized. Call set_global_io_semaphore first.")
    return _io_semaphore




@asynccontextmanager
async def io_semaphore():
    """Ergonomic async context manager for the global I/O semaphore.

    Usage:
        async with io_semaphore():
            # protected I/O
            ...
    """
    sem = get_global_io_semaphore()
    async with sem:
        yield


class LeaseToken:
    """Opaque token representing an actuation lease."""
    __slots__ = ("_ok",)
    def __init__(self) -> None:
        self._ok = True


class ActuationLease:
    """Acquire a single actuation lease ensuring only one component acts at a time.

    Usage:
        async with ActuationLease() as token:
            await actuator.execute(decision)  # pass token if required
    """

    def __init__(self, timeout: float = 5.0):
        self._timeout = timeout
        self._token: Optional[LeaseToken] = None

    async def __aenter__(self) -> LeaseToken:
        global _actuation_semaphore
        if _actuation_semaphore is None:
            set_single_actuation_semaphore()
        assert _actuation_semaphore is not None
        try:
            await asyncio.wait_for(_actuation_semaphore.acquire(), timeout=self._timeout)
        except asyncio.TimeoutError as e:
            raise LockTimeoutError("Failed to acquire ActuationLease within timeout") from e
        self._token = LeaseToken()
        return self._token

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool:
        global _actuation_semaphore
        if _actuation_semaphore is not None:
            try:
                _actuation_semaphore.release()
            except Exception:
                logger.debug("Actuation semaphore release error", exc_info=True)
        self._token = None
        return False
