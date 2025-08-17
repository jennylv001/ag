"""
Concurrency utilities for the browser_use agent.
Provides bulletproof lock wrapper and other concurrency primitives.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class LockTimeoutError(Exception):
    """Raised when a lock cannot be acquired within the timeout period."""
    pass


class bulletproof_lock:
    """
    A bulletproof asyncio.Lock wrapper with timeout and comprehensive error handling.

    Usage:
        async with bulletproof_lock(lock, timeout=3.0):
            # critical section
            pass
    """

    def __init__(self, lock: asyncio.Lock, timeout: float = 3.0):
        self.lock = lock
        self.timeout = timeout
        self._acquired = False

    async def __aenter__(self):
        try:
            # Try to acquire the lock with timeout
            await asyncio.wait_for(self.lock.acquire(), timeout=self.timeout)
            self._acquired = True
            return self

        except asyncio.TimeoutError:
            import traceback
            caller_info = traceback.format_stack()[-3].strip()  # Get calling location
            error_msg = f"Failed to acquire lock within {self.timeout}s timeout"
            logger.error(f"❌ LOCK TIMEOUT: {caller_info} - {error_msg}")
            raise LockTimeoutError(error_msg) from None
        except Exception as e:
            error_msg = f"Unexpected error acquiring lock: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            try:
                self.lock.release()
            except Exception as e:
                logger.error(f"Error releasing lock: {e}", exc_info=True)
                # Don't raise here as it would mask the original exception
            finally:
                self._acquired = False

        # Don't suppress any exceptions from the critical section
        return False


class ResourceSemaphore:
    """
    A wrapper around asyncio.Semaphore with additional tracking and logging.
    """

    def __init__(self, max_concurrent: int, name: str = "resource"):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.name = name

    async def __aenter__(self):
        await self.semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()
        return False

    def locked(self) -> bool:
        """Check if the semaphore is currently at its limit."""
        return self.semaphore.locked()

    @property
    def available_permits(self) -> int:
        """Get the number of available permits."""
        return self.semaphore._value if hasattr(self.semaphore, '_value') else 0


# Global semaphore for I/O operations - will be initialized by Supervisor
_io_semaphore: Optional[ResourceSemaphore] = None


def set_global_io_semaphore(max_io: int):
    """Initialize the global I/O semaphore. Called by Supervisor during setup."""
    global _io_semaphore
    _io_semaphore = ResourceSemaphore(max_io, "global_io")
    logger.info(f"Global I/O semaphore initialized with {max_io} permits")


def get_global_io_semaphore() -> ResourceSemaphore:
    """Get the global I/O semaphore. Raises if not initialized."""
    if _io_semaphore is None:
        raise RuntimeError("Global I/O semaphore not initialized. Call set_global_io_semaphore first.")
    return _io_semaphore


async def with_io_semaphore():
    """Context manager for I/O operations using the global semaphore."""
    semaphore = get_global_io_semaphore()
    return semaphore


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
