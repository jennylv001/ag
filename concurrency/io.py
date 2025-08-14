"""
Concurrency utilities for I/O management.

This module provides async context managers and utilities for managing
concurrent I/O operations with semaphore-based flow control.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional

# Global I/O semaphore for protecting concurrent operations
_io_semaphore: Optional[asyncio.Semaphore] = None


def _get_default_semaphore_count() -> int:
    """
    Get the default semaphore count based on system resources.
    
    Returns:
        Conservative semaphore count for I/O operations
    """
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 8  # Conservative fallback
    return max(1, cpu_count)


def _ensure_semaphore() -> asyncio.Semaphore:
    """
    Ensure the global I/O semaphore is initialized.
    
    Returns:
        The global I/O semaphore
    """
    global _io_semaphore
    if _io_semaphore is None:
        count = _get_default_semaphore_count()
        _io_semaphore = asyncio.Semaphore(count)
    return _io_semaphore


def set_io_semaphore_count(count: int) -> None:
    """
    Set the I/O semaphore count for tuning concurrency.
    
    Args:
        count: Number of concurrent I/O operations to allow
        
    Raises:
        ValueError: If count is less than 1
    """
    if count < 1:
        raise ValueError("Semaphore count must be at least 1")
        
    global _io_semaphore
    _io_semaphore = asyncio.Semaphore(count)


@asynccontextmanager
async def io_semaphore():
    """
    Async context manager for I/O semaphore protection.
    
    This context manager acquires the global I/O semaphore before entering
    and releases it when exiting, providing flow control for concurrent
    I/O operations.
    
    Usage:
        async with io_semaphore():
            # Perform I/O operation here
            result = await some_io_operation()
    """
    semaphore = _ensure_semaphore()
    async with semaphore:
        yield


async def with_io_semaphore() -> asyncio.Semaphore:
    """
    Get the I/O semaphore for backward compatibility.
    
    This function returns the global I/O semaphore for cases where
    direct semaphore access is needed for more complex scenarios.
    
    Returns:
        The global I/O semaphore
        
    Usage:
        semaphore = await with_io_semaphore()
        async with semaphore:
            # Perform I/O operation
            pass
    """
    return _ensure_semaphore()


def get_io_semaphore_stats() -> dict:
    """
    Get statistics about the I/O semaphore usage.
    
    Returns:
        Dictionary with semaphore statistics
    """
    global _io_semaphore
    if _io_semaphore is None:
        return {
            'initialized': False,
            'count': None,
            'available': None,
            'waiting': None
        }
    
    # Handle both internal semaphore structures
    waiters = getattr(_io_semaphore, '_waiters', None) or []
    
    return {
        'initialized': True,
        'count': _io_semaphore._value + len(waiters),  # Total count
        'available': _io_semaphore._value,  # Available permits
        'waiting': len(waiters)  # Waiting tasks
    }