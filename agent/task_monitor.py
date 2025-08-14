"""
Task Monitor for robust restart capability of long-lived agent tasks.

This module provides a TaskMonitor that manages component tasks and restarts them
when they fail or get cancelled unexpectedly, with exponential backoff and 
proper observability.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Callable, Set, Iterable, Optional, Any

logger = logging.getLogger(__name__)


class TaskMonitor:
    """
    Monitors and restarts component tasks with exponential backoff.
    
    This class manages a mapping of component names to asyncio.Task instances,
    automatically restarting failed tasks using provided factory functions.
    """

    def __init__(self, max_restart_attempts: int = 5, base_backoff: float = 2.0, max_backoff: float = 60.0):
        """
        Initialize the TaskMonitor.
        
        Args:
            max_restart_attempts: Maximum restart attempts per component per hour
            base_backoff: Base backoff time in seconds for exponential backoff
            max_backoff: Maximum backoff time in seconds
        """
        self._tasks: Dict[str, asyncio.Task] = {}
        self._restart_factories: Dict[str, Callable[[], asyncio.Task]] = {}
        self._enabled_components: Set[str] = set()
        self._restart_counts: Dict[str, int] = {}
        self._last_restart_times: Dict[str, float] = {}
        self._is_restarting: Dict[str, bool] = {}
        
        self._max_restart_attempts = max_restart_attempts
        self._base_backoff = base_backoff
        self._max_backoff = max_backoff
        self._restart_window = 3600.0  # 1 hour in seconds
        self._grace_period = 3.0  # Grace period after restart to suppress immediate re-restarts
        
        self._closed = False

    def register(self, name: str, task: asyncio.Task) -> None:
        """
        Register a task for monitoring.
        
        Args:
            name: Component name for the task
            task: The asyncio.Task to monitor
        """
        if self._closed:
            logger.warning(f"TaskMonitor is closed, ignoring registration of {name}")
            return
            
        if name in self._tasks:
            logger.warning(f"Component {name} already registered, replacing existing task")
            old_task = self._tasks[name]
            if not old_task.done():
                old_task.cancel()
        
        self._tasks[name] = task
        task.add_done_callback(lambda t: self._on_task_done(name, t))
        
        # Initialize counters if not present
        if name not in self._restart_counts:
            self._restart_counts[name] = 0
        if name not in self._is_restarting:
            self._is_restarting[name] = False
            
        logger.info(f"Registered task for component: {name}")

    def set_restart_factories(self, factories: Dict[str, Callable[[], asyncio.Task]]) -> None:
        """
        Set the restart factory functions for components.
        
        Args:
            factories: Mapping of component names to factory functions
        """
        self._restart_factories.update(factories)
        logger.info(f"Set restart factories for components: {list(factories.keys())}")

    def enable(self, component_names: Iterable[str]) -> None:
        """
        Enable restart monitoring for specified components.
        
        Args:
            component_names: Iterable of component names to enable for restart
        """
        new_components = set(component_names) - self._enabled_components
        self._enabled_components.update(component_names)
        
        if new_components:
            logger.info(f"Enabled restart monitoring for components: {list(new_components)}")

    async def close(self) -> None:
        """
        Stop monitoring and cancel all tracked tasks.
        """
        if self._closed:
            return
            
        self._closed = True
        logger.info("Closing TaskMonitor, cancelling all tasks")
        
        # Cancel all tasks
        for name, task in self._tasks.items():
            if not task.done():
                logger.debug(f"Cancelling task for component: {name}")
                task.cancel()
        
        # Wait for all tasks to complete with timeout
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks.values(), return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not complete within timeout during shutdown")
        
        # Clear state
        self._tasks.clear()
        self._enabled_components.clear()
        self._is_restarting.clear()

    def _on_task_done(self, component_name: str, task: asyncio.Task) -> None:
        """
        Callback invoked when a monitored task completes.
        
        Args:
            component_name: Name of the component whose task completed
            task: The completed task
        """
        if self._closed:
            return
            
        try:
            # Check if task completed normally or with exception
            if task.cancelled():
                logger.debug(f"Task for component {component_name} was cancelled")
                if component_name in self._enabled_components and not self._is_restarting.get(component_name, False):
                    asyncio.create_task(self._restart_component(component_name, "Task was cancelled"))
            elif task.exception():
                exception = task.exception()
                logger.error(f"Task for component {component_name} failed with exception: {exception}")
                if component_name in self._enabled_components and not self._is_restarting.get(component_name, False):
                    asyncio.create_task(self._restart_component(component_name, str(exception)))
            else:
                logger.info(f"Task for component {component_name} completed normally")
        except Exception as e:
            logger.error(f"Error in task completion callback for {component_name}: {e}")

    async def _restart_component(self, component_name: str, reason: str) -> None:
        """
        Restart a component with exponential backoff.
        
        Args:
            component_name: Name of the component to restart
            reason: Reason for the restart
        """
        if self._closed or self._is_restarting.get(component_name, False):
            return
            
        self._is_restarting[component_name] = True
        
        try:
            # Check restart rate limiting
            current_time = time.time()
            if self._should_rate_limit_restart(component_name, current_time):
                logger.error(
                    f"Component {component_name} has exceeded restart limit "
                    f"({self._max_restart_attempts} attempts per hour). Stopping auto-restart.",
                    extra={
                        'component': component_name,
                        'restart_count': self._restart_counts[component_name],
                        'reason': reason
                    }
                )
                return
            
            # Calculate backoff delay
            attempt = self._restart_counts[component_name]
            backoff_delay = min(self._base_backoff * (2 ** attempt), self._max_backoff)
            
            logger.info(
                f"Restarting component {component_name} (attempt {attempt + 1}) after {backoff_delay}s delay",
                extra={
                    'component': component_name,
                    'attempt': attempt + 1,
                    'backoff': backoff_delay,
                    'reason': reason
                }
            )
            
            # Wait for backoff period
            await asyncio.sleep(backoff_delay)
            
            # Check if we're still supposed to restart (not closed, still enabled)
            if self._closed or component_name not in self._enabled_components:
                return
                
            # Get factory function
            factory = self._restart_factories.get(component_name)
            if not factory:
                logger.error(f"No restart factory found for component {component_name}")
                return
            
            # Create new task
            try:
                new_task = factory()
                if not isinstance(new_task, asyncio.Task):
                    logger.error(f"Factory for {component_name} returned non-Task object: {type(new_task)}")
                    return
                    
                # Update tracking
                self._tasks[component_name] = new_task
                self._restart_counts[component_name] += 1
                self._last_restart_times[component_name] = current_time
                
                # Set up monitoring for new task
                new_task.add_done_callback(lambda t: self._on_task_done(component_name, t))
                
                logger.info(
                    f"Successfully restarted component {component_name}",
                    extra={
                        'component': component_name,
                        'new_task_id': id(new_task),
                        'restart_count': self._restart_counts[component_name]
                    }
                )
                
                # Grace period to prevent immediate re-restart
                await asyncio.sleep(self._grace_period)
                
            except Exception as e:
                logger.error(f"Failed to create new task for component {component_name}: {e}")
                
        except Exception as e:
            logger.error(f"Error during restart of component {component_name}: {e}")
        finally:
            self._is_restarting[component_name] = False

    def _should_rate_limit_restart(self, component_name: str, current_time: float) -> bool:
        """
        Check if component restarts should be rate limited.
        
        Args:
            component_name: Name of the component
            current_time: Current timestamp
            
        Returns:
            True if restart should be rate limited
        """
        # Clean up old restart times outside the window
        last_restart = self._last_restart_times.get(component_name, 0)
        if current_time - last_restart > self._restart_window:
            self._restart_counts[component_name] = 0
            
        return self._restart_counts[component_name] >= self._max_restart_attempts

    def get_component_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all monitored components.
        
        Returns:
            Dictionary mapping component names to their statistics
        """
        stats = {}
        for component_name in self._tasks:
            task = self._tasks[component_name]
            stats[component_name] = {
                'enabled': component_name in self._enabled_components,
                'restart_count': self._restart_counts.get(component_name, 0),
                'is_restarting': self._is_restarting.get(component_name, False),
                'task_done': task.done() if task else True,
                'task_cancelled': task.cancelled() if task and task.done() else False,
                'last_restart_time': self._last_restart_times.get(component_name),
            }
        return stats