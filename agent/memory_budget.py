"""
Memory budget enforcement for agent history management.
"""
from __future__ import annotations

import logging
import sys
from typing import Deque, Any, TYPE_CHECKING
from collections import deque

if TYPE_CHECKING:
    from browser_use.agent.views import AgentHistory

logger = logging.getLogger(__name__)


class MemoryBudgetEnforcer:
    """
    Enforces memory budget constraints on agent history by pruning old items
    when the estimated memory usage exceeds configured limits.
    """
    
    def __init__(self, max_memory_mb: float = 100.0):
        """
        Initialize the memory budget enforcer.
        
        Args:
            max_memory_mb: Maximum memory budget in megabytes
        """
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)  # Convert MB to bytes
        self.max_memory_mb = max_memory_mb
        
    def estimate_object_size(self, obj: Any) -> int:
        """
        Estimate the memory size of an object in bytes.
        This is a heuristic estimation, not exact measurement.
        """
        try:
            # Get the basic size
            size = sys.getsizeof(obj)
            
            # For complex objects, recursively estimate size of contents
            if hasattr(obj, '__dict__'):
                # Add size of instance dictionary
                size += sys.getsizeof(obj.__dict__)
                # Add size of attribute values
                for value in obj.__dict__.values():
                    size += self.estimate_object_size(value)
                    
            elif isinstance(obj, (list, tuple)):
                # Add size of container elements
                for item in obj:
                    size += self.estimate_object_size(item)
                    
            elif isinstance(obj, dict):
                # Add size of dictionary contents
                for key, value in obj.items():
                    size += self.estimate_object_size(key)
                    size += self.estimate_object_size(value)
                    
            elif isinstance(obj, str):
                # String size is already handled by sys.getsizeof
                pass
                
            return size
            
        except Exception as e:
            # Fallback to basic size if estimation fails
            logger.debug(f"Failed to estimate object size: {e}")
            return sys.getsizeof(obj)
    
    def estimate_history_size(self, history_deque: Deque[AgentHistory]) -> int:
        """
        Estimate the total memory size of the history deque in bytes.
        """
        total_size = sys.getsizeof(history_deque)
        
        for history_item in history_deque:
            total_size += self.estimate_object_size(history_item)
            
        return total_size
    
    def enforce_budget(self, history_deque: Deque[AgentHistory]) -> int:
        """
        Enforce memory budget by pruning old history items if needed.
        
        Args:
            history_deque: The history deque to prune
            
        Returns:
            Number of items pruned
        """
        if not history_deque:
            return 0
            
        current_size = self.estimate_history_size(history_deque)
        
        if current_size <= self.max_memory_bytes:
            logger.debug(f"Memory usage within budget: {current_size / 1024 / 1024:.2f}MB / {self.max_memory_mb:.2f}MB")
            return 0
        
        # Memory budget exceeded, start pruning from the left (oldest items)
        pruned_count = 0
        
        logger.info(f"Memory budget exceeded: {current_size / 1024 / 1024:.2f}MB / {self.max_memory_mb:.2f}MB, pruning history...")
        
        while history_deque and current_size > self.max_memory_bytes:
            # Remove the oldest item
            removed_item = history_deque.popleft()
            pruned_count += 1
            
            # Recalculate size
            current_size = self.estimate_history_size(history_deque)
            
            logger.debug(f"Pruned history item {pruned_count}, new size: {current_size / 1024 / 1024:.2f}MB")
            
            # Safety check to prevent infinite loop
            if pruned_count > 1000:
                logger.error("Pruned too many items, breaking to prevent infinite loop")
                break
        
        if pruned_count > 0:
            final_size = self.estimate_history_size(history_deque)
            logger.info(f"Memory budget enforcement complete: pruned {pruned_count} items, "
                       f"final size: {final_size / 1024 / 1024:.2f}MB")
        
        return pruned_count
    
    def get_current_usage_mb(self, history_deque: Deque[AgentHistory]) -> float:
        """Get current memory usage in MB."""
        size_bytes = self.estimate_history_size(history_deque)
        return size_bytes / 1024 / 1024
    
    def get_budget_usage_percent(self, history_deque: Deque[AgentHistory]) -> float:
        """Get current memory usage as percentage of budget."""
        current_size = self.estimate_history_size(history_deque)
        return (current_size / self.max_memory_bytes) * 100 if self.max_memory_bytes > 0 else 0
