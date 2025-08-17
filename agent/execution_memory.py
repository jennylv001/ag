"""
Enhanced Memory System with Execution Tracking
Addresses: Incomplete follow-through, Limited error recovery
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ServiceAttempt:
    """Track detailed attempts on specific services."""
    service_name: str
    url: str
    step_number: int
    actions_tried: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, failed, success, abandoned
    notes: str = ""

@dataclass
class StrategyContext:
    """Track high-level strategy progression."""
    strategy_name: str
    started_step: int
    actions_taken: int = 0
    services_tried: List[str] = field(default_factory=list)
    outcome: Optional[str] = None
    next_steps: List[str] = field(default_factory=list)

class ExecutionMemory:
    """Enhanced memory system focused on execution tracking."""

    def __init__(self):
        self.service_attempts: Dict[str, ServiceAttempt] = {}
        self.strategies: List[StrategyContext] = []
        self.current_strategy: Optional[str] = None
        self.global_progress: Dict[str, Any] = {}

    def start_service_attempt(self, service_name: str, url: str, step_number: int):
        """Begin tracking a new service attempt."""
        self.service_attempts[service_name] = ServiceAttempt(
            service_name=service_name,
            url=url,
            step_number=step_number
        )

    def add_service_action(self, service_name: str, action_description: str, success: Optional[bool] = None):
        """Record an action taken on a service."""
        if service_name in self.service_attempts:
            attempt = self.service_attempts[service_name]
            attempt.actions_tried.append(action_description)

            if success is False:
                attempt.failures.append(action_description)
                # Auto-suggest alternatives
                if len(attempt.failures) >= 2 and attempt.status == "pending":
                    attempt.status = "needs_alternative_approach"
                    attempt.notes = f"Multiple failures: {', '.join(attempt.failures)}. Try different method."

    def mark_service_status(self, service_name: str, status: str, notes: str = ""):
        """Update service attempt status."""
        if service_name in self.service_attempts:
            self.service_attempts[service_name].status = status
            self.service_attempts[service_name].notes = notes

    def start_strategy(self, strategy_name: str, step_number: int):
        """Begin tracking a new strategy."""
        strategy = StrategyContext(strategy_name=strategy_name, started_step=step_number)
        self.strategies.append(strategy)
        self.current_strategy = strategy_name

    def get_service_summary(self) -> str:
        """Get formatted summary of service attempts."""
        if not self.service_attempts:
            return "No services attempted yet."

        summary = "**SERVICE ATTEMPT SUMMARY:**\\n"
        for name, attempt in self.service_attempts.items():
            summary += f"- **{name}** ({attempt.status}): "
            summary += f"{len(attempt.actions_tried)} actions, {len(attempt.failures)} failures\\n"
            if attempt.notes:
                summary += f"  Notes: {attempt.notes}\\n"

        return summary

    def get_next_actions(self) -> List[str]:
        """Get recommended next actions based on history."""
        recommendations = []

        # Check for services needing alternative approaches
        for name, attempt in self.service_attempts.items():
            if attempt.status == "needs_alternative_approach":
                recommendations.append(f"Try alternative registration method on {name}")
            elif attempt.status == "pending" and len(attempt.actions_tried) == 0:
                recommendations.append(f"Complete registration attempt on {name}")

        # Suggest new services if current ones are failing
        attempted_services = set(self.service_attempts.keys())
        known_services = {"protonmail", "gmail", "atomicmail", "tuta", "outlook", "yahoo"}
        untried_services = known_services - attempted_services

        if len([a for a in self.service_attempts.values() if a.status == "failed"]) >= 2:
            if untried_services:
                recommendations.append(f"Try new email service: {list(untried_services)[0]}")

        return recommendations

    def should_abandon_current_service(self, service_name: str) -> bool:
        """Determine if current service should be abandoned."""
        if service_name not in self.service_attempts:
            return False

        attempt = self.service_attempts[service_name]
        # Abandon if 3+ failures without trying alternatives
        return len(attempt.failures) >= 3 and attempt.status != "needs_alternative_approach"

    def get_memory_context(self) -> str:
        """Get formatted memory context for LLM prompt."""
        context = ""

        # Service summary
        context += self.get_service_summary()

        # Strategy progress
        if self.current_strategy:
            current_strat = next((s for s in self.strategies if s.strategy_name == self.current_strategy), None)
            if current_strat:
                context += f"\\n**CURRENT STRATEGY:** {current_strat.strategy_name}\\n"
                context += f"Actions taken: {current_strat.actions_taken}\\n"

        # Recommendations
        next_actions = self.get_next_actions()
        if next_actions:
            context += "\\n**RECOMMENDED ACTIONS:**\\n"
            for action in next_actions[:3]:  # Limit to top 3
                context += f"- {action}\\n"

        return context

# Integration with existing system
class EnhancedMemoryBudgetEnforcer:
    """Enhanced memory budget enforcer with execution context."""

    def __init__(self, original_enforcer):
        self.original_enforcer = original_enforcer
        self.execution_memory = ExecutionMemory()

    def add_execution_context(self, formatted_history: str) -> str:
        """Add execution context to formatted history."""
        execution_context = self.execution_memory.get_memory_context()
        if execution_context:
            return f"{formatted_history}\\n\\n{execution_context}"
        return formatted_history
