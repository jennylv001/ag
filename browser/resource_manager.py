"""
Browser Resource Management Enhancement
Addresses: Excessive tab creation, Resource waste, Poor browser hygiene
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import time
from urllib.parse import urlparse

class TabPurpose(Enum):
    MAIN_TASK = "main_task"
    RESEARCH = "research"
    SERVICE_REGISTRATION = "service_registration"
    SEARCH = "search"
    TEMPORARY = "temporary"

@dataclass
class TabContext:
    """Track tab purpose and usage for intelligent management."""
    tab_id: str
    url: str
    purpose: TabPurpose
    created_at: float
    last_accessed: float
    service_name: Optional[str] = None
    is_active: bool = False
    attempt_count: int = 0

class BrowserResourceManager:
    """Manages browser tabs intelligently to prevent resource waste."""

    def __init__(self, max_tabs: int = 8, max_research_tabs: int = 3):
        self.max_tabs = max_tabs
        self.max_research_tabs = max_research_tabs
        self.tab_contexts: Dict[str, TabContext] = {}
        self.service_tabs: Dict[str, str] = {}  # service_name -> tab_id
        self.research_tabs: List[str] = []

    def should_reuse_tab(self, url: str, purpose: TabPurpose) -> Optional[str]:
        """Determine if an existing tab should be reused instead of creating new one."""

        # Always reuse tabs for the same service
        if purpose == TabPurpose.SERVICE_REGISTRATION:
            service_name = self._extract_service_name(url)
            if service_name in self.service_tabs:
                return self.service_tabs[service_name]

        # Reuse search tabs for similar searches
        if purpose == TabPurpose.SEARCH:
            for tab_id, context in self.tab_contexts.items():
                if (context.purpose == TabPurpose.SEARCH and
                    self._is_similar_search(url, context.url)):
                    return tab_id

        # Reuse research tabs if we have too many
        if purpose == TabPurpose.RESEARCH and len(self.research_tabs) >= self.max_research_tabs:
            # Reuse oldest research tab
            oldest_tab = min(self.research_tabs,
                           key=lambda tid: self.tab_contexts[tid].last_accessed)
            return oldest_tab

        return None

    def register_tab(self, tab_id: str, url: str, purpose: TabPurpose) -> None:
        """Register a new or reused tab."""
        current_time = time.time()

        if tab_id in self.tab_contexts:
            # Update existing tab
            context = self.tab_contexts[tab_id]
            context.url = url
            context.last_accessed = current_time
            context.attempt_count += 1
        else:
            # Create new tab context
            context = TabContext(
                tab_id=tab_id,
                url=url,
                purpose=purpose,
                created_at=current_time,
                last_accessed=current_time
            )
            self.tab_contexts[tab_id] = context

        # Update service mapping
        if purpose == TabPurpose.SERVICE_REGISTRATION:
            service_name = self._extract_service_name(url)
            if service_name:
                self.service_tabs[service_name] = tab_id
                context.service_name = service_name

        # Update research tabs list
        if purpose == TabPurpose.RESEARCH:
            if tab_id not in self.research_tabs:
                self.research_tabs.append(tab_id)

    def get_tabs_to_close(self) -> List[str]:
        """Get list of tabs that should be closed to free resources."""
        tabs_to_close = []
        current_time = time.time()

        # Close tabs older than 10 minutes that haven't been accessed recently
        for tab_id, context in self.tab_contexts.items():
            age = current_time - context.created_at
            idle_time = current_time - context.last_accessed

            # Close old temporary tabs
            if (context.purpose == TabPurpose.TEMPORARY and
                age > 300):  # 5 minutes
                tabs_to_close.append(tab_id)

            # Close idle research tabs
            elif (context.purpose == TabPurpose.RESEARCH and
                  idle_time > 600 and  # 10 minutes idle
                  len(self.research_tabs) > 2):
                tabs_to_close.append(tab_id)

            # Close failed service registration tabs
            elif (context.purpose == TabPurpose.SERVICE_REGISTRATION and
                  context.attempt_count >= 3 and
                  idle_time > 300):  # 5 minutes after 3 failed attempts
                tabs_to_close.append(tab_id)

        # If we're still over limit, close oldest research tabs
        total_tabs = len(self.tab_contexts)
        if total_tabs > self.max_tabs:
            research_tabs_by_age = sorted(
                [tid for tid in self.research_tabs if tid not in tabs_to_close],
                key=lambda tid: self.tab_contexts[tid].last_accessed
            )
            needed_closures = total_tabs - self.max_tabs
            tabs_to_close.extend(research_tabs_by_age[:needed_closures])

        return tabs_to_close

    def cleanup_closed_tab(self, tab_id: str) -> None:
        """Clean up context for a closed tab."""
        if tab_id in self.tab_contexts:
            context = self.tab_contexts[tab_id]

            # Remove from service mapping
            if context.service_name and context.service_name in self.service_tabs:
                del self.service_tabs[context.service_name]

            # Remove from research tabs
            if tab_id in self.research_tabs:
                self.research_tabs.remove(tab_id)

            # Remove context
            del self.tab_contexts[tab_id]

    def _extract_service_name(self, url: str) -> Optional[str]:
        """Extract service name from URL."""
        try:
            domain = urlparse(url).netloc.lower()

            # Map common email services
            service_mapping = {
                'proton.me': 'protonmail',
                'mail.proton.me': 'protonmail',
                'tuta.com': 'tuta',
                'mail.tuta.com': 'tuta',
                'atomicmail.io': 'atomicmail',
                'gmail.com': 'gmail',
                'accounts.google.com': 'gmail',
                'outlook.com': 'outlook',
                'yahoo.com': 'yahoo'
            }

            for domain_pattern, service in service_mapping.items():
                if domain_pattern in domain:
                    return service

            return None
        except:
            return None

    def _is_similar_search(self, url1: str, url2: str) -> bool:
        """Check if two URLs are similar search queries."""
        try:
            # Both should be Google search URLs
            if 'google.com/search' not in url1 or 'google.com/search' not in url2:
                return False

            # Extract search terms
            from urllib.parse import parse_qs, urlparse

            query1 = parse_qs(urlparse(url1).query).get('q', [''])[0].lower()
            query2 = parse_qs(urlparse(url2).query).get('q', [''])[0].lower()

            # Check for common terms
            terms1 = set(query1.split())
            terms2 = set(query2.split())

            # Consider similar if 60%+ terms overlap
            if len(terms1) == 0 or len(terms2) == 0:
                return False

            overlap = len(terms1.intersection(terms2))
            min_terms = min(len(terms1), len(terms2))

            return overlap / min_terms >= 0.6

        except:
            return False

    def get_resource_guidance(self) -> str:
        """Get guidance about browser resource usage."""
        total_tabs = len(self.tab_contexts)
        research_count = len(self.research_tabs)
        service_count = len(self.service_tabs)

        guidance = f"**BROWSER RESOURCE STATUS:**\\n"
        guidance += f"- Total tabs: {total_tabs}/{self.max_tabs}\\n"
        guidance += f"- Research tabs: {research_count}/{self.max_research_tabs}\\n"
        guidance += f"- Service tabs: {service_count}\\n"

        if total_tabs > self.max_tabs * 0.8:
            guidance += "⚠️ **HIGH TAB COUNT** - Consider closing unused tabs\\n"

        tabs_to_close = self.get_tabs_to_close()
        if tabs_to_close:
            guidance += f"- Recommend closing {len(tabs_to_close)} tabs\\n"

        # Service-specific guidance
        if self.service_tabs:
            guidance += "\\n**ACTIVE SERVICE TABS:**\\n"
            for service, tab_id in self.service_tabs.items():
                context = self.tab_contexts.get(tab_id)
                if context:
                    guidance += f"- {service}: {context.attempt_count} attempts\\n"

        return guidance

# Enhanced controller integration
class ResourceAwareController:
    """Enhanced controller that manages tabs intelligently."""

    def __init__(self):
        self.resource_manager = BrowserResourceManager()

    def should_use_new_tab(self, url: str, action_type: str) -> tuple[bool, Optional[str]]:
        """Determine if new tab should be used and which tab to reuse if not."""

        # Determine purpose
        if 'search' in action_type.lower() or 'google.com' in url:
            purpose = TabPurpose.SEARCH
        elif any(service in url.lower() for service in ['proton', 'tuta', 'atomic', 'gmail', 'outlook']):
            purpose = TabPurpose.SERVICE_REGISTRATION
        elif 'research' in action_type.lower():
            purpose = TabPurpose.RESEARCH
        else:
            purpose = TabPurpose.TEMPORARY

        # Check if we should reuse existing tab
        reuse_tab_id = self.resource_manager.should_reuse_tab(url, purpose)

        if reuse_tab_id:
            self.resource_manager.register_tab(reuse_tab_id, url, purpose)
            return False, reuse_tab_id

        # Check if we're at tab limit
        if len(self.resource_manager.tab_contexts) >= self.resource_manager.max_tabs:
            # Force reuse oldest research tab if possible
            if purpose in [TabPurpose.RESEARCH, TabPurpose.SEARCH]:
                if self.resource_manager.research_tabs:
                    oldest_tab = min(self.resource_manager.research_tabs,
                                   key=lambda tid: self.resource_manager.tab_contexts[tid].last_accessed)
                    self.resource_manager.register_tab(oldest_tab, url, purpose)
                    return False, oldest_tab

        # Create new tab
        return True, None

    def handle_tab_cleanup(self) -> List[str]:
        """Get tabs that should be closed for resource management."""
        return self.resource_manager.get_tabs_to_close()
