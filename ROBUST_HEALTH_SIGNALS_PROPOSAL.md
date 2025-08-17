# Robust Health Signal System Architecture

## Current Issues with Health Signals

The current health monitoring system causes false positives that unnecessarily interrupt operations:

1. **Hardcoded 3-second timeout** (`timeout_ms = min(3000, user_timeout_ms)`)
2. **Binary responsive/unresponsive** classification
3. **Single-point failure detection** (`page.evaluate('1')` test)
4. **No context awareness** (simple sites vs complex web apps)
5. **No progressive degradation** (immediate failure vs graceful degradation)

## Proposed Multi-Dimensional Health Assessment

### 1. **Page Health Levels** (Instead of Binary)

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
import time

class PageHealthLevel(Enum):
    FULLY_RESPONSIVE = "fully_responsive"     # <1s response, all features work
    FUNCTIONAL = "functional"                 # 1-3s response, core features work
    SLOW_BUT_USABLE = "slow_but_usable"      # 3-8s response, basic interaction possible
    DEGRADED = "degraded"                     # 8-15s response, limited functionality
    UNRESPONSIVE = "unresponsive"            # >15s or no response

@dataclass
class PageHealthMetrics:
    health_level: PageHealthLevel
    js_execution_time: Optional[float]        # Time for simple JS execution
    dom_interaction_time: Optional[float]     # Time for DOM queries
    network_activity: str                     # "idle", "moderate", "heavy"
    page_complexity: str                      # "simple", "medium", "complex"
    functional_capabilities: Dict[str, bool]  # click, type, scroll, extract
    last_assessment_time: float
    confidence_score: float                   # 0.0-1.0 reliability of assessment
```

### 2. **Context-Aware Timeout Configuration**

```python
@dataclass
class ContextualTimeoutConfig:
    base_timeout: float = 5.0

    # Domain-specific adjustments
    domain_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "google.com": 2.5,      # Search engines need more time
        "facebook.com": 2.0,    # Social media platforms
        "amazon.com": 1.8,      # E-commerce sites
        "github.com": 1.5,      # Developer platforms
        # Simple domains get default 1.0
    })

    # Page type adjustments
    page_type_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "search_results": 2.0,
        "e_commerce": 1.5,
        "social_media": 1.8,
        "static_content": 0.8,
    })

    # Network condition adjustments
    network_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "fast": 1.0,
        "moderate": 1.5,
        "slow": 2.5,
        "poor": 4.0,
    })

    def calculate_timeout(self, url: str, page_type: str = "unknown",
                         network_condition: str = "moderate") -> float:
        domain = urlparse(url).netloc.lower()

        timeout = self.base_timeout
        timeout *= self.domain_multipliers.get(domain, 1.0)
        timeout *= self.page_type_multipliers.get(page_type, 1.0)
        timeout *= self.network_multipliers.get(network_condition, 1.0)

        return min(timeout, 30.0)  # Cap at 30s maximum
```

### 3. **Progressive Health Assessment**

```python
class ProgressiveHealthAssessor:
    def __init__(self, page: Page):
        self.page = page

    async def assess_health(self, context: ContextualTimeoutConfig,
                          url: str) -> PageHealthMetrics:
        start_time = time.monotonic()

        # Level 1: Basic Responsiveness (1s timeout)
        js_time = await self._test_js_execution(timeout=1.0)
        if js_time and js_time < 1.0:
            return PageHealthMetrics(
                health_level=PageHealthLevel.FULLY_RESPONSIVE,
                js_execution_time=js_time,
                last_assessment_time=start_time,
                confidence_score=0.9
            )

        # Level 2: Functional Test (3s timeout)
        dom_time = await self._test_dom_interaction(timeout=3.0)
        capabilities = await self._test_capabilities(timeout=2.0)

        if js_time and js_time < 3.0 and capabilities.get("click", False):
            return PageHealthMetrics(
                health_level=PageHealthLevel.FUNCTIONAL,
                js_execution_time=js_time,
                dom_interaction_time=dom_time,
                functional_capabilities=capabilities,
                last_assessment_time=start_time,
                confidence_score=0.8
            )

        # Level 3: Extended Assessment (8s timeout)
        extended_timeout = context.calculate_timeout(url)
        js_time = await self._test_js_execution(timeout=extended_timeout)

        if js_time and js_time < 8.0:
            return PageHealthMetrics(
                health_level=PageHealthLevel.SLOW_BUT_USABLE,
                js_execution_time=js_time,
                functional_capabilities=capabilities,
                last_assessment_time=start_time,
                confidence_score=0.7
            )

        # Level 4: Degraded but might recover
        if js_time and js_time < 15.0:
            return PageHealthMetrics(
                health_level=PageHealthLevel.DEGRADED,
                js_execution_time=js_time,
                last_assessment_time=start_time,
                confidence_score=0.5
            )

        # Level 5: Truly unresponsive
        return PageHealthMetrics(
            health_level=PageHealthLevel.UNRESPONSIVE,
            last_assessment_time=start_time,
            confidence_score=0.9
        )

    async def _test_js_execution(self, timeout: float) -> Optional[float]:
        """Test basic JavaScript execution performance."""
        start = time.monotonic()
        try:
            eval_task = asyncio.create_task(self.page.evaluate('1 + 1'))
            done, pending = await asyncio.wait([eval_task], timeout=timeout)

            if eval_task in done:
                await eval_task  # Get result or exception
                return time.monotonic() - start
            else:
                eval_task.cancel()
                return None
        except Exception:
            return None

    async def _test_dom_interaction(self, timeout: float) -> Optional[float]:
        """Test DOM query performance."""
        start = time.monotonic()
        try:
            eval_task = asyncio.create_task(
                self.page.evaluate('document.body ? document.body.tagName : null')
            )
            done, pending = await asyncio.wait([eval_task], timeout=timeout)

            if eval_task in done:
                result = await eval_task
                if result:  # Successfully got DOM info
                    return time.monotonic() - start
            else:
                eval_task.cancel()
            return None
        except Exception:
            return None

    async def _test_capabilities(self, timeout: float) -> Dict[str, bool]:
        """Test specific functional capabilities."""
        capabilities = {
            "click": False,
            "type": False,
            "scroll": False,
            "extract": False
        }

        try:
            # Test if we can find clickable elements
            clickable_task = asyncio.create_task(
                self.page.evaluate('document.querySelectorAll("a, button, input").length > 0')
            )
            done, pending = await asyncio.wait([clickable_task], timeout=timeout/4)

            if clickable_task in done:
                capabilities["click"] = bool(await clickable_task)
            else:
                clickable_task.cancel()

            # Test if we can access input elements
            input_task = asyncio.create_task(
                self.page.evaluate('document.querySelectorAll("input, textarea").length > 0')
            )
            done, pending = await asyncio.wait([input_task], timeout=timeout/4)

            if input_task in done:
                capabilities["type"] = bool(await input_task)
            else:
                input_task.cancel()

            # Always assume scroll/extract work if basic JS works
            capabilities["scroll"] = True
            capabilities["extract"] = True

        except Exception:
            pass

        return capabilities
```

### 4. **Enhanced Health Signal Integration**

```python
class EnhancedHealthSignalManager:
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.timeout_config = ContextualTimeoutConfig()
        self.health_history: Dict[str, List[PageHealthMetrics]] = {}

    async def assess_page_health(self, page: Page, url: str,
                               operation_type: str = "general") -> PageHealthMetrics:
        """Assess page health with context awareness."""
        assessor = ProgressiveHealthAssessor(page)

        # Determine page type from URL patterns
        page_type = self._classify_page_type(url)

        # Get network condition estimate
        network_condition = await self._estimate_network_condition(url)

        # Perform assessment
        health_metrics = await assessor.assess_health(self.timeout_config, url)

        # Store in history for learning
        domain = urlparse(url).netloc
        if domain not in self.health_history:
            self.health_history[domain] = []
        self.health_history[domain].append(health_metrics)

        # Keep only recent history (last 10 assessments per domain)
        self.health_history[domain] = self.health_history[domain][-10:]

        # Emit appropriate health signals
        await self._emit_health_signals(health_metrics, operation_type)

        return health_metrics

    def _classify_page_type(self, url: str) -> str:
        """Classify page type for context-aware timeouts."""
        url_lower = url.lower()

        if "search" in url_lower or "google.com/search" in url_lower:
            return "search_results"
        elif any(x in url_lower for x in ["shop", "buy", "cart", "amazon", "ebay"]):
            return "e_commerce"
        elif any(x in url_lower for x in ["facebook", "twitter", "linkedin", "instagram"]):
            return "social_media"
        elif url_lower.endswith((".html", ".htm", ".txt", ".md")):
            return "static_content"
        else:
            return "unknown"

    async def _estimate_network_condition(self, url: str) -> str:
        """Estimate network condition based on domain history."""
        domain = urlparse(url).netloc

        if domain in self.health_history:
            recent_times = [
                h.js_execution_time for h in self.health_history[domain][-3:]
                if h.js_execution_time is not None
            ]

            if recent_times:
                avg_time = sum(recent_times) / len(recent_times)
                if avg_time < 0.5:
                    return "fast"
                elif avg_time < 1.5:
                    return "moderate"
                elif avg_time < 3.0:
                    return "slow"
                else:
                    return "poor"

        return "moderate"  # Default assumption

    async def _emit_health_signals(self, health_metrics: PageHealthMetrics,
                                 operation_type: str):
        """Emit appropriate health signals based on assessment."""

        # Only emit negative signals for truly problematic states
        if health_metrics.health_level == PageHealthLevel.UNRESPONSIVE:
            await self.state_manager.ingest_signal('io_timeout')
        elif health_metrics.health_level == PageHealthLevel.DEGRADED:
            # Only emit warning signal, don't trigger full timeout
            await self.state_manager.ingest_signal('performance_warning', {
                'level': 'degraded',
                'js_time': health_metrics.js_execution_time
            })
        else:
            # Emit positive signal for functional pages
            await self.state_manager.ingest_signal('io_ok')

    def should_continue_operation(self, health_metrics: PageHealthMetrics,
                                operation_type: str) -> bool:
        """Determine if operation should continue based on health level."""

        # Allow operations to continue even with degraded performance
        if operation_type in ["search", "extract", "navigate"]:
            return health_metrics.health_level != PageHealthLevel.UNRESPONSIVE

        # More strict requirements for interactive operations
        elif operation_type in ["click", "type", "upload"]:
            return health_metrics.health_level in [
                PageHealthLevel.FULLY_RESPONSIVE,
                PageHealthLevel.FUNCTIONAL,
                PageHealthLevel.SLOW_BUT_USABLE
            ]

        # Default: allow unless truly unresponsive
        return health_metrics.health_level != PageHealthLevel.UNRESPONSIVE
```

### 5. **Updated Browser Session Integration**

```python
# Update the browser session's _is_page_responsive method:

async def _is_page_responsive_enhanced(self, page: Page, timeout: float = 5.0,
                                     operation_type: str = "general") -> tuple[bool, PageHealthMetrics]:
    """Enhanced page responsiveness check with context awareness."""

    health_manager = EnhancedHealthSignalManager(self.state_manager)
    health_metrics = await health_manager.assess_page_health(page, page.url, operation_type)

    # Decision based on operation requirements
    should_continue = health_manager.should_continue_operation(health_metrics, operation_type)

    return should_continue, health_metrics

# Update the navigate method:
async def navigate_enhanced(self, url: str = 'about:blank', new_tab: bool = False,
                          timeout_ms: int | None = None, operation_type: str = "navigate") -> Page:
    """Enhanced navigation with robust health assessment."""

    # Calculate context-aware timeout
    timeout_config = ContextualTimeoutConfig()
    contextual_timeout_ms = int(timeout_config.calculate_timeout(url, "navigation") * 1000)

    # Use contextual timeout instead of hardcoded 3000ms
    if timeout_ms is not None:
        user_timeout_ms = int(timeout_ms)
    elif self.browser_profile.default_navigation_timeout is not None:
        user_timeout_ms = int(self.browser_profile.default_navigation_timeout)
    else:
        user_timeout_ms = contextual_timeout_ms

    # Remove the hardcoded min(3000, user_timeout_ms) limit!
    timeout_ms = min(contextual_timeout_ms, user_timeout_ms)

    # ... existing navigation logic ...

    # After navigation timeout, use enhanced health assessment
    if nav_task in pending:
        self.logger.warning(
            f"⚠️ Loading {_log_pretty_url(normalized_url)} didn't finish after {timeout_ms / 1000}s, assessing page health..."
        )

        # Enhanced health check instead of simple responsiveness test
        should_continue, health_metrics = await self._is_page_responsive_enhanced(
            page, timeout=5.0, operation_type=operation_type
        )

        if should_continue:
            self.logger.info(
                f"✅ Page is {health_metrics.health_level.value} and usable for {operation_type} operations"
            )
        else:
            self.logger.error(
                f"❌ Page is {health_metrics.health_level.value} and not suitable for {operation_type} operations"
            )
            raise RuntimeError(f'Page health level {health_metrics.health_level.value} insufficient for {operation_type}')
```

## Implementation Benefits

1. **Eliminates False Positives**: Google search pages won't be marked as "unresponsive" when they're functionally usable
2. **Context-Aware Decisions**: Different timeout expectations for different types of sites
3. **Progressive Degradation**: Operations can continue with slower pages rather than failing immediately
4. **Operation-Specific Requirements**: Search/extract can work with slower pages, but form filling needs responsive pages
5. **Learning System**: Builds domain-specific performance expectations over time
6. **Better Observability**: Rich health metrics instead of binary pass/fail

This approach transforms health signals from rigid binary gates into intelligent, context-aware operational guidance that supports the agent's actual needs rather than fighting against modern web application realities.
