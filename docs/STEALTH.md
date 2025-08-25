# Stealth Mode Documentation

## Overview

Browser-use's stealth mode provides enhanced automation capabilities designed to operate more naturally and avoid detection by anti-bot systems. This mode implements human-like interaction patterns and coordinate safety mechanisms.

## Quick Start

### Basic Configuration

```python
from browser_use import Agent, AgentSettings, BrowserProfile

# Configure stealth mode
browser_profile = BrowserProfile(
    stealth=True,                           # Enable stealth mode
    advanced_stealth=True,                  # Apply sensible behavioral defaults (optional)
    executable_path="/path/to/chrome",      # Point to Chrome executable
    user_data_dir="/path/to/stealth/profile",  # Dedicated profile directory
    headless=False,                         # Use headed mode for better stealth
    enable_default_extensions=True,        # Allow extensions for realistic browsing
)

# Create agent with stealth profile
settings = AgentSettings(
    task="Your automation task",
    llm=your_llm_instance,
    browser_profile=browser_profile
)
agent = Agent(settings)

# Run the agent
result = await agent.run()
```

### Chrome Executable Path Examples

**Windows:**
```python
executable_path = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
```

**macOS:**
```python
executable_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
```

**Linux:**
```python
executable_path = "/usr/bin/google-chrome"
# or
executable_path = "/opt/google/chrome/chrome"
```

## Stealth Features

### Human-like Interactions

When stealth mode is enabled, the following interactions become more human-like:

1. **Stealth Clicking**: Natural coordinate-based clicking with slight randomization
2. **Stealth Typing**: Character-by-character input with realistic timing delays
3. **Coordinate Safety**: Intelligent fallback when element coordinates become stale

### Observability & Monitoring

#### Stealth Activity Logs

Watch for these log indicators during stealth operations:

```
🤖 Stealth click on element <index> at coordinates (x, y)
⌨️ Stealth typing: "<text>" into element <index>
🎯 Using viewport coordinates for reliable clicking
📍 Falling back to element center due to stale coordinates
```

#### Stealth Metrics

Stealth mode tracks usage statistics via the browser session's internal counters:

```python
# Get current stealth usage counts from browser session
browser_session = agent.browser_session  # After agent setup
counters = browser_session._stealth_counters

print(f"Stealth clicks used: {counters['stealth.click.used']}")
print(f"Click fallbacks: {counters['stealth.click.fallback']}")
print(f"Stealth typing used: {counters['stealth.type.used']}")
print(f"Typing fallbacks: {counters['stealth.type.fallback']}")

# Additional robustness counters from Task 7
print(f"Bounding box recovery attempts: {counters['stealth.click.rebbox_attempts']}")
print(f"No bbox fallbacks: {counters['stealth.click.no_bbox_fallback']}")
```

#### ActionResult Integration

Stealth actions are logged in ActionResult.extracted_content:

```python
# After agent execution, check for stealth markers
for step in agent.history.history:
    for result in step.result:
        if result.extracted_content in ['stealth_click', 'stealth_typing']:
            print(f"Stealth action detected: {result.extracted_content}")
```

## Best Practices

### Profile Configuration

1. **Dedicated User Data Directory**: Always use a separate profile directory for stealth operations
   ```python
   user_data_dir = "/path/to/dedicated/stealth/profile"
   ```

2. **Avoid "Botty" Flags**: Keep these settings for maximum stealth:
   ```python
   BrowserProfile(
       stealth=True,
       headless=False,                 # Use headed mode for better stealth
       enable_default_extensions=True, # Allow extensions for realistic browsing
       disable_security=False,         # Keep security enabled
   )
   ```

3. **Chrome Executable**: Use full Chrome rather than Chromium for better compatibility
   ```python
   executable_path = "/path/to/google-chrome"  # Not chromium
   ```

### Configuration model (flags only)

Stealth is governed by just two profile flags:

- stealth=True: enables core human-like interactions (typing, scrolling) and stealth launch hygiene.
- advanced_stealth=True: when paired with stealth, enables behavioral features (entropy, planning, exploration, error simulation, navigation).

No per-feature environment variables are required anymore. For deterministic runs you may still set:

```bash
set STEALTH_RUN_SEED=12345  # optional: make stealth engines deterministic
```

## Troubleshooting

### Profile Lock Issues

If you encounter profile lock errors:

1. **Ensure unique user_data_dir**: Each browser instance needs its own directory
2. **Close existing browsers**: Make sure no other Chrome instances are using the profile
3. **Check permissions**: Verify write access to the user_data_dir
4. **Clean restart**: Delete the profile directory if it becomes corrupted

```python
import shutil
from pathlib import Path

# Clean profile directory if needed
profile_dir = Path("/path/to/stealth/profile")
if profile_dir.exists():
    shutil.rmtree(profile_dir)
```

### Coordinate Safety Issues

If clicking fails with coordinate errors:

1. **Check viewport_coordinates preference**: Stealth mode automatically prefers viewport coordinates
2. **Verify element visibility**: Ensure target elements are in the viewport
3. **Monitor logs**: Look for coordinate fallback messages

### Common Issues

#### Chrome Not Found
```
Error: Executable doesn't exist at /path/to/chrome
```
**Solution**: Verify Chrome installation path and update `executable_path`

#### Profile Already in Use
```
Error: The profile appears to be in use by another browser
```
**Solution**: Use a unique `user_data_dir` or close other browser instances

#### Stealth Not Working
```
Warning: Stealth mode requested but not fully enabled
```
**Solution**: Ensure Chrome executable path is correct and stealth=True is set

## Advanced Configuration

### Custom Stealth Parameters

```python
# Fine-tune stealth behavior
browser_profile = BrowserProfile(
    stealth=True,
    executable_path="/path/to/chrome",
    user_data_dir="/path/to/profile",

    # Viewport settings for consistency
    viewport={'width': 1920, 'height': 1080},

    # Timing settings
    minimum_wait_page_load_time=2.0,
    default_navigation_timeout=10000,

    # Resource settings
    include_dynamic_attributes=True,
)
```

### Integration with Existing Code

Stealth mode is designed to be backward compatible:

```python
# Existing code continues to work
settings = AgentSettings(
    task="Navigate and extract data",
    llm=llm,
    browser_profile=BrowserProfile(stealth=True, executable_path="/path/to/chrome")
)
agent = Agent(settings)

# Stealth enhancements are automatic
result = await agent.run()
```

## Testing Stealth Mode

### Verification Script

```python
import asyncio
from browser_use import Agent, AgentSettings, BrowserProfile

async def test_stealth():
    # Configure stealth
    profile = BrowserProfile(
        stealth=True,
        executable_path="/path/to/chrome",
        user_data_dir="/tmp/stealth_test"
    )

    # Create agent
    settings = AgentSettings(
        task="Go to google.com and search for 'test'",
        llm=your_llm,
        browser_profile=profile
    )
    agent = Agent(settings)

    # Run the agent
    result = await agent.run()

    # Check stealth usage
    browser_session = agent.browser_session
    counters = browser_session._stealth_counters

    print(f"Stealth clicks used: {counters['stealth.click.used']}")
    print(f"Stealth typing used: {counters['stealth.type.used']}")

    # Verify stealth markers in history
    stealth_actions = 0
    for step in result.history:
        for action_result in step.result:
            if action_result.extracted_content in ['stealth_click', 'stealth_typing']:
                stealth_actions += 1

    print(f"Stealth actions in history: {stealth_actions}")
    return result

# Run test
result = asyncio.run(test_stealth())
```

## Migration Guide

### From Standard to Stealth Mode

1. **Update BrowserProfile**:
   ```python
   # Before
   profile = BrowserProfile()

   # After
   profile = BrowserProfile(
       stealth=True,
       executable_path="/path/to/chrome",
       user_data_dir="/path/to/stealth/profile"
   )
   ```

2. **No Code Changes Required**: Existing agent code works unchanged

3. **Monitor Logs**: Watch for stealth activity indicators

4. **Check Metrics**: Use stealth stats to verify functionality

## Support

For issues related to stealth mode:

1. Check the troubleshooting section above
2. Verify Chrome installation and paths
3. Ensure profile directory permissions
4. Review stealth logs for diagnostic information

For additional help, refer to the main browser-use documentation or submit an issue with:
- Chrome version and installation path
- Browser profile configuration
- Relevant log output
- Stealth statistics from get_stealth_stats()
