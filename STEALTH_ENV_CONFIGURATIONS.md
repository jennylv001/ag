# Browser Session Stealth Environment Configurations

## 🔧 **Complete Stealth .env Configuration Reference**

Based on analysis of `browser/session.py`, here are all the stealth environment variables available:

### 🎯 **Core Stealth Features**

```bash
# Basic Stealth Behaviors (defaults shown)
STEALTH_TYPE=true                    # Enable human-like typing patterns
STEALTH_SCROLL=true                  # Enable human-like scrolling physics
STEALTH_NAVIGATION=false             # Enable human-like URL navigation (EXPERIMENTAL)
STEALTH_COORD_CLICK=true             # Enable coordinate-based clicking with human movement

# Advanced Behavioral Planning
STEALTH_BEHAVIORAL_PLANNING=false   # Enable context-aware behavioral planning
STEALTH_PAGE_EXPLORATION=false      # Enable exploration sequences before actions
STEALTH_ERROR_SIMULATION=false      # Enable human-like error simulation and correction
```

### 🧠 **Behavioral Planning System**

```bash
# Context Collection & Planning
STEALTH_BEHAVIORAL_PLANNING=true    # Collect nearby element context for smarter decisions
STEALTH_PAGE_EXPLORATION=true       # Execute exploration sequences (hover, scan patterns)
STEALTH_ERROR_SIMULATION=true       # Simulate human errors (wrong clicks, corrections)
```

### 📊 **Current Default States**

| Variable | Default | Status | Impact |
|----------|---------|---------|--------|
| `STEALTH_TYPE` | `true` | ✅ Active | Human typing patterns |
| `STEALTH_SCROLL` | `true` | ✅ Active | Natural scrolling physics |
| `STEALTH_COORD_CLICK` | `true` | ✅ Active | Mouse movement simulation |
| `STEALTH_NAVIGATION` | `false` | ⚠️ Disabled | URL typing (experimental) |
| `STEALTH_BEHAVIORAL_PLANNING` | `false` | ⚠️ Disabled | Context-aware decisions |
| `STEALTH_PAGE_EXPLORATION` | `false` | ⚠️ Disabled | Pre-action exploration |
| `STEALTH_ERROR_SIMULATION` | `false` | ⚠️ Disabled | Human error patterns |

### 🎛️ **Recommended Production .env**

```bash
# Maximum Stealth Configuration
# Copy to .env file in project root

# Core stealth features (safe for production)
STEALTH_TYPE=true
STEALTH_SCROLL=true
STEALTH_COORD_CLICK=true

# Advanced features (use carefully - may impact performance)
STEALTH_BEHAVIORAL_PLANNING=true
STEALTH_PAGE_EXPLORATION=false      # Keep disabled unless needed
STEALTH_ERROR_SIMULATION=false      # Keep disabled unless testing human-like errors

# Experimental features (not recommended for production)
STEALTH_NAVIGATION=false             # May cause navigation issues
```

### 🔍 **Environment Variable Detection**

The system reads these variables at runtime in several locations:

1. **Stealth Manager Initialization** (`setup_playwright()`)
   - Logs current env var states for monitoring
   - Initializes stealth manager with configuration

2. **Action Execution** (typing, clicking, scrolling)
   - Checks environment variables on each action
   - Applies stealth behaviors conditionally

3. **Behavioral Planning** (context collection, exploration)
   - Dynamic feature toggling based on env vars
   - Real-time adaptation to configuration changes

### ⚡ **Performance Impact**

| Feature | CPU Impact | Latency | Detection Risk |
|---------|------------|---------|----------------|
| `STEALTH_TYPE=true` | Low | +50-200ms | Very Low |
| `STEALTH_SCROLL=true` | Low | +100-300ms | Very Low |
| `STEALTH_COORD_CLICK=true` | Medium | +200-500ms | Low |
| `STEALTH_BEHAVIORAL_PLANNING=true` | Medium | +500-1000ms | Very Low |
| `STEALTH_PAGE_EXPLORATION=true` | High | +1-3s | Very Low |
| `STEALTH_ERROR_SIMULATION=true` | Low | +200-800ms | Very Low |

### 🛡️ **Security Considerations**

- **Environment variables are read at runtime** - changes require restart
- **No .env file found** - configuration via system environment only
- **Stealth counters track usage** - metrics available for monitoring
- **Fallback behavior** - graceful degradation if stealth features fail

### 🚀 **Quick Setup**

```bash
# Windows PowerShell
$env:STEALTH_TYPE="true"
$env:STEALTH_BEHAVIORAL_PLANNING="true"

# Linux/Mac
export STEALTH_TYPE=true
export STEALTH_BEHAVIORAL_PLANNING=true

# Or create .env file in project root
echo "STEALTH_TYPE=true" > .env
echo "STEALTH_BEHAVIORAL_PLANNING=true" >> .env
```

### 📈 **Monitoring & Observability**

The session tracks stealth usage with counters:
- `stealth.type.used` / `stealth.type.fallback`
- `stealth.click.used` / `stealth.click.fallback`
- `stealth.behavioral_planning.used`
- `stealth.exploration.sequences_executed`
- `stealth.error_simulation.*_triggered`

View session summary in logs when browser session ends.
