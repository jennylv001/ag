## Flow Map Template

Purpose: Document logic flows, data transformations, modules, dependencies, and side effects for a change. Use this before implementation.

- Change title:
- Owner/date:
- Related issue/PR:

### 1) Scope and entry points
- Trigger(s): user action, API call, schedule, event bus, CLI, etc.
- Primary entry function(s)/module(s):

### 2) Modules and responsibilities
- Modules involved and their responsibilities (one line each):
  - module.function: purpose
  - ...

### 3) Data flow
- Inputs (shape, types, source):
- Transformations (where/how data changes):
- Outputs (shape, destination, consumers):

### 4) Control flow and interactions
- Call sequence (high level):
  1. Component A -> B -> C
  2. External I/O (files, network, DB):
- Concurrency/async considerations (locks, tasks, queues):

### 5) Dependencies and side effects
- External services/libraries:
- Filesystem/Env/Process impacts:
- Caching/stateful components:

### 6) Edge cases and failure modes
- Inputs: empty/null, large, malformed
- Timeouts/retries/backoff
- Permission/auth errors
- Partial failure handling and cleanup

### 7) Observed problem and hypothesized root cause
- Symptom(s):
- Evidence (code refs, logs, failing tests):
- Root cause hypothesis (specific line(s)/condition(s)):

### 8) Proposed solution plan (pre-implementation)
- Minimal, robust change(s):
- Alternatives considered:
- Risk assessment and mitigations:

### 9) Validation plan
- New/updated tests (name, coverage target):
- Manual or e2e checks (steps, expected results):
- Performance/security implications and checks:

---
Notes:
- Keep functions single-responsibility and add type hints.
- Avoid unsafe calls (eval/exec/unsafe serialization).
- Add guards for known edge cases to prevent regressions.
