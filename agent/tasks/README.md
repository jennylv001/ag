# agent/tasks

Minimal task system for modular behaviors.

- BaseTask: Protocol with step()/is_done()/succeeded()
- TaskResult: lightweight outcome container
- LowLevelExecutionTask: wraps one orchestrator iteration
- SolveCaptchaTask: LLM-guided CAPTCHA solver (via tool shim)

Notes:
- Keep tasks small and idempotent per step.
- Prefer constructor injection from the orchestrator.
- Registry mapping in `agent/tasks/__init__.py` enables planner lookup.
