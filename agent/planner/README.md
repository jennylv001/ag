# agent/planner

CDAD planner (Context-Driven Abstraction & Decomposition).

- Generates a markdown table plan from context.
- Parses to plan rows and maps to BaseTask instances via TASK_REGISTRY.
- Feature-flagged: controlled by `AgentSettings.use_task_planner` (default False).

Caveats:
- Parser is forgiving; rows without known tasks are ignored.
- Keep off by default to avoid behavior drift.
