# Human Guidance Integration

This document explains how human guidance flows through the agent and how to use it at runtime.

## Flow
- Enqueue: `Agent.inject_human_guidance(text)` places text onto an internal queue (StateManager.human_guidance_queue).
- Pause loop: While the agent is `PAUSED`, Supervisor's pause handler polls the queue and, when present, forwards guidance to `MessageManager.add_human_guidance`, appending a system note.
- Prompt: MessageManager merges system notes into the final prompt context for the next LLM request.

## Runtime usage
- Pause the agent: `agent.pause()`
- Inject guidance: `agent.inject_human_guidance("Use the top navbar search")`
- Resume the agent: `agent.resume()`

Guidance is now visible to the LLM as a system note and used for the next decision.

## Observability
- Enqueue log: "Human guidance enqueued"
- Consume log: "Human guidance consumed during pause"

## Notes
- Guidance is processed only while paused to mimic a realistic supervision flow.
- Guidance is stored in `MessageManager.state.local_system_notes` and included in prompts by `update_history_representation`.
