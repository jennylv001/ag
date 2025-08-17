1. Core Identity

Purpose: This module provides a comprehensive framework for an autonomous web automation agent, handling everything from state management and browser perception to LLM-based decision-making and action execution through a resilient, event-driven architecture.

Key Patterns: Event Sourcing, Publish-Subscribe, State Management, Singleton (for global semaphores), Asynchronous Task Coordination.

2. Component & API Surface (YAML)

A machine-parsable block defining the module's structure.

```yaml
# Static analysis of module components and contracts
module: "/agent and /agent/message_manager"
dependencies:
  external:
    - "asyncio"
    - "logging"
    - "pydantic"
    - "anyio"
    - "itertools"
    - "typing"
    - "pathlib"
    - "uuid"
    - "PIL"
  internal:
    - "browser_use.agent.views"
    - "browser_use.agent.prompts"
    - "browser_use.agent.state_manager"
    - "browser_use.agent.events"
    - "browser_use.llm.messages"
    - "browser_use.utils"
    - "browser_use.browser.views"
    - "browser_use.filesystem.file_system"
    - "browser_use.agent.concurrency"
    - "browser_use.agent.memory_budget"
    - "browser_use.agent.supervisor"
    - "browser_use.agent.settings"
    - "browser_use.agent.actuator"
    - "browser_use.agent.decision_maker"
    - "browser_use.agent.llm_caller"
    - "browser_use.agent.reactor_vitals"
    - "browser_use.agent.perception"
    - "browser_use.agent.gif"
    - "browser_use.controller.service"
    - "browser_use.browser.session"
    - "browser_use.exceptions"
    - "browser_use.config"
public_api:

  - name: "Agent"
    type: "Class"
    properties:
      - "settings: AgentSettings"
      - "supervisor: Supervisor"
    methods:
      - "async def run(self) -> AgentHistoryList"
      - "def pause(self) -> None"
      - "def resume(self) -> None"
      - "def stop(self) -> None"
      - "def inject_human_guidance(self, text: str) -> None"
      - "async def add_new_task(self, new_task: str) -> None"
      - "async def close(self) -> None"
      - "async def load_and_rerun(self, history_file: str, **kwargs) -> list[ActionResult]"
      - "async def rerun_history(self, history: AgentHistoryList, max_retries: int = 3, skip_failures: bool = True, delay_between_actions: float = 2.0) -> list[ActionResult]"
    description: "Public-facing wrapper for the agent architecture, delegating all major operations to the Supervisor."

  - name: "Supervisor"
    type: "Class"
    properties:
      - "settings: AgentSettings"
      - "state_manager: StateManager"
      - "browser_session: BrowserSession"
      - "agent_bus: asyncio.Queue"
      - "perception: Perception"
      - "message_manager: MessageManager"
      - "decision_maker: DecisionMaker"
      - "llm_caller: LLMCaller"
      - "actuator: Actuator"
      - "reactor_vitals: ReactorVitals"
    methods:
      - "async def run(self) -> AgentHistoryList"
      - "def pause(self) -> None"
      - "def resume(self) -> None"
      - "def stop(self) -> None"
      - "async def close(self) -> None"
    description: "The central orchestrator that initializes and manages the lifecycle of all agent components and the main event loop."
data_contracts:

  - name: "AgentSettings"
    schema:
      task: "str"
      llm: "BaseChatModel"
      use_planner: "bool"
      max_steps: "int"
      max_failures: "int"
      lock_timeout_seconds: "float"
      memory_budget_mb: "float"
      max_concurrent_io: "int"
    description: "A Pydantic model defining the static configuration for an agent's run."

  - name: "AgentState"
    schema:
      agent_id: "str"
      task: "str"
      status: "AgentStatus (Enum)"
      load_status: "LoadStatus (Enum)"
      task_stack: "TaskStack"
      n_steps: "int"
      consecutive_failures: "int"
      history: "AgentHistoryList"
      human_guidance_queue: "asyncio.Queue[str]"
    description: "A Pydantic model representing the dynamic state of the agent, managed exclusively by the StateManager."

  - name: "Event"
    schema:
      step_token: "int"
      task_id: "str"
      timestamp: "float"
    description: "A base dataclass for all events, ensuring they carry essential context like the step number and task identifier."

  - name: "PerceptionSnapshot"
    schema:
      browser_state: "BrowserStateSummary"
      new_downloaded_files: "Optional[list[str]]"
    description: "Emitted by the Perception component, containing a snapshot of the browser and file system state."

  - name: "DecisionPlan"
    schema:
      messages_to_llm: "list[BaseMessage]"
      llm_output: "Optional[AgentOutput]"
      decision_type: "str"
    description: "Emitted by the DecisionMaker after a successful LLM call, containing the actions to be executed."

  - name: "LLMRequest"
    schema:
      messages: "list[BaseMessage]"
      output_schema: "Any"
      request_id: "str"
      request_type: "str"
    description: "A request placed on the event bus for the LLMCaller to process, decoupling decision logic from the actual API call."

  - name: "ActionExecuted"
    schema:
      action_results: "list[Any]"
      success: "bool"
    description: "Emitted by the Actuator upon completion of a set of actions, signaling the end of a sub-step."
```

3. Static Analysis & Architecture

The /agent module implements a decoupled, event-driven architecture for an autonomous web agent. The Supervisor class acts as the central orchestrator, initializing and managing the lifecycle of several distinct, asynchronous components that communicate primarily via an asyncio.Queue serving as an event bus (agent_bus). This pub/sub model decouples the major functional areas: Perception (observing the environment), DecisionMaker (generating LLM prompts), LLMCaller (executing LLM calls), and Actuator (performing actions in the browser).

State is centralized within the StateManager class, which holds an AgentState Pydantic model. All state modifications are serialized through an asynchronous lock (bulletproof_lock) to prevent race conditions. This makes StateManager the single source of truth. A key feature is the TaskStack within the state, which provides a formal mechanism for hierarchical task management, allowing the agent to suspend and resume objectives to handle interruptions like pop-ups or CAPTCHAs.

A secondary event bus (heartbeat_bus) and a dedicated ReactorVitals component implement a health monitoring protocol. Components periodically emit Heartbeat events; if a component goes silent, ReactorVitals can initiate recovery procedures or flag a critical failure, enhancing system resilience.

The MessageManager component is responsible for prompt engineering. It consumes the agent's history and current browser state to construct elaborate prompts for the DecisionMaker, guided by templates found in .md files and configured by the MessageManagerSettings model.

4. Data and Control Flow

The agent's operation follows a circular, event-driven flow, which begins when Supervisor.run() starts all component tasks.



Initiation: The main loop is kick-started by the Perception component, which emits an initial StepFinalized event.

Perception & Snapshot: The Perception task, upon receiving a StepFinalized event, captures the current state of the BrowserSession and file system. It then publishes this information as a PerceptionSnapshot event onto the agent_bus.

Decision & Prompting: The DecisionMaker consumes the PerceptionSnapshot. It collaborates with the MessageManager to build a detailed prompt based on the agent's history, current goal, and the new browser state. This process culminates in the emission of an LLMRequest event.

LLM Invocation: The LLMCaller is the sole component responsible for handling LLMRequest events. It performs the asynchronous call to the language model, including retry logic, and publishes the outcome as an LLMResponse event.

Planning & Action: The DecisionMaker receives the LLMResponse, validates it, and formulates an actionable plan, which it emits as a DecisionPlan event containing a list of actions.

Actuation: The Actuator component listens for DecisionPlan events. It uses the Controller service to execute the specified browser actions. The LongIOWatcher class within the Actuator monitors these potentially long-running I/O operations for timeouts. Upon completion, the Actuator publishes an ActionExecuted event (or an ErrorEvent if it fails).

Coordination & Loop Finalization: The Supervisor's _event_coordinator task listens for ActionExecuted and ErrorEvent. Receiving either signals the end of a step's active phase. The supervisor then calls _finalize_step, which records the outcome in the StateManager, increments the step counter, and publishes a new StepFinalized event, thereby re-initiating the cycle.

5. Integrity and Architectural Audit

Architectural Contradiction: The most significant issue is the presence of a dual execution model within the DecisionMaker and Actuator components. Both contain a primary event-driven run method that integrates with the Supervisor's event bus. However, they also contain obsolete public methods (decide in DecisionMaker and execute in Actuator) that belong to a legacy, direct-call architecture. This legacy code is not used by the Supervisor and appears to be dead code from a previous version, creating a point of confusion for future maintenance.

Dependency Cohesion: The module is highly cohesive, with Pydantic models and dataclasses (AgentState, AgentSettings, and all Event subtypes) serving as robust data contracts that ensure component interoperability. While there is tight coupling, particularly around the StateManager, this is a deliberate design choice for a centralized state architecture.

Execution Path Viability: The core event loop is sound. The use of a step_barrier asyncio.Event is intended to prevent the Perception component from running ahead of the Actuator, ensuring one logical "step" completes before the next begins. However, this introduces a potential deadlock risk if an unhandled error prevents the barrier from being set. The Supervisor's top-level exception handling within its TaskGroup is the primary defense against this, ensuring that component failures are caught and the system can transition to a FAILED state rather than hanging indefinitely.

State Management Integrity: The StateManager's use of a single, fine-grained lock for all state access is a robust and effective strategy to guarantee data consistency and prevent race conditions among the concurrent components. The introduction of TaskStack for managing sub-goals is a sophisticated feature that provides a structured solution to a common problem in agent execution (handling unexpected interruptions) that is often handled with less formal, ad-hoc logic.
