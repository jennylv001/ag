import re
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from browser_use.agent.views import AgentOutput, TaskLog, TaskLogItem


def test_agent_output_ensures_structured_log_exists():
    # Build a dynamic AgentOutput with a minimal Pydantic ActionModel
    class DummyActionModel(BaseModel):
        done: Optional[Dict[str, Any]] = Field(default_factory=lambda: {"success": False, "text": "noop"})

    Out = AgentOutput.type_with_custom_actions(DummyActionModel)
    out = Out(
        prior_action_assessment="ok",
        task_log="tlog",
        next_goal="next",
        action=[DummyActionModel()],
    )
    assert out.task_log_structured is not None
    assert isinstance(out.task_log_structured, TaskLog)


def test_task_log_progress_parsing_and_clamp():
    # String percent
    t1 = TaskLog(progress_pct="35%")
    assert t1.progress_pct == 35.0

    # Ratio to percent
    t2 = TaskLog(progress_pct=0.42)
    assert abs(t2.progress_pct - 42.0) < 1e-9

    # Clamp above 100 (validator coerces instead of raising)
    t3 = TaskLog(progress_pct=1000)
    assert t3.progress_pct == 100.0

    # Clamp below 0
    t4 = TaskLog(progress_pct=-5)
    assert t4.progress_pct == 0.0

    # Accept numeric string
    t5 = TaskLog(progress_pct="12.5")
    assert abs(t5.progress_pct - 12.5) < 1e-9


def test_task_log_sets_updated_at_iso8601():
    t = TaskLog()
    assert t.updated_at is not None
    # Basic ISO-8601 with timezone suffix
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", t.updated_at) is not None


def test_task_log_item_normalization_and_id_generation():
    # Missing id -> auto-generated; stray casing/spacing normalized
    item = TaskLogItem(id="", status="In Progress", priority="medium")
    assert isinstance(item.id, str) and len(item.id) > 0
    assert item.status == "in-progress"
    assert item.priority == "med"

    # Synonyms mapping
    item2 = TaskLogItem(id="x", status="done", priority="MID")
    assert item2.status == "completed"
    assert item2.priority == "med"
