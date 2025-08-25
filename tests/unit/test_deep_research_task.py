from __future__ import annotations

import asyncio
import json
import pytest

from browser_use.agent.tasks.deep_research_task import (
    DeepResearchSynthesisTask,
    ResearchInput,
    ResearchConstraints,
)
from browser_use.agent.tasks.base_task import TaskResult


class DummyLLM:
    model = "dummy-llm"

    async def ainvoke(self, messages, output_format=None):  # type: ignore[override]
        # Return simple JSON strings depending on prompt content
        content = ""
        for m in messages:
            try:
                if isinstance(m.content, list):
                    content = "\n".join(getattr(p, "text", "") for p in m.content)
                else:
                    content = str(m.content)
            except Exception:
                pass
        if "Return JSON with keys: broad" in content:
            class Resp:
                completion = json.dumps({
                    "broad": ["overview"],
                    "narrow": ["specific"],
                    "adversarial": ["counter"]
                })
            return Resp()
        elif "Return JSON with keys: text, confidence" in content:
            class Resp:
                completion = json.dumps({
                    "text": "Summary",
                    "confidence": 0.7,
                    "uncertainties": ["gap1", "gap2", "gap3"]
                })
            return Resp()
        else:
            class Resp:
                completion = "{}"
            return Resp()


@pytest.mark.asyncio
async def test_deep_research_task_runs_and_succeeds():
    ri = ResearchInput(
        research_question="What are best practices for web automation?",
        constraints=ResearchConstraints(recency_window_days=180, regions=["global"], sectors=["tech"]),
    )
    task = DeepResearchSynthesisTask(controller=None, browser=None, llm=DummyLLM(), research_input=ri)
    res = await task.run()
    assert isinstance(res, TaskResult)
    assert res.success is True
    assert res.message == "deep_research_synthesis_complete"
    assert res.data and "output" in res.data
    out = res.data["output"]
    assert "executive_summary" in out
    assert out["coverage"]["recency_window_days"] == 180


def test_task_registry_includes_deep_research():
    from browser_use.agent.tasks import TASK_REGISTRY

    assert "deep_research_synthesis" in TASK_REGISTRY
