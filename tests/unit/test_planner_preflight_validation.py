import pytest

from browser_use.agent.tasks.planner import preflight_validate_tasks
from browser_use.agent.tasks.low_level_execution_task import LowLevelExecutionTask
from browser_use.agent.tasks.solve_captcha_task import SolveCaptchaTask


class DummyOrchestrator:
    pass


def test_preflight_ok_for_low_level():
    t = LowLevelExecutionTask(orchestrator=DummyOrchestrator())
    ok, errs = preflight_validate_tasks([t])
    assert ok is True
    assert errs == []


def test_preflight_fails_for_unregistered_class():
    class NotRegistered:
        pass
    # Mock an instance that looks like a BaseTask but isn't in registry
    class FakeTask(NotRegistered):
        def step(self):
            return None
        def is_done(self):
            return True
        def succeeded(self):
            return True
    t = FakeTask()
    ok, errs = preflight_validate_tasks([t])
    assert ok is False
    assert any('not registered' in e.lower() for e in errs)


def test_preflight_missing_deps_captcha():
    # Missing required deps should trigger errors
    t = SolveCaptchaTask.__new__(SolveCaptchaTask)  # bypass __init__
    # Intentionally leave controller/browser/llm unset
    ok, errs = preflight_validate_tasks([t])
    assert ok is False
    assert any('missing dependency' in e.lower() for e in errs)
