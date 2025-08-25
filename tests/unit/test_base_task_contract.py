from __future__ import annotations

from browser_use.agent.tasks.base_task import BaseTask, TaskResult


class DummyTask(BaseTask):
    def __init__(self):
        self._done = False
        self._success = False
        self._n = 0

    def step(self):
        self._n += 1
        if self._n >= 2:
            self._done = True
            self._success = True
            return TaskResult(success=True, message="finished", data={"steps": self._n})
        return None

    def is_done(self) -> bool:
        return self._done

    def succeeded(self) -> bool:
        return self._success


def test_base_task_contract():
    t = DummyTask()
    # first step returns None, not done
    assert t.step() is None
    assert not t.is_done()
    assert not t.succeeded()

    # second step completes
    res = t.step()
    assert isinstance(res, TaskResult)
    assert t.is_done()
    assert t.succeeded()
    assert res.success is True
    assert res.data == {"steps": 2}
