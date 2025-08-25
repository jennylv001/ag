import asyncio
import time

from browser_use.controller.service import Controller


async def _call_action_async(ctrl: Controller, name: str, **kwargs):
    # Call the normalized action function directly to avoid extra wrappers
    action = ctrl.registry.registry.actions[name]
    model = action.param_model(**kwargs)
    return await action.function(params=model)


def test_controller_has_wait_action_registered():
    c = Controller()
    assert 'wait' in c.registry.registry.actions


def test_controller_wait_sleep_duration_is_honored():
    c = Controller()
    # Use a short sleep to keep tests fast
    start = time.monotonic()
    res = asyncio.run(_call_action_async(c, 'wait', seconds=1))
    elapsed = time.monotonic() - start
    assert getattr(res, 'extracted_content', '').startswith('🕒  Waited for 1 seconds')
    # Allow some wiggle for scheduling
    assert elapsed >= 0.9
