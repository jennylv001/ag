import asyncio
import pytest

from browser_use.utils import SignalHandler


@pytest.mark.asyncio
async def test_prompt_prints_once_and_single_waiter(monkeypatch, capsys):
    """
    Simulate Ctrl+C handling and ensure:
    - The resume/guidance prompt is printed only once per pause session
    - Only one waiter is created (no duplicates)
    - No process exit occurs (exit_on_second_int=False)
    """

    # Count prompt prints by monkeypatching the prompt method
    prompt_calls = {"count": 0}

    def fake_print_prompt(self):  # type: ignore[override]
        prompt_calls["count"] += 1

    monkeypatch.setattr(SignalHandler, "_print_resume_prompt", fake_print_prompt, raising=True)

    # Replace the async waiter to avoid blocking on input()
    waiter_started = asyncio.Event()
    waiter_calls = {"count": 0}

    async def fake_async_wait_for_resume(self):  # type: ignore[override]
        waiter_calls["count"] += 1
        waiter_started.set()
        # Emit the prompt as real waiter would
        self._print_resume_prompt()
        # Simulate quick resume path without user input
        await asyncio.sleep(0.01)
        self.reset()

    monkeypatch.setattr(SignalHandler, "_async_wait_for_resume", fake_async_wait_for_resume, raising=True)

    # Create a handler with exits disabled for safety
    loop = asyncio.get_event_loop()
    sh = SignalHandler(loop=loop, pause_callback=lambda: None, resume_callback=lambda: None, custom_exit_callback=None, exit_on_second_int=False)

    # Trigger first Ctrl+C flow
    sh.sigint_handler()

    # Wait for our fake waiter to start
    await asyncio.wait_for(waiter_started.wait(), timeout=1.0)
    # Allow prompt print to flush
    await asyncio.sleep(0.02)

    # Verify prompt printed exactly once for the session
    assert prompt_calls["count"] == 1

    # Trigger handler again while pause is active; should NOT spawn another waiter
    sh.sigint_handler()

    # Give the event loop a moment
    await asyncio.sleep(0.03)

    # Confirm only one waiter call was made
    assert waiter_calls["count"] == 1

    # No additional prompt prints expected
    assert prompt_calls["count"] == 1
