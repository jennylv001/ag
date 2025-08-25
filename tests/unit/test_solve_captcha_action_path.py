from __future__ import annotations

# Ensure the controller exposes the solve_captcha action and internally routes
# through the new task-based shim without import errors.

from browser_use.controller.service import Controller


def test_controller_has_solve_captcha_action():
    c = Controller()
    # Ensure registry includes the solve_captcha action (use internal ActionRegistry)
    assert 'solve_captcha' in c.registry.registry.actions
    # Creating the action model should succeed without errors
    ActionModel = c.registry.create_action_model()
    assert isinstance(ActionModel, type)
