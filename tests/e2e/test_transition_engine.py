"""E2E smoke test for transition engine module imports.

This keeps the quick sanity task green without launching the browser.
"""

import importlib


def test_transition_engine_imports_smoke():
	# Ensure key agent modules import without syntax/indentation errors.
	modules = [
		"agent.orchestrator",
		"agent.service",
		"agent.tasks.service",
		"agent.tasks.planner",
	]
	for name in modules:
		mod = importlib.import_module(name)
		assert mod is not None, f"Failed to import {name}"

