import os

from browser_use.logging_config import setup_logging

# Only set up logging if not in MCP mode or if explicitly requested
if os.environ.get('BROWSER_USE_SETUP_LOGGING', 'true').lower() != 'false':
	logger = setup_logging()
else:
	import logging

	logger = logging.getLogger('browser_use')

# Monkeypatch BaseSubprocessTransport.__del__ to handle closed event loops gracefully
from asyncio import base_subprocess

_original_del = base_subprocess.BaseSubprocessTransport.__del__


def _patched_del(self):
	"""Patched __del__ that handles closed event loops without throwing noisy red-herring errors like RuntimeError: Event loop is closed"""
	try:
		# Check if the event loop is closed before calling the original
		if hasattr(self, '_loop') and self._loop and self._loop.is_closed():
			# Event loop is closed, skip cleanup that requires the loop
			return
		_original_del(self)
	except RuntimeError as e:
		if 'Event loop is closed' in str(e):
			# Silently ignore this specific error
			pass
		else:
			raise


base_subprocess.BaseSubprocessTransport.__del__ = _patched_del


# --- Lightweight, lazy re-exports ---
# Avoid importing heavy optional dependencies (LLM providers, pydantic-based modules)
# at package import time. Provide names via lazy attribute access for backwards compatibility.

_LAZY_EXPORTS = {
	# Agent core
	'Agent': ('browser_use.agent.service', 'Agent'),
	'AgentSettings': ('browser_use.agent.settings', 'AgentSettings'),
	'MessageManagerSettings': ('browser_use.agent.message_manager.views', 'MessageManagerSettings'),
	'SystemPrompt': ('browser_use.agent.prompts', 'SystemPrompt'),
	'ActionModel': ('browser_use.agent.views', 'ActionModel'),
	'ActionResult': ('browser_use.agent.views', 'ActionResult'),
	'AgentHistoryList': ('browser_use.agent.views', 'AgentHistoryList'),
	# Browser
	'Browser': ('browser_use.browser', 'Browser'),
	'BrowserConfig': ('browser_use.browser', 'BrowserConfig'),
	'BrowserSession': ('browser_use.browser', 'BrowserSession'),
	'BrowserProfile': ('browser_use.browser', 'BrowserProfile'),
	'BrowserContext': ('browser_use.browser', 'BrowserContext'),
	'BrowserContextConfig': ('browser_use.browser', 'BrowserContextConfig'),
	# Controller and DOM
	'Controller': ('browser_use.controller.service', 'Controller'),
	'DomService': ('browser_use.dom.service', 'DomService'),
	# Chat models (remain lazy and optional)
	'ChatOpenAI': ('browser_use.llm', 'ChatOpenAI'),
	'ChatGoogle': ('browser_use.llm', 'ChatGoogle'),
	'ChatAnthropic': ('browser_use.llm', 'ChatAnthropic'),
	'ChatGroq': ('browser_use.llm', 'ChatGroq'),
	'ChatAzureOpenAI': ('browser_use.llm', 'ChatAzureOpenAI'),
	'ChatOllama': ('browser_use.llm', 'ChatOllama'),
}


def __getattr__(name: str):
	entry = _LAZY_EXPORTS.get(name)
	if not entry:
		raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
	module_path, attr_name = entry
	try:
		from importlib import import_module
		module = import_module(module_path)
		attr = getattr(module, attr_name)
		# Cache for future lookups
		globals()[name] = attr
		return attr
	except Exception as e:
		# Provide a helpful error that preserves the original exception
		raise ImportError(f"Failed to import {name} from {module_path}: {e}") from e


__all__ = list(_LAZY_EXPORTS.keys())

# Best-effort: rebuild pydantic models if available without importing eagerly
try:
	from browser_use.agent.settings import AgentSettings as _AS
	from browser_use.agent.message_manager.views import MessageManagerSettings as _MMS
	_AS.model_rebuild()
	_MMS.model_rebuild()
except Exception:
	pass
