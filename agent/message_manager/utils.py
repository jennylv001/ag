from __future__ import annotations

import json
import logging
import base64
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio

from browser_use.llm.messages import BaseMessage

logger = logging.getLogger(__name__)


async def save_conversation(
	input_messages: list[BaseMessage],
	response: Any,
	target: str | Path,
	encoding: str | None = None,
	step_number: int | None = None,
	screenshot_path: str | None = None,
) -> None:
	"""Save conversation history to file asynchronously with enhanced structure."""
	target_path = Path(target)

	# Check if target should be organized into datetime folders
	if target_path.suffix == '.md' and step_number is not None:
		# Create datetime-organized folder structure
		session_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
		session_folder = target_path.parent / f"session_{session_datetime}"
		session_folder.mkdir(parents=True, exist_ok=True)

		# Save main conversation file
		conversation_file = session_folder / "conversation.md"
		step_folder = session_folder / f"step_{step_number:03d}"
		step_folder.mkdir(exist_ok=True)

		# Save step-specific details
		step_file = step_folder / "step.md"
		await anyio.Path(step_file).write_text(
			await _format_conversation(input_messages, response),
			encoding=encoding or 'utf-8',
		)

		# Copy screenshot if available
		if screenshot_path:
			screenshot_source = Path(screenshot_path)
			if screenshot_source.exists():
				screenshot_dest = step_folder / f"screenshot_{step_number:03d}.png"
				try:
					screenshot_data = screenshot_source.read_bytes()
					await anyio.Path(screenshot_dest).write_bytes(screenshot_data)
				except Exception as e:
					logger.warning(f"Failed to copy screenshot: {e}")

		# Update main conversation file with step summary
		await _update_conversation_summary(conversation_file, step_number, input_messages, response, encoding)
	else:
		# Standard single-file saving
		if target_path.parent:
			await anyio.Path(target_path.parent).mkdir(parents=True, exist_ok=True)

		await anyio.Path(target_path).write_text(
			await _format_conversation(input_messages, response),
			encoding=encoding or 'utf-8',
		)


async def _update_conversation_summary(
	conversation_file: Path,
	step_number: int,
	input_messages: list[BaseMessage],
	response: Any,
	encoding: str | None = None,
) -> None:
	"""Update the main conversation summary file with step information."""
	# Read existing content if file exists
	existing_content = ""
	if conversation_file.exists():
		existing_content = await anyio.Path(conversation_file).read_text(encoding=encoding or 'utf-8')

	# Extract key information from the step
	step_summary = _extract_step_summary(input_messages, response, step_number)

	# Append step summary
	step_entry = f"\n## Step {step_number}\n\n{step_summary}\n"
	new_content = existing_content + step_entry

	await anyio.Path(conversation_file).write_text(new_content, encoding=encoding or 'utf-8')


def _extract_step_summary(input_messages: list[BaseMessage], response: Any, step_number: int) -> str:
	"""Extract a concise summary from the step's messages and response."""
	# Extract action from response if available
	action_summary = "No action"
	try:
		response_data = json.loads(response.model_dump_json(exclude_unset=True))
		actions = response_data.get('action', [])
		if actions:
			action_names = [list(action.keys())[0] for action in actions if isinstance(action, dict)]
			action_summary = f"Actions: {', '.join(action_names)}"
	except Exception:
		pass

	# Extract key browser state information
	browser_info = "Browser state not available"
	for message in input_messages:
		if hasattr(message, 'text') and 'Current URL:' in message.text:
			# Extract URL from browser state
			lines = message.text.split('\n')
			for line in lines:
				if line.strip().startswith('Current URL:'):
					browser_info = line.strip()
					break
			break

	return f"**{action_summary}**\n\n{browser_info}\n\nScreenshot: `step_{step_number:03d}/screenshot_{step_number:03d}.png`"


async def _format_conversation(messages: list[BaseMessage], response: Any) -> str:
	"""Format the conversation including messages and response."""
	lines = []

	# Format messages
	for message in messages:
		lines.append(f'## {message.role}')
		lines.append('')
		lines.append(message.text)
		lines.append('')  # Empty line after each message

	# Format response
	lines.append('## RESPONSE')
	lines.append('')
	lines.append(json.dumps(json.loads(response.model_dump_json(exclude_unset=True)), indent=2))

	return '\n'.join(lines)


# Note: _write_messages_to_file and _write_response_to_file have been merged into _format_conversation
# This is more efficient for async operations and reduces file I/O
