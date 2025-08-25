import base64

from google.genai.types import Content, ContentListUnion, Part

from browser_use.llm.messages import (
	AssistantMessage,
	BaseMessage,
	SystemMessage,
	UserMessage,
)


class GoogleMessageSerializer:
	"""Serializer for converting messages to Google Gemini format."""

	@staticmethod
	def serialize_messages(messages: list[BaseMessage | str | dict]) -> tuple[ContentListUnion, str | None]:
		"""
		Convert a list of BaseMessages to Google format, extracting system message.

		Google handles system instructions separately from the conversation, so we need to:
		1. Extract any system messages and return them separately as a string
		2. Convert the remaining messages to Content objects

		Args:
		    messages: List of messages to convert

		Returns:
		    A tuple of (formatted_messages, system_message) where:
		    - formatted_messages: List of Content objects for the conversation
		    - system_message: System instruction string or None
		"""

		# Normalize inputs first: allow strings and simple dicts
		normalized: list[BaseMessage] = []
		for m in messages:
			# Convert plain strings to UserMessage
			if isinstance(m, str):
				normalized.append(UserMessage(content=m))
				continue

			# Convert simple dict representations {role, content}
			if isinstance(m, dict):
				role = m.get('role')
				content = m.get('content')
				if role in ('system', 'developer'):
					normalized.append(SystemMessage(content=str(content) if content is not None else ''))
				elif role in ('assistant', 'model'):
					normalized.append(AssistantMessage(content=str(content) if content is not None else ''))
				else:
					normalized.append(UserMessage(content=str(content) if content is not None else ''))
				continue

			# Assume already a BaseMessage or compatible object
			normalized.append(m)  # type: ignore[arg-type]

		# Handle cases where messages might contain non-Pydantic BaseMessage-like objects
		messages_copy: list[BaseMessage] = []
		for m in normalized:
			if hasattr(m, 'model_copy'):
				messages_copy.append(m.model_copy(deep=True))  # type: ignore[attr-defined]
			else:
				# For non-Pydantic objects, just append as-is
				messages_copy.append(m)
		messages = messages_copy

		formatted_messages: ContentListUnion = []
		system_message: str | None = None

		for message in messages:
			# Determine role defensively
			role = getattr(message, 'role', None)

			# Handle system/developer messages
			if isinstance(message, SystemMessage) or role in ['system', 'developer']:
				# Extract system message content as string
				if hasattr(message, 'content') and isinstance(message.content, str):
					system_message = message.content
				elif hasattr(message, 'content') and message.content is not None:
					# Handle Iterable of content parts
					parts = []
					for part in message.content:  # type: ignore[assignment]
						ptype = getattr(part, 'type', None)
						if ptype == 'text':
							parts.append(getattr(part, 'text', ''))
					system_message = '\n'.join(parts)
				continue

			# Determine the role for non-system messages
			if isinstance(message, UserMessage):
				role = 'user'
			elif isinstance(message, AssistantMessage):
				role = 'model'
			else:
				# Default to user for any unknown message types
				role = 'user'

			# Initialize message parts
			message_parts: list[Part] = []

			# Extract content and create parts
			if hasattr(message, 'content') and isinstance(message.content, str):
				# Regular text content
				message_parts = [Part.from_text(text=message.content)]
			elif hasattr(message, 'content') and message.content is not None:
				# Handle Iterable of content parts
				for part in message.content:
					ptype = getattr(part, 'type', None)
					if ptype == 'text':
						message_parts.append(Part.from_text(text=getattr(part, 'text', '')))
					elif ptype == 'refusal':
						message_parts.append(Part.from_text(text=f"[Refusal] {getattr(part, 'refusal', '')}"))
					elif ptype == 'image_url':
						# Handle images
						image_url = getattr(getattr(part, 'image_url', None), 'url', None)
						if not image_url:
							continue

						# Format: data:image/png;base64,<data>
						try:
							_, data = image_url.split(',', 1)
							# Decode base64 to bytes
							image_bytes = base64.b64decode(data)
							# Add image part
							image_part = Part.from_bytes(data=image_bytes, mime_type='image/png')
							message_parts.append(image_part)
						except Exception:
							# Skip malformed image parts
							pass

			# Create the Content object
			if message_parts:
				final_message = Content(role=role, parts=message_parts)
				formatted_messages.append(final_message)

		return formatted_messages, system_message
