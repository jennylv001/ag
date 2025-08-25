import importlib.resources
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional

import importlib.resources
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional

from browser_use.llm.messages import BaseMessage, ContentPartImageParam, ContentPartTextParam, ImageURL, SystemMessage, UserMessage
from browser_use.observability import observe_debug
from browser_use.utils import is_new_tab_page

if TYPE_CHECKING:
	from browser_use.agent.views import AgentStepInfo, AgentHistoryList
	from browser_use.browser.views import BrowserStateSummary
	from browser_use.filesystem.file_system import FileSystem


class SystemPrompt:
	def __init__(
		self,
		action_description: str,
		max_actions_per_step: int = 10,
		override_system_message: str | None = None,
		extend_system_message: str | None = None,
		use_thinking: bool = True,
		flash_mode: bool = False,
	):
		self.default_action_description = action_description
		self.max_actions_per_step = max_actions_per_step
		self.use_thinking = use_thinking
		self.flash_mode = flash_mode
		prompt = ''
		if override_system_message:
			prompt = override_system_message
		else:
			self._load_prompt_template()
			prompt = self.prompt_template.format(max_actions=self.max_actions_per_step)

		if extend_system_message:
			prompt += f'\n{extend_system_message}'

		self.system_message = SystemMessage(content=prompt, cache=True)

	def _load_prompt_template(self) -> None:
		"""Load the prompt template from the markdown file."""
		try:
			# Choose the appropriate template based on flash_mode and use_thinking settings
			if self.flash_mode:
				template_filename = 'system_prompt_flash.md'
			elif self.use_thinking:
				template_filename = 'system_prompt.md'
			else:
				template_filename = 'system_prompt_no_thinking.md'

			# This works both in development and when installed as a package
			with importlib.resources.files('browser_use.agent').joinpath(template_filename).open('r', encoding='utf-8') as f:
				self.prompt_template = f.read()
		except Exception as e:
			raise RuntimeError(f'Failed to load system prompt template: {e}')

	def get_system_message(self) -> SystemMessage:
		"""
		Get the system prompt for the agent.

		Returns:
			SystemMessage: Formatted system prompt
		"""
		return self.system_message


class AgentMessagePrompt:
	vision_detail_level: Literal['auto', 'low', 'high']

	def __init__(
		self,
		browser_state_summary: 'BrowserStateSummary',
		file_system: 'FileSystem',
		agent_history_description: str | None = None,
		read_state_description: str | None = None,
		task: str | None = None,
		current_task_id: str = "root",
		task_context: str | None = None,
		include_attributes: list[str] | None = None,
		step_info: Optional['AgentStepInfo'] = None,
		page_filtered_actions: str | None = None,
		max_clickable_elements_length: int = 40000,
		sensitive_data: str | None = None,
		available_file_paths: list[str] | None = None,
		screenshots: list[str] | None = None,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
	):
		self.browser_state: 'BrowserStateSummary' = browser_state_summary
		self.file_system: 'FileSystem | None' = file_system
		self.agent_history_description: str | None = agent_history_description
		self.read_state_description: str | None = read_state_description
		self.task: str | None = task
		self.current_task_id: str = current_task_id
		self.task_context: str | None = task_context
		self.include_attributes = include_attributes
		self.step_info = step_info
		self.page_filtered_actions: str | None = page_filtered_actions
		self.max_clickable_elements_length: int = max_clickable_elements_length
		self.sensitive_data: str | None = sensitive_data
		self.available_file_paths: list[str] | None = available_file_paths
		self.screenshots = screenshots or []
		self.vision_detail_level = vision_detail_level
		assert self.browser_state

	@observe_debug(ignore_input=True, ignore_output=True, name='_get_browser_state_description')
	def _get_browser_state_description(self) -> str:
		# Be defensive: element_tree may be None in minimal stubs/tests
		try:
			element_tree = getattr(self.browser_state, 'element_tree', None)
			if element_tree and hasattr(element_tree, 'clickable_elements_to_string'):
				elements_text = element_tree.clickable_elements_to_string(include_attributes=self.include_attributes)
			else:
				elements_text = ''
		except Exception:
			elements_text = ''

		if len(elements_text) > self.max_clickable_elements_length:
			elements_text = elements_text[: self.max_clickable_elements_length]
			truncated_text = f' (truncated to {self.max_clickable_elements_length} characters)'
		else:
			truncated_text = ''

		has_content_above = (getattr(self.browser_state, 'pixels_above', 0) or 0) > 0
		has_content_below = (getattr(self.browser_state, 'pixels_below', 0) or 0) > 0

		# Enhanced page information for the model
		page_info_text = ''
		if self.browser_state.page_info:
			pi = self.browser_state.page_info
			# Compute page statistics dynamically
			pages_above = pi.pixels_above / pi.viewport_height if pi.viewport_height > 0 else 0
			pages_below = pi.pixels_below / pi.viewport_height if pi.viewport_height > 0 else 0
			total_pages = pi.page_height / pi.viewport_height if pi.viewport_height > 0 else 0
			current_page_position = pi.scroll_y / max(pi.page_height - pi.viewport_height, 1)
			page_info_text = f'Page info: {pi.viewport_width}x{pi.viewport_height}px viewport, {pi.page_width}x{pi.page_height}px total page size, {pages_above:.1f} pages above, {pages_below:.1f} pages below, {total_pages:.1f} total pages, at {current_page_position:.0%} of page'

		if elements_text != '':
			if has_content_above:
				if self.browser_state.page_info:
					pi = self.browser_state.page_info
					pages_above = pi.pixels_above / pi.viewport_height if pi.viewport_height > 0 else 0
					elements_text = f'... {self.browser_state.pixels_above} pixels above ({pages_above:.1f} pages) - scroll to see more or extract structured data if you are looking for specific information ...\n{elements_text}'
				else:
					elements_text = f'... {self.browser_state.pixels_above} pixels above - scroll to see more or extract structured data if you are looking for specific information ...\n{elements_text}'
			else:
				elements_text = f'[Start of page]\n{elements_text}'
			if has_content_below:
				if self.browser_state.page_info:
					pi = self.browser_state.page_info
					pages_below = pi.pixels_below / pi.viewport_height if pi.viewport_height > 0 else 0
					elements_text = f'{elements_text}\n... {self.browser_state.pixels_below} pixels below ({pages_below:.1f} pages) - scroll to see more or extract structured data if you are looking for specific information ...'
				else:
					elements_text = f'{elements_text}\n... {self.browser_state.pixels_below} pixels below - scroll to see more or extract structured data if you are looking for specific information ...'
			else:
				elements_text = f'{elements_text}\n[End of page]'
		else:
			elements_text = 'empty page'

		tabs_text = ''
		current_tab_candidates = []

		# Find tabs that match both URL and title to identify current tab more reliably
		for tab in self.browser_state.tabs:
			if tab.url == self.browser_state.url and tab.title == self.browser_state.title:
				current_tab_candidates.append(tab.page_id)

		# If we have exactly one match, mark it as current
		# Otherwise, don't mark any tab as current to avoid confusion
		current_tab_id = current_tab_candidates[0] if len(current_tab_candidates) == 1 else None

		for tab in self.browser_state.tabs:
			tabs_text += f'Tab {tab.page_id}: {tab.url} - {tab.title[:30]}\n'

		current_tab_text = f'Current tab: {current_tab_id}' if current_tab_id is not None else ''

		# Check if current page is a PDF viewer and add appropriate message
		pdf_message = ''
		if getattr(self.browser_state, 'is_pdf_viewer', False):
			pdf_message = 'PDF viewer cannot be rendered. In this page, DO NOT use the extract_structured_data action as PDF content cannot be rendered. Use the read_file action on the downloaded PDF in available_file_paths to read the full content.\n\n'

		# Render merged DOM+AX affordances if available (compact)
		affordances_text = ''
		try:
			affs = getattr(self.browser_state, 'affordances', None)
			if isinstance(affs, list) and affs:
				lines = []
				for a in affs:
					if not isinstance(a, dict):
						continue
					idx = a.get('index')
					role = a.get('role')
					name = a.get('name')
					vp = a.get('viewport_coordinates') or {}
					# show minimal bbox for grounding
					tl = (vp.get('top_left') or {}).get('y'), (vp.get('top_left') or {}).get('x')
					br = (vp.get('bottom_right') or {}).get('y'), (vp.get('bottom_right') or {}).get('x')
					lines.append(f"[{idx}] role={role or '-'} name={name or '-'} bbox=({tl[1]},{tl[0]})-({br[1]},{br[0]})")
				affordances_text = 'Affordances (DOM+AX merged where available):\n' + '\n'.join(lines)
		except Exception:
			pass

		browser_state = f"""{current_tab_text}
Available tabs:
{tabs_text}
{page_info_text}
{pdf_message}Interactive elements from top layer of the current page inside the viewport{truncated_text}:
{elements_text}
{affordances_text}
"""
		return browser_state

	def _get_agent_state_description(self) -> str:
		if self.step_info:
			step_info_description = f'Step {self.step_info.step_number + 1} of {self.step_info.max_steps} max possible steps\n'
		else:
			step_info_description = ''
		time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
		step_info_description += f'Current date and time: {time_str}'

		_todo_contents = self.file_system.get_todo_contents() if self.file_system else ''
		if not len(_todo_contents):
			_todo_contents = '[Current todo.md is empty, fill it with your plan when applicable]'

		agent_state = f"""
<user_request>
{self.task}
</user_request>
<task_context>
Current Task ID: {self.current_task_id}
{self.task_context if self.task_context else "Task Context: root (main task)"}
</task_context>
<file_system>
{self.file_system.describe() if self.file_system else 'No file system available'}
</file_system>
<todo_contents>
{_todo_contents}
</todo_contents>
"""
		if self.sensitive_data:
			agent_state += f'<sensitive_data>\n{self.sensitive_data}\n</sensitive_data>\n'

		agent_state += f'<step_info>\n{step_info_description}\n</step_info>\n'
		if self.available_file_paths:
			agent_state += '<available_file_paths>\n' + '\n'.join(self.available_file_paths) + '\n</available_file_paths>\n'
		return agent_state

	@observe_debug(ignore_input=True, ignore_output=True, name='get_user_message')
	def get_user_message(self, use_vision: bool = True) -> UserMessage:
		# Don't pass screenshot to model if page is a new tab page, step is 0, and there's only one tab
		if (
			is_new_tab_page(self.browser_state.url)
			and self.step_info is not None
			and self.step_info.step_number == 0
			and len(self.browser_state.tabs) == 1
		):
			use_vision = False

		state_description = (
			'<agent_history>\n'
			+ (self.agent_history_description.strip('\n') if self.agent_history_description else '')
			+ '\n</agent_history>\n'
		)
		state_description += '<agent_state>\n' + self._get_agent_state_description().strip('\n') + '\n</agent_state>\n'
		state_description += '<browser_state>\n' + self._get_browser_state_description().strip('\n') + '\n</browser_state>\n'
		state_description += (
			'<read_state>\n'
			+ (self.read_state_description.strip('\n') if self.read_state_description else '')
			+ '\n</read_state>\n'
		)
		if self.page_filtered_actions:
			state_description += 'For this page, these additional actions are available:\n'
			state_description += self.page_filtered_actions + '\n'

		if use_vision is True and self.screenshots:
			# Start with text description
			content_parts: list[ContentPartTextParam | ContentPartImageParam] = [ContentPartTextParam(text=state_description)]

			# Add screenshots with labels
			for i, screenshot in enumerate(self.screenshots):
				if i == len(self.screenshots) - 1:
					label = 'Current screenshot:'
				else:
					# Use simple, accurate labeling since we don't have actual step timing info
					label = 'Previous screenshot:'

				# Add label as text content
				content_parts.append(ContentPartTextParam(text=label))

				# Add the screenshot
				content_parts.append(
					ContentPartImageParam(
						image_url=ImageURL(
							url=f'data:image/png;base64,{screenshot}',
							media_type='image/png',
							detail=self.vision_detail_level,
						),
					)
				)

			return UserMessage(content=content_parts)

		return UserMessage(content=state_description)


class PlannerPrompt:
	"""Planner prompt builder with doctrine-specific templates (Scout/Medic).

	Includes CurrentIntent, RecentHistory, SystemAssessmentVector, and EnvironmentState.
	Optionally attaches screenshots when use_vision is True.
	"""

	# Scout (Proactive) template
	SCOUT_TEMPLATE = """# ROLE: Proactive Strategist (Scout Doctrine)
# INPUT: Strategic Planning Package

<CurrentIntent>
	{CurrentIntent}
</CurrentIntent>

<RecentHistory>
	{RecentHistory}
</RecentHistory>

<SystemAssessmentVector>
	risk: {SAV_risk}
	opportunity: {SAV_opportunity}
	confidence: {SAV_confidence}
	stagnation: {SAV_stagnation}
	looping: {SAV_looping}
	contributors: {SAV_contributors}
</SystemAssessmentVector>

<EnvironmentState>
	<BrowserSummary>
		URL: {Env_URL}
		Clickable Elements: {Env_Clickable}
		New Downloads: {Env_NewDownloads}
	</BrowserSummary>
	<VisualContext>
		screenshots: {Env_Screenshots}
		change_map: {Env_ChangeMap}
	</VisualContext>
</EnvironmentState>

# COGNITIVE PROCESS
1. Holistic Synthesis — unify Intent, History, Signals, and Environment.
2. Declare Imperative — choose CONTINUE, PIVOT, or FORTIFY.
3. Formulate next_goal aligned to imperative and evidence.
4. Justify using specific signals/visuals (cite screenshots or metrics).

Output ONLY the JSON object with keys: memory_summary, next_goal, effective_strategy.
"""

	# Medic (Reactive) template
	MEDIC_TEMPLATE = """# ROLE: Reactive Analyst (Medic Doctrine)
# INPUT: Complete Post-Mortem Package

<FailureContext>
	ErrorType: {Failure_ErrorType}
	ErrorMessage: {Failure_ErrorMessage}
	LastAction: {Failure_LastAction}
</FailureContext>

<CurrentIntent>
	{CurrentIntent}
</CurrentIntent>

<RecentHistory>
	{RecentHistory}
</RecentHistory>

<SystemAssessmentVector>
	risk: {SAV_risk}
	opportunity: {SAV_opportunity}
	confidence: {SAV_confidence}
	stagnation: {SAV_stagnation}
	looping: {SAV_looping}
	contributors: {SAV_contributors}
</SystemAssessmentVector>

<EnvironmentState>
	<BrowserSummary>
		URL: {Env_URL}
		Clickable Elements: {Env_Clickable}
		New Downloads: {Env_NewDownloads}
	</BrowserSummary>
	<VisualContext>
		screenshots: {Env_Screenshots}
		change_map: {Env_ChangeMap}
	</VisualContext>
</EnvironmentState>

# COGNITIVE PROCESS
1. Root Cause Diagnosis — correlate failure context with signals and environment.
2. Invalidate Flawed Assumption — state the incorrect belief that led to failure.
3. Robust Recovery — formulate corrective next_goal directly addressing ground truth.
4. Prevention Guard — propose one simple guard to avoid recurrence.

Output ONLY the JSON object with keys: memory_summary, next_goal, effective_strategy.
"""

	def __init__(
		self,
		task: str,
		history: 'AgentHistoryList',
		last_error: Optional[str],
		current_task_id: Optional[str] = None,
		task_context: Optional[str] = None,
		screenshots: Optional[list[str]] = None,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
		use_vision: bool = False,
		doctrine: Literal['scout', 'medic'] = 'scout',
		failure_context: Optional[dict] = None,
		# New optional inputs for doctrine templates
		current_intent: Optional[str] = None,
		browser_summary: Optional[dict] = None,
	):
		self.task = task
		self.history = history
		self.last_error = last_error
		self.current_task_id = current_task_id or "root"
		self.task_context = task_context or "Task Context: root (main task)"
		self.screenshots = screenshots or []
		self.vision_detail_level = vision_detail_level
		self.use_vision = use_vision
		self.doctrine = doctrine
		self.failure_context = failure_context
		# Optional mode hint from assessor-driven planner
		self.extra_context_hint: Optional[str] = None
		# Optional assessment context (floats, contributors, visual summary)
		self.assessment_context: Optional[dict] = None
		# Doctrine fields
		self.current_intent = current_intent or task
		# Expected keys: url, clickable_elements, new_downloads
		self.browser_summary = browser_summary or {}

	def get_messages(self) -> list[BaseMessage]:
		"""Formats the inputs into doctrine template messages."""
		# 1. Synthesize the 'recent_history' string from the history object
		history_summary_lines = []
		for item in self.history.history:
			if item.model_output:
				actions = [a.model_dump(exclude_none=True, exclude_defaults=True) for a in item.model_output.action]
				result_str = "Success" if all(r.success is not False for r in item.result) else "Failure"
				history_summary_lines.append(
					f"  - Step Goal: {item.model_output.next_goal}\n"
					f"    Action(s): {actions}\n"
					f"    Outcome: {result_str}"
				)
		recent_history_str = "\n".join(history_summary_lines) if history_summary_lines else "No actions taken in this reflection window."

		# 2. Environment summary
		env_url = None
		if isinstance(self.browser_summary, dict):
			env_url = self.browser_summary.get('url')
		if not env_url:
			env_url = "Not available"
			if self.history.history and self.history.history[-1].state:
				env_url = self.history.history[-1].state.url or "about:blank"
		env_clickable = None
		if isinstance(self.browser_summary, dict):
			env_clickable = self.browser_summary.get('clickable_elements')
		if env_clickable is None:
			env_clickable = "unknown"
		env_new_downloads = []
		if isinstance(self.browser_summary, dict):
			env_new_downloads = self.browser_summary.get('new_downloads') or []

		# 3. Assessment values
		ctx = self.assessment_context or {}
		sav_vals = {
			'risk': ctx.get('risk'),
			'opportunity': ctx.get('opportunity'),
			'confidence': ctx.get('confidence'),
			'stagnation': ctx.get('stagnation'),
			'looping': ctx.get('looping'),
			'contributors': ctx.get('contributors') or [],
			'screenshot_refs': ctx.get('screenshot_refs') or [],
			'change_map_ref': ctx.get('change_map_ref') or ctx.get('change_map') or None,
		}

		# Avoid dumping raw image data in text prompts; present a compact summary instead
		screenshots_repr = "none"
		try:
			refs = sav_vals.get('screenshot_refs') or []
			if isinstance(refs, list):
				screenshots_repr = f"{len(refs)} image(s) available"
			elif refs:
				screenshots_repr = "1 image available"
		except Exception:
			screenshots_repr = "unknown"

		# 4. Choose template and render
		template = self.MEDIC_TEMPLATE if self.doctrine == 'medic' else self.SCOUT_TEMPLATE
		fc = self.failure_context or {
			'error_type': None,
			'error_message': self.last_error or None,
			'last_action': None,
		}
		prompt_text = template.format(
			CurrentIntent=self.current_intent,
			RecentHistory=recent_history_str,
			SAV_risk=sav_vals['risk'],
			SAV_opportunity=sav_vals['opportunity'],
			SAV_confidence=sav_vals['confidence'],
			SAV_stagnation=sav_vals['stagnation'],
			SAV_looping=sav_vals['looping'],
			SAV_contributors=sav_vals['contributors'],
			Env_URL=env_url,
			Env_Clickable=env_clickable,
			Env_NewDownloads=env_new_downloads,
			Env_Screenshots=screenshots_repr,
			Env_ChangeMap=sav_vals['change_map_ref'],
			Failure_ErrorType=fc.get('error_type'),
			Failure_ErrorMessage=fc.get('error_message'),
			Failure_LastAction=fc.get('last_action'),
		)

		# Optional mode/budget hint
		if getattr(self, 'extra_context_hint', None):
			prompt_text = f"{prompt_text}\n\n[Mode Hint]\n{self.extra_context_hint}"
		if self.use_vision and self.screenshots:
			parts: list[ContentPartTextParam | ContentPartImageParam] = [ContentPartTextParam(text=prompt_text)]
			for i, screenshot in enumerate(self.screenshots):
				label = 'Current screenshot:' if i == len(self.screenshots) - 1 else 'Previous screenshot:'
				parts.append(ContentPartTextParam(text=label))
				parts.append(
					ContentPartImageParam(
						image_url=ImageURL(
							url=f'data:image/png;base64,{screenshot}',
							media_type='image/png',
							detail=self.vision_detail_level,
						),
					)
				)
			return [UserMessage(content=parts)]
		else:
			return [UserMessage(content=prompt_text)]
