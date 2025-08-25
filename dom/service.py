import logging
import os
import asyncio
from importlib import resources
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
	from browser_use.browser.types import Page


from browser_use.dom.views import (
	DOMBaseNode,
	DOMElementNode,
	DOMState,
	DOMTextNode,
	SelectorMap,
	ViewportInfo,
)
from browser_use.dom.history_tree_processor.view import CoordinateSet, Coordinates
from browser_use.observability import observe_debug
from browser_use.utils import is_new_tab_page, time_execution_async

# @dataclass
# class ViewportInfo:
# 	width: int
# 	height: int


class DomService:
	logger: logging.Logger

	def __init__(self, page: 'Page', logger: logging.Logger | None = None):
		self.page = page
		self.xpath_cache = {}
		self.logger = logger or logging.getLogger(__name__)

		self.js_code = resources.files('browser_use.dom.dom_tree').joinpath('index.js').read_text()

	async def _wait_for_page_stable(self) -> None:
		"""Best-effort wait to let the page settle before DOM indexing.
		Avoids DOM churn during extraction with minimal latency.
		"""
		# Load states
		try:
			await self.page.wait_for_load_state('domcontentloaded')
		except Exception:
			pass
		# Optional very small networkidle budget (env-tunable); disabled by default to keep latency low
		networkidle_ms = int(os.environ.get('DOM_WAIT_NETWORKIDLE_MS', '0') or '0')
		if networkidle_ms > 0:
			try:
				await self.page.wait_for_load_state('networkidle', timeout=networkidle_ms)
			except Exception:
				pass
		# MutationObserver quiet window + two rAFs; fonts.ready bounded
		quiet_ms = int(os.environ.get('DOM_WAIT_QUIET_MS', '120') or '120')
		fonts_ms = int(os.environ.get('DOM_WAIT_FONTS_MS', '150') or '150')
		try:
			await self.page.evaluate(
				"""
				(opts) => new Promise((resolve) => {
				  const quietMs = Math.max(60, Math.min(600, (opts && opts.quietMs) || 120));
				  const fontsMs = Math.max(50, Math.min(600, (opts && opts.fontsMs) || 150));
				  const finish = () => requestAnimationFrame(() => requestAnimationFrame(resolve));
				  let lastMutation = performance.now();
				  let observer;
				  try {
				    observer = new MutationObserver(() => { lastMutation = performance.now(); });
				    observer.observe(document, { subtree: true, childList: true, attributes: true, characterData: true });
				  } catch {}
				  const check = () => {
				    const now = performance.now();
				    if (now - lastMutation >= quietMs) {
				      try { observer && observer.disconnect(); } catch {}
				      finish();
				    } else {
				      setTimeout(check, Math.min(quietMs, 60));
				    }
				  };
				  setTimeout(check, Math.min(quietMs, 60));
				  try {
				    if (document.fonts && document.fonts.ready) {
				      const to = setTimeout(() => {}, fontsMs);
				      document.fonts.ready.finally(() => { try { clearTimeout(to); } catch {} });
				    }
				  } catch {}
				})
				""",
				{'quietMs': quiet_ms, 'fontsMs': fonts_ms}
			)
		except Exception:
			pass

	# region - Clickable elements
	@observe_debug(ignore_input=True, ignore_output=True, name='get_clickable_elements')
	@time_execution_async('--get_clickable_elements')
	async def get_clickable_elements(
		self,
		highlight_elements: bool = False,
		focus_element: int = -1,
		viewport_expansion: int = 0,
	) -> DOMState:
		element_tree, selector_map = await self._build_dom_tree(
			highlight_elements, focus_element, viewport_expansion
		)
		return DOMState(element_tree=element_tree, selector_map=selector_map)

	@time_execution_async('--get_cross_origin_iframes')
	async def get_cross_origin_iframes(self) -> list[str]:
		# invisible cross-origin iframes are used for ads and tracking, dont open those
		hidden_frame_urls = await self.page.locator('iframe').filter(visible=False).evaluate_all('e => e.map(e => e.src)')

		is_ad_url = lambda url: any(
			domain in urlparse(url).netloc for domain in ('doubleclick.net', 'adroll.com', 'googletagmanager.com')
		)

		return [
			frame.url
			for frame in self.page.frames
			if urlparse(frame.url).netloc  # exclude data:urls and new tab pages
			and urlparse(frame.url).netloc != urlparse(self.page.url).netloc  # exclude same-origin iframes
			and frame.url not in hidden_frame_urls  # exclude hidden frames
			and not is_ad_url(frame.url)  # exclude most common ad network tracker frame URLs
		]

	@time_execution_async('--get_affordances')
	async def get_affordances(
		self,
		viewport_expansion: int = 0,
		interesting_only: bool = True,
		ax_timeout_ms: int = 300,
		semantic_mode: bool = False,
	) -> list[dict]:
		"""Return a flat list of affordances.

		- Takes a fast AX snapshot per frame (interestingOnly) with a small time budget.
		- Merges AX only when there's a clear role+name match to DOM candidates; otherwise leaves DOM-only entries.
		- If semantic_mode=True, returns canonicalized roles and essential states for LLM consumption.
		- Implements graceful timeout handling that preserves partial DOM+AX results.
		"""
		# Ensure page is relatively stable first
		await self._wait_for_page_stable()

		# Build DOM clickable elements (read-only, no overlays)
		dom_state = await self.get_clickable_elements(
			highlight_elements=False,
			focus_element=-1,
			viewport_expansion=viewport_expansion,
		)

		# Prepare DOM candidates: index -> (node, role, name)
		def _infer_role(node: DOMElementNode) -> str | None:
			attrs = node.attributes or {}
			role = (attrs.get('role') or '').strip().lower() or None
			if role:
				return role
			tag = node.tag_name.lower()
			if tag == 'a':
				return 'link'
			if tag == 'button':
				return 'button'
			if tag == 'select':
				return 'combobox'
			if tag == 'textarea':
				return 'textbox'
			if tag == 'input':
				t = (attrs.get('type') or '').strip().lower()
				if t in ('submit', 'button', 'image', 'reset'):
					return 'button'
				if t in ('checkbox',):
					return 'checkbox'
				if t in ('radio',):
					return 'radio'
				# default text-like inputs
				return 'textbox'
			if tag == 'label':
				return 'label'
			if tag == 'iframe':
				return 'iframe'
			return None

		def _infer_name(node: DOMElementNode) -> str | None:
			attrs = node.attributes or {}
			# Priority order: aria-label, alt, title, placeholder, value (for inputs), then text
			name = (
				attrs.get('aria-label')
				or attrs.get('alt')
				or attrs.get('title')
				or attrs.get('placeholder')
			)
			if not name and node.tag_name.lower() == 'input':
				val = attrs.get('value')
				if val:
					name = val
			if not name:
				# Use a shallow text sample to keep it fast
				try:
					name = node.get_all_text_till_next_clickable_element(max_depth=1)
				except Exception:
					name = None
			if name:
				name = ' '.join(str(name).split()).strip()
				return name if name else None
			return None

		def _norm(s: str | None) -> str:
			return ' '.join((s or '').split()).strip().lower()

		dom_candidates: dict[int, tuple[DOMElementNode, str | None, str | None]] = {}
		for idx, node in dom_state.selector_map.items():
			role = _infer_role(node)
			name = _infer_name(node)
			dom_candidates[idx] = (node, role, name)

		# Collect AX nodes per frame with graceful timeout handling
		ax_nodes: list[dict] = []
		ax_pairs: set[tuple[str, str]] = set()

		async def _ax_snapshot_frame(frm) -> dict | None:
			try:
				# Use the page-level accessibility with root=frame.document() when available
				root = await frm.evaluate_handle('document.documentElement')
				page_obj = getattr(self, 'page', None)
				if page_obj and getattr(page_obj, 'accessibility', None):
					return await asyncio.wait_for(page_obj.accessibility.snapshot(root=root, interesting_only=interesting_only), timeout=ax_timeout_ms / 1000)
				# Fallback to frame.accessibility if available
				if getattr(frm, 'accessibility', None):
					return await asyncio.wait_for(frm.accessibility.snapshot(interesting_only=interesting_only), timeout=ax_timeout_ms / 1000)
			except Exception:
				return None

		# Run snapshots in parallel with graceful timeout handling
		frames = list(getattr(self.page, 'frames', []))
		try:
			ax_roots = await asyncio.wait_for(
				asyncio.gather(*[_ax_snapshot_frame(f) for f in frames], return_exceptions=True),
				timeout=(ax_timeout_ms / 1000) * 1.5  # Allow some buffer for parallel execution
			)
		except asyncio.TimeoutError:
			# If we timeout, use whatever DOM data we have
			ax_roots = []
			self.logger.debug(f"AX tree collection timed out after {ax_timeout_ms}ms, proceeding with DOM-only affordances")

		# Flatten snapshot trees for role/name pairs (gracefully handle partial results)
		for root in ax_roots:
			if not isinstance(root, (dict, list)) or isinstance(root, Exception):
				continue
			stack = [root]
			while stack:
				item = stack.pop()
				if isinstance(item, dict):
					ax_nodes.append(item)
					children = item.get('children') or []
					if isinstance(children, list):
						stack.extend(children)
				elif isinstance(item, list):
					stack.extend(item)

		# Build a set of clear AX pairs (role+name) for matching
		for n in ax_nodes:
			role = n.get('role')
			name = n.get('name')
			if role and name:
				ax_pairs.add((_norm(role), _norm(name)))

		# Enhanced semantic processing if requested
		if semantic_mode:
			return self._process_semantic_affordances(dom_candidates, ax_nodes, ax_pairs)

		# Standard affordances processing
		affordances: list[dict] = []
		for idx, (node, role, name) in dom_candidates.items():
			role_n = _norm(role) if role else ''
			name_n = _norm(name) if name else ''
			merged = (role_n, name_n) in ax_pairs and role_n != '' and name_n != ''

			# Prepare coordinates as plain dicts
			vp = node.viewport_coordinates.model_dump() if node.viewport_coordinates else None
			pg = node.page_coordinates.model_dump() if node.page_coordinates else None

			# Build the final affordance item
			affordances.append({
				'index': idx,
				'role': role if merged else (role or None),
				'name': name if merged else (name or None),
				'viewport_coordinates': vp,
				'page_coordinates': pg,
			})

		return affordances

	def _process_semantic_affordances(
		self,
		dom_candidates: dict[int, tuple[DOMElementNode, str | None, str | None]],
		ax_nodes: list[dict],
		ax_pairs: set[tuple[str, str]]
	) -> list[dict]:
		"""Process affordances with semantic enhancements for LLM consumption."""

		# Role canonicalization mapping
		CANONICAL_ROLES = {
			'a': 'link', 'button': 'button', 'input[type=text]': 'textbox', 'input[type=email]': 'textbox',
			'input[type=password]': 'textbox', 'input[type=search]': 'textbox', 'input[type=checkbox]': 'checkbox',
			'input[type=radio]': 'radio', 'input[type=submit]': 'button', 'input[type=button]': 'button',
			'textarea': 'textbox', 'select': 'combobox', 'h1': 'heading', 'h2': 'heading', 'h3': 'heading',
			'nav': 'navigation', 'main': 'main', 'header': 'banner', 'footer': 'contentinfo',
			'aside': 'complementary', 'form': 'form', 'article': 'article', 'section': 'section',
			'img': 'img', 'figure': 'figure', 'table': 'table', 'ul': 'list', 'ol': 'list', 'li': 'listitem'
		}

		def _canonicalize_role(node: DOMElementNode, role: str | None) -> str:
			"""Canonicalize role for consistent LLM consumption."""
			if role and role in CANONICAL_ROLES.values():
				return role

			tag = node.tag_name.lower()
			attrs = node.attributes or {}
			input_type = attrs.get('type', '').lower()

			if tag == 'input' and input_type:
				lookup_key = f"{tag}[type={input_type}]"
				return CANONICAL_ROLES.get(lookup_key, CANONICAL_ROLES.get(tag, 'generic'))

			return CANONICAL_ROLES.get(tag, role or 'generic')

		def _extract_essential_states(node: DOMElementNode, ax_data: dict | None = None) -> dict:
			"""Extract essential states for semantic understanding."""
			states = {}
			attrs = node.attributes or {}

			# Extract from AX data (preferred)
			if ax_data:
				for state_key in ['disabled', 'checked', 'expanded', 'selected', 'pressed', 'focused']:
					if state_key in ax_data:
						states[state_key] = ax_data[state_key]

			# Supplement with DOM data
			if 'disabled' not in states:
				states['disabled'] = attrs.get('disabled') is not None
			if 'checked' not in states:
				states['checked'] = attrs.get('checked') is not None
			if 'expanded' not in states:
				states['expanded'] = attrs.get('aria-expanded') == 'true'
			if 'selected' not in states:
				states['selected'] = attrs.get('aria-selected') == 'true'
			if 'required' not in states:
				states['required'] = attrs.get('required') is not None
			if 'readonly' not in states:
				states['readonly'] = attrs.get('readonly') is not None
			if 'invalid' not in states:
				states['invalid'] = attrs.get('aria-invalid') == 'true'
			if 'hidden' not in states:
				states['hidden'] = attrs.get('hidden') is not None or attrs.get('aria-hidden') == 'true'

			# Only return non-default states to reduce noise
			return {k: v for k, v in states.items() if v}

		def _normalize_name(name: str | None) -> str:
			"""Normalize name to essential tokens."""
			if not name:
				return ""
			# Remove extra whitespace and truncate if too long
			normalized = ' '.join(name.split()).strip()
			return normalized[:50] + "..." if len(normalized) > 50 else normalized

		# Index AX elements by (role, name) for fast lookup
		ax_index: dict[tuple[str, str], dict] = {}
		for ax_elem in ax_nodes:
			role = ax_elem.get('role', '').strip().lower()
			name = ax_elem.get('name', '').strip().lower()
			if role and name:
				ax_index[(role, name)] = ax_elem

		semantic_affordances = []
		id_counter = 1

		for idx, (node, role, name) in dom_candidates.items():
			canonical_role = _canonicalize_role(node, role)
			normalized_name = _normalize_name(name)

			# Find matching AX data
			role_norm = canonical_role.lower()
			name_norm = normalized_name.lower()
			ax_data = ax_index.get((role_norm, name_norm))

			# Extract essential states
			essential_states = _extract_essential_states(node, ax_data)

			# Prepare coordinates (simplified)
			coordinates = None
			if node.viewport_coordinates:
				vp = node.viewport_coordinates.model_dump()
				coordinates = {
					'center': vp.get('center'),
					'width': vp.get('width'),
					'height': vp.get('height')
				}

			# Extract essential attributes only
			attrs = node.attributes or {}
			essential_attrs = {}
			for key in ['href', 'src', 'alt', 'title', 'value', 'placeholder', 'type', 'id']:
				if key in attrs and attrs[key]:
					val = attrs[key]
					essential_attrs[key] = val[:100] + "..." if len(str(val)) > 100 else val

			semantic_affordance = {
				'id': f"{canonical_role}_{id_counter}",
				'index': idx,  # Keep original index for action compatibility
				'role': canonical_role,
				'name': normalized_name,
			}

			# Add optional fields only if they have content
			if essential_states:
				semantic_affordance['states'] = essential_states
			if coordinates:
				semantic_affordance['coords'] = coordinates
			if essential_attrs:
				semantic_affordance['attrs'] = essential_attrs
			if ax_data:
				semantic_affordance['ax_merged'] = True

			semantic_affordances.append(semantic_affordance)
			id_counter += 1

		return semantic_affordances

	@time_execution_async('--build_dom_tree')
	async def _build_dom_tree(
		self,
		highlight_elements: bool,
		focus_element: int,
		viewport_expansion: int,
	) -> tuple[DOMElementNode, SelectorMap]:
		# Ensure the page is reasonably stable before probing
		await self._wait_for_page_stable()
		if await self.page.evaluate('1+1') != 2:
			raise ValueError('The page cannot evaluate javascript code properly')

		if is_new_tab_page(self.page.url) or self.page.url.startswith('chrome://'):
			# short-circuit if the page is a new empty tab or chrome:// page for speed, no need to inject buildDomTree.js
			return (
				DOMElementNode(
					tag_name='body',
					xpath='',
					attributes={},
					children=[],
					is_visible=False,
					parent=None,
				),
				{},
			)

		# NOTE: We execute JS code in the browser to extract important DOM information.
		#       The returned hash map contains information about the DOM tree and the
		#       relationship between the DOM elements.
		debug_mode = self.logger.getEffectiveLevel() == logging.DEBUG
		args = {
			'doHighlightElements': highlight_elements,
			'focusHighlightIndex': focus_element,
			'viewportExpansion': viewport_expansion,
			'debugMode': debug_mode,
		}

		try:
			self.logger.debug(f'🔧 Starting JavaScript DOM analysis for {self.page.url[:50]}...')
			eval_page: dict = await self.page.evaluate(self.js_code, args)
			self.logger.debug('✅ JavaScript DOM analysis completed')
		except Exception as e:
			self.logger.error('Error evaluating JavaScript: %s', e)
			raise

		# Merge DOMs from all frames (including cross-origin) to index inner iframe content
		# Always start from the map produced by the top-level document so we have a baseline even if merging fails
		main_map: dict = dict(eval_page.get('map', {}) or {})
		try:
			# Capture top-level viewport scroll and size once for coordinate normalization
			# IMPORTANT: wrap object literal to avoid SyntaxError: Unexpected token ':'
			# page.evaluate expects a function or a valid expression; an unwrapped object literal is parsed as a block.
			_top_metrics = await self.page.evaluate('() => ({ sx: window.scrollX, sy: window.scrollY, w: window.innerWidth, h: window.innerHeight })')
			top_scroll_x = int(_top_metrics.get('sx', 0))
			top_scroll_y = int(_top_metrics.get('sy', 0))
			top_vp_w = int(_top_metrics.get('w', 0))
			top_vp_h = int(_top_metrics.get('h', 0))

			main_map: dict = eval_page.get('map', {}) or {}
			# Helper to get current max id in the map
			def current_max_id(m: dict) -> int:
				try:
					return max((int(k) for k in m.keys()), default=-1)
				except Exception:
					return -1

			max_id = current_max_id(main_map)

			# Helper to get current max highlight index in the map
			def current_max_hl(m: dict) -> int:
				max_hl = -1
				for v in m.values():
					if isinstance(v, dict):
						hl = v.get('highlightIndex')
						if isinstance(hl, int) and hl > max_hl:
							max_hl = hl
				return max_hl

			max_hl_index = current_max_hl(main_map)

			# Build a quick lookup of iframe nodes in main map by (tagName, xpath)
			iframe_nodes_by_xpath: dict[str, str] = {}
			for k, v in main_map.items():
				if isinstance(v, dict) and v.get('tagName') == 'iframe' and 'xpath' in v:
					iframe_nodes_by_xpath[v['xpath']] = k

			# Small XPath builder executed in the parent context for a given iframe element
			xpath_builder_fn = """
			(el) => {
			  function getXPathTree(element) {
			    const segments = [];
			    let currentElement = element;
			    while (currentElement && currentElement.nodeType === Node.ELEMENT_NODE) {
			      let index = 0;
			      let sibling = currentElement.previousSibling;
			      while (sibling) {
			        if (sibling.nodeType === Node.ELEMENT_NODE && sibling.nodeName === currentElement.nodeName) {
			          index++;
			        }
			        sibling = sibling.previousSibling;
			      }
			      const tagName = currentElement.nodeName.toLowerCase();
			      const xpathIndex = index > 0 ? `[${index + 1}]` : "";
			      segments.unshift(`${tagName}${xpathIndex}`);
			      currentElement = currentElement.parentNode;
			    }
			    return segments.join("/");
			  }
			  return getXPathTree(el);
			}
			"""

			# Helper: resilient frame evaluation with brief retry to handle navigation/context loss
			async def _safe_frame_eval(_frame, _script: str, _arg):
				"""Evaluate in frame with small retry budget for transient 'execution context was destroyed' errors."""
				attempts = 0
				last_exc = None
				while attempts < 3:
					try:
						return await _frame.evaluate(_script, _arg)
					except Exception as e:  # Playwright may raise generic Error; keep broad but short retries
						msg = str(e).lower()
						last_exc = e
						# Retry only for common navigation/context-loss cases
						if 'execution context was destroyed' in msg or 'frame was detached' in msg or 'navigation' in msg:
							try:
								await asyncio.sleep(0.15 * (attempts + 1))
							except Exception:
								pass
							attempts += 1
							continue
						raise
				# If we exhausted retries, re-raise last exception
				raise last_exc if last_exc else RuntimeError('frame evaluate failed')

			for frame in getattr(self.page, 'frames', []):
				# Skip the main frame by URL match to avoid double-processing
				try:
					if getattr(frame, 'url', None) == self.page.url:
						continue
					# Evaluate DOM extraction inside this frame
					frame_eval: dict = await _safe_frame_eval(frame, self.js_code, args)
					# Find this frame's corresponding iframe xpath in the parent
					iframe_el = await frame.frame_element()
					iframe_xpath: str = await iframe_el.evaluate(xpath_builder_fn)
					# Locate the iframe node id in the main map
					iframe_node_id = iframe_nodes_by_xpath.get(iframe_xpath)
					if not iframe_node_id:
						# If not found, skip stitching for this frame
						continue

					# Compute cumulative iframe offset up to the top-level viewport for nested frames
					total_off_x = 0.0
					total_off_y = 0.0
					try:
						cur = frame
						while getattr(cur, 'parent_frame', None) is not None and cur.parent_frame is not None:
							# get the iframe element representing this frame in its parent
							cur_iframe_el = await cur.frame_element()
							rect = await cur_iframe_el.evaluate('(el) => { const r = el.getBoundingClientRect(); return { x: r.left, y: r.top }; }')
							total_off_x += float(rect.get('x', 0) or 0)
							total_off_y += float(rect.get('y', 0) or 0)
							cur = cur.parent_frame
					except Exception:
						pass

					# Remap this frame's map ids to avoid collisions and merge into main map
					frame_map: dict = frame_eval.get('map', {}) or {}
					if not frame_map:
						continue

					id_offset = max_id + 1
					new_map: dict = {}
					id_mapping: dict[int | str, int] = {}

					# Build id mapping first
					for old_id in frame_map.keys():
						try:
							old_int = int(old_id)
						except Exception:
							continue
						id_mapping[old_id] = old_int + id_offset

					# Remap nodes and their children lists
					for old_id, node_data in frame_map.items():
						if old_id not in id_mapping:
							continue
						new_id = id_mapping[old_id]
						nd = dict(node_data)
						children = nd.get('children', []) or []
						remapped_children = []
						for cid in children:
							try:
								remapped_children.append(id_mapping.get(cid, int(cid) + id_offset))
							except Exception:
								# If child id parsing fails, skip
								pass
						nd['children'] = remapped_children
						# Offset highlightIndex to avoid collisions with main document
						if 'highlightIndex' in nd and isinstance(nd['highlightIndex'], int):
							nd['highlightIndex'] = nd['highlightIndex'] + (max_hl_index + 1)

						# Normalize coordinates from frame space to top-level space when present
						vpc = nd.get('viewportCoordinates')
						if isinstance(vpc, dict):
							def _shift_coords(coords: dict, dx: float, dy: float) -> dict:
								def _pt(p):
									return {'x': float(p.get('x', 0)) + dx, 'y': float(p.get('y', 0)) + dy}
								return {
									'topLeft': _pt(coords.get('topLeft', {})),
									'topRight': _pt(coords.get('topRight', {})),
									'bottomLeft': _pt(coords.get('bottomLeft', {})),
									'bottomRight': _pt(coords.get('bottomRight', {})),
									'center': _pt(coords.get('center', {})),
									'width': float(coords.get('width', 0)),
									'height': float(coords.get('height', 0)),
								}

							# Step 1: move from frame viewport to top-level viewport by adding cumulative iframe offsets
							const_top_viewport = _shift_coords(vpc, total_off_x, total_off_y)
							nd['viewportCoordinates'] = const_top_viewport

							# Step 2: compute top-level page coordinates by adding top-level scroll
							nd['pageCoordinates'] = _shift_coords(const_top_viewport, float(top_scroll_x), float(top_scroll_y))

							# Replace viewportInfo with top-level metrics
							nd['viewportInfo'] = {
								'scrollX': top_scroll_x,
								'scrollY': top_scroll_y,
								'width': top_vp_w,
								'height': top_vp_h,
							}
						new_map[str(new_id)] = nd

					# Merge
					main_map.update(new_map)
					max_id = current_max_id(main_map)
					max_hl_index = current_max_hl(main_map)

					# Set iframeContentRootId on the parent iframe node to stitch trees in Python
					root_old = frame_eval.get('rootId')
					root_new = id_mapping.get(root_old)
					if root_new is not None and iframe_node_id in main_map:
						main_map[iframe_node_id]['iframeContentRootId'] = root_new
				except Exception as e:
					# If anything fails for a frame, continue without breaking main extraction
					self.logger.debug(f"Skipping frame merge due to error: {e}")

		except Exception as e:
			# If anything fails during the overall frame merge process, proceed with main document only
			self.logger.debug(f"Skipping overall frame merge due to error: {e}")

		# Assign merged map back
		eval_page['map'] = main_map

		# Only log performance metrics in debug mode
		if debug_mode and 'perfMetrics' in eval_page:
			perf = eval_page['perfMetrics']

			# Get key metrics for summary
			total_nodes = perf.get('nodeMetrics', {}).get('totalNodes', 0)
			# processed_nodes = perf.get('nodeMetrics', {}).get('processedNodes', 0)

			# Count interactive elements from the DOM map
			interactive_count = 0
			if 'map' in eval_page:
				for node_data in eval_page['map'].values():
					if isinstance(node_data, dict) and node_data.get('isInteractive'):
						interactive_count += 1

			# Create concise summary
			url_short = self.page.url[:50] + '...' if len(self.page.url) > 50 else self.page.url
			self.logger.debug(
				'🔎 Ran buildDOMTree.js interactive element detection on: %s interactive=%d/%d\n',
				url_short,
				interactive_count,
				total_nodes,
				# processed_nodes,
			)

		self.logger.debug('🔄 Starting Python DOM tree construction...')
		result = await self._construct_dom_tree(eval_page)
		self.logger.debug('✅ Python DOM tree construction completed')
		return result

	@time_execution_async('--construct_dom_tree')
	async def _construct_dom_tree(
		self,
		eval_page: dict,
	) -> tuple[DOMElementNode, SelectorMap]:
		js_node_map = eval_page['map']
		js_root_id = eval_page['rootId']

		selector_map = {}
		node_map = {}

		for id, node_data in js_node_map.items():
			node, children_ids = self._parse_node(node_data)
			if node is None:
				continue

			node_map[id] = node

			if isinstance(node, DOMElementNode) and node.highlight_index is not None:
				selector_map[node.highlight_index] = node

			# NOTE: We know that we are building the tree bottom up
			#       and all children are already processed.
			if isinstance(node, DOMElementNode):
				for child_id in children_ids:
					child_key = str(child_id)
					if child_key not in node_map:
						continue

					child_node = node_map[child_key]

					child_node.parent = node
					node.children.append(child_node)

				# Handle iframe content linking using iframeContentRootId if present
				if isinstance(node, DOMElementNode):
					raw = js_node_map.get(id, {})
					if isinstance(raw, dict) and raw.get('tagName', '').lower() == 'iframe' and 'iframeContentRootId' in raw:
						iframe_content_root_id = str(raw['iframeContentRootId'])
						if iframe_content_root_id in node_map:
							iframe_content_root_node = node_map[iframe_content_root_id]
							iframe_content_root_node.parent = node
							if iframe_content_root_node not in node.children:
								node.children.append(iframe_content_root_node)

		html_to_dict = node_map[str(js_root_id)]

		del node_map
		del js_node_map
		del js_root_id

		if html_to_dict is None or not isinstance(html_to_dict, DOMElementNode):
			raise ValueError('Failed to parse HTML to dictionary')

		return html_to_dict, selector_map

	def _parse_node(
		self,
		node_data: dict,
	) -> tuple[DOMBaseNode | None, list[int]]:
		if not node_data:
			return None, []

		# Process text nodes immediately
		if node_data.get('type') == 'TEXT_NODE':
			text_node = DOMTextNode(
				text=node_data['text'],
				is_visible=node_data['isVisible'],
				parent=None,
			)
			return text_node, []

		# Process coordinates if they exist for element nodes

		viewport_info = None
		page_coordinates = None
		viewport_coordinates = None

		def _to_coordinates_set(coords: dict | None) -> CoordinateSet | None:
			if not isinstance(coords, dict):
				return None
			def _pt(d: dict) -> Coordinates:
				return Coordinates(x=int(float(d.get('x', 0))), y=int(float(d.get('y', 0))))
			# Support both camelCase and snake_case keys
			top_left = coords.get('topLeft') or coords.get('top_left') or {}
			top_right = coords.get('topRight') or coords.get('top_right') or {}
			bottom_left = coords.get('bottomLeft') or coords.get('bottom_left') or {}
			bottom_right = coords.get('bottomRight') or coords.get('bottom_right') or {}
			center = coords.get('center') or {}
			width = int(float(coords.get('width', 0)))
			height = int(float(coords.get('height', 0)))
			return CoordinateSet(
				top_left=_pt(top_left),
				top_right=_pt(top_right),
				bottom_left=_pt(bottom_left),
				bottom_right=_pt(bottom_right),
				center=_pt(center),
				width=width,
				height=height,
			)

		vpi = node_data.get('viewportInfo') or node_data.get('viewport_info') or node_data.get('viewport')
		if isinstance(vpi, dict):
			viewport_info = ViewportInfo(
				scroll_x=int(vpi.get('scrollX') or vpi.get('scroll_x') or 0) if 'scrollX' in vpi or 'scroll_x' in vpi else None,
				scroll_y=int(vpi.get('scrollY') or vpi.get('scroll_y') or 0) if 'scrollY' in vpi or 'scroll_y' in vpi else None,
				width=int(vpi.get('width') or 0),
				height=int(vpi.get('height') or 0),
			)

		page_coordinates = _to_coordinates_set(node_data.get('pageCoordinates') or node_data.get('page_coordinates'))
		viewport_coordinates = _to_coordinates_set(node_data.get('viewportCoordinates') or node_data.get('viewport_coordinates'))

		element_node = DOMElementNode(
			tag_name=node_data['tagName'],
			xpath=node_data['xpath'],
			attributes=node_data.get('attributes', {}),
			children=[],
			is_visible=node_data.get('isVisible', False),
			is_interactive=node_data.get('isInteractive', False),
			is_top_element=node_data.get('isTopElement', False),
			is_in_viewport=node_data.get('isInViewport', False),
			highlight_index=node_data.get('highlightIndex'),
			shadow_root=node_data.get('shadowRoot', False),
			parent=None,
			viewport_info=viewport_info,
			page_coordinates=page_coordinates,
			viewport_coordinates=viewport_coordinates,
		)

		children_ids = node_data.get('children', [])

		return element_node, children_ids
