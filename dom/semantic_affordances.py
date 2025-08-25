"""
Enhanced semantic affordances service for DOM/AX tree optimization.

This module provides semantic compactness improvements for LLM inference effectiveness:
- Canonicalized element roles (button, link, checkbox, textbox, etc.)
- Normalized names with essential tokens
- Essential states (disabled, checked, expanded, selected)
- DOM ↔ Accessibility Tree seamless merging with stable ID unification
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CanonicalRole(Enum):
    """Canonicalized interactive element roles for consistent LLM consumption."""
    BUTTON = "button"
    LINK = "link"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    TEXTBOX = "textbox"
    LISTBOX = "listbox"
    COMBOBOX = "combobox"
    SLIDER = "slider"
    MENU = "menu"
    MENUITEM = "menuitem"
    TAB = "tab"
    TABLIST = "tablist"
    TABPANEL = "tabpanel"
    DIALOG = "dialog"
    ALERT = "alert"
    REGION = "region"
    NAVIGATION = "navigation"
    MAIN = "main"
    BANNER = "banner"
    CONTENTINFO = "contentinfo"
    COMPLEMENTARY = "complementary"
    FORM = "form"
    SEARCH = "search"
    APPLICATION = "application"
    DOCUMENT = "document"
    IMG = "img"
    FIGURE = "figure"
    TABLE = "table"
    GRID = "grid"
    TREE = "tree"
    TREEITEM = "treeitem"
    LIST = "list"
    LISTITEM = "listitem"
    ARTICLE = "article"
    SECTION = "section"
    HEADING = "heading"
    SEPARATOR = "separator"
    PROGRESSBAR = "progressbar"
    STATUS = "status"
    TIMER = "timer"
    MARQUEE = "marquee"
    LOG = "log"
    MATH = "math"
    NOTE = "note"
    TOOLTIP = "tooltip"
    GENERIC = "generic"


@dataclass
class EssentialStates:
    """Essential element states for semantic understanding."""
    disabled: Optional[bool] = None
    checked: Optional[bool] = None
    expanded: Optional[bool] = None
    selected: Optional[bool] = None
    pressed: Optional[bool] = None
    focused: Optional[bool] = None
    required: Optional[bool] = None
    readonly: Optional[bool] = None
    invalid: Optional[bool] = None
    hidden: Optional[bool] = None

    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class SemanticAffordance:
    """Semantic affordance with canonicalized role and essential information."""
    id: str
    role: CanonicalRole
    name: str
    states: EssentialStates
    coordinates: Optional[Dict[str, Any]] = None
    attributes: Optional[Dict[str, str]] = None
    dom_id: Optional[str] = None
    ax_id: Optional[str] = None
    xpath: Optional[str] = None

    def to_compact_dict(self) -> Dict[str, Any]:
        """Convert to compact dictionary for LLM consumption."""
        result = {
            "id": self.id,
            "role": self.role.value,
            "name": self.name
        }

        states = self.states.to_dict()
        if states:
            result["states"] = states

        if self.coordinates:
            result["coords"] = self.coordinates

        # Include only essential attributes
        if self.attributes:
            essential_attrs = self._extract_essential_attributes()
            if essential_attrs:
                result["attrs"] = essential_attrs

        return result

    def _extract_essential_attributes(self) -> Dict[str, str]:
        """Extract only essential attributes for LLM understanding."""
        if not self.attributes:
            return {}

        essential_keys = {
            'href', 'src', 'alt', 'title', 'value', 'placeholder',
            'type', 'name', 'id', 'class', 'for', 'data-*'
        }

        essential = {}
        for key, value in self.attributes.items():
            if key in essential_keys or key.startswith('data-') or key.startswith('aria-'):
                # Truncate long values to prevent noise
                if isinstance(value, str) and len(value) > 100:
                    essential[key] = value[:97] + "..."
                else:
                    essential[key] = value

        return essential


class SemanticAffordancesProcessor:
    """Processes DOM and AX tree data into semantic affordances."""

    # Role mapping from various sources to canonical roles
    ROLE_MAPPINGS = {
        # HTML tag mappings
        'a': CanonicalRole.LINK,
        'button': CanonicalRole.BUTTON,
        'input[type=button]': CanonicalRole.BUTTON,
        'input[type=submit]': CanonicalRole.BUTTON,
        'input[type=reset]': CanonicalRole.BUTTON,
        'input[type=checkbox]': CanonicalRole.CHECKBOX,
        'input[type=radio]': CanonicalRole.RADIO,
        'input[type=text]': CanonicalRole.TEXTBOX,
        'input[type=email]': CanonicalRole.TEXTBOX,
        'input[type=password]': CanonicalRole.TEXTBOX,
        'input[type=search]': CanonicalRole.TEXTBOX,
        'input[type=url]': CanonicalRole.TEXTBOX,
        'input[type=tel]': CanonicalRole.TEXTBOX,
        'input[type=number]': CanonicalRole.TEXTBOX,
        'input[type=range]': CanonicalRole.SLIDER,
        'textarea': CanonicalRole.TEXTBOX,
        'select': CanonicalRole.COMBOBOX,
        'option': CanonicalRole.LISTITEM,
        'h1': CanonicalRole.HEADING,
        'h2': CanonicalRole.HEADING,
        'h3': CanonicalRole.HEADING,
        'h4': CanonicalRole.HEADING,
        'h5': CanonicalRole.HEADING,
        'h6': CanonicalRole.HEADING,
        'nav': CanonicalRole.NAVIGATION,
        'main': CanonicalRole.MAIN,
        'header': CanonicalRole.BANNER,
        'footer': CanonicalRole.CONTENTINFO,
        'aside': CanonicalRole.COMPLEMENTARY,
        'form': CanonicalRole.FORM,
        'article': CanonicalRole.ARTICLE,
        'section': CanonicalRole.SECTION,
        'img': CanonicalRole.IMG,
        'figure': CanonicalRole.FIGURE,
        'table': CanonicalRole.TABLE,
        'ul': CanonicalRole.LIST,
        'ol': CanonicalRole.LIST,
        'li': CanonicalRole.LISTITEM,
        'dialog': CanonicalRole.DIALOG,
        'hr': CanonicalRole.SEPARATOR,
        'progress': CanonicalRole.PROGRESSBAR,

        # ARIA role mappings
        'button': CanonicalRole.BUTTON,
        'link': CanonicalRole.LINK,
        'checkbox': CanonicalRole.CHECKBOX,
        'radio': CanonicalRole.RADIO,
        'textbox': CanonicalRole.TEXTBOX,
        'searchbox': CanonicalRole.TEXTBOX,
        'spinbutton': CanonicalRole.TEXTBOX,
        'listbox': CanonicalRole.LISTBOX,
        'combobox': CanonicalRole.COMBOBOX,
        'slider': CanonicalRole.SLIDER,
        'menu': CanonicalRole.MENU,
        'menubar': CanonicalRole.MENU,
        'menuitem': CanonicalRole.MENUITEM,
        'menuitemcheckbox': CanonicalRole.MENUITEM,
        'menuitemradio': CanonicalRole.MENUITEM,
        'tab': CanonicalRole.TAB,
        'tablist': CanonicalRole.TABLIST,
        'tabpanel': CanonicalRole.TABPANEL,
        'dialog': CanonicalRole.DIALOG,
        'alertdialog': CanonicalRole.DIALOG,
        'alert': CanonicalRole.ALERT,
        'region': CanonicalRole.REGION,
        'navigation': CanonicalRole.NAVIGATION,
        'main': CanonicalRole.MAIN,
        'banner': CanonicalRole.BANNER,
        'contentinfo': CanonicalRole.CONTENTINFO,
        'complementary': CanonicalRole.COMPLEMENTARY,
        'form': CanonicalRole.FORM,
        'search': CanonicalRole.SEARCH,
        'application': CanonicalRole.APPLICATION,
        'document': CanonicalRole.DOCUMENT,
        'img': CanonicalRole.IMG,
        'figure': CanonicalRole.FIGURE,
        'table': CanonicalRole.TABLE,
        'grid': CanonicalRole.GRID,
        'tree': CanonicalRole.TREE,
        'treeitem': CanonicalRole.TREEITEM,
        'list': CanonicalRole.LIST,
        'listitem': CanonicalRole.LISTITEM,
        'article': CanonicalRole.ARTICLE,
        'section': CanonicalRole.SECTION,
        'heading': CanonicalRole.HEADING,
        'separator': CanonicalRole.SEPARATOR,
        'progressbar': CanonicalRole.PROGRESSBAR,
        'status': CanonicalRole.STATUS,
        'timer': CanonicalRole.TIMER,
        'marquee': CanonicalRole.MARQUEE,
        'log': CanonicalRole.LOG,
        'math': CanonicalRole.MATH,
        'note': CanonicalRole.NOTE,
        'tooltip': CanonicalRole.TOOLTIP,
        'generic': CanonicalRole.GENERIC,
    }

    def __init__(self):
        self.id_counter = 0
        self.unified_ids: Dict[Tuple[str, str], str] = {}  # (role, name) -> unified_id

    def process_affordances(
        self,
        dom_elements: List[Dict[str, Any]],
        ax_elements: Optional[List[Dict[str, Any]]] = None
    ) -> List[SemanticAffordance]:
        """
        Process DOM and AX elements into unified semantic affordances.

        Strategy:
        1. Prefer Accessibility Tree (AX) for semantic information
        2. Use DOM for geometry & attributes if AX is missing
        3. Merge based on stable ID unification (role + name matching)
        """
        unified_elements = self._merge_dom_ax_trees(dom_elements, ax_elements or [])
        affordances = []

        for element in unified_elements:
            affordance = self._create_semantic_affordance(element)
            if affordance:
                affordances.append(affordance)

        return affordances

    def _merge_dom_ax_trees(
        self,
        dom_elements: List[Dict[str, Any]],
        ax_elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge DOM and AX trees with preference for AX semantic data."""

        # Index AX elements by (role, name) for fast lookup
        ax_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for ax_elem in ax_elements:
            role = self._normalize_text(ax_elem.get('role', ''))
            name = self._normalize_text(ax_elem.get('name', ''))
            if role and name:
                ax_index[(role, name)] = ax_elem

        unified = []
        processed_ax_keys: Set[Tuple[str, str]] = set()

        # Process DOM elements and merge with AX when available
        for dom_elem in dom_elements:
            dom_role = self._infer_canonical_role_from_dom(dom_elem)
            dom_name = self._extract_name_from_dom(dom_elem)

            if not dom_role:
                continue

            role_norm = self._normalize_text(dom_role.value)
            name_norm = self._normalize_text(dom_name)

            key = (role_norm, name_norm)
            ax_elem = ax_index.get(key) if name_norm else None

            if ax_elem:
                # Merge: AX for semantics, DOM for geometry
                merged = self._merge_element_data(dom_elem, ax_elem)
                processed_ax_keys.add(key)
            else:
                # DOM-only element
                merged = {
                    'source': 'dom',
                    'role': dom_role.value,
                    'name': dom_name,
                    'dom_data': dom_elem,
                    'ax_data': None
                }

            unified.append(merged)

        # Add remaining AX-only elements
        for key, ax_elem in ax_index.items():
            if key not in processed_ax_keys:
                role, name = key
                unified.append({
                    'source': 'ax',
                    'role': role,
                    'name': name,
                    'dom_data': None,
                    'ax_data': ax_elem
                })

        return unified

    def _merge_element_data(
        self,
        dom_elem: Dict[str, Any],
        ax_elem: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge DOM and AX element data, preferring AX for semantics."""
        return {
            'source': 'merged',
            'role': ax_elem.get('role', self._infer_canonical_role_from_dom(dom_elem).value),
            'name': ax_elem.get('name', self._extract_name_from_dom(dom_elem)),
            'dom_data': dom_elem,
            'ax_data': ax_elem
        }

    def _create_semantic_affordance(self, element: Dict[str, Any]) -> Optional[SemanticAffordance]:
        """Create a semantic affordance from unified element data."""
        role_str = element.get('role', '')
        name = element.get('name', '')

        if not role_str:
            return None

        # Get canonical role
        canonical_role = self._get_canonical_role(role_str)
        if not canonical_role:
            return None

        # Generate stable unified ID
        unified_id = self._get_unified_id(canonical_role.value, name)

        # Extract states from both DOM and AX
        states = self._extract_essential_states(element)

        # Extract coordinates (prefer DOM)
        coordinates = self._extract_coordinates(element)

        # Extract attributes (from DOM)
        attributes = self._extract_attributes(element)

        # Extract IDs for traceability
        dom_id = self._extract_dom_id(element)
        ax_id = self._extract_ax_id(element)
        xpath = self._extract_xpath(element)

        return SemanticAffordance(
            id=unified_id,
            role=canonical_role,
            name=self._normalize_name(name),
            states=states,
            coordinates=coordinates,
            attributes=attributes,
            dom_id=dom_id,
            ax_id=ax_id,
            xpath=xpath
        )

    def _get_unified_id(self, role: str, name: str) -> str:
        """Generate stable unified ID for element based on role and name."""
        key = (role, self._normalize_text(name))

        if key not in self.unified_ids:
            self.id_counter += 1
            self.unified_ids[key] = f"{role}_{self.id_counter}"

        return self.unified_ids[key]

    def _get_canonical_role(self, role_str: str) -> Optional[CanonicalRole]:
        """Map role string to canonical role."""
        normalized = role_str.lower().strip()

        # Direct mapping
        if normalized in [r.value for r in CanonicalRole]:
            return CanonicalRole(normalized)

        # Lookup in mappings
        return self.ROLE_MAPPINGS.get(normalized, CanonicalRole.GENERIC)

    def _infer_canonical_role_from_dom(self, dom_elem: Dict[str, Any]) -> Optional[CanonicalRole]:
        """Infer canonical role from DOM element."""
        # Check explicit role attribute first
        attrs = dom_elem.get('attributes', {})
        role = attrs.get('role')
        if role:
            canonical = self._get_canonical_role(role)
            if canonical and canonical != CanonicalRole.GENERIC:
                return canonical

        # Infer from tag name and type
        tag = dom_elem.get('tagName', '').lower()
        input_type = attrs.get('type', '').lower()

        if tag == 'input' and input_type:
            lookup_key = f"{tag}[type={input_type}]"
            return self.ROLE_MAPPINGS.get(lookup_key, self.ROLE_MAPPINGS.get(tag))

        return self.ROLE_MAPPINGS.get(tag, CanonicalRole.GENERIC)

    def _extract_name_from_dom(self, dom_elem: Dict[str, Any]) -> str:
        """Extract name from DOM element using priority order."""
        attrs = dom_elem.get('attributes', {})

        # Priority order for name extraction
        name_sources = [
            attrs.get('aria-label'),
            attrs.get('aria-labelledby'),  # Would need resolution
            attrs.get('alt'),
            attrs.get('title'),
            attrs.get('placeholder'),
            attrs.get('value') if dom_elem.get('tagName', '').lower() == 'input' else None,
            dom_elem.get('text', ''),  # Simplified text content
        ]

        for source in name_sources:
            if source and isinstance(source, str) and source.strip():
                return source.strip()

        return ""

    def _extract_essential_states(self, element: Dict[str, Any]) -> EssentialStates:
        """Extract essential states from DOM and AX data."""
        states = EssentialStates()

        # Extract from AX data (preferred)
        ax_data = element.get('ax_data')
        if ax_data:
            states.disabled = ax_data.get('disabled')
            states.checked = ax_data.get('checked')
            states.expanded = ax_data.get('expanded')
            states.selected = ax_data.get('selected')
            states.pressed = ax_data.get('pressed')
            states.focused = ax_data.get('focused')

        # Supplement with DOM data
        dom_data = element.get('dom_data')
        if dom_data:
            attrs = dom_data.get('attributes', {})

            if states.disabled is None:
                states.disabled = attrs.get('disabled') is not None
            if states.checked is None:
                states.checked = attrs.get('checked') is not None
            if states.expanded is None:
                states.expanded = attrs.get('aria-expanded') == 'true'
            if states.selected is None:
                states.selected = attrs.get('aria-selected') == 'true'
            if states.pressed is None:
                states.pressed = attrs.get('aria-pressed') == 'true'
            if states.required is None:
                states.required = attrs.get('required') is not None
            if states.readonly is None:
                states.readonly = attrs.get('readonly') is not None
            if states.invalid is None:
                states.invalid = attrs.get('aria-invalid') == 'true'
            if states.hidden is None:
                states.hidden = attrs.get('hidden') is not None or attrs.get('aria-hidden') == 'true'

        return states

    def _extract_coordinates(self, element: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract coordinates from DOM data."""
        dom_data = element.get('dom_data')
        if not dom_data:
            return None

        viewport_coords = dom_data.get('viewportCoordinates')
        if viewport_coords:
            # Return simplified coordinate structure
            return {
                'center': viewport_coords.get('center'),
                'width': viewport_coords.get('width'),
                'height': viewport_coords.get('height')
            }

        return None

    def _extract_attributes(self, element: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Extract essential attributes from DOM data."""
        dom_data = element.get('dom_data')
        if not dom_data:
            return None

        return dom_data.get('attributes', {})

    def _extract_dom_id(self, element: Dict[str, Any]) -> Optional[str]:
        """Extract DOM element ID for traceability."""
        dom_data = element.get('dom_data')
        if dom_data:
            return dom_data.get('id') or dom_data.get('highlightIndex')
        return None

    def _extract_ax_id(self, element: Dict[str, Any]) -> Optional[str]:
        """Extract AX element ID for traceability."""
        ax_data = element.get('ax_data')
        if ax_data:
            return ax_data.get('nodeId') or ax_data.get('backendNodeId')
        return None

    def _extract_xpath(self, element: Dict[str, Any]) -> Optional[str]:
        """Extract XPath from DOM data."""
        dom_data = element.get('dom_data')
        if dom_data:
            return dom_data.get('xpath')
        return None

    def _normalize_name(self, name: str) -> str:
        """Normalize name to essential tokens."""
        if not name:
            return ""

        # Remove extra whitespace and normalize
        normalized = re.sub(r'\s+', ' ', name.strip())

        # Truncate if too long (keep it concise for LLM)
        if len(normalized) > 50:
            normalized = normalized[:47] + "..."

        return normalized

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text.lower().strip())
