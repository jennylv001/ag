import hashlib
import json
from typing import Dict, List, Set

from browser_use.dom.views import DOMElementNode, DOMBaseNode


class ClickableElementProcessor:
	@staticmethod
	def _hash_string(text: str) -> str:
		return hashlib.sha256(text.encode('utf-8')).hexdigest()

	@staticmethod
	def _get_parent_branch_path_str(dom_element: DOMElementNode) -> str:
		parents_tags: List[str] = []
		current_element: DOMElementNode | None = dom_element.parent
		while current_element is not None:
			parents_tags.append(current_element.tag_name.lower())
			current_element = current_element.parent
		parents_tags.reverse()
		return '/'.join(parents_tags)

	@staticmethod
	def get_element_feature_hashes(dom_element: DOMElementNode) -> Dict[str, str]:
		if not dom_element or not isinstance(dom_element, DOMElementNode):
			return {}

		features: Dict[str, str] = {}

		# 1. Tag Name Hash
		features['tag_name'] = ClickableElementProcessor._hash_string(dom_element.tag_name.lower())

		# 2. Core Attributes Hash
		core_attr_keys = ['id', 'name', 'data-testid', 'data-cy', 'data-qa', 'data-test-id', 'automation-id', 'for', 'label']
		core_attrs_payload = {}
		if dom_element.attributes:
			for k in core_attr_keys:
				attr_val = dom_element.attributes.get(k)
				if attr_val:
					core_attrs_payload[k] = attr_val
		features['core_attrs'] = ClickableElementProcessor._hash_string(
			json.dumps(core_attrs_payload, sort_keys=True)
		)
		stable_core_identifier_exists = bool(core_attrs_payload)

		# 3. Semantic Attributes Hash
		semantic_attr_keys = ['role', 'type', 'aria-label', 'aria-labelledby', 'aria-describedby', 'alt', 'title', 'placeholder', 'href', 'action', 'method', 'value']
		semantic_attrs_payload = {}
		if dom_element.attributes:
			for k in semantic_attr_keys:
				attr_val = dom_element.attributes.get(k)
				if attr_val is not None:  # include empty string cases
					semantic_attrs_payload[k] = attr_val
		features['semantic_attrs'] = ClickableElementProcessor._hash_string(
			json.dumps(semantic_attrs_payload, sort_keys=True)
		)

		# 4. Structural Hashes
		parent_branch_str = ClickableElementProcessor._get_parent_branch_path_str(dom_element)
		features['parent_branch_hash'] = ClickableElementProcessor._hash_string(parent_branch_str)
		structural_info_full = f"parent_branch:{parent_branch_str}|xpath_self:{dom_element.xpath}"
		features['structure_full_hash'] = ClickableElementProcessor._hash_string(structural_info_full)

		# 5. Text Sample Hash (normalize whitespace and lowercase for stability)
		text_content = (dom_element.get_all_text_till_next_clickable_element(max_depth=1) or '')
		text_norm = ' '.join(text_content.lower().split())[:100]
		features['text_sample'] = ClickableElementProcessor._hash_string(text_norm)

		# 6. Class Names Hash
		class_names_str = ''
		if dom_element.attributes:
			class_attr_val = dom_element.attributes.get('class')
			if class_attr_val:
				classes = sorted(filter(None, class_attr_val.lower().split()))
				class_names_str = ' '.join(classes)
		features['class_names_hash'] = ClickableElementProcessor._hash_string(class_names_str)

		# 7. Sibling Index Hash
		sibling_index_str = f"{dom_element.tag_name.lower()}_root"
		if dom_element.parent:
			count_preceding = 0
			for sibling in dom_element.parent.children:
				if sibling is dom_element:
					break
				if isinstance(sibling, DOMElementNode) and sibling.tag_name.lower() == dom_element.tag_name.lower():
					count_preceding += 1
			sibling_index_str = f"{dom_element.tag_name.lower()}_{count_preceding}"
		features['sibling_index_hash'] = ClickableElementProcessor._hash_string(sibling_index_str)

		# 8. Primary Adaptive Hash (compose)
		parts: List[str] = [f"tag:{features['tag_name']}", f"core_attrs:{features['core_attrs']}"]
		if stable_core_identifier_exists:
			parts.append(f"parent_branch:{features['parent_branch_hash']}")
		else:
			parts.append(f"struct_full:{features['structure_full_hash']}")
			parts.append(f"sibling_idx:{features['sibling_index_hash']}")
		parts.append(f"text_sample:{features['text_sample']}")
		parts.append(f"semantic_attrs:{features['semantic_attrs']}")
		parts.append(f"classes:{features['class_names_hash']}")

		primary_input = "|".join(sorted(parts))
		features['primary_adaptive_hash'] = ClickableElementProcessor._hash_string(primary_input)

		return features

	@staticmethod
	def hash_dom_element(dom_element: DOMElementNode) -> str:
		feature_hashes = ClickableElementProcessor.get_element_feature_hashes(dom_element)
		return feature_hashes.get('primary_adaptive_hash', ClickableElementProcessor._hash_string(str(dom_element)))

	@staticmethod
	def get_clickable_elements(dom_element: DOMElementNode) -> list[DOMElementNode]:
		"""Depth-first traversal to collect elements with a highlight_index."""
		result: List[DOMElementNode] = []
		for child in dom_element.children:
			if isinstance(child, DOMElementNode):
				if child.highlight_index is not None:
					result.append(child)
				result.extend(ClickableElementProcessor.get_clickable_elements(child))
		return result

	@staticmethod
	def get_clickable_elements_hashes(dom_element: DOMElementNode) -> Set[str]:
		clickable_elements = ClickableElementProcessor.get_clickable_elements(dom_element)
		return {ClickableElementProcessor.hash_dom_element(el) for el in clickable_elements}
