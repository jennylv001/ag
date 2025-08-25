"""Test script for semantic affordances functionality."""

import asyncio
import json
from browser_use.dom.semantic_affordances import SemanticAffordancesProcessor, CanonicalRole, EssentialStates

def test_semantic_affordances():
    """Test the semantic affordances processor."""
    processor = SemanticAffordancesProcessor()

    # Sample DOM elements
    dom_elements = [
        {
            'tagName': 'button',
            'attributes': {'id': 'submit-btn', 'class': 'btn-primary'},
            'viewportCoordinates': {'center': {'x': 100, 'y': 200}, 'width': 80, 'height': 30},
            'text': 'Submit',
            'xpath': '//button[@id="submit-btn"]'
        },
        {
            'tagName': 'input',
            'attributes': {'type': 'text', 'placeholder': 'Enter name', 'required': 'true'},
            'viewportCoordinates': {'center': {'x': 200, 'y': 150}, 'width': 150, 'height': 25},
            'text': '',
            'xpath': '//input[@type="text"]'
        },
        {
            'tagName': 'a',
            'attributes': {'href': 'https://example.com', 'title': 'Visit Example'},
            'viewportCoordinates': {'center': {'x': 300, 'y': 100}, 'width': 100, 'height': 20},
            'text': 'Example Link',
            'xpath': '//a[@href="https://example.com"]'
        }
    ]

    # Sample AX elements
    ax_elements = [
        {
            'role': 'button',
            'name': 'Submit',
            'disabled': False,
            'focused': False
        },
        {
            'role': 'textbox',
            'name': 'Enter name',
            'required': True,
            'invalid': False
        }
    ]

    # Process affordances
    affordances = processor.process_affordances(dom_elements, ax_elements)

    print("Generated Semantic Affordances:")
    print("=" * 50)

    for affordance in affordances:
        compact = affordance.to_compact_dict()
        print(f"ID: {compact['id']}")
        print(f"Role: {compact['role']}")
        print(f"Name: {compact['name']}")
        if 'states' in compact:
            print(f"States: {compact['states']}")
        if 'coords' in compact:
            print(f"Coordinates: {compact['coords']}")
        if 'attrs' in compact:
            print(f"Attributes: {compact['attrs']}")
        print("-" * 30)

    # Test role canonicalization
    print("\nRole Canonicalization Tests:")
    print("=" * 50)

    test_roles = ['button', 'link', 'textbox', 'input[type=checkbox]', 'select', 'h1']
    for role in test_roles:
        canonical = processor._get_canonical_role(role)
        print(f"{role} -> {canonical.value if canonical else 'None'}")

if __name__ == "__main__":
    test_semantic_affordances()
