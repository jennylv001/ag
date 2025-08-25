"""
Test the enhanced get_affordances method with semantic mode.
"""

import asyncio
from unittest.mock import Mock, AsyncMock
from browser_use.dom.service import DomService
from browser_use.dom.views import DOMElementNode, SelectorMap
from browser_use.dom.history_tree_processor.view import CoordinateSet


async def test_enhanced_affordances():
    """Test enhanced affordances with semantic mode."""

    # Mock page and browser objects
    mock_page = Mock()
    mock_page.url = "https://example.com"
    mock_page.frames = []
    mock_page.evaluate = AsyncMock(return_value=2)  # For page stability check
    mock_page.wait_for_load_state = AsyncMock()

    # Create test DOM elements
    button_element = DOMElementNode(
        tag_name="button",
        xpath="//button[@id='submit']",
        attributes={"id": "submit", "type": "submit", "class": "btn-primary"},
        children=[],
        is_visible=True,
        parent=None,
        viewport_coordinates=CoordinateSet(
            top_left={"x": 60, "y": 185},
            top_right={"x": 140, "y": 185},
            bottom_left={"x": 60, "y": 215},
            bottom_right={"x": 140, "y": 215},
            center={"x": 100, "y": 200},
            width=80,
            height=30
        )
    )
    button_element.get_all_text_till_next_clickable_element = Mock(return_value="Submit Form")

    input_element = DOMElementNode(
        tag_name="input",
        xpath="//input[@id='email']",
        attributes={"id": "email", "type": "email", "placeholder": "Enter email", "required": ""},
        children=[],
        is_visible=True,
        parent=None,
        viewport_coordinates=CoordinateSet(
            top_left={"x": 125, "y": 135},
            top_right={"x": 275, "y": 135},
            bottom_left={"x": 125, "y": 165},
            bottom_right={"x": 275, "y": 165},
            center={"x": 200, "y": 150},
            width=150,
            height=30
        )
    )
    input_element.get_all_text_till_next_clickable_element = Mock(return_value="")

    # Create mock selector map
    selector_map = SelectorMap({1: button_element, 2: input_element})

    # Create DOM service
    dom_service = DomService(mock_page)

    # Mock the get_clickable_elements method
    mock_dom_state = Mock()
    mock_dom_state.selector_map = selector_map
    dom_service.get_clickable_elements = AsyncMock(return_value=mock_dom_state)

    print("Testing Standard Affordances Mode...")
    print("=" * 50)

    # Test standard affordances
    standard_affordances = await dom_service.get_affordances(
        viewport_expansion=0,
        interesting_only=True,
        ax_timeout_ms=100,  # Short timeout for test
        semantic_mode=False
    )

    for affordance in standard_affordances:
        print(f"Index: {affordance['index']}")
        print(f"Role: {affordance['role']}")
        print(f"Name: {affordance['name']}")
        print(f"Coordinates: {affordance['viewport_coordinates']}")
        print("-" * 30)

    print("\nTesting Semantic Affordances Mode...")
    print("=" * 50)

    # Test semantic affordances
    semantic_affordances = await dom_service.get_affordances(
        viewport_expansion=0,
        interesting_only=True,
        ax_timeout_ms=100,  # Short timeout for test
        semantic_mode=True
    )

    for affordance in semantic_affordances:
        print(f"ID: {affordance['id']}")
        print(f"Index: {affordance['index']}")
        print(f"Role: {affordance['role']}")
        print(f"Name: {affordance['name']}")
        if 'states' in affordance:
            print(f"States: {affordance['states']}")
        if 'coords' in affordance:
            print(f"Coordinates: {affordance['coords']}")
        if 'attrs' in affordance:
            print(f"Attributes: {affordance['attrs']}")
        print("-" * 30)

    print("\nTesting Timeout Handling...")
    print("=" * 50)

    # Test timeout handling by setting very short timeout
    timeout_affordances = await dom_service.get_affordances(
        viewport_expansion=0,
        interesting_only=True,
        ax_timeout_ms=1,  # Very short timeout to force timeout
        semantic_mode=True
    )

    print(f"Retrieved {len(timeout_affordances)} affordances despite AX timeout")
    for affordance in timeout_affordances:
        has_ax_merged = affordance.get('ax_merged', False)
        print(f"ID: {affordance['id']}, AX Merged: {has_ax_merged}")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_enhanced_affordances())
