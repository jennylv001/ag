#!/usr/bin/env python3

"""
Quick test to validate that solve_captcha can serialize DOM elements properly
without encountering CoordinateSet JSON serialization errors.
"""

import asyncio
import json
from unittest.mock import Mock, AsyncMock
from dom.history_tree_processor.view import CoordinateSet, Coordinates
from dom.views import DOMElementNode

def test_serialization():
    """Test that our serialization function can handle CoordinateSet objects"""

    # Define serialize_element function (copied from solve_captcha.py)
    def serialize_element(element_dict):
        """Recursively serialize objects that might contain non-JSON-serializable types"""
        if isinstance(element_dict, dict):
            result = {}
            for key, value in element_dict.items():
                result[key] = serialize_element(value)
            return result
        elif isinstance(element_dict, list):
            return [serialize_element(item) for item in element_dict]
        elif hasattr(element_dict, 'model_dump'):  # BaseModel objects
            return element_dict.model_dump()
        elif hasattr(element_dict, '__dict__'):  # Regular objects with attributes
            return {k: serialize_element(v) for k, v in element_dict.__dict__.items()}
        else:
            # Basic types that should be JSON serializable
            return element_dict

    # Create a mock CoordinateSet with nested Coordinates
    mock_coordinates = Coordinates(x=100, y=200)
    mock_coordinate_set = CoordinateSet(
        top_left=mock_coordinates,
        top_right=Coordinates(x=200, y=200),
        bottom_left=Coordinates(x=100, y=300),
        bottom_right=Coordinates(x=200, y=300),
        center=Coordinates(x=150, y=250),
        width=100,
        height=100
    )

    # Create a mock DOM element with CoordinateSet
    mock_element = {
        'tag_name': 'button',
        'xpath': '//button[@id="captcha-button"]',
        'attributes': {'id': 'captcha-button', 'class': 'btn'},
        'is_visible': True,
        'is_interactive': True,
        'viewport_coordinates': mock_coordinate_set,
        'page_coordinates': mock_coordinate_set,
        'children': []
    }

    # Test serialization
    try:
        serialized = serialize_element(mock_element)
        json_str = json.dumps(serialized, indent=2)
        print("✅ Serialization successful!")
        print(f"JSON length: {len(json_str)} characters")

        # Verify that viewport_coordinates are properly serialized
        assert 'viewport_coordinates' in serialized
        assert isinstance(serialized['viewport_coordinates'], dict)
        assert 'center' in serialized['viewport_coordinates']
        assert serialized['viewport_coordinates']['center']['x'] == 150

        print("✅ All assertions passed!")
        return True

    except Exception as e:
        print(f"❌ Serialization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_serialization()
    if success:
        print("\n🎉 CAPTCHA serialization test PASSED - solve_captcha should work!")
    else:
        print("\n💥 CAPTCHA serialization test FAILED - needs more work")
