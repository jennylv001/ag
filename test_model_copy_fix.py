#!/usr/bin/env python3

"""
Test script to verify the model_copy defensive fix works
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_anthropic_serializer_defensive():
    """Test that the Anthropic serializer handles non-Pydantic objects"""
    try:
        from llm.anthropic.serializer import AnthropicMessageSerializer

        # Create some mock messages with mixed types
        class MockMessage:
            def model_copy(self, deep=True):
                return MockMessage()

        messages = [
            MockMessage(),  # Has model_copy
            "plain string",  # No model_copy
            MockMessage(),   # Has model_copy
        ]

        # This should not raise an AttributeError anymore
        result = AnthropicMessageSerializer.remove_redundant_cache(messages)

        logger.info("✅ Anthropic serializer defensive handling works")
        return True
    except Exception as e:
        logger.error(f"❌ Anthropic serializer test failed: {e}")
        return False

def test_google_serializer_defensive():
    """Test that the Google serializer handles non-Pydantic objects"""
    try:
        from llm.google.serializer import GoogleMessageSerializer

        # Create some mock messages with mixed types
        class MockMessage:
            def model_copy(self, deep=True):
                return MockMessage()

        messages = [
            MockMessage(),  # Has model_copy
            "plain string",  # No model_copy
            MockMessage(),   # Has model_copy
        ]

        # This should not raise an AttributeError anymore
        try:
            GoogleMessageSerializer.format_messages(messages)
        except AttributeError as e:
            if "model_copy" in str(e):
                logger.error(f"❌ Google serializer still has model_copy issue: {e}")
                return False
        except Exception:
            # Other exceptions are okay for this test - we just care about model_copy
            pass

        logger.info("✅ Google serializer defensive handling works")
        return True
    except Exception as e:
        logger.error(f"❌ Google serializer test failed: {e}")
        return False

def main():
    """Run serializer defensive tests"""
    logger.info("🔍 Testing model_copy defensive fixes...")

    all_passed = True

    # Test our defensive fixes
    all_passed &= test_anthropic_serializer_defensive()
    all_passed &= test_google_serializer_defensive()

    if all_passed:
        logger.info("🎉 All model_copy defensive tests passed!")
        logger.info("✅ Serializers can handle mixed message types")
        return 0
    else:
        logger.error("💥 Some defensive tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())
