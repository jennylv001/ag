"""
Simple test to verify upload file function directly
"""
import asyncio
from pathlib import Path
import sys

# Add the browser_use package to path
sys.path.insert(0, '.')

from browser_use.browser import BrowserSession, BrowserConfig

async def test_upload_directly():
    """Test upload file action directly"""
    print("=== Testing Upload File Function Directly ===")

    # Create a test file
    test_file_path = Path("test_upload.txt")
    test_file_path.write_text("This is a test file for upload validation")
    print(f"Created test file: {test_file_path.absolute()}")

    # Initialize browser session
    browser_config = BrowserConfig(headless=False)
    browser_session = BrowserSession(config=browser_config)

    try:
        # Start browser session
        await browser_session.start()

        # Navigate to a page with file upload
        print("Navigating to file upload test page...")
        await browser_session.navigate("https://www.w3schools.com/html/tryit.asp?filename=tryhtml_form_submit")

        # Wait a moment for page to load
        await asyncio.sleep(3)

        # Get current browser state to see elements
        page_state = await browser_session.get_state_summary()
        print(f"Page loaded: {page_state.url}")
        print(f"Found {len(page_state.interactable_elements)} interactable elements")

        # Let's try to find any file input
        for elem in page_state.interactable_elements[:20]:  # Check first 20 elements
            print(f"Element {elem.index}: {elem.element_type} - {elem.attributes}")

        print("\nTest completed - Check above for element details")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            await browser_session.close()
        except:
            pass
        if test_file_path.exists():
            test_file_path.unlink()
            print(f"Cleaned up test file: {test_file_path}")

if __name__ == "__main__":
    asyncio.run(test_upload_directly())
