"""
Simple test to verify upload file logging accuracy
"""
import asyncio
from pathlib import Path
from browser_use import Agent, AgentSettings
from browser_use.controller.service import Controller
from browser_use.browser import BrowserSession, BrowserConfig
from browser_use.llm.llm_manager import LLMManager

async def test_upload_logging():
    """Test upload file action logging accuracy"""
    print("=== Testing Upload File Logging ===")

    # Create a test file
    test_file_path = Path("test_upload.txt")
    test_file_path.write_text("This is a test file for upload validation")
    print(f"Created test file: {test_file_path.absolute()}")

    # Initialize browser session
    browser_config = BrowserConfig(headless=False)
    browser_session = BrowserSession(config=browser_config)
    controller = Controller()

    try:
        # Start browser session
        await browser_session.start()

        # Navigate to a page with file upload
        print("Navigating to file upload test page...")
        await browser_session.go_to_url("https://httpbin.org/forms/post")

        # Get current browser state to see elements
        page_state = await browser_session.get_state_summary()
        print(f"Page loaded: {page_state.url}")
        print(f"Found {len(page_state.interactable_elements)} interactable elements")

        # Look for file input element
        file_inputs = [elem for elem in page_state.interactable_elements
                      if elem.element_type == "input" and elem.attributes.get("type") == "file"]

        if file_inputs:
            file_input = file_inputs[0]
            print(f"Found file input at index {file_input.index}")

            # Try upload file action using controller
            print("\n--- Testing Upload File Action ---")
            result = await controller.upload_file(
                browser_session=browser_session,
                index=file_input.index,
                file_paths=[str(test_file_path.absolute())]
            )

            print(f"Upload result: success={result.success}")
            print(f"Upload error: {result.error}")
            print(f"Upload extracted_content: {result.extracted_content}")

            # Check if the file input has the file selected
            updated_state = await browser_session.get_state_summary()
            updated_file_input = next((elem for elem in updated_state.interactable_elements
                                     if elem.index == file_input.index), None)
            if updated_file_input:
                print(f"Updated file input value: {updated_file_input.attributes.get('value', 'No value')}")
        else:
            print("No file input found on this page")

    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        # Cleanup
        await browser_session.close()
        if test_file_path.exists():
            test_file_path.unlink()
            print(f"Cleaned up test file: {test_file_path}")

if __name__ == "__main__":
    asyncio.run(test_upload_logging())
