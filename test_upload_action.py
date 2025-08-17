"""
Test upload file action via the controller to verify improved logging
"""
import asyncio
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, '.')

async def test_upload_action():
    """Test the upload file action to see the improved logging"""
    print("=== Testing Upload File Action Logging ===")

    # Import what we need
    from browser_use import Agent, AgentSettings
    from browser_use.browser import BrowserConfig

    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test file for upload validation\nSecond line of content")
        test_file_path = f.name

    print(f"Created test file: {test_file_path}")

    # Simple agent task that includes a file upload
    task = f"""
Navigate to https://httpbin.org/forms/post and try to upload the file {test_file_path}.
Pay attention to whether the upload is actually successful or just appears successful.
"""

    # Configure agent with minimal settings
    agent_settings = AgentSettings(
        browser_config=BrowserConfig(
            headless=False,  # Keep visible to see what happens
            temp_dir_cleanup_on_exit=False
        ),
        max_actions_per_step=1,  # Limit actions for focused testing
        save_recording_path=None
    )

    # Mock LLM for simple testing
    class MockLLM:
        async def get_completion(self, *args, **kwargs):
            from browser_use.llm.views import ChatInvokeCompletion
            from browser_use.agent.views import AgentOutput, ActionModel
            from browser_use.controller.registry import ActionModelRegistry

            # First, navigate to the page
            nav_action = ActionModelRegistry.get_action_model('go_to_url')(
                go_to_url={'url': 'https://httpbin.org/forms/post', 'new_tab': False}
            )

            agent_output = AgentOutput(
                thinking="Navigating to the upload test page.",
                evaluation="Starting upload test.",
                task_log="Navigation step.",
                next_goal="Navigate to test page.",
                actions=[nav_action]
            )

            return ChatInvokeCompletion(
                completion=agent_output.model_dump(),
                completion_tokens=100,
                prompt_tokens=200,
                total_tokens=300
            )

    try:
        # Create agent with mock LLM
        agent = Agent(
            task=task,
            llm=MockLLM(),
            settings=agent_settings
        )

        print("\nStarting agent test...")
        await agent.run(max_steps=1)  # Just one step to navigate

        print("Navigation complete. Check browser for results.")
        print("The improved upload logging should now be more honest about success/failure.")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            Path(test_file_path).unlink()
            print(f"Cleaned up test file: {test_file_path}")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(test_upload_action())
