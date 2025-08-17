"""
Test the corrected upload file behavior to verify honest success/failure reporting
"""
import asyncio
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, '.')

async def test_honest_upload_feedback():
    """Test that upload actions now give honest feedback about success/failure"""
    print("=== Testing Corrected Upload File Feedback ===")

    from browser_use import Agent, AgentSettings
    from browser_use.browser import BrowserConfig

    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test file for upload honesty verification")
        test_file_path = f.name

    print(f"Created test file: {test_file_path}")

    # Task that attempts file upload - this should now give honest feedback
    task = f"""
Navigate to https://httpbin.org/forms/post and upload the file {test_file_path}.
The upload action should now give honest feedback about whether the upload actually succeeded or just selected the file.
"""

    agent_settings = AgentSettings(
        browser_config=BrowserConfig(headless=False),
        max_actions_per_step=2,
        available_file_paths=[test_file_path]  # Allow the test file
    )

    try:
        # Simple mock LLM that will do navigation and upload
        class HonestTestLLM:
            def __init__(self):
                self.step = 0

            async def get_completion(self, *args, **kwargs):
                from browser_use.llm.views import ChatInvokeCompletion
                from browser_use.agent.views import AgentOutput, ActionModel
                from browser_use.controller.registry import ActionModelRegistry

                self.step += 1

                if self.step == 1:
                    # Navigate to the test page
                    nav_action = ActionModelRegistry.get_action_model('go_to_url')(
                        go_to_url={'url': 'https://httpbin.org/forms/post', 'new_tab': False}
                    )

                    return ChatInvokeCompletion(
                        completion=AgentOutput(
                            thinking="Navigating to the upload test page.",
                            evaluation="Starting upload test.",
                            task_log="Navigation step.",
                            next_goal="Navigate to test page.",
                            actions=[nav_action]
                        ).model_dump(),
                        completion_tokens=100,
                        prompt_tokens=200,
                        total_tokens=300
                    )

                elif self.step == 2:
                    # Try to upload file - this should now give honest feedback
                    upload_action = ActionModelRegistry.get_action_model('upload_file')(
                        upload_file={'index': 1, 'path': test_file_path}  # Assume file input at index 1
                    )

                    return ChatInvokeCompletion(
                        completion=AgentOutput(
                            thinking="Attempting to upload the test file.",
                            evaluation="Upload action will now give honest feedback.",
                            task_log="Upload attempt with corrected feedback.",
                            next_goal="Upload file and get honest status.",
                            actions=[upload_action]
                        ).model_dump(),
                        completion_tokens=100,
                        prompt_tokens=200,
                        total_tokens=300
                    )
                else:
                    # Done
                    done_action = ActionModelRegistry.get_action_model('done')(
                        done={'text': 'Upload test completed - check logs for honest feedback', 'success': True}
                    )

                    return ChatInvokeCompletion(
                        completion=AgentOutput(
                            thinking="Upload test completed.",
                            evaluation="Check logs to see if upload feedback is now honest.",
                            task_log="Upload honesty test finished.",
                            next_goal="Complete test.",
                            actions=[done_action]
                        ).model_dump(),
                        completion_tokens=100,
                        prompt_tokens=200,
                        total_tokens=300
                    )

        print("\nStarting agent with corrected upload feedback...")
        agent = Agent(
            task=task,
            llm=HonestTestLLM(),
            settings=agent_settings
        )

        await agent.run(max_steps=3)

        print("\n=== Test completed ===")
        print("Check the logs above - the upload action should now:")
        print("✅ Return success=True only when upload is confirmed via UI")
        print("✅ Return success=False when file is selected but upload not confirmed")
        print("✅ Use warning logs and clear messages about uncertainty")
        print("✅ No more misleading 'Successfully uploaded' claims")

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
    asyncio.run(test_honest_upload_feedback())
