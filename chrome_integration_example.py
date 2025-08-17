"""
Example: Using your own Chrome browser with browser-use Agent

This demonstrates different ways to connect the Agent to your existing Chrome installation.
"""
import asyncio
from browser_use import Agent, AgentSettings
from browser_use.browser import BrowserConfig

async def use_your_chrome_example():
    """Example of connecting Agent to your own Chrome browser"""

    print("=== Using Your Own Chrome Browser with Agent ===\n")

    # Option 1: Use your Chrome installation with custom profile
    print("Option 1: Custom Chrome executable and profile")
    agent_settings_custom = AgentSettings(
        browser_config=BrowserConfig(
            # Point to your Chrome installation
            executable_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",

            # Use your existing Chrome profile directory (be careful with this!)
            # user_data_dir="C:\\Users\\HeadOffice\\AppData\\Local\\Google\\Chrome\\User Data",

            # Better: Create a dedicated profile for automation
            user_data_dir="C:\\Users\\HeadOffice\\Desktop\\B\\Dir\\AgentChrome",

            # Browser settings
            headless=False,  # Keep visible
            keep_alive=True,  # Don't close when done
            devtools=True,   # Open DevTools

            # Optional: Custom Chrome arguments
            args=[
                "--start-maximized",
                "--disable-web-security",  # Only if needed
            ]
        ),
        max_actions_per_step=2,
    )

    print("Configuration created for custom Chrome setup.")
    print("Chrome will use:")
    print(f"  - Executable: {agent_settings_custom.browser_config.executable_path}")
    print(f"  - Profile: {agent_settings_custom.browser_config.user_data_dir}")
    print(f"  - Keep alive: {agent_settings_custom.browser_config.keep_alive}")
    print()

    # Option 2: Connect to already running Chrome
    print("Option 2: Connect to running Chrome via CDP")
    print("To use this option:")
    print("1. Start Chrome with: chrome.exe --remote-debugging-port=9222")
    print("2. Then use this configuration:")

    agent_settings_cdp = AgentSettings(
        browser_config=BrowserConfig(
            cdp_url="http://localhost:9222",  # Connect to running Chrome
            headless=False,
        )
    )
    print(f"  - CDP URL: {agent_settings_cdp.browser_config.cdp_url}")
    print()

    # Mock LLM for demonstration
    class DemoLLM:
        async def get_completion(self, *args, **kwargs):
            from browser_use.llm.views import ChatInvokeCompletion
            from browser_use.agent.views import AgentOutput, ActionModel
            from browser_use.controller.registry import ActionModelRegistry

            # Simple navigation task
            nav_action = ActionModelRegistry.get_action_model('go_to_url')(
                go_to_url={'url': 'https://www.google.com', 'new_tab': False}
            )

            return ChatInvokeCompletion(
                completion=AgentOutput(
                    thinking="Navigating to Google using your Chrome browser.",
                    evaluation="Testing custom Chrome connection.",
                    task_log="Demo navigation.",
                    next_goal="Navigate to Google.",
                    actions=[nav_action]
                ).model_dump(),
                completion_tokens=100,
                prompt_tokens=200,
                total_tokens=300
            )

    # Test the custom Chrome setup
    try:
        print("Testing custom Chrome setup...")

        agent = Agent(
            task="Navigate to https://www.google.com using my Chrome browser",
            llm=DemoLLM(),
            settings=agent_settings_custom
        )

        print("Agent created successfully with custom Chrome configuration!")
        print("You can now run: await agent.run(max_steps=1)")
        print()

        # Uncomment to actually run:
        # await agent.run(max_steps=1)

    except Exception as e:
        print(f"Error setting up agent: {e}")
        print("Make sure Chrome path and profile directory are correct.")

    print("=== Key Benefits of Using Your Own Chrome ===")
    print("✅ Maintain your existing cookies and login sessions")
    print("✅ Use your installed extensions")
    print("✅ Preserve browsing history and preferences")
    print("✅ Better stealth (looks like normal browsing)")
    print("✅ Reuse existing Chrome configuration")

if __name__ == "__main__":
    asyncio.run(use_your_chrome_example())
