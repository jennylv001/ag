"""
Simple example showing how to configure Agent to use your own Chrome browser
"""

def show_chrome_configuration_options():
    """Show different ways to configure Agent to use your Chrome browser"""

    print("=" * 70)
    print("HOW TO USE YOUR OWN CHROME BROWSER WITH BROWSER-USE AGENT")
    print("=" * 70)
    print()

    print("🔧 METHOD 1: Use Your Chrome Installation")
    print("-" * 40)
    print("""
from browser_use import Agent, AgentSettings
from browser_use.browser import BrowserConfig

agent_settings = AgentSettings(
    browser_config=BrowserConfig(
        executable_path="C:\\\\Program Files\\\\Google\\\\Chrome\\\\Application\\\\chrome.exe",
        user_data_dir="C:\\\\Users\\\\YourName\\\\Desktop\\\\ChromeProfile",  # Custom profile
        headless=False,        # Keep visible
        keep_alive=True,       # Don't close when done
        devtools=True,         # Open DevTools
    )
)

agent = Agent(task="Your task", llm=your_llm, settings=agent_settings)
""")

    print("🔧 METHOD 2: Connect to Running Chrome")
    print("-" * 40)
    print("""
# First, start Chrome with debugging:
# chrome.exe --remote-debugging-port=9222

agent_settings = AgentSettings(
    browser_config=BrowserConfig(
        cdp_url="http://localhost:9222",  # Connect to existing Chrome
    )
)
""")

    print("🔧 METHOD 3: Connect by Process ID")
    print("-" * 40)
    print("""
# Find Chrome process ID first
import psutil
chrome_pids = [p.info['pid'] for p in psutil.process_iter(['pid', 'name'])
               if 'chrome' in p.info['name'].lower()]

agent_settings = AgentSettings(
    browser_config=BrowserConfig(
        browser_pid=chrome_pids[0],  # Use first Chrome process
    )
)
""")

    print("📋 COMPLETE WORKING EXAMPLE:")
    print("-" * 40)
    print("""
import asyncio
from browser_use import Agent, AgentSettings
from browser_use.browser import BrowserConfig

async def use_my_chrome():
    # Configure to use your Chrome
    settings = AgentSettings(
        browser_config=BrowserConfig(
            executable_path="C:\\\\Program Files\\\\Google\\\\Chrome\\\\Application\\\\chrome.exe",
            user_data_dir="C:\\\\temp\\\\agent_chrome",  # Dedicated profile
            headless=False,
            keep_alive=True,
        )
    )

    agent = Agent(
        task="Navigate to Google and search for 'browser automation'",
        llm=your_llm_instance,
        settings=settings
    )

    await agent.run()

# Run it
asyncio.run(use_my_chrome())
""")

    print("✅ BENEFITS:")
    print("- Keep your existing cookies and sessions")
    print("- Use your installed Chrome extensions")
    print("- Maintain browsing history and preferences")
    print("- Better stealth (looks like normal browsing)")
    print("- Full control over Chrome configuration")
    print()

    print("⚠️  IMPORTANT NOTES:")
    print("- Use a separate user_data_dir for safety")
    print("- Don't use your main Chrome profile during automation")
    print("- set keep_alive=True to preserve browser state")
    print("- executable_path should point to your Chrome installation")

if __name__ == "__main__":
    show_chrome_configuration_options()
