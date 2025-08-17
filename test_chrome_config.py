"""
Test script to verify your Chrome configuration works with the Agent
"""
import asyncio
import sys
import os
from pathlib import Path

# Add browser_use to path
sys.path.insert(0, '.')

async def test_chrome_config():
    """Test different Chrome configuration options"""
    print("🧪 Testing Chrome Configuration Options\n")

    try:
        from browser_use import Agent, AgentSettings
        from browser_use.browser import BrowserConfig
        print("✅ Successfully imported browser_use modules")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return

    # Test 1: Custom Chrome executable path
    print("\n📋 Test 1: Custom Chrome Executable Path")
    try:
        chrome_path = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        if Path(chrome_path).exists():
            print(f"✅ Chrome found at: {chrome_path}")

            test_config = BrowserConfig(
                executable_path=chrome_path,
                user_data_dir="C:\\Users\\HeadOffice\\Desktop\\B\\Dir\\TestChrome",
                headless=False,
                keep_alive=True,
            )
            print("✅ Chrome configuration created successfully")

            test_settings = AgentSettings(browser_config=test_config)
            print("✅ Agent settings created successfully")

        else:
            print(f"❌ Chrome not found at: {chrome_path}")
            print("💡 Try finding Chrome with: where chrome")

    except Exception as e:
        print(f"❌ Chrome config error: {e}")

    # Test 2: Check for running Chrome processes
    print("\n📋 Test 2: Running Chrome Processes")
    try:
        import psutil
        chrome_procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'chrome' in proc.info['name'].lower():
                    chrome_procs.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if chrome_procs:
            print(f"✅ Found {len(chrome_procs)} Chrome processes:")
            for proc in chrome_procs[:3]:  # Show first 3
                print(f"   PID: {proc['pid']} - {proc['name']}")

            # Test connecting by PID
            test_pid_config = BrowserConfig(browser_pid=chrome_procs[0]['pid'])
            print("✅ PID-based configuration created")
        else:
            print("ℹ️  No running Chrome processes found")

    except ImportError:
        print("❌ psutil not available (pip install psutil)")
    except Exception as e:
        print(f"❌ Process check error: {e}")

    # Test 3: CDP connection test
    print("\n📋 Test 3: CDP Connection Test")
    print("💡 To test CDP connection:")
    print("   1. Start Chrome with: chrome.exe --remote-debugging-port=9222")
    print("   2. Then use: BrowserConfig(cdp_url='http://localhost:9222')")

    try:
        import requests
        response = requests.get("http://localhost:9222/json", timeout=2)
        if response.status_code == 200:
            print("✅ Chrome debug port is accessible!")
            tabs = response.json()
            print(f"   Found {len(tabs)} open tabs/pages")
        else:
            print("❌ Chrome debug port responded with error")
    except requests.exceptions.ConnectionError:
        print("ℹ️  Chrome debug port not accessible (not running with --remote-debugging-port=9222)")
    except ImportError:
        print("❌ requests not available (pip install requests)")
    except Exception as e:
        print(f"❌ CDP test error: {e}")

    print("\n🎯 RECOMMENDED CONFIGURATION FOR YOUR SETUP:")
    print("=" * 50)
    print("""
from browser_use import Agent, AgentSettings
from browser_use.browser import BrowserConfig

# Use your Chrome with a dedicated profile
agent_settings = AgentSettings(
    browser_config=BrowserConfig(
        executable_path="C:\\\\Program Files\\\\Google\\\\Chrome\\\\Application\\\\chrome.exe",
        user_data_dir="C:\\\\Users\\\\HeadOffice\\\\Desktop\\\\B\\\\Dir\\\\AgentChrome",
        headless=False,      # Keep visible
        keep_alive=True,     # Don't close when done
        devtools=False,      # Set to True if you want DevTools
    )
)

# Create your agent
agent = Agent(
    task="Your automation task here",
    llm=your_llm_instance,
    settings=agent_settings
)

# Run the agent
await agent.run()
""")

if __name__ == "__main__":
    asyncio.run(test_chrome_config())
