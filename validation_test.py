import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# Setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
load_dotenv()

from browser_use.llm.google.chat import ChatGoogle
from browser_use import Agent, AgentSettings, setup_logging
from browser_use.browser import BrowserConfig, BrowserProfile

setup_logging()
logger = logging.getLogger("validation_test")

async def main():
    logger.info("=== VALIDATION TEST: Confirming Agent Functionality ===")
    
    try:
        # Single API key for simple test
        api_key = os.getenv("GEMINI_API_KEY_1")
        if not api_key:
            logger.error("No GEMINI_API_KEY_1 found")
            return
            
        llm = ChatGoogle(
            model="gemini-2.0-flash-exp",
            api_key=api_key
        )

        browser_profile = BrowserProfile(
            headless=False,  # Show browser window
            stealth=True,
            browser_args=[]
        )

        agent_settings = AgentSettings(
            task="Take a screenshot of google.com and describe what you see briefly",
            llm=llm,
            browser_profile=browser_profile,
            max_steps=3,  # Just a few steps
            use_planner=False,
            enable_long_running_mode=False,  # Simple mode
        )

        agent = Agent(settings=agent_settings)

        # Run to completion; agent will stop itself for this short task
        result = await agent.run()

        if result and result.history:
            logger.info("✅ VALIDATION PASSED: Agent completed successfully")
            logger.info(f"Steps executed: {len(result.history)}")
        else:
            logger.error("❌ VALIDATION FAILED: No result or history")

    except asyncio.TimeoutError:
        logger.error("❌ VALIDATION FAILED: Agent timed out")
    except Exception as e:
        logger.error(f"❌ VALIDATION FAILED: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
