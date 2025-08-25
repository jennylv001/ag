import asyncio
import logging
import os
import sys
try:
    from dotenv import load_dotenv  # optional
except Exception:  # pragma: no cover
    def load_dotenv():
        return None

# --- Setup: This ensures all imports work correctly ---
# Assuming the script is in a 'tests' directory, and the library root is one level up.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
load_dotenv()


# --- DEEPER FIX FOR ASYNCIO/WINDOWS ---
# Set the policy at the absolute start of the script, before any other imports
# that might interact with asyncio.


# We import the necessary classes
from browser_use.llm.google.chat import ChatGoogle
# CORRECTED: Import AgentSettings along with Agent
from browser_use import Agent, AgentSettings, setup_logging
from browser_use.browser import BrowserConfig, BrowserProfile

setup_logging()
logger = logging.getLogger("main_test_script")

# --- THE DEFINITIVE SOLUTION: A COMPLETE AND CORRECT WRAPPER CLASS ---
class RotatingGoogleClientManager:
    """
    Acts as a perfect, drop-in replacement for an LLM client.
    This version includes all required attributes and correct method signatures.
    """
    def __init__(self, api_keys: list[str], model: str, **kwargs):
        if not api_keys: raise ValueError("API keys list cannot be empty.")
        self.api_keys = api_keys
        self.model = model
        self.model_name = model
        self.kwargs = kwargs
        self._current_key_index = 0
        logger.info(f"Client Manager initialized with {len(self.api_keys)} keys for model {self.model_name}.")

    @property
    def provider(self) -> str: return 'google'

    @property
    def name(self) -> str: return self.model_name

    def _get_next_key(self) -> str:
        key = self.api_keys[self._current_key_index]
        logger.info(f"Selecting API key at index {self._current_key_index} for this agent task.")
        self._current_key_index = (self._current_key_index + 1) % len(self.api_keys)
        return key

    async def ainvoke(self, messages, output_format=None, **kwargs):
        """
        Creates a temporary ChatGoogle instance with one key and uses it.
        The signature now correctly matches the Agent's internal call.
        """
        api_key_for_this_call = self._get_next_key()

        # This wrapper does not need to handle structured output itself,
        # it just passes the arguments along to the real client.
        temp_client = ChatGoogle(
            model=self.model,
            api_key=api_key_for_this_call,
            **self.kwargs
        )

        # Pass the output_format argument to the underlying client
        return await temp_client.ainvoke(messages, output_format=output_format, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        # This is a pass-through method. The actual logic is handled by the underlying client.
        # The wrapper just needs to return itself so the next call in the chain works.
        return self

def get_keys_from_env() -> list[str]:
    keys = []
    i = 1
    while True:
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key: keys.append(key)
        else: break
        i += 1
    return keys

async def main():
    logger.info("--- Starting Final Test with Corrected Rotating Client Manager ---")

    api_keys_list = get_keys_from_env()
    if not api_keys_list:
        logger.error("FATAL: No 'GEMINI_API_KEY_n' variables found in .env file.")
        return

    logger.info(f"Successfully loaded {len(api_keys_list)} API keys.")

    # Add a debug log to confirm which event loop is running
    logger.debug(f"Current asyncio event loop: {asyncio.get_event_loop().__class__.__name__}")

    agent_result = None
    llm_manager = RotatingGoogleClientManager(
        api_keys=api_keys_list,
        model=os.getenv("LLM_MODEL", "gemini-1.5-flash-latest")
    )

    browser_profile = BrowserProfile(
        headless=False,
        stealth=True,
        user_data_dir=r"C:\Users\HeadOffice\Desktop\B\Directory",
        executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        browser_args=[]
    )

    # --- CORRECTED INITIALIZATION ---
    # 1. Create the AgentSettings object first.
    # Note: The original test passed `browser_config` and `telemetry_enabled`, which are not
    # standard AgentSettings fields. I've mapped them correctly. `browser_config` is used
    # to create a `BrowserProfile` for the agent. `telemetry_enabled` is not a direct
    # setting, so it's omitted as it's likely handled by observability elsewhere.
    # Directory that contains files allowed for upload. You can also whitelist specific files.
    available_file_path = r"C:\Users\HeadOffice\Desktop\B\Dir"
    available_file_paths = [available_file_path]

    agent_settings = AgentSettings(
        task=(
            '''Find free job posting websites and aggregate a list of ten links then attempt job posting on each of them as Isla McKennon, hiring manager at Ginkgo Bioworks, email:"hiring@ginkgobioworks.us", Phone:"7318592040", City:"Boston", Zipcode:"02210", Job Post Title:"Remote Administrative Assistant". You are not an alumni of the University. [Rinse and Repeat, Ensure no website from the first set is included in the next and so on, save all succesful job posts to results.md."
            "When filling up job posts, Use the logo in your file directory for uploads, path=C:\\Users\\HeadOffice\\Desktop\\B\\Dir\\logo.jpeg, if required."
            "PROCEED WITH ACTIONS FROM UP TO DOWN, FIELD BY FIELD FILLING OUT THE REQUIRED ONES, DO NOT JUMP FIELDS, IF TWO CONSECUTIVE ERRORS OCCUR, "
            "USE SCREENSHOT FOR VISUAL ANALYSIS OF THE PAGE AND MERGE INTO STATE, UPDATE YOUR REASONING, DONT RESTART PROCESSES, INSTEAD PERFORM ITERATIVE ACTION REFINEMENT"
            "NOTE: THIS IS A LONG-RUNNING TASK, DO NOT GET STUCK, PIVOT, AVOID REPITITION, IF YOU HIT A DEADEND, PIVOT. DONT FIGHT THE SYSTEM, ADAPT. USE TODO.MD TO KEEP TRACK OF PROGRESS AND INSIGHTS.MD TO LEARN AS YOU PROGRESS. USE ONLY ACTIONS PROVIDED'''
        ),
        llm=llm_manager,
        browser_profile=browser_profile,
        use_planner=False,
        use_task_planner=False,
        task_layer_enabled=True,
        available_file_paths=available_file_paths,
        save_conversation_path=r"C:\Users\HeadOffice\Desktop\B\browser_use\Conversation",
        default_search_engine='google',

        # Enable failure-proof mode
        enable_long_running_mode=True,
        long_running_checkpoint_interval=300.0,  # 5 min checkpoints
        long_running_cpu_threshold_warning=95.0,  # Conservative thresholds
        max_steps=10000,  # Allow extended operations
        enable_modes=True,  # Health-aware behavior
        long_running_enable_autonomous_continuation=True,  # The key setting!
        long_running_max_consecutive_failures=20,  # Tolerant of failures
        long_running_failure_escalation_delay=300.0,  # 5 minutes between escalations
    )

    # 2. Pass the single settings object to the Agent.
    agent = Agent(settings=agent_settings)

    # Run without an external per-test timeout; rely on internal timeouts
    agent_result = await agent.run()

    if agent_result and agent_result.history:
        logger.info("--- ✅ Test PASSED ---")
        logger.info("Agent run completed and produced a history.")
    else:
        logger.error("--- ❌ Test FAILED ---")
        logger.error("The agent did not return a valid result. This is likely due to the browser failing to start or the LLM call failing repeatedly.")

if __name__ == "__main__":
    asyncio.run(main())
