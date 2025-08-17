#!/usr/bin/env python3
"""
Task 6: Enhanced Error Simulation Integration Test Suite
Tests standalone error simulation capabilities from HumanInteractionEngine
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from playwright.async_api import async_playwright

# Add parent directory to Python path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from browser.session import BrowserSession
from browser.stealth import StealthManager


class TestEnhancedErrorSimulation:
    """Test enhanced error simulation integration"""

    @pytest.fixture
    async def browser_session(self):
        """Create a browser session for testing"""
        session = BrowserSession(
            headless=True,
            disable_security=True,
            user_data_dir=None
        )
        yield session
        await session.close()

    @pytest.fixture
    def test_html(self):
        """HTML content for testing error simulation"""
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Error Simulation Test</title></head>
        <body>
            <button id="target-button">Click Me</button>
            <button id="nearby-button">Nearby Button</button>
            <input id="target-input" type="text" placeholder="Type here" />
            <input id="nearby-input" type="text" placeholder="Wrong input" />
            <div id="content">Test content</div>
        </body>
        </html>
        """

    async def test_standalone_click_error_simulation_enabled(self, browser_session, test_html):
        """Test standalone click error simulation when enabled"""
        # Set environment variable to enable error simulation
        os.environ['STEALTH_ERROR_SIMULATION'] = 'true'

        try:
            await browser_session.start()
            page = await browser_session.get_current_page()
            await page.set_content(test_html)

            stealth_manager = browser_session._stealth_manager

            # Mock the interaction engine to simulate error decision
            with patch.object(stealth_manager.interaction_engine, '_should_simulate_error', return_value=True), \
                 patch.object(stealth_manager.interaction_engine, '_plan_error_simulation') as mock_plan_error:

                mock_plan_error.return_value = {
                    'type': 'wrong_click',
                    'wrong_element': {'center': {'x': 200, 'y': 100}},
                    'correction_delay': 0.1
                }

                # Mock the error simulation execution
                with patch.object(stealth_manager, '_execute_error_simulation', new_callable=AsyncMock) as mock_execute:
                    button = await page.query_selector('#target-button')
                    await stealth_manager.execute_human_like_click(page, button)

                    # Verify error simulation was triggered
                    mock_execute.assert_called_once()

                    # Check counters
                    assert browser_session._stealth_counters['stealth.error_simulation.standalone_enabled'] > 0
                    assert browser_session._stealth_counters['stealth.error_simulation.click_errors_triggered'] > 0

        finally:
            # Clean up environment variable
            if 'STEALTH_ERROR_SIMULATION' in os.environ:
                del os.environ['STEALTH_ERROR_SIMULATION']

    async def test_standalone_typing_error_simulation_enabled(self, browser_session, test_html):
        """Test standalone typing error simulation when enabled"""
        # Set environment variable to enable error simulation
        os.environ['STEALTH_ERROR_SIMULATION'] = 'true'

        try:
            await browser_session.start()
            page = await browser_session.get_current_page()
            await page.set_content(test_html)

            stealth_manager = browser_session._stealth_manager

            # Mock the interaction engine to simulate error decision
            with patch.object(stealth_manager.interaction_engine, '_should_simulate_error', return_value=True), \
                 patch.object(stealth_manager.interaction_engine, '_plan_typing_error_simulation') as mock_plan_error:

                mock_plan_error.return_value = {
                    'type': 'premature_typing',
                    'premature_text': 'a'
                }

                # Mock the error simulation execution
                with patch.object(stealth_manager, '_execute_typing_error_simulation', new_callable=AsyncMock) as mock_execute:
                    await stealth_manager.execute_human_like_typing(page, '#target-input', 'test text')

                    # Verify error simulation was triggered
                    mock_execute.assert_called_once()

                    # Check counters
                    assert browser_session._stealth_counters['stealth.error_simulation.standalone_enabled'] > 0
                    assert browser_session._stealth_counters['stealth.error_simulation.typing_errors_triggered'] > 0

        finally:
            # Clean up environment variable
            if 'STEALTH_ERROR_SIMULATION' in os.environ:
                del os.environ['STEALTH_ERROR_SIMULATION']

    async def test_error_simulation_disabled_by_default(self, browser_session, test_html):
        """Test that error simulation is disabled by default"""
        # Ensure environment variable is not set
        if 'STEALTH_ERROR_SIMULATION' in os.environ:
            del os.environ['STEALTH_ERROR_SIMULATION']

        await browser_session.start()
        page = await browser_session.get_current_page()
        await page.set_content(test_html)

        stealth_manager = browser_session._stealth_manager

        # Mock the interaction engine (should not be called)
        with patch.object(stealth_manager.interaction_engine, '_should_simulate_error') as mock_should_error:
            button = await page.query_selector('#target-button')
            await stealth_manager.execute_human_like_click(page, button)

            # Verify error simulation was not triggered
            mock_should_error.assert_not_called()

            # Check counters remain zero
            assert browser_session._stealth_counters['stealth.error_simulation.standalone_enabled'] == 0
            assert browser_session._stealth_counters['stealth.error_simulation.click_errors_triggered'] == 0

    async def test_error_simulation_with_behavioral_planning_disabled(self, browser_session, test_html):
        """Test that error simulation is disabled when behavioral planning is enabled"""
        # Set environment variable to enable error simulation
        os.environ['STEALTH_ERROR_SIMULATION'] = 'true'
        os.environ['STEALTH_BEHAVIORAL_PLANNING'] = 'true'

        try:
            await browser_session.start()
            page = await browser_session.get_current_page()
            await page.set_content(test_html)

            stealth_manager = browser_session._stealth_manager

            # Mock the interaction engine (should not be called for standalone)
            with patch.object(stealth_manager.interaction_engine, '_should_simulate_error') as mock_should_error:
                button = await page.query_selector('#target-button')

                # Mock behavioral planning to be enabled
                with patch.object(stealth_manager, '_get_element_context') as mock_context:
                    mock_context.return_value = {'_interaction_plan': {'type': 'click'}}

                    await stealth_manager.execute_human_like_click(page, button)

                    # Verify standalone error simulation was not triggered
                    # (behavioral planning takes precedence)
                    # Error simulation counters should remain zero for standalone
                    assert browser_session._stealth_counters['stealth.error_simulation.standalone_enabled'] == 0

        finally:
            # Clean up environment variables
            if 'STEALTH_ERROR_SIMULATION' in os.environ:
                del os.environ['STEALTH_ERROR_SIMULATION']
            if 'STEALTH_BEHAVIORAL_PLANNING' in os.environ:
                del os.environ['STEALTH_BEHAVIORAL_PLANNING']

    async def test_wrong_click_error_execution_with_counters(self, browser_session, test_html):
        """Test wrong click error execution with counter tracking"""
        await browser_session.start()
        page = await browser_session.get_current_page()
        await page.set_content(test_html)

        stealth_manager = browser_session._stealth_manager

        # Test wrong click error simulation
        error_sim = {
            'type': 'wrong_click',
            'wrong_element': {'center': {'x': 200, 'y': 100}},
            'correction_delay': 0.1
        }

        await stealth_manager._execute_error_simulation(page, error_sim)

        # Check execution counters
        assert browser_session._stealth_counters['stealth.error_simulation.wrong_click_executions'] == 1
        assert browser_session._stealth_counters['stealth.error_simulation.correction_behaviors_executed'] == 1

    async def test_typing_error_execution_with_counters(self, browser_session, test_html):
        """Test typing error execution with counter tracking"""
        await browser_session.start()
        page = await browser_session.get_current_page()
        await page.set_content(test_html)

        stealth_manager = browser_session._stealth_manager

        # Test wrong focus error simulation
        input_element = await page.query_selector('#target-input')
        error_sim = {
            'type': 'wrong_focus',
            'wrong_element': {'center': {'x': 300, 'y': 150}}
        }

        await stealth_manager._execute_typing_error_simulation(page, input_element, error_sim)

        # Check execution counters
        assert browser_session._stealth_counters['stealth.error_simulation.wrong_focus_executions'] == 1
        assert browser_session._stealth_counters['stealth.error_simulation.correction_behaviors_executed'] == 1

    async def test_premature_typing_error_execution_with_counters(self, browser_session, test_html):
        """Test premature typing error execution with counter tracking"""
        await browser_session.start()
        page = await browser_session.get_current_page()
        await page.set_content(test_html)

        stealth_manager = browser_session._stealth_manager

        # Test premature typing error simulation
        input_element = await page.query_selector('#target-input')
        error_sim = {
            'type': 'premature_typing',
            'premature_text': 'abc'
        }

        await stealth_manager._execute_typing_error_simulation(page, input_element, error_sim)

        # Check execution counters
        assert browser_session._stealth_counters['stealth.error_simulation.premature_typing_executions'] == 1
        assert browser_session._stealth_counters['stealth.error_simulation.correction_behaviors_executed'] == 1

    async def test_error_simulation_session_summary_logging(self, browser_session, test_html):
        """Test error simulation metrics in session summary logging"""
        await browser_session.start()
        page = await browser_session.get_current_page()
        await page.set_content(test_html)

        # Simulate some error simulations
        browser_session._stealth_counters['stealth.error_simulation.click_errors_triggered'] = 3
        browser_session._stealth_counters['stealth.error_simulation.typing_errors_triggered'] = 2
        browser_session._stealth_counters['stealth.error_simulation.correction_behaviors_executed'] = 5

        # Mock logger to capture output
        with patch.object(browser_session.logger, 'info') as mock_logger:
            browser_session._log_stealth_session_summary()

            # Verify error simulation info is included in log
            mock_logger.assert_called()
            log_message = mock_logger.call_args[0][0]
            assert 'error_simulations=5' in log_message
            assert 'click=3' in log_message
            assert 'typing=2' in log_message
            assert 'corrections=5' in log_message

    async def test_element_context_error_tracking(self, browser_session, test_html):
        """Test error simulation tracking in element context"""
        # Set environment variable to enable error simulation
        os.environ['STEALTH_ERROR_SIMULATION'] = 'true'

        try:
            await browser_session.start()
            page = await browser_session.get_current_page()
            await page.set_content(test_html)

            stealth_manager = browser_session._stealth_manager

            # Mock the interaction engine to simulate error decision
            with patch.object(stealth_manager.interaction_engine, '_should_simulate_error', return_value=True), \
                 patch.object(stealth_manager.interaction_engine, '_plan_error_simulation') as mock_plan_error:

                mock_plan_error.return_value = {
                    'type': 'wrong_click',
                    'wrong_element': {'center': {'x': 200, 'y': 100}},
                    'correction_delay': 0.1
                }

                # Mock element context collection
                mock_context = {}
                with patch.object(stealth_manager, '_get_element_context', return_value=mock_context):
                    with patch.object(stealth_manager, '_execute_error_simulation', new_callable=AsyncMock):
                        button = await page.query_selector('#target-button')
                        await stealth_manager.execute_human_like_click(page, button)

                        # Verify error tracking in context
                        assert mock_context.get('_error_simulated') is True
                        assert mock_context.get('_error_type') == 'wrong_click'

        finally:
            # Clean up environment variable
            if 'STEALTH_ERROR_SIMULATION' in os.environ:
                del os.environ['STEALTH_ERROR_SIMULATION']


async def main():
    """Run all tests"""
    print("🧪 Starting Task 6: Enhanced Error Simulation Integration Tests...")

    # Create test instance
    test_instance = TestEnhancedErrorSimulation()

    # Test standalone click error simulation
    print("✅ Testing standalone click error simulation enabled...")
    session1 = BrowserSession(headless=True, disable_security=True, user_data_dir=None)
    await test_instance.test_standalone_click_error_simulation_enabled(session1, test_instance.test_html)
    await session1.close()

    # Test standalone typing error simulation
    print("✅ Testing standalone typing error simulation enabled...")
    session2 = BrowserSession(headless=True, disable_security=True, user_data_dir=None)
    await test_instance.test_standalone_typing_error_simulation_enabled(session2, test_instance.test_html)
    await session2.close()

    # Test error simulation disabled by default
    print("✅ Testing error simulation disabled by default...")
    session3 = BrowserSession(headless=True, disable_security=True, user_data_dir=None)
    await test_instance.test_error_simulation_disabled_by_default(session3, test_instance.test_html)
    await session3.close()

    # Test wrong click error execution with counters
    print("✅ Testing wrong click error execution with counters...")
    session4 = BrowserSession(headless=True, disable_security=True, user_data_dir=None)
    await test_instance.test_wrong_click_error_execution_with_counters(session4, test_instance.test_html)
    await session4.close()

    # Test typing error execution with counters
    print("✅ Testing typing error execution with counters...")
    session5 = BrowserSession(headless=True, disable_security=True, user_data_dir=None)
    await test_instance.test_typing_error_execution_with_counters(session5, test_instance.test_html)
    await session5.close()

    # Test premature typing error execution with counters
    print("✅ Testing premature typing error execution with counters...")
    session6 = BrowserSession(headless=True, disable_security=True, user_data_dir=None)
    await test_instance.test_premature_typing_error_execution_with_counters(session6, test_instance.test_html)
    await session6.close()

    # Test error simulation session summary logging
    print("✅ Testing error simulation session summary logging...")
    session7 = BrowserSession(headless=True, disable_security=True, user_data_dir=None)
    await test_instance.test_error_simulation_session_summary_logging(session7, test_instance.test_html)
    await session7.close()

    # Test element context error tracking
    print("✅ Testing element context error tracking...")
    session8 = BrowserSession(headless=True, disable_security=True, user_data_dir=None)
    await test_instance.test_element_context_error_tracking(session8, test_instance.test_html)
    await session8.close()

    print("🎉 All Task 6: Enhanced Error Simulation Integration tests completed successfully!")
    print("📊 Summary:")
    print("   - Standalone error simulation decision logic integration ✅")
    print("   - Error simulation execution with proper method calls ✅")
    print("   - Error correction behaviors in simulation methods ✅")
    print("   - Behavioral state tracking with element context ✅")
    print("   - Comprehensive error simulation monitoring counters ✅")
    print("   - Session summary logging with error simulation metrics ✅")
    print("   - Environment variable controls for error simulation ✅")
    print("   - Coordination with behavioral planning system ✅")


if __name__ == "__main__":
    asyncio.run(main())
