"""
Fixtures for end-to-end tests.
"""

import os
import tempfile
import shutil
import pytest
from typing import Generator


@pytest.fixture
def temp_profile_dir() -> Generator[str, None, None]:
    """Create a temporary Chrome profile directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="stealth_e2e_test_")
    try:
        yield temp_dir
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


@pytest.fixture
def chrome_exec() -> str:
    """Find Chrome executable for testing."""
    chrome_paths = [
        # Windows
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
        # macOS
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        # Linux
        "/usr/bin/google-chrome",
        "/opt/google/chrome/chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
    ]

    for path in chrome_paths:
        if os.path.exists(path):
            return path

    # If no Chrome found, try using playwright's chromium
    import shutil
    chromium_path = shutil.which("chromium")
    if chromium_path:
        return chromium_path

    chromium_browser = shutil.which("chromium-browser")
    if chromium_browser:
        return chromium_browser

    pytest.skip("Chrome/Chromium executable not found for stealth testing")
