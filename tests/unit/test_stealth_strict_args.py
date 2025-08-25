import re
from browser_use.browser.profile import BrowserProfile

# Flags that should not appear when stealth=True
FORBIDDEN_FLAGS = [
    "--disable-background-networking",
    "--metrics-recording-only",
    "--no-first-run",
    "--password-store=basic",
    "--use-mock-keychain",
    "--test-type=gpu",
    "--disable-sync",
]

PREFIX_FORBIDDEN = [
    "--enable-features=NetworkService,NetworkServiceInProcess",
    "--disable-features=",
]

def test_stealth_filters_automation_flags_headful_persistent():
    profile = BrowserProfile(stealth=True, headless=False, user_data_dir="C:/tmp/profile")
    args = profile.get_args()

    # Ensure none of the forbidden flags are present
    for flag in FORBIDDEN_FLAGS:
        assert flag not in args, f"flag should be stripped in stealth: {flag}"

    for prefix in PREFIX_FORBIDDEN:
        assert all(not a.startswith(prefix) for a in args), f"prefix should be stripped in stealth: {prefix}"

    # But we still expect AutomationControlled to be disabled explicitly for stealth
    assert any(
        a.startswith("--disable-blink-features=") and "AutomationControlled" in a for a in args
    ), "stealth should disable Blink AutomationControlled"


def test_non_stealth_keeps_flags_by_default():
    profile = BrowserProfile(stealth=False, headless=True)
    args = profile.get_args()

    # Non-stealth should still include at least some of these defaults (sanity check)
    assert any(a.startswith("--disable-features=") for a in args)
    assert "--metrics-recording-only" in args


def test_unsupported_flags_stripped_in_all_modes():
    # The flag below triggers Chrome's unsupported banner and should never be present
    banned = "--extensions-on-chrome-urls"

    p1 = BrowserProfile(stealth=True, headless=False, user_data_dir="C:/tmp/profile")
    a1 = p1.get_args()
    assert banned not in a1

    p2 = BrowserProfile(stealth=False, headless=True)
    a2 = p2.get_args()
    assert banned not in a2

    # Even if a user tries to pass it explicitly, it should be filtered out
    p3 = BrowserProfile(stealth=False, headless=True, args=[banned])
    a3 = p3.get_args()
    assert banned not in a3
