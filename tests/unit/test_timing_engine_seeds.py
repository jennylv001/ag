import os
import math
from browser.stealth import HumanProfile, AgentBehavioralState, CognitiveTimingEngine


def test_per_run_seed_determinism():
    profile = HumanProfile.create_random_profile()
    state = AgentBehavioralState()
    os.environ.pop('STEALTH_ENTROPY', None)

    eng1 = CognitiveTimingEngine(profile, state, entropy_enabled=False, run_seed=1234)
    eng2 = CognitiveTimingEngine(profile, state, entropy_enabled=False, run_seed=1234)

    # Without action seed, sequences should be identical given same run_seed
    seq1 = [eng1.get_deliberation_delay(1.0, 0.5) for _ in range(3)]
    seq2 = [eng2.get_deliberation_delay(1.0, 0.5) for _ in range(3)]
    assert seq1 == seq2


def test_per_action_seed_variation_and_reset():
    profile = HumanProfile.create_random_profile()
    state = AgentBehavioralState()
    os.environ.pop('STEALTH_ENTROPY', None)

    eng = CognitiveTimingEngine(profile, state, entropy_enabled=False, run_seed=42)

    eng.set_action_seed(1, 'click')
    a1 = [eng.get_keystroke_interval('a') for _ in range(3)]
    eng.clear_action_seed()

    eng.set_action_seed(2, 'click')
    a2 = [eng.get_keystroke_interval('a') for _ in range(3)]
    eng.clear_action_seed()

    assert a1 != a2  # different action seeds produce different sequences

    # Reset to same action seed yields determinism
    eng.set_action_seed(1, 'click')
    a1b = [eng.get_keystroke_interval('a') for _ in range(3)]
    assert a1 == a1b


def test_mouse_settle_time_bounds():
    profile = HumanProfile.create_random_profile()
    state = AgentBehavioralState()
    eng = CognitiveTimingEngine(profile, state, entropy_enabled=True, run_seed=7)

    eng.set_action_seed(99, 'scroll')
    vals = [eng.get_mouse_settle_time(d) for d in [10, 100, 500, 1000]]
    eng.clear_action_seed()

    for v in vals:
        assert 0.02 <= v <= 0.5
