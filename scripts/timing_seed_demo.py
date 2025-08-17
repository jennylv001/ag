import os
import importlib.util
import sys
from pathlib import Path

# Load stealth.py directly to avoid importing package __init__
ROOT = Path(__file__).resolve().parents[1]
STEALTH_PATH = ROOT / "browser" / "stealth.py"
spec = importlib.util.spec_from_file_location("stealth_mod", str(STEALTH_PATH))
stealth_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(stealth_mod)  # type: ignore

CognitiveTimingEngine = stealth_mod.CognitiveTimingEngine
HumanProfile = stealth_mod.HumanProfile
AgentBehavioralState = stealth_mod.AgentBehavioralState


def sample(engine: CognitiveTimingEngine):
    # Sample a few values to demonstrate determinism
    vals = {
        "delib": [round(engine.get_deliberation_delay(1.0, 0.5), 4) for _ in range(3)],
        "keys": [round(engine.get_keystroke_interval('e', 'h'), 4) for _ in range(5)],
        "settle": [round(engine.get_mouse_settle_time(d), 4) for d in (50, 200, 600)],
    }
    return vals


def main():
    os.environ.setdefault("STEALTH_ENTROPY", "true")
    run_seed = int(os.environ.get("STEALTH_RUN_SEED", "12345"))

    profile = HumanProfile.create_expert_profile()
    state = AgentBehavioralState()
    eng1 = CognitiveTimingEngine(profile, state, entropy_enabled=True, run_seed=run_seed)

    # Per-action seeded samples
    eng1.set_action_seed(1, "typing")
    s1_a = sample(eng1)
    eng1.clear_action_seed()

    eng1.set_action_seed(2, "typing")
    s1_b = sample(eng1)
    eng1.clear_action_seed()

    # Recreate engine with same run seed, expect identical per-action samples
    profile2 = HumanProfile.create_expert_profile()
    state2 = AgentBehavioralState()
    eng2 = CognitiveTimingEngine(profile2, state2, entropy_enabled=True, run_seed=run_seed)
    eng2.set_action_seed(1, "typing")
    s2_a = sample(eng2)
    eng2.clear_action_seed()
    eng2.set_action_seed(2, "typing")
    s2_b = sample(eng2)

    print("run_seed=", run_seed)
    print("action 1:", s1_a, "==", s2_a, "=>", s1_a == s2_a)
    print("action 2:", s1_b, "==", s2_b, "=>", s1_b == s2_b)


if __name__ == "__main__":
    main()
