import os
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEALTH_PATH = ROOT / "browser" / "stealth.py"
spec = importlib.util.spec_from_file_location("stealth_mod", str(STEALTH_PATH))
stealth_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(stealth_mod)  # type: ignore

BiometricMotionEngine = stealth_mod.BiometricMotionEngine
HumanProfile = stealth_mod.HumanProfile
AgentBehavioralState = stealth_mod.AgentBehavioralState


def path_sample(engine: BiometricMotionEngine):
    # Generate a small path and timestamps
    path = engine.generate_movement_path(100, 100, 400, 320, num_points=20)
    # Reduce to first few rounded points for comparison readability
    rounded = [(round(x, 1), round(y, 1), round(t, 3)) for x, y, t in path[:6]]
    # Also compute settle time to see variability link
    settle = round(engine.behavioral_state.get_confidence_modifier() * 1.0, 4)
    return {"points": rounded, "state_factor": settle}


def main():
    os.environ.setdefault("STEALTH_ENTROPY", "true")
    run_seed = int(os.environ.get("STEALTH_RUN_SEED", "98765"))

    profile = HumanProfile.create_expert_profile()
    state = AgentBehavioralState()
    eng1 = BiometricMotionEngine(profile, state, entropy_enabled=True, run_seed=run_seed)

    eng1.set_action_seed(1, "click")
    s1_a = path_sample(eng1)
    eng1.clear_action_seed()

    eng1.set_action_seed(2, "click")
    s1_b = path_sample(eng1)
    eng1.clear_action_seed()

    # Recreate engine with same run seed
    profile2 = HumanProfile.create_expert_profile()
    state2 = AgentBehavioralState()
    eng2 = BiometricMotionEngine(profile2, state2, entropy_enabled=True, run_seed=run_seed)

    eng2.set_action_seed(1, "click")
    s2_a = path_sample(eng2)
    eng2.clear_action_seed()

    eng2.set_action_seed(2, "click")
    s2_b = path_sample(eng2)

    print("run_seed=", run_seed)
    print("action 1:", s1_a, "==", s2_a, "=>", s1_a == s2_a)
    print("action 2:", s1_b, "==", s2_b, "=>", s1_b == s2_b)


if __name__ == "__main__":
    main()
