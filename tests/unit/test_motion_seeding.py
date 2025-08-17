import math
import pytest

np = pytest.importorskip("numpy")  # Required by _bezier_curve implementation

from browser_use.browser.stealth import (
    BiometricMotionEngine,
    HumanProfile,
    AgentBehavioralState,
)


def _paths_equal(p1, p2, tol=1e-9):
    if len(p1) != len(p2):
        return False
    for (x1, y1, t1), (x2, y2, t2) in zip(p1, p2):
        if not (
            math.isclose(x1, x2, rel_tol=0, abs_tol=tol)
            and math.isclose(y1, y2, rel_tol=0, abs_tol=tol)
            and math.isclose(t1, t2, rel_tol=0, abs_tol=tol)
        ):
            return False
    return True


def test_biometric_motion_engine_seeding_deterministic():
    profile = HumanProfile.create_expert_profile()
    state = AgentBehavioralState()

    # Same run_seed across engines
    run_seed = 12345
    e1 = BiometricMotionEngine(profile, state, entropy_enabled=False, run_seed=run_seed)
    e2 = BiometricMotionEngine(profile, state, entropy_enabled=False, run_seed=run_seed)

    # Same action seed and same inputs -> identical paths
    e1.set_action_seed(1, action_kind="move")
    e2.set_action_seed(1, action_kind="move")

    p1 = e1.generate_movement_path(10.0, 10.0, 200.0, 140.0, num_points=60)
    p2 = e2.generate_movement_path(10.0, 10.0, 200.0, 140.0, num_points=60)

    assert _paths_equal(p1, p2), "Paths should be identical with same run/action seeds"

    # Different action seed should change the path
    e2.set_action_seed(2, action_kind="move")
    p3 = e2.generate_movement_path(10.0, 10.0, 200.0, 140.0, num_points=60)
    assert not _paths_equal(p1, p3), "Different action seeds should produce different paths"

    # Clearing and re-applying the same action seed restores determinism
    e1.clear_action_seed()
    e1.set_action_seed(1, action_kind="move")
    p4 = e1.generate_movement_path(10.0, 10.0, 200.0, 140.0, num_points=60)
    assert _paths_equal(p1, p4), "Re-applying same action seed should produce same path"
