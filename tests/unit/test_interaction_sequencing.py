import os
import copy
import pytest

from browser.stealth import HumanInteractionEngine, HumanProfile, AgentBehavioralState


@pytest.fixture(autouse=True)
def deterministic_env(monkeypatch):
    # Enable entropy to exercise bounded jitter, but deterministically via seeds
    monkeypatch.setenv('STEALTH_ENTROPY', 'true')
    yield


def make_engine(seed: int):
    profile = HumanProfile.create_random_profile()
    state = AgentBehavioralState()
    engine = HumanInteractionEngine(profile, state, entropy_enabled=True, run_seed=seed)
    engine.set_action_seed(action_id=42, action_kind="test")
    return engine


def sample_elements(n=5):
    elems = []
    for i in range(n):
        elems.append({
            'center': {'x': 50 + i * 15, 'y': 100 + i * 10},
            'tag_name': 'div',
            'size': {'width': 80, 'height': 20}
        })
    return elems


def test_exploration_sequence_determinism_same_seed():
    engine1 = make_engine(1234)
    engine2 = make_engine(1234)

    target = {'center': {'x': 200, 'y': 150}, 'tag_name': 'button', 'size': {'width': 100, 'height': 30}}
    nearby = sample_elements(5)

    seq1 = engine1._plan_exploration_sequence(target, copy.deepcopy(nearby))
    seq2 = engine2._plan_exploration_sequence(target, copy.deepcopy(nearby))

    assert seq1 == seq2, "Exploration sequence should be identical for same run/action seed"


def test_exploration_sequence_variation_different_action_seed():
    engine1 = make_engine(1234)
    engine2 = make_engine(1234)

    # Different action seeds should alter sequence
    engine1.set_action_seed(100, "test")
    engine2.set_action_seed(101, "test")

    target = {'center': {'x': 200, 'y': 150}, 'tag_name': 'button', 'size': {'width': 100, 'height': 30}}
    nearby = sample_elements(5)

    seq1 = engine1._plan_exploration_sequence(target, copy.deepcopy(nearby))
    seq2 = engine2._plan_exploration_sequence(target, copy.deepcopy(nearby))

    assert seq1 != seq2, "Exploration sequence should differ for different action seeds"


def test_post_action_behavior_determinism_and_bounds():
    engine = make_engine(5678)
    target = {'center': {'x': 200, 'y': 150}, 'tag_name': 'button', 'size': {'width': 100, 'height': 30}}

    behaviors = engine._plan_post_action_behavior(target)

    # Deterministic across same seeds
    engine2 = make_engine(5678)
    behaviors2 = engine2._plan_post_action_behavior(target)
    assert behaviors == behaviors2

    # Bounds: duration and radius within specified ranges
    for b in behaviors:
        if b['type'] == 'observation_pause':
            assert 0.5 <= b['duration'] <= 2.0
        if b['type'] == 'micro_adjustment':
            assert 5 <= b['movement_radius'] <= 20


def test_pre_post_primary_delays_present_and_bounded():
    engine = make_engine(9999)
    target = {'center': {'x': 200, 'y': 150}, 'tag_name': 'button', 'size': {'width': 100, 'height': 30}}
    plan = engine.get_interaction_plan(target, nearby_elements=sample_elements(4), page_context={'action_type': 'click'})

    # Presence
    assert 'pre_primary_delay' in plan and 'post_primary_delay' in plan
    # Bounds
    assert 0.0 <= plan['pre_primary_delay'] <= 0.35
    assert 0.0 <= plan['post_primary_delay'] <= 0.6
