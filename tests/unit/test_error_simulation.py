import copy
import math
import pytest

from browser.stealth import HumanInteractionEngine, HumanProfile, AgentBehavioralState


def make_engine(seed: int, stress=0.0, familiarity=1.0, impulsivity=None, motor_precision=None):
    profile = HumanProfile.create_random_profile()
    if impulsivity is not None:
        profile.impulsivity = impulsivity
    if motor_precision is not None:
        profile.motor_precision = motor_precision
    state = AgentBehavioralState(stress_level=stress, familiarity_score=familiarity)
    engine = HumanInteractionEngine(profile, state, entropy_enabled=True, run_seed=seed)
    engine.set_action_seed(action_id=77, action_kind="error-test")
    return engine


def sample_elements(n=5):
    elems = []
    for i in range(n):
        elems.append({
            'center': {'x': 100 + i * 20, 'y': 100 + i * 20},
            'tag_name': 'input' if i % 2 == 0 else 'div',
            'size': {'width': 60, 'height': 20}
        })
    return elems


def test_error_probability_bounds_and_ordering():
    e_low = make_engine(1, stress=0.0, familiarity=1.0, impulsivity=0.0)
    e_high = make_engine(1, stress=1.0, familiarity=0.0, impulsivity=1.0)
    p_low = e_low._compute_error_probability()
    p_high = e_high._compute_error_probability()
    assert 0.01 <= p_low <= 0.2
    assert 0.01 <= p_high <= 0.2
    assert p_high > p_low


def test_click_context_uses_click_errors():
    eng = make_engine(1234, stress=0.5, familiarity=0.2, impulsivity=0.8, motor_precision=0.5)
    target = {'center': {'x': 200, 'y': 150}, 'tag_name': 'button', 'size': {'width': 100, 'height': 30}}
    nearby = sample_elements(5)

    sim = eng._plan_error_simulation(target, copy.deepcopy(nearby))
    # In click context, we should get wrong_click or premature_action
    assert sim['type'] in {'wrong_click', 'premature_action'}


def test_typing_context_uses_typing_errors():
    eng = make_engine(1234, stress=0.2, familiarity=0.4, impulsivity=0.3)
    target = {'center': {'x': 200, 'y': 150}, 'tag_name': 'input', 'text_content': '', 'size': {'width': 100, 'height': 30}}
    nearby = sample_elements(5)

    sim = eng._plan_error_simulation(target, copy.deepcopy(nearby))
    assert sim['type'] in {'wrong_focus', 'premature_typing', 'typo_sequence', 'typo'}


def test_wrong_target_is_plausible_distance():
    eng = make_engine(9876, stress=0.5, familiarity=0.2)
    target = {'center': {'x': 200, 'y': 200}, 'tag_name': 'button', 'size': {'width': 80, 'height': 30}}
    nearby = sample_elements(8)

    sim = eng._plan_error_simulation(target, copy.deepcopy(nearby))
    if sim['type'] == 'wrong_click':
        wrong_el = sim['wrong_element']
        cx, cy = wrong_el['center']['x'], wrong_el['center']['y']
        dist = math.sqrt((cx - 200)**2 + (cy - 200)**2)
        assert 30 <= dist <= 100


def test_determinism_same_seed():
    e1 = make_engine(2222, stress=0.6, familiarity=0.3, impulsivity=0.6)
    e2 = make_engine(2222, stress=0.6, familiarity=0.3, impulsivity=0.6)
    t_click = {'center': {'x': 200, 'y': 150}, 'tag_name': 'button', 'size': {'width': 100, 'height': 30}}
    t_type = {'center': {'x': 200, 'y': 150}, 'tag_name': 'input', 'text_content': '', 'size': {'width': 100, 'height': 30}}
    nearby = sample_elements(6)

    s1c = e1._plan_error_simulation(t_click, copy.deepcopy(nearby))
    s2c = e2._plan_error_simulation(t_click, copy.deepcopy(nearby))
    assert s1c == s2c
    s1t = e1._plan_error_simulation(t_type, copy.deepcopy(nearby))
    s2t = e2._plan_error_simulation(t_type, copy.deepcopy(nearby))
    assert s1t == s2t
