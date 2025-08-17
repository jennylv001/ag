#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3 Behavioral Planning Integration Demo
==========================================

This demo shows how the behavioral planning integration works when clicking elements.

Usage:
    python demo_task3_behavioral_planning.py

Environment Variables:
    STEALTH_BEHAVIORAL_PLANNING=true    # Enable behavioral planning (default: false)
    STEALTH_PAGE_EXPLORATION=true       # Enable page exploration (default: false)
    STEALTH_ERROR_SIMULATION=true       # Enable error simulation (default: false)

Features Demonstrated:
1. Environment variable activation of behavioral planning
2. Context collection from nearby elements
3. Interaction plan generation with exploration and errors
4. Counter tracking for monitoring and observability
"""

import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

def demo_environment_setup():
    """Demo 1: Show how environment variables activate behavioral planning"""
    print("🎯 Demo 1: Environment Variable Configuration")
    print("=" * 50)

    # Show current environment state
    current_behavioral = os.environ.get('STEALTH_BEHAVIORAL_PLANNING', 'false')
    current_exploration = os.environ.get('STEALTH_PAGE_EXPLORATION', 'false')
    current_error_sim = os.environ.get('STEALTH_ERROR_SIMULATION', 'false')

    print(f"Current environment settings:")
    print(f"  STEALTH_BEHAVIORAL_PLANNING = {current_behavioral}")
    print(f"  STEALTH_PAGE_EXPLORATION    = {current_exploration}")
    print(f"  STEALTH_ERROR_SIMULATION    = {current_error_sim}")
    print()

    # Show activation logic
    behavioral_enabled = current_behavioral.lower() == 'true'
    exploration_enabled = current_exploration.lower() == 'true'
    error_sim_enabled = current_error_sim.lower() == 'true'

    print(f"Behavioral Planning Active: {behavioral_enabled}")
    print(f"Page Exploration Active:    {exploration_enabled}")
    print(f"Error Simulation Active:    {error_sim_enabled}")
    print()

    if behavioral_enabled:
        print("✅ Behavioral planning is ENABLED")
        print("   → Interaction plans will be generated")
        print("   → Context collection will be performed")
        print("   → Exploration and error simulation may occur")
    else:
        print("⏸️  Behavioral planning is DISABLED")
        print("   → Standard stealth click behavior will be used")
        print("   → No interaction planning overhead")
    print()


def demo_context_structure():
    """Demo 2: Show the context structure used for behavioral planning"""
    print("🎯 Demo 2: Context Structure for Behavioral Planning")
    print("=" * 50)

    # Example context as collected by session.py
    example_context = {
        # Basic stealth properties
        "complexity": 0.6,
        "familiarity": 0.4,

        # Behavioral planning activation
        "behavioral_planning": True,

        # Element information
        "tag_name": "button",
        "size": {"width": 120, "height": 40},

        # Nearby elements for exploration
        "nearby_elements": [
            {
                "tag_name": "input",
                "center": {"x": 180, "y": 95},
                "size": {"width": 200, "height": 30},
                "attributes": {"type": "text", "placeholder": "Search..."}
            },
            {
                "tag_name": "a",
                "center": {"x": 95, "y": 140},
                "size": {"width": 80, "height": 25},
                "text": "Learn More"
            },
            {
                "tag_name": "div",
                "center": {"x": 200, "y": 120},
                "size": {"width": 150, "height": 60},
                "text": "Related Content"
            }
        ]
    }

    print("Example context passed to execute_human_like_click():")
    print(f"  Target Element: {example_context['tag_name']} ({example_context['size']['width']}x{example_context['size']['height']})")
    print(f"  Complexity Level: {example_context['complexity']}")
    print(f"  Behavioral Planning: {example_context['behavioral_planning']}")
    print(f"  Nearby Elements: {len(example_context['nearby_elements'])} found")

    for i, element in enumerate(example_context['nearby_elements']):
        print(f"    {i+1}. {element['tag_name']} at ({element['center']['x']}, {element['center']['y']}) - {element.get('text', 'no text')}")
    print()


async def demo_interaction_plan():
    """Demo 3: Show how interaction plans are generated and executed"""
    print("🎯 Demo 3: Interaction Plan Generation and Execution")
    print("=" * 50)

    try:
        from browser.stealth import StealthManager, HumanProfile

        # Create stealth manager
        profile = HumanProfile.create_random_profile()
        manager = StealthManager(profile)

        print(f"Created StealthManager with profile:")
        print(f"  Tech Savviness: {profile.tech_savviness:.2f}")
        print(f"  Motor Precision: {profile.motor_precision:.2f}")
        print(f"  Error Proneness: {profile.error_proneness:.2f}")
        print()

        # Example target element
        target_element = {
            'center': {'x': 300, 'y': 200},
            'tag_name': 'button',
            'size': {'width': 120, 'height': 40}
        }

        # Example nearby elements
        nearby_elements = [
            {
                'center': {'x': 280, 'y': 160},
                'tag_name': 'input',
                'size': {'width': 200, 'height': 30}
            },
            {
                'center': {'x': 350, 'y': 180},
                'tag_name': 'a',
                'size': {'width': 80, 'height': 25}
            }
        ]

        # Generate interaction plan
        plan = manager.interaction_engine.get_interaction_plan(
            target_element, nearby_elements, {'complexity': 0.7}
        )

        print("Generated Interaction Plan:")
        print(f"  Exploration Steps: {len(plan['exploration_steps'])}")
        for i, step in enumerate(plan['exploration_steps']):
            print(f"    {i+1}. {step['type']} for {step['duration']:.2f}s ({step['purpose']})")

        print(f"  Primary Action: {plan['primary_action']['type']}")
        print(f"  Error Simulation: {'Yes' if plan['error_simulation'] else 'None'}")
        if plan['error_simulation']:
            print(f"    Type: {plan['error_simulation']['type']}")

        print(f"  Post-Action Behaviors: {len(plan['post_action_behavior'])}")
        for i, behavior in enumerate(plan['post_action_behavior']):
            print(f"    {i+1}. {behavior['type']} ({behavior['purpose']})")
        print()

    except ImportError as e:
        print(f"⚠️ Import not available in test environment: {e}")

        # Show example plan structure
        example_plan = {
            'exploration_steps': [
                {'type': 'hover', 'duration': 1.2, 'purpose': 'context_gathering'},
                {'type': 'scan_to', 'duration': 0.4, 'purpose': 'context_gathering'}
            ],
            'primary_action': {
                'type': 'click',
                'approach': 'direct'
            },
            'error_simulation': {
                'type': 'wrong_click',
                'correction_delay': 1.5
            },
            'post_action_behavior': [
                {'type': 'observation_pause', 'duration': 0.8, 'purpose': 'verify_action_result'}
            ]
        }

        print("Example Interaction Plan (simulated):")
        print(f"  Exploration Steps: {len(example_plan['exploration_steps'])}")
        for i, step in enumerate(example_plan['exploration_steps']):
            print(f"    {i+1}. {step['type']} for {step['duration']}s ({step['purpose']})")

        print(f"  Primary Action: {example_plan['primary_action']['type']}")
        print(f"  Error Simulation: {example_plan['error_simulation']['type']}")
        print(f"  Post-Action Behaviors: {len(example_plan['post_action_behavior'])}")
        print()


def demo_counter_tracking():
    """Demo 4: Show stealth counter tracking for behavioral planning"""
    print("🎯 Demo 4: Stealth Counter Tracking")
    print("=" * 50)

    # Simulate counter initialization
    counters = {
        'stealth.click.used': 0,
        'stealth.click.fallback': 0,
        'stealth.planning.used': 0,
        'stealth.planning.fallback': 0,
        'stealth.exploration.steps': 0,
        'stealth.error.simulated': 0,
        'stealth.click.context_collected': 0
    }

    print("Initial Counter State:")
    for counter, value in counters.items():
        print(f"  {counter}: {value}")
    print()

    # Simulate behavioral planning usage
    print("Simulating Click with Behavioral Planning...")

    # Context indicates planning was used
    context = {
        '_planning_used': True,
        '_interaction_plan': {
            'exploration_steps': [{'type': 'hover'}, {'type': 'scan'}],
            'error_simulation': {'type': 'wrong_click'}
        }
    }

    # Update counters based on context
    if context.get('_planning_used', False):
        counters['stealth.planning.used'] += 1

        interaction_plan = context.get('_interaction_plan', {})
        exploration_steps = len(interaction_plan.get('exploration_steps', []))
        if exploration_steps > 0:
            counters['stealth.exploration.steps'] += exploration_steps

        if interaction_plan.get('error_simulation'):
            counters['stealth.error.simulated'] += 1

    print("Updated Counter State:")
    for counter, value in counters.items():
        if value > 0:
            print(f"  {counter}: {value} ✅")
        else:
            print(f"  {counter}: {value}")
    print()

    # Simulate session summary
    total_planning = counters['stealth.planning.used'] + counters['stealth.planning.fallback']
    planning_efficiency = f"{counters['stealth.planning.used']}/{total_planning}" if total_planning > 0 else "0/0"

    print(f"Session Summary:")
    print(f"  Planning Efficiency: {planning_efficiency}")
    print(f"  Exploration Steps: {counters['stealth.exploration.steps']}")
    print(f"  Errors Simulated: {counters['stealth.error.simulated']}")
    print()


def demo_integration_flow():
    """Demo 5: Show the complete integration flow"""
    print("🎯 Demo 5: Complete Behavioral Planning Flow")
    print("=" * 50)

    print("Step-by-step behavioral planning execution:")
    print()

    print("1. 🌍 Environment Check")
    print("   → STEALTH_BEHAVIORAL_PLANNING=true detected")
    print("   → context['behavioral_planning']=true confirmed")
    print()

    print("2. 📊 Context Collection (session.py)")
    print("   → _get_nearby_elements() executed")
    print("   → Element attributes and positions collected")
    print("   → stealth.click.context_collected += 1")
    print()

    print("3. 🧠 Plan Generation (stealth.py)")
    print("   → interaction_engine.get_interaction_plan() called")
    print("   → Target element + nearby elements analyzed")
    print("   → Exploration, error simulation, and post-action planned")
    print()

    print("4. 🎭 Plan Execution")
    print("   → execute_interaction_plan() orchestrates full sequence")
    print("   → Exploration steps: hover → scan → examine")
    print("   → Error simulation: wrong click → pause → correction")
    print("   → Primary action: human-like click with movement")
    print("   → Post-action: observation pause → micro-adjustment")
    print()

    print("5. 📈 Counter Updates")
    print("   → stealth.planning.used += 1")
    print("   → stealth.exploration.steps += 3")
    print("   → stealth.error.simulated += 1")
    print()

    print("6. ✅ Success Response")
    print("   → behavioral_state.record_action_result(True)")
    print("   → Return True to session.py")
    print("   → Session summary includes behavioral planning metrics")
    print()


def main():
    """Run all behavioral planning demos"""
    print("🚀 TASK 3: BEHAVIORAL PLANNING INTEGRATION DEMO")
    print("=" * 60)
    print()

    # Run demos
    demo_environment_setup()
    demo_context_structure()

    print("Running async demo...")
    asyncio.run(demo_interaction_plan())

    demo_counter_tracking()
    demo_integration_flow()

    print("=" * 60)
    print("✅ DEMO COMPLETE")
    print("=" * 60)
    print()
    print("To enable behavioral planning in your browser automation:")
    print()
    print("1. Set environment variables:")
    print("   export STEALTH_BEHAVIORAL_PLANNING=true")
    print("   export STEALTH_PAGE_EXPLORATION=true")
    print("   export STEALTH_ERROR_SIMULATION=true")
    print()
    print("2. Use browser_session._click_element_node() - it will:")
    print("   ✅ Collect context from nearby elements")
    print("   ✅ Generate comprehensive interaction plans")
    print("   ✅ Execute exploration, errors, and human-like actions")
    print("   ✅ Track detailed counters for monitoring")
    print()
    print("3. Monitor session summary for behavioral planning metrics")
    print()


if __name__ == "__main__":
    main()
