"""
Agent Stealth Module - Project HYDRA
====================================

Advanced human behavioral emulation for web automation agents.
Designed to defeat advanced biometric analysis systems like reCAPTCHA v3.

This module provides sophisticated human-like interaction patterns including:
- Stochastic timing with cognitive modeling
- Biometric mouse movement with Bézier curves and velocity profiles
- Context-aware DOM exploration and error simulation
- Adaptive behavioral state management

Author: Agent "Silus" - Stealth-Integrated Logic and User Simulation
"""

import asyncio
import logging
import math
import random
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    np = None  # type: ignore
try:
    from scipy import interpolate  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    class _InterpFallback:  # type: ignore
        def splprep(self, *args, **kwargs):
            raise RuntimeError("scipy is not available")

        def splev(self, *args, **kwargs):
            raise RuntimeError("scipy is not available")

    interpolate = _InterpFallback()  # type: ignore
from pydantic import BaseModel


@dataclass
class AgentBehavioralState:
    """
    Stateful, Dynamic Emulation with Adaptive Behavioral Feedback

    Tracks agent performance and dynamically adjusts behavioral parameters.
    Agents that succeed become more confident; agents that fail become more cautious.
    This creates non-deterministic adaptation that mimics human learning.
    """

    # Performance tracking
    recent_actions: List[bool] = field(default_factory=list)  # Success/failure history
    session_start_time: float = field(default_factory=time.time)
    total_actions: int = 0
    successful_actions: int = 0

    # Dynamic behavioral parameters
    confidence_level: float = 0.5  # 0.0 = very cautious, 1.0 = very confident
    stress_level: float = 0.0      # 0.0 = relaxed, 1.0 = highly stressed
    familiarity_score: float = 0.0 # 0.0 = unfamiliar, 1.0 = very familiar

    # Adaptation parameters
    max_history_length: int = 20
    confidence_adaptation_rate: float = 0.1
    stress_decay_rate: float = 0.95

    def record_action_result(self, success: bool) -> None:
        """Record the result of an action and update behavioral state."""
        self.recent_actions.append(success)
        self.total_actions += 1

        if success:
            self.successful_actions += 1

        # Trim history to prevent unbounded growth
        if len(self.recent_actions) > self.max_history_length:
            self.recent_actions.pop(0)

        self._update_confidence()
        self._update_stress()
        self._update_familiarity()

    def _update_confidence(self) -> None:
        """Update confidence based on recent success rate."""
        if len(self.recent_actions) >= 3:
            recent_success_rate = sum(self.recent_actions[-10:]) / min(10, len(self.recent_actions))
            target_confidence = 0.3 + (recent_success_rate * 0.6)  # Range: 0.3-0.9
            self.confidence_level += (target_confidence - self.confidence_level) * self.confidence_adaptation_rate
            self.confidence_level = max(0.1, min(0.9, self.confidence_level))

    def _update_stress(self) -> None:
        """Update stress based on recent failures."""
        if self.recent_actions and not self.recent_actions[-1]:
            # Failed action increases stress
            self.stress_level = min(1.0, self.stress_level + 0.2)
        else:
            # Successful actions reduce stress
            self.stress_level *= self.stress_decay_rate

    def _update_familiarity(self) -> None:
        """Update familiarity based on session duration and action count."""
        session_duration = time.time() - self.session_start_time
        # Familiarity increases with time and successful actions
        base_familiarity = min(1.0, (session_duration / 300.0))  # 5 minutes to become familiar
        success_rate = self.successful_actions / max(1, self.total_actions)
        self.familiarity_score = min(1.0, base_familiarity * success_rate)

    def get_confidence_modifier(self) -> float:
        """
        Get timing modifier based on current confidence and stress levels.
        Returns multiplier for base timing values.
        """
        # Confident agents act faster, stressed agents act slower
        confidence_factor = 0.7 + (self.confidence_level * 0.6)  # 0.7-1.3x
        stress_factor = 1.0 + (self.stress_level * 0.5)          # 1.0-1.5x
        familiarity_factor = 0.8 + (self.familiarity_score * 0.4) # 0.8-1.2x

        return confidence_factor * stress_factor * familiarity_factor


@dataclass
class HumanProfile:
    """
    Individual Human Characteristics Profile

    Defines the base personality and behavioral patterns for a simulated human.
    Each profile represents a different "user type" with distinct interaction patterns.
    """

    # Basic personality traits
    typing_speed_wpm: float = 65.0          # Words per minute baseline
    reaction_time_ms: float = 250.0         # Base reaction time
    motor_precision: float = 0.85           # 0.0 = very clumsy, 1.0 = very precise
    impulsivity: float = 0.3                # 0.0 = very deliberate, 1.0 = very impulsive
    tech_savviness: float = 0.7             # 0.0 = tech novice, 1.0 = tech expert

    # Timing characteristics
    deliberation_tendency: float = 0.6      # How much they think before acting
    multitasking_ability: float = 0.5       # Likelihood of task switching
    error_proneness: float = 0.15           # Base error rate

    # Movement characteristics
    movement_smoothness: float = 0.8        # How smooth their mouse movements are
    overshoot_tendency: float = 0.2         # Likelihood of overshooting targets
    correction_speed: float = 0.7           # How quickly they correct mistakes

    @classmethod
    def create_random_profile(cls) -> 'HumanProfile':
        """Create a randomized but realistic human profile."""
        return cls(
            typing_speed_wpm=random.gauss(65, 20),
            reaction_time_ms=random.gauss(250, 50),
            motor_precision=random.betavariate(3, 1),  # Skewed toward higher precision
            impulsivity=random.betavariate(2, 3),      # Skewed toward lower impulsivity
            tech_savviness=random.betavariate(2, 2),   # Normal distribution
            deliberation_tendency=random.betavariate(3, 2),
            multitasking_ability=random.betavariate(2, 2),
            error_proneness=max(0.05, random.gauss(0.15, 0.05)),
            movement_smoothness=random.betavariate(4, 1),
            overshoot_tendency=random.betavariate(2, 4),
            correction_speed=random.betavariate(3, 2)
        )

    @classmethod
    def create_expert_profile(cls) -> 'HumanProfile':
        """Create a profile for a tech-savvy, experienced user."""
        return cls(
            typing_speed_wpm=85.0,
            reaction_time_ms=200.0,
            motor_precision=0.95,
            impulsivity=0.4,
            tech_savviness=0.9,
            deliberation_tendency=0.4,
            multitasking_ability=0.8,
            error_proneness=0.08,
            movement_smoothness=0.9,
            overshoot_tendency=0.1,
            correction_speed=0.9
        )

    @classmethod
    def create_novice_profile(cls) -> 'HumanProfile':
        """Create a profile for a tech novice, cautious user."""
        return cls(
            typing_speed_wpm=35.0,
            reaction_time_ms=400.0,
            motor_precision=0.6,
            impulsivity=0.1,
            tech_savviness=0.3,
            deliberation_tendency=0.9,
            multitasking_ability=0.2,
            error_proneness=0.25,
            movement_smoothness=0.6,
            overshoot_tendency=0.4,
            correction_speed=0.4
        )


class CognitiveTimingEngine:
    """
    Advanced Cognitive Timing with Context-Aware Statistical Models

    Implements sophisticated timing patterns that model human cognitive processes.
    Uses different statistical distributions for different types of actions.
    """

    def __init__(self, profile: HumanProfile, behavioral_state: AgentBehavioralState, entropy_enabled: bool = False, run_seed: Optional[int] = None):
        self.profile = profile
        self.behavioral_state = behavioral_state
        # Entropy mode introduces slight, bounded randomness to break static patterns
        self._entropy_enabled = entropy_enabled

        # Seeded RNGs: one per-run, optional per-action overlay
        self._run_seed: int = int(run_seed) if run_seed is not None else int(time.time_ns() & 0xFFFFFFFF)
        # Base RNG for the run/session
        self._rng: random.Random = random.Random(self._run_seed)
        # Optional action-scoped RNG that, when set, drives timing for that action
        self._action_rng: Optional[random.Random] = None
        # Simple call counter used only when no action seed is set (keeps variability without global RNG)
        self._call_counter: int = 0

        # Cognitive timing parameters
        # Base thinking time in seconds
        self.base_deliberation_time: float = 0.5
        # Average chars per second derived from WPM (approx 5 chars per word)
        # Then invert to get base inter-keystroke interval (seconds per char)
        self.keystroke_base_interval: float = 60.0 / (self.profile.typing_speed_wpm * 5)

    def set_run_seed(self, seed: int | None) -> None:
        """Set/reset the per-run seed; resets the base RNG deterministically."""
        if seed is None:
            seed = int(time.time_ns() & 0xFFFFFFFF)
        self._run_seed = int(seed)
        self._rng.seed(self._run_seed)

    def set_action_seed(self, action_id: int, action_kind: str = "") -> None:
        """Derive and set a deterministic per-action RNG from run_seed, action_id, and kind."""
        try:
            payload = f"{self._run_seed}:{action_kind}:{int(action_id)}".encode("utf-8")
            # Use a stable 32-bit seed derived from SHA-256
            seed32 = int(hashlib.sha256(payload).hexdigest()[:8], 16)
            self._action_rng = random.Random(seed32)
        except Exception:
            # Fallback: perturb the base RNG reproducibly
            self._action_rng = random.Random(self._run_seed ^ (action_id & 0xFFFFFFFF))

    def clear_action_seed(self) -> None:
        """Clear action-scoped RNG; fall back to run RNG for subsequent calls."""
        self._action_rng = None

    def get_deliberation_delay(self, complexity: float = 1.0, element_familiarity: float = 0.5) -> float:
        """
        Model high-level "thinking" time using log-normal distribution.

        Args:
            complexity: Task complexity multiplier (0.5=simple, 2.0=complex)
            element_familiarity: How familiar this element type is (0.0=new, 1.0=familiar)

        Returns:
            Delay in seconds
        """
        # Base deliberation affected by personality and state
        base_time = self.base_deliberation_time * self.profile.deliberation_tendency

        # Apply complexity and familiarity modifiers
        complexity_factor = 0.5 + (complexity * 0.5)
        familiarity_factor = 1.5 - (element_familiarity * 0.8)  # Less familiar = more thinking

        # Apply behavioral state modifiers
        confidence_modifier = self.behavioral_state.get_confidence_modifier()

        # Log-normal distribution for deliberation (right-skewed, non-negative)
        mu = math.log(base_time * complexity_factor * familiarity_factor)
        sigma = 0.3 + (self.profile.impulsivity * 0.2)  # More impulsive = more variable

        # Choose RNG source
        rng = self._action_rng or self._rng
        self._call_counter += 1

        # Entropy jitter: slight offsets to parameters per call
        if self._entropy_enabled:
            mu += rng.uniform(-0.15, 0.15)
            sigma *= rng.uniform(0.9, 1.15)

        raw_delay = rng.lognormvariate(mu, sigma)

        # Apply confidence modifier and reasonable bounds
        final_delay = raw_delay * confidence_modifier
        if self._entropy_enabled:
            # Lightly vary caps to avoid hard boundaries (seeded)
            lower = 0.08 + rng.uniform(0.0, 0.04)
            upper = 4.8 + rng.uniform(0.0, 0.4)
            return max(lower, min(upper, final_delay))
        return max(0.1, min(5.0, final_delay))

    def get_keystroke_interval(self, char: str, prev_char: str = '') -> float:
        """
        Model inter-keystroke timing using Gamma distribution.

        Args:
            char: Current character being typed
            prev_char: Previous character (for bigram analysis)

        Returns:
            Delay in seconds before typing this character
        """
        base_interval = self.keystroke_base_interval

        # Character-specific modifiers
        char_difficulty = self._get_character_difficulty(char, prev_char)

        # Apply personality and state modifiers
        precision_factor = 0.8 + (self.profile.motor_precision * 0.4)
        confidence_modifier = self.behavioral_state.get_confidence_modifier()

        # Gamma distribution (right-skewed, good for modeling intervals)
        # Shape parameter controls skewness, scale controls timing
        rng = self._action_rng or self._rng
        self._call_counter += 1
        shape = 2.0 + (self.profile.motor_precision * 2.0)  # Higher precision = less variable
        if self._entropy_enabled:
            shape *= rng.uniform(0.9, 1.2)
        scale = (base_interval * char_difficulty) / shape

        raw_interval = rng.gammavariate(shape, scale)

        # Apply modifiers and bounds
        final_interval = raw_interval * precision_factor * confidence_modifier
        if self._entropy_enabled:
            # Tiny random floors/ceilings to avoid crisp bounds
            rng = self._action_rng or self._rng
            floor = 0.018 + rng.uniform(0.0, 0.006)
            ceil = 0.48 + rng.uniform(0.0, 0.06)
            return max(floor, min(ceil, final_interval))
        return max(0.02, min(0.5, final_interval))  # 20ms to 500ms per keystroke

    def _get_character_difficulty(self, char: str, prev_char: str) -> float:
        """Calculate typing difficulty for character transitions."""
        difficulty = 1.0

        # Capital letters are slower
        if char.isupper():
            difficulty *= 1.3

        # Numbers and special characters are slower
        if char.isdigit():
            difficulty *= 1.2
        elif not char.isalnum() and char != ' ':
            difficulty *= 1.5

        # Common bigrams are faster
        bigram = prev_char + char
        fast_bigrams = {'th', 'he', 'in', 'er', 'an', 'ed', 'nd', 'to', 'en', 'ti'}
        if bigram.lower() in fast_bigrams:
            difficulty *= 0.8

        # Same finger sequences are slower
        if self._same_finger_sequence(prev_char, char):
            difficulty *= 1.4

        return difficulty

    def _same_finger_sequence(self, char1: str, char2: str) -> bool:
        """Check if two characters use the same finger (simplified model)."""
        finger_map = {
            'q': 0, 'a': 0, 'z': 0,
            'w': 1, 's': 1, 'x': 1,
            'e': 2, 'd': 2, 'c': 2,
            'r': 3, 'f': 3, 'v': 3, 't': 3, 'g': 3, 'b': 3,
            'y': 4, 'h': 4, 'n': 4, 'u': 4, 'j': 4, 'm': 4,
            'i': 5, 'k': 5,
            'o': 6, 'l': 6,
            'p': 7
        }
        return finger_map.get(char1.lower()) == finger_map.get(char2.lower())

    def get_mouse_settle_time(self, movement_distance: float) -> float:
        """
        Calculate time to settle mouse after movement.
        Longer movements require more settling time.
        """
        base_settle = 0.05  # 50ms base settling
        distance_factor = min(2.0, movement_distance / 200.0)  # Scale with distance
        precision_factor = 2.0 - self.profile.motor_precision

        # Beta distribution for settle time (bounded, realistic shape)
        # Add entropy variation in distribution shape
        rng = self._action_rng or self._rng
        self._call_counter += 1
        if self._entropy_enabled:
            alpha = 1.8 + rng.uniform(0.0, 0.6)
            beta = 4.5 + rng.uniform(0.0, 0.8)
        else:
            alpha, beta = 2.0, 5.0  # Right-skewed, most values near minimum
        raw_settle = rng.betavariate(alpha, beta) * 0.3  # 0-300ms range

        final_settle = base_settle + (raw_settle * distance_factor * precision_factor)
        return max(0.02, min(0.5, final_settle))


class BiometricMotionEngine:
    """
    Advanced Biometric Motion with Velocity Profiles and Micro-corrections

    Generates realistic mouse movements using Bézier curves with human-like
    velocity profiles and natural imperfections.
    """

    def __init__(self, profile: HumanProfile, behavioral_state: AgentBehavioralState, entropy_enabled: bool = False, run_seed: Optional[int] = None):
        self.profile = profile
        self.behavioral_state = behavioral_state
        self._entropy_enabled = entropy_enabled
        # Seeded RNGs to align with timing engine for reproducible-but-variable motion
        self._run_seed: int = int(run_seed) if run_seed is not None else int(time.time_ns() & 0xFFFFFFFF)
        self._rng: random.Random = random.Random(self._run_seed)
        self._action_rng: Optional[random.Random] = None

    def set_action_seed(self, action_id: int, action_kind: str = "") -> None:
        try:
            payload = f"{self._run_seed}:{action_kind}:{int(action_id)}".encode("utf-8")
            seed32 = int(hashlib.sha256(payload).hexdigest()[:8], 16)
            self._action_rng = random.Random(seed32)
        except Exception:
            self._action_rng = random.Random(self._run_seed ^ (action_id & 0xFFFFFFFF))

    def clear_action_seed(self) -> None:
        self._action_rng = None

    def generate_movement_path(self, start_x: float, start_y: float,
                             end_x: float, end_y: float,
                             num_points: int = 50) -> List[Tuple[float, float, float]]:
        """
        Generate realistic mouse movement path with timing.

        Returns:
            List of (x, y, timestamp) tuples representing the movement path
        """
        # Validate all inputs
        for name, value in [
            ("start_x", start_x),
            ("start_y", start_y),
            ("end_x", end_x),
            ("end_y", end_y)
        ]:
            try:
                # Convert to float if needed
                value = float(value)

                # Check for NaN/Inf
                if math.isnan(value) or math.isinf(value):
                    raise ValueError(f"Invalid {name}: {value} (NaN or Infinity)")
            except (ValueError, TypeError):
                raise ValueError(f"Invalid {name}: {value} is not a valid number")

        # Ensure num_points is a positive integer
        if not isinstance(num_points, int) or num_points <= 0:
            num_points = 50  # Use default if invalid

        # Calculate distance between points
        try:
            distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        except (ValueError, TypeError) as e:
            # This should not happen after validation, but just in case
            raise ValueError(f"Failed to calculate distance: {str(e)}")

        # Log the movement path parameters
        logging.debug(f"🔍 Movement path: ({start_x:.1f}, {start_y:.1f}) → ({end_x:.1f}, {end_y:.1f}), distance: {distance:.1f}px")

        if distance < 5:
            # Very short movements can be direct
            return [(end_x, end_y, 0.05)]

        try:
            # Generate control points for Bézier curve
            control_points = self._generate_control_points(start_x, start_y, end_x, end_y, distance)

            # Generate smooth path using Bézier curve
            path_points = self._bezier_curve(control_points, num_points)

            # Add micro-corrections and imperfections
            path_points = self._add_movement_imperfections(path_points)

            # Generate velocity profile and timing
            timed_path = self._apply_velocity_profile(path_points, distance)

            # Final validation of generated path
            for i, (x, y, t) in enumerate(timed_path):
                if math.isnan(x) or math.isnan(y) or math.isnan(t) or math.isinf(x) or math.isinf(y) or math.isinf(t):
                    logging.warning(f"⚠️ Generated invalid path point at index {i}: ({x}, {y}, {t})")
                    # Replace with valid interpolated point
                    if i > 0 and i < len(timed_path) - 1:
                        prev = timed_path[i-1]
                        next_point = timed_path[i+1]
                        timed_path[i] = ((prev[0] + next_point[0])/2, (prev[1] + next_point[1])/2, (prev[2] + next_point[2])/2)

            return timed_path

        except Exception as e:
            logging.error(f"❌ Failed to generate movement path: {str(e)}")
            # Fallback to simple direct path with a small delay
            return [(start_x, start_y, 0.0), (end_x, end_y, 0.1)]

    def _generate_control_points(self, start_x: float, start_y: float,
                               end_x: float, end_y: float, distance: float) -> List[Tuple[float, float]]:
        """Generate control points for natural Bézier curves."""
        # Start point
        points = [(start_x, start_y)]

        # Calculate movement vector
        dx = end_x - start_x
        dy = end_y - start_y

        # Generate intermediate control points based on distance and personality
        if distance > 100:
            # Longer movements get more control points for natural arcs
            num_controls = min(4, int(distance / 100))

            for i in range(num_controls):
                t = (i + 1) / (num_controls + 1)

                # Base point along direct line
                base_x = start_x + (dx * t)
                base_y = start_y + (dy * t)

                # Add perpendicular offset for natural curve
                perp_x = -dy / distance
                perp_y = dx / distance

                # Offset amount based on personality and movement phase
                curve_strength = 20 * (1.0 - self.profile.motor_precision)
                phase_modifier = math.sin(math.pi * t)  # Strongest curve in middle
                offset = curve_strength * phase_modifier

                # Add some randomness
                rng = self._action_rng or self._rng
                offset *= rng.uniform(0.5, 1.5)

                control_x = base_x + (perp_x * offset)
                control_y = base_y + (perp_y * offset)

                points.append((control_x, control_y))

        # End point
        points.append((end_x, end_y))
        return points

    def _bezier_curve(self, control_points: List[Tuple[float, float]], num_points: int) -> List[Tuple[float, float]]:
        """Generate smooth Bézier curve through control points."""
        # Validate inputs
        if not control_points:
            logging.warning("⚠️ Empty control points provided to _bezier_curve")
            return []

        if len(control_points) < 2:
            logging.debug(f"🔍 Not enough control points ({len(control_points)}), returning as-is")
            return control_points

        # Validate each control point
        valid_control_points = []
        for i, point in enumerate(control_points):
            try:
                # Check point structure
                if not isinstance(point, (tuple, list)) or len(point) != 2:
                    logging.warning(f"⚠️ Invalid control point at index {i}: {point}")
                    continue

                x, y = point

                # Validate coordinates
                if (not isinstance(x, (int, float)) or not isinstance(y, (int, float)) or
                    math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y)):
                    logging.warning(f"⚠️ Invalid coordinates in control point at index {i}: ({x}, {y})")
                    continue

                valid_control_points.append((float(x), float(y)))
            except Exception as e:
                logging.warning(f"⚠️ Error validating control point at index {i}: {str(e)}")

        if len(valid_control_points) < 2:
            logging.warning(f"⚠️ Not enough valid control points ({len(valid_control_points)})")
            return valid_control_points

        try:
            # Convert to numpy array safely
            points = np.array(valid_control_points, dtype=np.float64)

            # Parameter values for interpolation
            t_control = np.linspace(0, 1, len(points))
            t_interp = np.linspace(0, 1, max(2, num_points))  # Ensure at least 2 points

            # Use a simpler interpolation method if we have fewer than 4 points
            kind = 'cubic' if len(points) >= 4 else 'linear'

            # Interpolate x and y coordinates separately with error handling
            try:
                fx = interpolate.interp1d(t_control, points[:, 0], kind=kind,
                                         bounds_error=False, fill_value='extrapolate')
                fy = interpolate.interp1d(t_control, points[:, 1], kind=kind,
                                         bounds_error=False, fill_value='extrapolate')

                smooth_points = []
                for t in t_interp:
                    try:
                        x, y = float(fx(t)), float(fy(t))

                        # Validate generated points
                        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                            continue

                        smooth_points.append((x, y))
                    except Exception:
                        continue

                if not smooth_points:
                    # Fallback to linear interpolation between first and last
                    start, end = valid_control_points[0], valid_control_points[-1]
                    return [start, end]

                return smooth_points

            except Exception as e:
                logging.warning(f"⚠️ Interpolation error: {str(e)}")
                # Fallback to linear interpolation
                return [valid_control_points[0], valid_control_points[-1]]

        except Exception as e:
            logging.error(f"❌ Bezier curve generation failed: {str(e)}")
            # Return original points as fallback
            return valid_control_points

    def _add_movement_imperfections(self, path_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Add realistic micro-corrections and tremor to movement."""
        if len(path_points) < 3:
            return path_points

        imperfect_points = []
        tremor_strength = (1.0 - self.profile.motor_precision) * 2.0
        rng = self._action_rng or self._rng

        for i, (x, y) in enumerate(path_points):
            # Add small random tremor
            tremor_x = rng.gauss(0, tremor_strength)
            tremor_y = rng.gauss(0, tremor_strength)

            # Reduce tremor at start and end (more controlled)
            edge_factor = min(i / 5, (len(path_points) - i) / 5, 1.0)
            tremor_x *= edge_factor
            tremor_y *= edge_factor

            # Add occasional larger corrections (simulate hand adjustments)
            if rng.random() < 0.1 * (1.0 - self.profile.motor_precision):
                correction_x = rng.gauss(0, tremor_strength * 3)
                correction_y = rng.gauss(0, tremor_strength * 3)
                tremor_x += correction_x
                tremor_y += correction_y

            imperfect_points.append((x + tremor_x, y + tremor_y))

        return imperfect_points

    def _apply_velocity_profile(self, path_points: List[Tuple[float, float]],
                              total_distance: float) -> List[Tuple[float, float, float]]:
        """Apply realistic velocity profile with acceleration and deceleration."""
        if len(path_points) < 2:
            x, y = path_points[0]
            return [(x, y, 0.05)]

        # Calculate total movement time based on distance and personality
        base_time = 0.3 + (total_distance / 1000.0)  # 300ms + distance factor
        confidence_modifier = self.behavioral_state.get_confidence_modifier()
        personality_modifier = 0.8 + (self.profile.reaction_time_ms / 500.0)

        # Add small run/action seeded jitter to total time within bounds
        rng = self._action_rng or self._rng
        jitter = rng.uniform(0.95, 1.1) if self._entropy_enabled else 1.0
        total_time = base_time * confidence_modifier * personality_modifier * jitter

        # Generate velocity profile using sine curve (natural acceleration/deceleration)
        timestamps = []
        cumulative_time = 0.0

        for i in range(len(path_points)):
            if i == 0:
                timestamps.append(0.0)
                continue

            # Progress through movement (0 to 1)
            progress = i / (len(path_points) - 1)

            # Sine-based velocity (accelerate then decelerate), with optional noise
            # Inject micro-variance into acceleration curve using seeded RNG
            # Randomize exponent slightly to vary the bell shape subtly
            exponent = 1.0
            if self._entropy_enabled:
                exponent = rng.uniform(0.9, 1.2)
            base_velocity = math.sin(progress * math.pi)
            velocity_factor = base_velocity ** exponent
            if self._entropy_enabled:
                velocity_factor = max(0.05, velocity_factor * rng.uniform(0.9, 1.12))

            # Add personality-based variations
            if self.profile.impulsivity > 0.7:
                # Impulsive users start fast and slow down more
                velocity_factor = velocity_factor ** 0.7
            elif self.profile.deliberation_tendency > 0.7:
                # Deliberate users accelerate more gradually
                velocity_factor = velocity_factor ** 1.3

            # Calculate time step
            segment_time = (total_time / len(path_points)) * (2.0 * velocity_factor)
            if self._entropy_enabled:
                segment_time *= rng.uniform(0.9, 1.1)
            cumulative_time += segment_time
            timestamps.append(cumulative_time)

        # Combine coordinates with timestamps
        timed_path = []
        for i, (x, y) in enumerate(path_points):
            timed_path.append((x, y, timestamps[i]))

        return timed_path

    def should_overshoot_target(self, distance: float) -> bool:
        """Determine if movement should overshoot target (natural human behavior)."""
        base_probability = self.profile.overshoot_tendency

        # Longer movements more likely to overshoot
        distance_factor = min(1.0, distance / 300.0)

        # Stress increases overshoot likelihood
        stress_factor = 1.0 + self.behavioral_state.stress_level

        rng = self._action_rng or self._rng
        final_probability = base_probability * distance_factor * stress_factor
        if self._entropy_enabled:
            final_probability *= rng.uniform(0.8, 1.25)
        return rng.random() < final_probability

    def generate_overshoot_correction(self, target_x: float, target_y: float,
                                    overshoot_distance: float = 15) -> List[Tuple[float, float, float]]:
        """Generate a correction movement after overshooting target."""
        # Calculate overshoot position
        rng = self._action_rng or self._rng
        angle = rng.uniform(0, 2 * math.pi)
        overshoot_x = target_x + (math.cos(angle) * overshoot_distance)
        overshoot_y = target_y + (math.sin(angle) * overshoot_distance)

        # Generate correction path back to target
        correction_path = self.generate_movement_path(
            overshoot_x, overshoot_y, target_x, target_y, num_points=10
        )

        # Correction is faster than initial movement
        for i in range(len(correction_path)):
            x, y, t = correction_path[i]
            correction_path[i] = (x, y, t * self.profile.correction_speed)

        return correction_path


class HumanInteractionEngine:
    """
    Context-Aware Interaction Planning with Error Simulation

    Generates realistic interaction patterns including page exploration,
    error simulation, and context-appropriate hesitation behaviors.
    """

    def __init__(self, profile: HumanProfile, behavioral_state: AgentBehavioralState, entropy_enabled: bool = False, run_seed: Optional[int] = None):
        self.profile = profile
        self.behavioral_state = behavioral_state
        self._entropy_enabled = entropy_enabled
        # Deterministic RNGs for planning
        self._run_seed: int = int(run_seed) if run_seed is not None else int(time.time_ns() & 0xFFFFFFFF)
        self._rng: random.Random = random.Random(self._run_seed)
        self._action_rng: Optional[random.Random] = None

    def set_run_seed(self, seed: int | None) -> None:
        if seed is None:
            seed = int(time.time_ns() & 0xFFFFFFFF)
        self._run_seed = int(seed)
        self._rng.seed(self._run_seed)

    def set_action_seed(self, action_id: int, action_kind: str = "") -> None:
        """Derive per-action deterministic RNG for planning sequences."""
        try:
            payload = f"{self._run_seed}:{action_kind}:{int(action_id)}".encode("utf-8")
            seed32 = int(hashlib.sha256(payload).hexdigest()[:8], 16)
            self._action_rng = random.Random(seed32)
        except Exception:
            self._action_rng = random.Random(self._run_seed ^ (action_id & 0xFFFFFFFF))

    def clear_action_seed(self) -> None:
        self._action_rng = None

    def get_interaction_plan(self, target_element: Dict[str, Any],
                           nearby_elements: List[Dict[str, Any]] = None,
                           page_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive interaction plan including exploration and potential errors.

        Args:
            target_element: The final target element to interact with
            nearby_elements: Elements near the target for exploration simulation
            page_context: Additional page context for decision making

        Returns:
            Dictionary containing interaction plan with steps
        """
        plan = {
            'exploration_steps': [],
            'primary_action': None,
            'error_simulation': None,
            'post_action_behavior': [],
            # Tiny, bounded delays to subtly separate phases (executed by StealthManager)
            'pre_primary_delay': 0.0,
            'post_primary_delay': 0.0,
        }

        # Determine if we should explore before acting
        should_explore = self._should_explore_page(target_element, page_context)

        if should_explore and nearby_elements:
            plan['exploration_steps'] = self._plan_exploration_sequence(target_element, nearby_elements)

        # Plan primary action
        plan['primary_action'] = self._plan_primary_action(target_element)

        # Determine if we should simulate an error
        should_error = self._should_simulate_error()
        if should_error and nearby_elements:
            plan['error_simulation'] = self._plan_error_simulation(target_element, nearby_elements)

        # Plan post-action behavior
        plan['post_action_behavior'] = self._plan_post_action_behavior(target_element)

        # Optional tiny pre/post primary delays (bounded) – deterministic via seeded RNG
        rng = self._action_rng or self._rng
        if self._entropy_enabled:
            plan['pre_primary_delay'] = max(0.0, min(0.35, rng.uniform(0.02, 0.25) * (0.8 + 0.4 * self.profile.deliberation_tendency)))
            plan['post_primary_delay'] = max(0.0, min(0.6, rng.uniform(0.05, 0.4) * (self.profile.reaction_time_ms / 300.0)))

        return plan

    def _should_explore_page(self, target_element: Dict[str, Any],
                           page_context: Dict[str, Any] = None) -> bool:
        """Determine if agent should explore page before taking action."""
        base_probability = 0.3

        # Novice users explore more
        tech_factor = 1.5 - self.profile.tech_savviness

        # Unfamiliar pages trigger more exploration
        familiarity_factor = 1.0 - self.behavioral_state.familiarity_score

        # Complex elements trigger more exploration
        element_complexity = self._estimate_element_complexity(target_element)
        complexity_factor = 1.0 + (element_complexity * 0.5)

        # Deliberate personalities explore more
        personality_factor = 0.5 + self.profile.deliberation_tendency

        final_probability = (base_probability * tech_factor * familiarity_factor *
                           complexity_factor * personality_factor)

        rng = getattr(self, "_action_rng", None) or getattr(self, "_rng", random)
        try:
            return rng.random() < min(0.8, final_probability)
        except Exception:
            return random.random() < min(0.8, final_probability)

    def _plan_exploration_sequence(self, target_element: Dict[str, Any],
                                 nearby_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Plan a sequence of exploratory interactions before main action."""
        rng = self._action_rng or self._rng
        exploration_steps: List[Dict[str, Any]] = []

        # Select 1-3 nearby elements to "examine"
        num_explorations = min(3, len(nearby_elements))
        base_num = max(1, int(rng.gauss(num_explorations * 0.6, 0.5)))
        if self._entropy_enabled:
            base_num += rng.choice([-1, 0, 0, 1])  # bias to small deviation
        num_explorations = max(1, min(3, base_num))

        # Deterministic sampling
        try:
            selected_elements = rng.sample(nearby_elements, min(num_explorations, len(nearby_elements)))
        except Exception:
            # Fallback deterministic selection by hashing center coords
            scored = []
            for el in nearby_elements:
                c = el.get('center', {})
                key = f"{c.get('x',0)}:{c.get('y',0)}"
                score = int(hashlib.sha256(key.encode('utf-8')).hexdigest()[:8], 16)
                scored.append((score, el))
            scored.sort(key=lambda t: t[0])
            selected_elements = [el for _, el in scored[:num_explorations]]

        for element in selected_elements:
            # Different types of exploration (fixed set; RNG decides, independent of profile)
            types = ['hover', 'brief_hover', 'scan_to']
            if self._entropy_enabled and rng.random() < 0.15:
                # Occasionally bias toward hover deterministically via RNG only
                types = types + ['hover']
            exploration_type = rng.choice(types)

            step = {
                'type': exploration_type,
                'element': element,
                'duration': self._get_exploration_duration(exploration_type),
                'purpose': 'context_gathering'
            }
            exploration_steps.append(step)

        # Optional micro reordering: at most one adjacent swap (bounded, deterministic)
        if self._entropy_enabled and len(exploration_steps) >= 2:
            i = rng.randrange(0, len(exploration_steps) - 1)
            if rng.random() < 0.35:
                exploration_steps[i], exploration_steps[i + 1] = exploration_steps[i + 1], exploration_steps[i]

        # Optional micro no-op hover: insert a brief_hover on one selected element
        if self._entropy_enabled and exploration_steps and rng.random() < 0.25:
            insert_idx = rng.randrange(0, len(exploration_steps) + 1)
            base_el = rng.choice(selected_elements)
            micro_step = {
                'type': 'brief_hover',
                'element': base_el,
                'duration': max(0.12, min(0.4, self._get_exploration_duration('brief_hover') * 0.5)),
                'purpose': 'micro_noop_hover'
            }
            exploration_steps.insert(insert_idx, micro_step)

        return exploration_steps

    def _plan_primary_action(self, target_element: Dict[str, Any]) -> Dict[str, Any]:
        """Plan the main interaction with the target element."""
        element_type = target_element.get('tag_name', 'unknown')

        action = {
            'type': 'click',
            'element': target_element,
            'approach': 'direct'
        }

        # Some elements benefit from different interaction approaches
        rng = self._action_rng or self._rng
        if element_type in ['input', 'textarea']:
            action['type'] = 'focus_and_type'
            # Sometimes users click to position cursor first
            if rng.random() < 0.3:
                action['approach'] = 'click_then_clear_then_type'
        elif element_type == 'select':
            action['type'] = 'click_and_select'

        return action

    def _should_simulate_error(self) -> bool:
        """Determine if we should simulate a user error."""
        p = self._compute_error_probability()
        rng = self._action_rng or self._rng
        return rng.random() < p

    def _compute_error_probability(self) -> float:
        """Compute low-probability error chance, modulated by stress/familiarity/impulsivity.

        Returns a bounded probability in [0.01, 0.2]. No RNG used to keep it stable.
        """
        base = float(self.profile.error_proneness)
        # Stress and low familiarity increase errors; impulsivity has a moderate effect
        stress_factor = 1.0 + 0.6 * float(self.behavioral_state.stress_level)
        familiarity = float(self.behavioral_state.familiarity_score)
        familiarity_factor = 1.0 + 0.8 * (1.0 - familiarity)
        impulsivity_factor = 1.0 + 0.4 * float(self.profile.impulsivity)
        p = base * stress_factor * familiarity_factor * impulsivity_factor
        # Bound to realistic, low-probability range
        return max(0.01, min(0.2, p))

    def _plan_error_simulation(self, target_element: Dict[str, Any],
                             nearby_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Plan a realistic error and correction sequence."""
        # Check if this is a typing action based on element context
        is_typing_action = target_element.get('text_content') is not None

        rng = self._action_rng or self._rng
        # Weighted, context-appropriate error types
        if is_typing_action:
            # Higher weights for wrong_focus when unfamiliar; premature_typing with impulsivity;
            # typo/typo_sequence with stress and error_proneness
            fam = float(self.behavioral_state.familiarity_score)
            # IMPORTANT: Avoid depending on profile.error_proneness here because profiles are
            # randomized per instance. Using it would break determinism across engines even
            # when run_seed/action_seed are identical (as seen in tests). Instead, derive
            # the "typo" tendency from behavioral state (stress) and a small base.
            stress = float(self.behavioral_state.stress_level)
            weights = {
                'wrong_focus': 0.3 + 0.7 * (1.0 - fam),
                'premature_typing': 0.2 + 0.6 * float(self.profile.impulsivity),
                'typo_sequence': 0.2 + 0.6 * stress,
                'typo': 0.25 + 0.5 * stress,
            }
            keys = list(weights.keys())
            total = sum(weights.values())
            r = rng.random() * total
            acc = 0.0
            error_type = keys[-1]
            for k in keys:
                acc += weights[k]
                if r <= acc:
                    error_type = k
                    break
        else:
            # Clicking context: wrong_click more likely when unfamiliar/low precision;
            # premature_action with higher impulsivity
            fam = float(self.behavioral_state.familiarity_score)
            weights = {
                'wrong_click': 0.5 + 0.7 * (1.0 - fam) + 0.5 * (1.0 - float(self.profile.motor_precision)),
                'premature_action': 0.3 + 0.7 * float(self.profile.impulsivity),
            }
            keys = list(weights.keys())
            total = sum(weights.values())
            r = rng.random() * total
            acc = 0.0
            error_type = keys[-1]
            for k in keys:
                acc += weights[k]
                if r <= acc:
                    error_type = k
                    break

        if error_type == 'wrong_click' and nearby_elements:
            # Click a nearby element by mistake
            wrong_element = self._select_plausible_wrong_target(target_element, nearby_elements)
            return {
                'type': 'wrong_click',
                'wrong_element': wrong_element,
                'correction_delay': rng.uniform(0.5, 2.0),
                'correction_action': 'click_correct_target'
            }
        elif error_type == 'wrong_focus' and nearby_elements:
            # Focus wrong input field first (typing-specific)
            wrong_element = self._select_plausible_wrong_target(target_element, nearby_elements)
            return {
                'type': 'wrong_focus',
                'wrong_element': wrong_element,
                'correction_delay': rng.uniform(0.3, 0.8),
                'correction_action': 'focus_correct_element'
            }
        elif error_type == 'premature_typing':
            # Start typing before fully focused (typing-specific)
            return {
                'type': 'premature_typing',
                'premature_text': rng.choice(['a', 'th', 'w']),
                'correction_delay': rng.uniform(0.5, 1.0),
                'correction_method': 'backspace'
            }
        elif error_type == 'typo_sequence':
            # Multiple typos in typing sequence (typing-specific)
            return {
                'type': 'typo_sequence',
                'typo_count': rng.randint(2, 4),
                'correction_method': rng.choice(['backspace', 'select_all_retype'])
            }
        elif error_type == 'typo':
            return {
                'type': 'typo',
                'typo_count': rng.randint(1, 3),
                'correction_method': rng.choice(['backspace', 'select_all_retype'])
            }
        elif error_type == 'premature_action':
            return {
                'type': 'premature_action',
                'pause_duration': rng.uniform(0.3, 1.0),
                'continuation': 'complete_intended_action'
            }

        return None

    def _select_plausible_wrong_target(self, target_element: Dict[str, Any],
                                     nearby_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select a plausible wrong target for error simulation."""
        target_pos = target_element.get('center', {})
        target_x, target_y = target_pos.get('x', 0), target_pos.get('y', 0)

        # Find elements within reasonable mistake distance (30-100 pixels)
        candidates = []
        for element in nearby_elements:
            elem_pos = element.get('center', {})
            elem_x, elem_y = elem_pos.get('x', 0), elem_pos.get('y', 0)

            distance = math.sqrt((elem_x - target_x)**2 + (elem_y - target_y)**2)
            if 30 <= distance <= 100:
                candidates.append(element)

        rng = self._action_rng or self._rng
        return rng.choice(candidates) if candidates else nearby_elements[0]

    def _plan_post_action_behavior(self, target_element: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan behavior after completing the main action."""
        rng = self._action_rng or self._rng
        behaviors: List[Dict[str, Any]] = []

        # Sometimes users pause to observe results
        if rng.random() < 0.4:
            behaviors.append({
                'type': 'observation_pause',
                'duration': rng.uniform(0.5, 2.0),
                'purpose': 'verify_action_result'
            })

        # Sometimes small corrective mouse movements
        if rng.random() < 0.2:
            behaviors.append({
                'type': 'micro_adjustment',
                'movement_radius': rng.uniform(5, 20),
                'purpose': 'hand_settling'
            })

        # Optional bounded reordering between behaviors (swap if both present)
        if self._entropy_enabled and len(behaviors) == 2 and rng.random() < 0.35:
            behaviors[0], behaviors[1] = behaviors[1], behaviors[0]

        return behaviors

    def _estimate_element_complexity(self, element: Dict[str, Any]) -> float:
        """Estimate the complexity of interacting with an element."""
        base_complexity = 0.5

        element_type = element.get('tag_name', '').lower()

        # Different element types have different complexity
        complexity_map = {
            'button': 0.3,
            'a': 0.3,
            'input': 0.6,
            'select': 0.7,
            'textarea': 0.8,
            'form': 0.9
        }

        type_complexity = complexity_map.get(element_type, base_complexity)

        # Size affects complexity (smaller = harder)
        size = element.get('size', {})
        width = size.get('width', 100)
        height = size.get('height', 30)
        area = width * height

        # Very small elements are harder to click
        if area < 1000:  # Less than ~30x30 pixels
            type_complexity *= 1.3

        return min(1.0, type_complexity)

    def _get_exploration_duration(self, exploration_type: str) -> float:
        """Get duration for different types of exploration actions."""
        rng = self._action_rng or self._rng
        durations = {
            'hover': rng.uniform(0.8, 2.0),
            'brief_hover': rng.uniform(0.3, 0.8),
            'scan_to': rng.uniform(0.2, 0.5)
        }
        if self._entropy_enabled:
            # Light jitter of the chosen duration distribution
            for k in durations:
                durations[k] *= rng.uniform(0.9, 1.15)

        base_duration = durations.get(exploration_type, 1.0)
        # For deterministic planning consistency across profiles, avoid personality scaling here.
        return base_duration


class StealthManager:
    """
    Main Stealth Coordination System

    Orchestrates all stealth components and provides the primary interface
    for the browser automation system to use human-like behaviors.
    """

    def __init__(self, human_profile: Optional[HumanProfile] = None):
        import os
        self.profile = human_profile or HumanProfile.create_random_profile()
        self.behavioral_state = AgentBehavioralState()

        # Feature flags (populated by session after construction)
        self.entropy_enabled = False
        self.behavioral_planning_enabled = False
        self.page_exploration_enabled = False
        self.error_simulation_enabled = False
        self.navigation_enabled = False
        self.typing_enabled = True
        self.scroll_enabled = True

        # Initialize engines
        try:
            run_seed_env = os.environ.get('STEALTH_RUN_SEED')
            run_seed = int(run_seed_env) if run_seed_env is not None and run_seed_env.strip() != '' else None
        except Exception:
            run_seed = None
        self.timing_engine = CognitiveTimingEngine(
            self.profile,
            self.behavioral_state,
            entropy_enabled=self.entropy_enabled,
            run_seed=run_seed,
        )
        self.motion_engine = BiometricMotionEngine(
            self.profile,
            self.behavioral_state,
            entropy_enabled=self.entropy_enabled,
            run_seed=run_seed,
        )
        self.interaction_engine = HumanInteractionEngine(
            self.profile,
            self.behavioral_state,
            entropy_enabled=self.entropy_enabled,
            run_seed=run_seed,
        )

        # Bounded profile drift to avoid a static personality fingerprint (entropy only)
        self._actions_since_last_drift = 0

        # Session tracking
        self.action_count = 0

        # Initialize RNG attributes used in exploration methods
        import random
        self._rng = random.Random()
        self._action_rng: Optional[random.Random] = None

        # Set up logger - integrate with main browser_use logging system
        self.logger = logging.getLogger("browser_use.stealth_manager")
        # Let the logger inherit from the main browser_use logging configuration
        # Don't add separate handlers - use the main logging system

    def _validate_coordinates(self, coordinates, coordinate_name="coordinates") -> Tuple[float, float]:
        """
        Validate coordinates to ensure they are valid numbers before using them in calculations.

        Args:
            coordinates: A tuple/list of (x, y) coordinates or individual x or y value
            coordinate_name: Name of the coordinate for logging purposes

        Returns:
            Validated coordinates as float values

        Raises:
            ValueError: If coordinates are invalid (None, NaN, inf, or not numeric)
        """
        if coordinates is None:
            raise ValueError(f"Invalid {coordinate_name}: None")

        # For tuple/list of coordinates
        if isinstance(coordinates, (tuple, list)):
            if len(coordinates) != 2:
                raise ValueError(f"Invalid {coordinate_name}: Expected (x,y) tuple, got {coordinates}")

            x, y = coordinates

            # Check if both values are numeric
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValueError(f"Invalid {coordinate_name}: Non-numeric values: ({type(x).__name__}, {type(y).__name__})")

            # Convert to float
            x = float(x)
            y = float(y)

            # Check for NaN or inf
            if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                raise ValueError(f"Invalid {coordinate_name}: Contains NaN or Infinity: ({x}, {y})")

            return (x, y)

        # For individual coordinate values
        elif isinstance(coordinates, (int, float)):
            value = float(coordinates)
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"Invalid {coordinate_name}: {value} is NaN or Infinity")
            return value

        else:
            raise ValueError(f"Invalid {coordinate_name} type: {type(coordinates).__name__}")

    async def execute_human_like_click(self, page, element_coordinates: Tuple[float, float],
                                     element_context: Dict[str, Any] = None) -> bool:
        """
        Execute a human-like click with full behavioral simulation.

        Args:
            page: Playwright page object
            element_coordinates: (x, y) coordinates of target element
            element_context: Additional context about the element and page

        Returns:
            Boolean indicating success
        """
        try:
            import os
            self.action_count += 1
            if getattr(self, 'entropy_enabled', False):
                self._maybe_profile_drift()
            # Seed timing per action
            try:
                self.timing_engine.set_action_seed(self.action_count, action_kind="click")
                self.motion_engine.set_action_seed(self.action_count, action_kind="click")
                try:
                    self.interaction_engine.set_action_seed(self.action_count, action_kind="click")
                except Exception:
                    pass
            except Exception:
                pass

            # Validate input coordinates
            try:
                validated_coordinates = self._validate_coordinates(element_coordinates, "element_coordinates")
                target_x, target_y = validated_coordinates
                self.logger.debug(f"🔍 Validated click coordinates: ({target_x}, {target_y})")
            except ValueError as e:
                self.logger.error(f"❌ Invalid click coordinates: {str(e)}")
                raise

            # Check if behavioral planning is enabled
            behavioral_planning_enabled = (
                element_context
                and element_context.get('behavioral_planning', False)
                and getattr(self, 'behavioral_planning_enabled', False)
            )

            if behavioral_planning_enabled:
                self.logger.debug("🧠 Behavioral planning enabled, generating interaction plan")

                # Prepare target element data for interaction planning
                target_element = {
                    'center': {'x': target_x, 'y': target_y},
                    'tag_name': element_context.get('tag_name', 'unknown'),
                    'size': element_context.get('size', {'width': 50, 'height': 30})
                }

                # Get nearby elements from context if available
                nearby_elements = element_context.get('nearby_elements', [])
                page_context = {
                    'url': page.url if hasattr(page, 'url') else '',
                    'complexity': element_context.get('complexity', 1.0)
                }

                try:
                    # Generate comprehensive interaction plan
                    interaction_plan = self.interaction_engine.get_interaction_plan(
                        target_element, nearby_elements, page_context
                    )

                    # Store a flag to track planning usage for counters
                    element_context['_planning_used'] = True
                    element_context['_interaction_plan'] = interaction_plan

                    # Execute the full interaction plan
                    await self.execute_interaction_plan(page, interaction_plan)

                    # Record success and return
                    self.behavioral_state.record_action_result(True)
                    try:
                        self.timing_engine.clear_action_seed()
                        self.motion_engine.clear_action_seed()
                        try:
                            self.interaction_engine.clear_action_seed()
                        except Exception:
                            pass
                    except Exception:
                        pass
                    self.logger.debug("🧠 Behavioral planning interaction completed successfully")
                    return True

                except Exception as planning_e:
                    self.logger.warning(f"⚠️ Behavioral planning failed, falling back to standard click: {type(planning_e).__name__}")
                    # Store flag for fallback tracking
                    element_context['_planning_fallback'] = True
                    # Fall through to standard click behavior

            # Standard click behavior (used when behavioral planning is disabled or fails)
            self.logger.debug("🖱️ Using standard human-like click behavior")

            # Get current mouse position
            current_mouse = await self._get_current_mouse_position(page)
            start_x, start_y = current_mouse

            # Pre-action deliberation
            complexity = element_context.get('complexity', 1.0) if element_context else 1.0
            familiarity = element_context.get('familiarity', 0.5) if element_context else 0.5

            deliberation_time = self.timing_engine.get_deliberation_delay(complexity, familiarity)
            await asyncio.sleep(deliberation_time)

            # Check for standalone error simulation (not through behavioral planning)
            standalone_error_simulation_enabled = bool(getattr(self, 'error_simulation_enabled', False))
            if standalone_error_simulation_enabled and not behavioral_planning_enabled:
                should_simulate_error = self.interaction_engine._should_simulate_error()
                if should_simulate_error:
                    self.logger.debug("🎭 Standalone error simulation triggered for click")

                    # Track standalone error simulation enabled
                    if hasattr(self, 'session') and self.session:
                        self.session._stealth_counters['stealth.error_simulation.standalone_enabled'] += 1
                        self.session._stealth_counters['stealth.error_simulation.click_errors_triggered'] += 1

                    # Generate error simulation plan
                    target_element = {
                        'center': {'x': target_x, 'y': target_y},
                        'tag_name': element_context.get('tag_name', 'button') if element_context else 'button'
                    }
                    nearby_elements = element_context.get('nearby_elements', []) if element_context else []

                    error_sim = self.interaction_engine._plan_error_simulation(target_element, nearby_elements)
                    if error_sim:
                        self.logger.debug(f"🎭 Executing standalone click error simulation: {error_sim['type']}")
                        await self._execute_error_simulation(page, error_sim)

                        # Track error simulation
                        if element_context:
                            element_context['_error_simulated'] = True
                            element_context['_error_type'] = error_sim['type']

            # Generate movement path
            movement_path = self.motion_engine.generate_movement_path(
                start_x, start_y, target_x, target_y
            )

            # Execute movement
            await self._execute_mouse_movement(page, movement_path)

            # Check for overshoot
            distance = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
            if self.motion_engine.should_overshoot_target(distance):
                # Execute overshoot and correction
                overshoot_path = self.motion_engine.generate_overshoot_correction(target_x, target_y)
                await self._execute_mouse_movement(page, overshoot_path)

            # Final settling time
            settle_time = self.timing_engine.get_mouse_settle_time(distance)
            await asyncio.sleep(settle_time)

            # Execute click
            await page.mouse.click(target_x, target_y)

            # Record success and return
            self.behavioral_state.record_action_result(True)
            try:
                self.timing_engine.clear_action_seed()
                self.motion_engine.clear_action_seed()
                try:
                    self.interaction_engine.clear_action_seed()
                except Exception:
                    pass
            except Exception:
                pass
            return True

        except Exception as e:
            self.behavioral_state.record_action_result(False)
            try:
                self.timing_engine.clear_action_seed()
                self.motion_engine.clear_action_seed()
                try:
                    self.interaction_engine.clear_action_seed()
                except Exception:
                    pass
            except Exception:
                pass
            raise e

    async def execute_human_like_typing(self, page, element_handle, text: str, element_context: Dict[str, Any] = None) -> bool:
        """
        Execute human-like typing with realistic timing, occasional errors, and behavioral planning.

        Args:
            page: Playwright page object
            element_handle: Element to type into
            text: Text to type
            element_context: Additional context about the element and page for behavioral planning

        Returns:
            Boolean indicating success
        """
        try:
            import os
            self.action_count += 1
            if getattr(self, 'entropy_enabled', False):
                self._maybe_profile_drift()
            # Seed timing per action
            try:
                self.timing_engine.set_action_seed(self.action_count, action_kind="typing")
                self.motion_engine.set_action_seed(self.action_count, action_kind="typing")
                try:
                    self.interaction_engine.set_action_seed(self.action_count, action_kind="typing")
                except Exception:
                    pass
            except Exception:
                pass

            # Check if behavioral planning is enabled for typing
            behavioral_planning_enabled = (
                element_context
                and element_context.get('behavioral_planning', False)
                and getattr(self, 'behavioral_planning_enabled', False)
            )

            if behavioral_planning_enabled:
                self.logger.debug("🧠 Behavioral planning enabled for typing, generating interaction plan")

                # Prepare target element data for typing interaction planning
                try:
                    # Get element position for planning
                    element_box = await element_handle.bounding_box()
                    if element_box:
                        center_x = element_box['x'] + element_box['width'] / 2
                        center_y = element_box['y'] + element_box['height'] / 2
                    else:
                        center_x, center_y = 200.0, 200.0  # Fallback coordinates

                    target_element = {
                        'center': {'x': center_x, 'y': center_y},
                        'tag_name': element_context.get('tag_name', 'input'),
                        'size': element_context.get('size', {'width': 200, 'height': 30}),
                        'text_content': text[:50] + '...' if len(text) > 50 else text  # Preview of text to type
                    }

                    # Get nearby elements from context if available
                    nearby_elements = element_context.get('nearby_elements', [])
                    page_context = {
                        'url': page.url if hasattr(page, 'url') else '',
                        'complexity': element_context.get('complexity', 0.8),  # Typing is generally more complex
                        'action_type': 'typing',
                        'text_length': len(text)
                    }

                    # Generate comprehensive interaction plan for typing
                    interaction_plan = self.interaction_engine.get_interaction_plan(
                        target_element, nearby_elements, page_context
                    )

                    # Execute pre-typing exploration if enabled and planned
                    exploration_enabled = bool(getattr(self, 'page_exploration_enabled', False))
                    if exploration_enabled and interaction_plan.get('exploration_steps'):
                        self.logger.debug(f"🔍 Executing {len(interaction_plan['exploration_steps'])} exploration steps before typing")
                        exploration_metrics = await self._execute_exploration_sequence(
                            page, interaction_plan['exploration_steps'], element_context
                        )
                        self.logger.debug(f"📊 Typing exploration completed: {exploration_metrics['steps_executed']}/{len(interaction_plan['exploration_steps'])} steps, "
                                        f"duration={exploration_metrics['total_duration']:.2f}s, "
                                        f"success_rate={exploration_metrics['timing_breakdown'].get('success_rate', 0):.2f}")
                        element_context['_typing_exploration_used'] = True
                        element_context['_typing_exploration_metrics'] = exploration_metrics

                    # Execute any planned pre-typing errors (e.g., clicking wrong field first)
                    error_sim = interaction_plan.get('error_simulation')
                    if error_sim and error_sim.get('type') in ['wrong_focus', 'premature_typing']:
                        self.logger.debug(f"🎭 Executing typing error simulation: {error_sim['type']}")
                        await self._execute_typing_error_simulation(page, element_handle, error_sim)

                    # Mark that planning was used
                    element_context['_typing_planning_used'] = True
                    element_context['_interaction_plan'] = interaction_plan

                    self.logger.debug("🧠 Behavioral planning typing interaction completed successfully")

                except Exception as planning_e:
                    self.logger.warning(f"⚠️ Behavioral planning failed for typing, falling back to standard typing: {type(planning_e).__name__}")
                    element_context['_typing_planning_fallback'] = True
                    # Fall through to standard typing behavior

            # Standard or fallback typing behavior
            if not behavioral_planning_enabled or element_context.get('_typing_planning_fallback', False):
                self.logger.debug("⌨️ Using standard human-like typing behavior")

            # Focus the element first
            await element_handle.focus()

            # Pre-typing deliberation (deciding what to type)
            complexity = element_context.get('complexity', 0.8) if element_context else 0.8
            familiarity = element_context.get('familiarity', 0.6) if element_context else 0.6
            deliberation_time = self.timing_engine.get_deliberation_delay(complexity, familiarity)
            await asyncio.sleep(deliberation_time)

            # Check for standalone error simulation (not through behavioral planning)
            standalone_error_simulation_enabled = bool(getattr(self, 'error_simulation_enabled', False))
            if standalone_error_simulation_enabled and not behavioral_planning_enabled:
                should_simulate_error = self.interaction_engine._should_simulate_error()
                if should_simulate_error:
                    self.logger.debug("🎭 Standalone error simulation triggered for typing")

                    # Track standalone error simulation enabled
                    if hasattr(self, 'session') and self.session:
                        self.session._stealth_counters['stealth.error_simulation.standalone_enabled'] += 1
                        self.session._stealth_counters['stealth.error_simulation.typing_errors_triggered'] += 1

                    # Generate typing error simulation
                    # For typing, we can use the same planning method with a simple element structure
                    target_element = {
                        'center': {'x': 100, 'y': 100},  # Dummy coordinates for typing
                        'tag_name': 'input'
                    }
                    nearby_elements = []  # No nearby elements for typing errors
                    error_sim = self.interaction_engine._plan_error_simulation(target_element, nearby_elements)
                    if error_sim:
                        self.logger.debug(f"🎭 Executing standalone typing error simulation: {error_sim['type']}")

                        # Use the existing element_handle for typing error simulation
                        await self._execute_typing_error_simulation(page, element_handle, error_sim)

                        # Track error simulation
                        if element_context:
                            element_context['_error_simulated'] = True
                            element_context['_error_type'] = error_sim['type']
                        return

            # Generate typing sequence with potential errors
            typing_sequence = self._generate_typing_sequence(text)

            # Execute typing sequence
            await self._execute_typing_sequence(page, typing_sequence)

            # Execute post-typing behaviors if planned
            if behavioral_planning_enabled and element_context.get('_interaction_plan'):
                interaction_plan = element_context['_interaction_plan']
                for behavior in interaction_plan.get('post_action_behavior', []):
                    await self._execute_post_action_behavior(page, behavior)

            # Record success and return
            self.behavioral_state.record_action_result(True)
            try:
                self.timing_engine.clear_action_seed()
                self.motion_engine.clear_action_seed()
                try:
                    self.interaction_engine.clear_action_seed()
                except Exception:
                    pass
            except Exception:
                pass
            return True

        except Exception as e:
            self.behavioral_state.record_action_result(False)
            try:
                self.timing_engine.clear_action_seed()
                self.motion_engine.clear_action_seed()
                try:
                    self.interaction_engine.clear_action_seed()
                except Exception:
                    pass
            except Exception:
                pass
            raise e

    async def execute_human_like_navigation(self, page, url: str, context: Dict[str, Any] = None) -> bool:
        """
        Execute human-like navigation with URL typing simulation and cognitive patterns.

        Args:
            page: Playwright page object
            url: URL to navigate to
            context: Additional context about navigation (complexity, familiarity, etc.)

        Returns:
            Boolean indicating success
        """
        try:
            self.action_count += 1
            if getattr(self, 'entropy_enabled', False):
                self._maybe_profile_drift()
            # Seed timing per action
            try:
                self.timing_engine.set_action_seed(self.action_count, action_kind="navigation")
                self.motion_engine.set_action_seed(self.action_count, action_kind="navigation")
                try:
                    self.interaction_engine.set_action_seed(self.action_count, action_kind="navigation")
                except Exception:
                    pass
            except Exception:
                pass

            # Validate URL input
            if not url or not isinstance(url, str):
                raise ValueError(f"Invalid URL: {url}")

            url = url.strip()
            if not url:
                raise ValueError("Empty URL after stripping")

            # Extract context parameters
            complexity = context.get('complexity', 1.0) if context else 1.0
            familiarity = context.get('familiarity', 0.5) if context else 0.5
            nav_type = context.get('nav_type', 'manual') if context else 'manual'  # manual, bookmark, link

            self.logger.debug(f"🧭 Starting human-like navigation to: {url[:50]}...")

            # Pre-navigation cognitive deliberation
            deliberation_time = self.timing_engine.get_deliberation_delay(complexity, familiarity)
            await asyncio.sleep(deliberation_time)

            # Focus address bar with human-like click timing
            try:
                # Get address bar element - try multiple selectors
                address_bar = None
                address_selectors = [
                    'input[name="url"]',
                    'input[type="url"]',
                    '.address-bar',
                    '[data-testid="address-bar"]',
                    'input.addressbar',
                    'omnibox'
                ]

                for selector in address_selectors:
                    try:
                        address_bar = await page.wait_for_selector(selector, timeout=500)
                        if address_bar:
                            break
                    except Exception:
                        continue

                if address_bar:
                    # Human-like focus with click simulation
                    box = await address_bar.bounding_box()
                    if box:
                        center_x = box['x'] + box['width'] / 2
                        center_y = box['y'] + box['height'] / 2

                        # Use existing mouse movement for address bar focus
                        current_mouse = await self._get_current_mouse_position(page)
                        movement_path = self.motion_engine.generate_movement_path(
                            current_mouse[0], current_mouse[1], center_x, center_y
                        )
                        await self._execute_mouse_movement(page, movement_path)
                        await page.mouse.click(center_x, center_y)

                        # Brief pause after clicking address bar
                        await asyncio.sleep(random.uniform(0.1, 0.3))
                else:
                    # Fallback: try to focus on page and use Ctrl+L
                    await page.keyboard.press('Control+l')
                    await asyncio.sleep(random.uniform(0.2, 0.4))

            except Exception as focus_e:
                self.logger.debug(f"Address bar focus failed, using keyboard shortcut: {type(focus_e).__name__}")
                # Fallback to Ctrl+L
                try:
                    await page.keyboard.press('Control+l')
                    await asyncio.sleep(random.uniform(0.2, 0.4))
                except Exception as keyboard_e:
                    self.logger.debug(f"Keyboard shortcut failed: {type(keyboard_e).__name__}")
                    # If address bar simulation fails completely, fall back to direct navigation
                    self.logger.debug("Address bar simulation failed completely, using direct navigation with stealth timing")

                    # Apply pre-navigation delay to maintain stealth timing
                    await asyncio.sleep(random.uniform(0.5, 1.5))

                    # Use direct navigation as fallback
                    await page.goto(url, wait_until='load', timeout=30000)

                    # Apply post-navigation delay
                    orientation_time = self.timing_engine.get_deliberation_delay(0.8, familiarity)
                    await asyncio.sleep(orientation_time)

                    # Record success and return
                    self.behavioral_state.record_action_result(True)
                    return True

            # Clear existing URL with human-like selection
            try:
                await page.keyboard.press('Control+a')
                await asyncio.sleep(random.uniform(0.05, 0.15))
            except Exception as select_e:
                self.logger.debug(f"URL selection failed: {type(select_e).__name__}")
                # Continue anyway - might be able to type URL

            # Generate URL typing sequence with potential errors
            url_typing_sequence = self._generate_url_typing_sequence(url)

            # Execute URL typing with human-like timing
            try:
                await self._execute_typing_sequence(page, url_typing_sequence)
            except Exception as typing_e:
                self.logger.debug(f"URL typing failed: {type(typing_e).__name__}")
                raise ValueError(f"Failed to type URL during navigation: {str(typing_e)}")

            # Post-typing hesitation (deciding whether to press Enter)
            hesitation_time = self.timing_engine.get_deliberation_delay(0.6, 0.8)
            await asyncio.sleep(hesitation_time)

            # Press Enter to navigate
            try:
                await page.keyboard.press('Enter')

                # Wait a moment for navigation to start
                await asyncio.sleep(random.uniform(0.5, 1.0))

                # Verify navigation actually occurred by checking URL change
                current_url = page.url
                if current_url == 'about:blank' or not any(part in current_url.lower() for part in url.lower().split('://')[1].split('/')[0].split('.')):
                    # Navigation didn't work, fall back to direct navigation
                    self.logger.debug(f"Address bar navigation failed - URL unchanged: {current_url}")
                    await page.goto(url, wait_until='load', timeout=30000)

            except Exception as enter_e:
                self.logger.debug(f"Enter key press failed: {type(enter_e).__name__}")
                # Fall back to direct navigation
                self.logger.debug("Falling back to direct navigation")
                await page.goto(url, wait_until='load', timeout=30000)

            # Post-navigation orientation delay
            orientation_time = self.timing_engine.get_deliberation_delay(0.8, familiarity)
            await asyncio.sleep(orientation_time)

            # Record success and return
            self.behavioral_state.record_action_result(True)
            try:
                self.timing_engine.clear_action_seed()
                self.motion_engine.clear_action_seed()
                try:
                    self.interaction_engine.clear_action_seed()
                except Exception:
                    pass
            except Exception:
                pass
            return True

        except ValueError:
            # Re-raise ValueError as-is (these are our custom error messages)
            self.behavioral_state.record_action_result(False)
            raise
        except Exception as e:
            self.behavioral_state.record_action_result(False)
            try:
                self.timing_engine.clear_action_seed()
                self.motion_engine.clear_action_seed()
                try:
                    self.interaction_engine.clear_action_seed()
                except Exception:
                    pass
            except Exception:
                pass
            raise e

    async def execute_human_like_scroll(self, page, pixels: int, context: Dict[str, Any] = None) -> bool:
        """
        Execute physics-based human scroll with momentum and natural patterns.

        Args:
            page: Playwright page object
            pixels: Number of pixels to scroll (positive = down, negative = up)
            context: Additional context about scroll behavior

        Returns:
            Boolean indicating success
        """
        try:
            self.action_count += 1
            if getattr(self, 'entropy_enabled', False):
                self._maybe_profile_drift()
            # Seed timing per action
            try:
                self.timing_engine.set_action_seed(self.action_count, action_kind="scroll")
                self.motion_engine.set_action_seed(self.action_count, action_kind="scroll")
                try:
                    self.interaction_engine.set_action_seed(self.action_count, action_kind="scroll")
                except Exception:
                    pass
            except Exception:
                pass

            # Validate scroll pixels
            if not isinstance(pixels, (int, float)):
                raise ValueError(f"Invalid scroll pixels: {pixels}")

            pixels = int(pixels)
            if pixels == 0:
                return True  # No scrolling needed

            # Extract context parameters
            content_density = context.get('content_density', 0.5) if context else 0.5  # 0=sparse, 1=dense
            reading_mode = context.get('reading_mode', False) if context else False
            urgency = context.get('urgency', 0.5) if context else 0.5  # 0=leisurely, 1=urgent

            self.logger.debug(f"📜 Starting human-like scroll: {pixels}px")

            # Calculate scroll characteristics
            abs_pixels = abs(pixels)
            scroll_direction = 1 if pixels > 0 else -1

            # Determine number of scroll increments based on distance and behavior
            if abs_pixels <= 200:
                num_increments = random.randint(2, 4)
            elif abs_pixels <= 600:
                num_increments = random.randint(3, 6)
            elif abs_pixels <= 1200:
                num_increments = random.randint(5, 8)
            else:
                num_increments = random.randint(6, 12)

            # Apply behavioral modifiers
            confidence_modifier = self.behavioral_state.get_confidence_modifier()

            # Stressed users scroll more erratically
            if self.behavioral_state.stress_level > 0.6:
                num_increments = int(num_increments * 1.3)

            # Confident users scroll more smoothly
            if confidence_modifier > 1.1:
                num_increments = max(2, int(num_increments * 0.8))

            # Generate scroll increments with natural variation
            increments = self._generate_scroll_increments(abs_pixels, num_increments)

            # Pre-scroll cognitive pause (deciding where to scroll)
            pre_scroll_pause = self.timing_engine.get_deliberation_delay(0.4, 0.7) * (1 - urgency)
            await asyncio.sleep(pre_scroll_pause)

            # Execute scroll increments with physics-based timing
            total_scrolled = 0
            for i, increment in enumerate(increments):
                current_scroll = increment * scroll_direction

                # Execute scroll step
                try:
                    await page.evaluate('(delta) => window.scrollBy(0, delta)', current_scroll)
                    total_scrolled += abs(current_scroll)
                except Exception as scroll_e:
                    self.logger.debug(f"Scroll step failed, using alternative: {type(scroll_e).__name__}")
                    # Fallback to mouse wheel simulation
                    await page.mouse.wheel(0, current_scroll)

                # Inter-scroll timing with physics simulation
                if i < len(increments) - 1:
                    # Generate natural pause between scroll increments
                    base_pause = random.uniform(0.08, 0.25)

                    # Content density affects pause duration
                    content_factor = 1.0 + (content_density * 0.5)  # Dense content = longer pauses

                    # Reading mode creates longer pauses
                    reading_factor = 1.5 if reading_mode else 1.0

                    # Apply behavioral state
                    behavior_factor = confidence_modifier * (1.0 - self.behavioral_state.stress_level * 0.3)

                    pause_duration = base_pause * content_factor * reading_factor * behavior_factor
                    pause_duration = max(0.02, min(0.8, pause_duration))

                    await asyncio.sleep(pause_duration)

            # Natural scroll overshoot and correction for longer scrolls
            if abs_pixels > 400 and random.random() < 0.3:
                overshoot_amount = random.randint(20, 80) * scroll_direction
                await page.evaluate('(delta) => window.scrollBy(0, delta)', overshoot_amount)

                # Correction pause
                await asyncio.sleep(random.uniform(0.2, 0.5))

                # Correction scroll
                correction = -overshoot_amount * random.uniform(0.7, 1.0)
                await page.evaluate('(delta) => window.scrollBy(0, delta)', correction)

            # Post-scroll settling pause
            settling_time = random.uniform(0.1, 0.4) * (1 - urgency)
            await asyncio.sleep(settling_time)

            # Record success and return
            self.behavioral_state.record_action_result(True)
            try:
                self.timing_engine.clear_action_seed()
                self.motion_engine.clear_action_seed()
                try:
                    self.interaction_engine.clear_action_seed()
                except Exception:
                    pass
            except Exception:
                pass
            return True

        except Exception as e:
            self.behavioral_state.record_action_result(False)
            try:
                self.timing_engine.clear_action_seed()
                self.motion_engine.clear_action_seed()
                try:
                    self.interaction_engine.clear_action_seed()
                except Exception:
                    pass
            except Exception:
                pass
            raise e

    async def execute_interaction_plan(self, page, interaction_plan: Dict[str, Any]) -> bool:
        """
        Execute a complete interaction plan including exploration and error simulation.

        Args:
            page: Playwright page object
            interaction_plan: Plan generated by HumanInteractionEngine

        Returns:
            Boolean indicating success
        """
        try:
            # Execute exploration sequence
            exploration_steps = interaction_plan.get('exploration_steps', [])
            if exploration_steps:
                exploration_metrics = await self._execute_exploration_sequence(page, exploration_steps)
                self.logger.debug(f"📊 Interaction plan exploration completed: {exploration_metrics['steps_executed']}/{len(exploration_steps)} steps, "
                                f"duration={exploration_metrics['total_duration']:.2f}s, "
                                f"success_rate={exploration_metrics['timing_breakdown'].get('success_rate', 0):.2f}")

            # Execute error simulation if planned
            error_sim = interaction_plan.get('error_simulation')
            if error_sim:
                await self._execute_error_simulation(page, error_sim)

            # Execute primary action
            primary_action = interaction_plan.get('primary_action')
            if primary_action:
                # Optional pre primary delay
                try:
                    pre_delay = float(interaction_plan.get('pre_primary_delay', 0.0))
                    if pre_delay > 0:
                        await asyncio.sleep(pre_delay)
                except Exception:
                    pass
                await self._execute_primary_action(page, primary_action)
                # Optional post primary delay
                try:
                    post_delay = float(interaction_plan.get('post_primary_delay', 0.0))
                    if post_delay > 0:
                        await asyncio.sleep(post_delay)
                except Exception:
                    pass

            # Execute post-action behaviors
            for behavior in interaction_plan.get('post_action_behavior', []):
                await self._execute_post_action_behavior(page, behavior)

            return True

        except Exception as e:
            self.behavioral_state.record_action_result(False)
            raise e

    # Helper methods for internal operations

    async def _get_current_mouse_position(self, page) -> Tuple[float, float]:
        """Get current mouse position or use a reasonable default."""
        try:
            # Try to get actual mouse position via JavaScript
            position = await page.evaluate("""
                () => {
                    if (window.lastMouseX !== undefined && window.lastMouseY !== undefined) {
                        return { x: window.lastMouseX, y: window.lastMouseY };
                    }
                    return { x: window.innerWidth / 2, y: window.innerHeight / 2 };
                }
            """)

            # Validate position before returning
            try:
                x = float(position.get('x', 0))
                y = float(position.get('y', 0))

                # Check for NaN/Inf
                if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                    raise ValueError(f"Invalid mouse position: ({x}, {y})")

                self.logger.debug(f"🔍 Current mouse position: ({x}, {y})")
                return (x, y)
            except (ValueError, TypeError) as e:
                self.logger.warning(f"⚠️ Invalid mouse position: {str(e)}, using fallback")
                raise  # Let the exception handler use the fallback

        except Exception:
            # Fallback to center of viewport
            viewport = page.viewport_size
            if viewport is None:
                self.logger.warning("⚠️ No viewport info, using default (500, 500)")
                return (500.0, 500.0)

            # Ensure the viewport values are valid
            width = float(viewport.get('width', 1024))
            height = float(viewport.get('height', 768))

            if width <= 0 or height <= 0 or math.isnan(width) or math.isnan(height):
                self.logger.warning(f"⚠️ Invalid viewport dimensions: {width}x{height}, using default")
                width, height = 1024.0, 768.0

            self.logger.debug(f"🔍 Using fallback mouse position: ({width/2}, {height/2})")
            return (width / 2, height / 2)

    async def _execute_mouse_movement(self, page, movement_path: List[Tuple[float, float, float]]) -> None:
        """Execute a mouse movement path with timing."""
        if not movement_path:
            self.logger.warning("⚠️ Empty movement path provided to _execute_mouse_movement")
            return

        self.logger.debug(f"🔍 Executing mouse movement path with {len(movement_path)} points")
        last_time = 0
        last_valid_point = None

        for i, point in enumerate(movement_path):
            try:
                # Validate the point structure
                if not isinstance(point, (tuple, list)) or len(point) != 3:
                    raise ValueError(f"Invalid point structure at index {i}: {point}")

                x, y, timestamp = point

                # Validate x, y coordinates and timestamp
                for name, value in [("x", x), ("y", y), ("timestamp", timestamp)]:
                    if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                        raise ValueError(f"Invalid {name} value: {value}")

                # Wait for the appropriate time
                time_delta = float(timestamp) - last_time
                if time_delta > 0:
                    # Cap time delta to prevent excessive waits
                    capped_delta = min(time_delta, 0.5)
                    await asyncio.sleep(capped_delta)

                # Move mouse to position
                await page.mouse.move(float(x), float(y))

                # Log each coordinate movement
                self.logger.debug(f"🖱️ Mouse moved to position: ({x:.1f}, {y:.1f})")

                # Track position for future reference with safer evaluation
                await page.evaluate("""
                    ([x, y]) => {
                        window.lastMouseX = x;
                        window.lastMouseY = y;
                    }
                """, [x, y])

                last_time = float(timestamp)
                last_valid_point = (x, y, timestamp)

            except Exception as e:
                self.logger.warning(f"⚠️ Error during mouse movement at point {i}: {str(e)}")
                if last_valid_point:
                    # Use last valid point to continue
                    try:
                        x, y, _ = last_valid_point
                        await page.mouse.move(float(x), float(y))
                    except Exception:
                        # If even this fails, just continue with next point
                        pass

    def _generate_typing_sequence(self, text: str) -> List[Dict[str, Any]]:
        """Generate a typing sequence with potential errors and corrections."""
        sequence = []
        prev_char = ''

        # Determine if we should simulate typing errors
        should_error = random.random() < (self.profile.error_proneness * 0.5)
        error_positions = set()

        if should_error:
            # Add 1-3 error positions
            num_errors = min(3, max(1, int(random.gauss(1.5, 0.5))))
            error_positions = set(random.sample(range(len(text)), min(num_errors, len(text))))

        i = 0
        while i < len(text):
            char = text[i]

            # Calculate timing for this character
            keystroke_delay = self.timing_engine.get_keystroke_interval(char, prev_char)

            # Check if this position should have an error
            if i in error_positions:
                # Insert error character
                error_char = self._generate_error_character(char)
                sequence.append({
                    'type': 'type',
                    'text': error_char,
                    'delay': keystroke_delay,
                    'is_error': True
                })

                # Add correction delay and backspace
                correction_delay = random.uniform(0.2, 0.8)
                sequence.append({
                    'type': 'wait',
                    'delay': correction_delay
                })
                sequence.append({
                    'type': 'key',
                    'key': 'Backspace',
                    'delay': 0.1
                })

                # Now add correct character
                sequence.append({
                    'type': 'type',
                    'text': char,
                    'delay': keystroke_delay * 1.2,  # Slightly slower after correction
                    'is_error': False
                })
            else:
                # Normal character
                sequence.append({
                    'type': 'type',
                    'text': char,
                    'delay': keystroke_delay,
                    'is_error': False
                })

            prev_char = char
            i += 1

        return sequence

    def _generate_url_typing_sequence(self, url: str) -> List[Dict[str, Any]]:
        """Generate a URL typing sequence with potential errors and corrections."""
        sequence = []
        prev_char = ''

        # URLs have different error patterns than regular text
        # Less likely to have errors on protocol and domain parts
        url_parts = url.split('/')
        domain_end = len(url_parts[0]) if '://' in url else len(url_parts[0]) if len(url_parts) > 0 else 0
        if '://' in url:
            domain_end = len(url_parts[0]) + 3 + len(url_parts[2]) if len(url_parts) > 2 else len(url)

        # Determine if we should simulate typing errors (lower rate for URLs)
        should_error = random.random() < (self.profile.error_proneness * 0.3)
        error_positions = set()

        if should_error:
            # Add 1-2 error positions, avoid critical parts
            num_errors = min(2, max(1, int(random.gauss(1.2, 0.3))))
            # Only add errors in path/query parts, not domain
            safe_positions = [i for i in range(len(url)) if i > domain_end]
            if safe_positions:
                error_positions = set(random.sample(safe_positions, min(num_errors, len(safe_positions))))

        i = 0
        while i < len(url):
            char = url[i]

            # Calculate timing for this character (URLs typed more carefully)
            base_keystroke_delay = self.timing_engine.get_keystroke_interval(char, prev_char)
            # URLs are typed more deliberately
            keystroke_delay = base_keystroke_delay * random.uniform(1.1, 1.4)

            # Check if this position should have an error
            if i in error_positions:
                # Insert error character
                error_char = self._generate_url_error_character(char)
                sequence.append({
                    'type': 'type',
                    'text': error_char,
                    'delay': keystroke_delay,
                    'is_error': True
                })

                # Add correction delay and backspace
                correction_delay = random.uniform(0.3, 1.0)  # Longer for URLs
                sequence.append({
                    'type': 'wait',
                    'delay': correction_delay
                })
                sequence.append({
                    'type': 'key',
                    'key': 'Backspace',
                    'delay': 0.1
                })

                # Now add correct character
                sequence.append({
                    'type': 'type',
                    'text': char,
                    'delay': keystroke_delay * 1.3,  # Even slower after URL correction
                    'is_error': False
                })
            else:
                # Normal character
                sequence.append({
                    'type': 'type',
                    'text': char,
                    'delay': keystroke_delay,
                    'is_error': False
                })

            prev_char = char
            i += 1

        return sequence

    def _generate_url_error_character(self, intended_char: str) -> str:
        """Generate a plausible URL typing error."""
        # URL-specific error patterns
        if intended_char == '.':
            return random.choice([',', '/'])
        elif intended_char == '/':
            return random.choice(['.', '\\'])
        elif intended_char == ':':
            return ';'
        elif intended_char == '-':
            return '_'
        elif intended_char == '_':
            return '-'
        else:
            # Use regular character error for alphanumeric
            return self._generate_error_character(intended_char)

    def _generate_scroll_increments(self, total_pixels: int, num_increments: int) -> List[int]:
        """Generate natural scroll increments with physics-based variation."""
        if num_increments <= 1:
            return [total_pixels]

        # Generate base increments
        base_increment = total_pixels / num_increments
        increments = []

        # Physics-based acceleration profile
        for i in range(num_increments):
            # Natural scroll has acceleration then deceleration
            # Peak velocity around 30-60% of the scroll
            progress = i / (num_increments - 1) if num_increments > 1 else 0

            if progress < 0.3:
                # Acceleration phase
                velocity_factor = 0.6 + (progress / 0.3) * 0.4  # 0.6 to 1.0
            elif progress < 0.6:
                # Peak velocity phase
                velocity_factor = 1.0 + random.uniform(-0.1, 0.1)  # Around 1.0
            else:
                # Deceleration phase
                remaining = (progress - 0.6) / 0.4
                velocity_factor = 1.0 - remaining * 0.4  # 1.0 to 0.6

            # Apply behavioral variation
            stress_variation = 1.0 + (self.behavioral_state.stress_level * random.uniform(-0.2, 0.2))
            confidence_variation = self.behavioral_state.get_confidence_modifier()

            # Calculate final increment
            increment = base_increment * velocity_factor * stress_variation * confidence_variation

            # Add natural random variation
            increment *= random.uniform(0.8, 1.2)

            # Ensure positive and reasonable bounds
            increment = max(10, int(increment))
            increments.append(increment)

        # Adjust to match total (distribute any remainder)
        current_total = sum(increments)
        if current_total != total_pixels:
            difference = total_pixels - current_total
            # Distribute difference across increments
            for i in range(abs(difference)):
                idx = i % len(increments)
                increments[idx] += 1 if difference > 0 else -1

        return increments

    def _generate_error_character(self, intended_char: str) -> str:
        """Generate a plausible typing error for the intended character."""
        # Common typo patterns
        qwerty_neighbors = {
            'a': 's', 's': 'ad', 'd': 'sf', 'f': 'dg', 'g': 'fh', 'h': 'gj',
            'j': 'hk', 'k': 'jl', 'l': 'k', 'q': 'w', 'w': 'qe', 'e': 'wr',
            'r': 'et', 't': 'ry', 'y': 'tu', 'u': 'yi', 'i': 'uo', 'o': 'ip',
            'p': 'o', 'z': 'x', 'x': 'zc', 'c': 'xv', 'v': 'cb', 'b': 'vn',
            'n': 'bm', 'm': 'n'
        }

        neighbors = qwerty_neighbors.get(intended_char.lower(), intended_char)
        if neighbors and len(neighbors) > 1:
            return random.choice(neighbors)
        else:
            # Return a random nearby letter
            return random.choice('abcdefghijklmnopqrstuvwxyz')

    async def _execute_typing_sequence(self, page, typing_sequence: List[Dict[str, Any]]) -> None:
        """Execute a typing sequence with delays and errors."""
        for action in typing_sequence:
            action_type = action['type']
            delay = action.get('delay', 0)

            # Wait for the delay
            if delay > 0:
                await asyncio.sleep(delay)

            if action_type == 'type':
                await page.keyboard.type(action['text'])
            elif action_type == 'key':
                await page.keyboard.press(action['key'])
            elif action_type == 'wait':
                # Additional waiting time (already handled above)
                pass

    async def _execute_exploration_sequence(self, page, exploration_steps: List[Dict[str, Any]],
                                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a complete exploration sequence with comprehensive error handling and monitoring.

        Args:
            page: Playwright page object
            exploration_steps: List of exploration steps to execute
            context: Optional context for exploration metrics and state

        Returns:
            Dict containing execution results and metrics
        """
        if not exploration_steps:
            return {
                'success': True,
                'steps_executed': 0,
                'total_duration': 0.0,
                'error_count': 0,
                'skipped_steps': 0
            }

        # Initialize execution metrics
        execution_metrics = {
            'success': True,
            'steps_executed': 0,
            'total_duration': 0.0,
            'error_count': 0,
            'skipped_steps': 0,
            'step_results': [],
            'timing_breakdown': {}
        }

        start_time = time.time()

        try:
            # Apply profile-based timing adjustments
            timing_modifier = self._calculate_exploration_timing_modifier()

            for i, step in enumerate(exploration_steps):
                step_start_time = time.time()
                step_result = {
                    'step_index': i,
                    'step_type': step.get('type', 'unknown'),
                    'success': False,
                    'duration': 0.0,
                    'error': None
                }

                try:
                    # Execute individual exploration step with enhanced timing
                    await self._execute_enhanced_exploration_step(page, step, timing_modifier, context)

                    step_result['success'] = True
                    execution_metrics['steps_executed'] += 1

                except Exception as e:
                    step_result['error'] = str(e)
                    execution_metrics['error_count'] += 1

                    # Log exploration step error but continue sequence
                    self.logger.debug(f"Exploration step {i} failed: {e}")

                    # Skip remaining steps if too many errors
                    if execution_metrics['error_count'] >= 3:
                        execution_metrics['skipped_steps'] = len(exploration_steps) - i - 1
                        self.logger.warning(f"Stopping exploration sequence after {execution_metrics['error_count']} errors")
                        break

                step_duration = time.time() - step_start_time
                step_result['duration'] = step_duration
                execution_metrics['total_duration'] += step_duration
                execution_metrics['step_results'].append(step_result)

                # Add inter-step delay based on profile characteristics
                inter_step_delay = self._calculate_inter_step_delay(i, len(exploration_steps))
                if inter_step_delay > 0:
                    await asyncio.sleep(inter_step_delay)
                    execution_metrics['total_duration'] += inter_step_delay

            # Calculate timing breakdown
            execution_metrics['timing_breakdown'] = {
                'average_step_duration': execution_metrics['total_duration'] / max(execution_metrics['steps_executed'], 1),
                'total_exploration_time': execution_metrics['total_duration'],
                'success_rate': execution_metrics['steps_executed'] / max(len(exploration_steps), 1)
            }

        except Exception as e:
            execution_metrics['success'] = False
            execution_metrics['error_count'] += 1
            self.logger.error(f"Exploration sequence failed completely: {e}")

            if context:
                context['_exploration_sequence_error'] = str(e)

        finally:
            execution_metrics['total_duration'] = time.time() - start_time

            # Update context with exploration metrics if provided (always execute this)
            if context:
                context['_exploration_metrics'] = execution_metrics
                context['_exploration_sequence_completed'] = execution_metrics['success']

        return execution_metrics

    async def _execute_enhanced_exploration_step(self, page, step: Dict[str, Any],
                                               timing_modifier: float = 1.0,
                                               context: Dict[str, Any] = None) -> None:
        """
        Execute an enhanced exploration step with profile-based timing and movement.

        Args:
            page: Playwright page object
            step: Exploration step definition
            timing_modifier: Profile-based timing adjustment factor
            context: Optional context for tracking metrics
        """
        step_type = step['type']
        element = step['element']
        base_duration = step.get('duration', 0.5)

        # Apply timing modifier based on profile characteristics
        adjusted_duration = base_duration * timing_modifier

        # Get element coordinates with validation
        coords = element.get('center', {})
        x, y = coords.get('x', 0), coords.get('y', 0)

        if x == 0 and y == 0:
            raise ValueError(f"Invalid coordinates for exploration step: {coords}")

        if step_type == 'hover':
            await self._execute_hover_exploration(page, x, y, adjusted_duration, context)
        elif step_type == 'brief_hover':
            await self._execute_brief_hover_exploration(page, x, y, adjusted_duration, context)
        elif step_type == 'scan_to':
            await self._execute_scan_to_exploration(page, x, y, adjusted_duration, context)
        else:
            self.logger.warning(f"Unknown exploration step type: {step_type}")
            raise ValueError(f"Unsupported exploration step type: {step_type}")

    async def _execute_hover_exploration(self, page, x: float, y: float, duration: float,
                                       context: Dict[str, Any] = None) -> None:
        """Execute hover exploration with profile-based movement characteristics."""
        # Generate human-like movement to target
        current_pos = await self._get_current_mouse_position(page)

        # Apply movement smoothness from profile
        num_points = max(8, int(15 * self.profile.movement_smoothness))
        path = self.motion_engine.generate_movement_path(
            current_pos[0], current_pos[1], x, y, num_points=num_points
        )

        # Execute movement with potential overshoot correction
        await self._execute_mouse_movement_with_overshoot(page, path, x, y, context)

        # Apply profile-based hover duration
        hover_duration = self._calculate_profile_adjusted_duration(duration, 'hover')
        await asyncio.sleep(hover_duration)

    async def _execute_brief_hover_exploration(self, page, x: float, y: float, duration: float,
                                             context: Dict[str, Any] = None) -> None:
        """Execute brief hover exploration with quicker movement."""
        current_pos = await self._get_current_mouse_position(page)

        # Brief hover uses fewer movement points for quicker action
        num_points = max(5, int(8 * self.profile.movement_smoothness))
        path = self.motion_engine.generate_movement_path(
            current_pos[0], current_pos[1], x, y, num_points=num_points
        )

        await self._execute_mouse_movement(page, path)

        # Brief hover has shorter duration
        brief_duration = self._calculate_profile_adjusted_duration(duration * 0.6, 'brief_hover')
        await asyncio.sleep(brief_duration)

    async def _execute_scan_to_exploration(self, page, x: float, y: float, duration: float,
                                         context: Dict[str, Any] = None) -> None:
        """Execute scan-to exploration with scanning movement pattern."""
        current_pos = await self._get_current_mouse_position(page)

        # Scanning movement is more direct but with profile-based variation
        num_points = max(6, int(10 * self.profile.movement_smoothness))
        path = self.motion_engine.generate_movement_path(
            current_pos[0], current_pos[1], x, y, num_points=num_points
        )

        # Execute scanning movement (slightly faster than hover)
        for i, (px, py) in enumerate(path):
            await page.mouse.move(px, py)
            # Shorter delays between movement points for scanning
            if i < len(path) - 1:
                await asyncio.sleep(0.01 * self.profile.movement_smoothness)

        # Scan duration is typically shorter
        scan_duration = self._calculate_profile_adjusted_duration(duration * 0.8, 'scan_to')
        await asyncio.sleep(scan_duration)

    def _calculate_exploration_timing_modifier(self) -> float:
        """Calculate timing modifier based on profile characteristics."""
        # Base modifier starts at 1.0
        modifier = 1.0

        # Deliberation tendency affects exploration thoroughness
        modifier *= (0.7 + 0.6 * self.profile.deliberation_tendency)

        # Tech savviness affects exploration speed (more experienced = faster)
        modifier *= (1.2 - 0.4 * self.profile.tech_savviness)

        # Impulsivity affects exploration patience (more impulsive = faster)
        modifier *= (1.1 - 0.3 * self.profile.impulsivity)

        # Reaction time affects overall timing
        reaction_factor = self.profile.reaction_time_ms / 250.0  # Normalized to base 250ms
        modifier *= (0.8 + 0.4 * reaction_factor)

        # Ensure modifier stays within reasonable bounds
        return max(0.5, min(2.0, modifier))

    def _calculate_inter_step_delay(self, step_index: int, total_steps: int) -> float:
        """Calculate delay between exploration steps based on profile."""
        # Base inter-step delay
        base_delay = 0.1 + (0.2 * self.profile.deliberation_tendency)

        # Longer delays for first and last steps
        if step_index == 0 or step_index == total_steps - 1:
            base_delay *= 1.3

        # Add small random variation
        rng = self._action_rng or self._rng
        variation = rng.uniform(0.8, 1.2)

        return base_delay * variation

    def _calculate_profile_adjusted_duration(self, base_duration: float, action_type: str) -> float:
        """Calculate profile-adjusted duration for specific exploration actions."""
        # Start with base duration
        adjusted = base_duration

        # Apply profile-specific adjustments
        if action_type == 'hover':
            # Deliberate users hover longer
            adjusted *= (0.8 + 0.4 * self.profile.deliberation_tendency)
            # Less tech-savvy users need more time to process
            adjusted *= (1.0 + 0.3 * (1.0 - self.profile.tech_savviness))
        elif action_type == 'brief_hover':
            # Brief hovers are less affected by deliberation
            adjusted *= (0.9 + 0.2 * self.profile.deliberation_tendency)
        elif action_type == 'scan_to':
            # Scanning is affected by reaction time and motor precision
            adjusted *= (self.profile.reaction_time_ms / 250.0)
            adjusted *= (1.1 - 0.1 * self.profile.motor_precision)

        # Add small random variation
        rng = self._action_rng or self._rng
        adjusted *= rng.uniform(0.85, 1.15)

        return max(0.1, adjusted)  # Minimum duration to avoid too-fast actions

    async def _execute_mouse_movement_with_overshoot(self, page, path: List[Tuple[float, float]],
                                                   target_x: float, target_y: float,
                                                   context: Dict[str, Any] = None) -> None:
        """Execute mouse movement with potential overshoot based on profile."""
        # Execute normal path
        await self._execute_mouse_movement(page, path)

        # Check for overshoot based on profile
        if random.random() < self.profile.overshoot_tendency:
            # Calculate overshoot amount
            overshoot_distance = random.uniform(5, 20) * (1.0 - self.profile.motor_precision)
            overshoot_angle = random.uniform(0, 2 * math.pi)

            overshoot_x = target_x + overshoot_distance * math.cos(overshoot_angle)
            overshoot_y = target_y + overshoot_distance * math.sin(overshoot_angle)

            # Move to overshoot position
            await page.mouse.move(overshoot_x, overshoot_y)

            # Correction delay based on profile
            correction_delay = (0.1 + 0.2 * (1.0 - self.profile.correction_speed))
            await asyncio.sleep(correction_delay)

            # Correct back to target
            await page.mouse.move(target_x, target_y)

            # Track overshoot correction in context if provided
            if context:
                if '_exploration_overshoot_corrections' not in context:
                    context['_exploration_overshoot_corrections'] = 0
                context['_exploration_overshoot_corrections'] += 1

    async def _execute_exploration_step(self, page, step: Dict[str, Any]) -> None:
        """Execute an exploration step (hover, scan, etc.)."""
        step_type = step['type']
        element = step['element']
        duration = step['duration']

        # Get element coordinates
        coords = element.get('center', {})
        x, y = coords.get('x', 0), coords.get('y', 0)

        if step_type in ['hover', 'brief_hover']:
            # Move to element and hover
            await page.mouse.move(x, y)
            await asyncio.sleep(duration)
        elif step_type == 'scan_to':
            # Quick movement toward element (scanning)
            current_pos = await self._get_current_mouse_position(page)
            path = self.motion_engine.generate_movement_path(
                current_pos[0], current_pos[1], x, y, num_points=10
            )
            await self._execute_mouse_movement(page, path)
            await asyncio.sleep(duration)

    async def _execute_error_simulation(self, page, error_sim: Dict[str, Any]) -> None:
        """Execute error simulation (wrong click, typo, etc.)."""
        error_type = error_sim['type']

        if error_type == 'wrong_click':
            wrong_element = error_sim['wrong_element']
            coords = wrong_element.get('center', {})
            x, y = coords.get('x', 0), coords.get('y', 0)

            # Click wrong element
            await page.mouse.click(x, y)

            # Track wrong click execution
            if hasattr(self, 'session') and self.session:
                self.session._stealth_counters['stealth.error_simulation.wrong_click_executions'] += 1

            # Wait (realization delay)
            await asyncio.sleep(error_sim['correction_delay'])

            # Track correction behavior
            if hasattr(self, 'session') and self.session:
                self.session._stealth_counters['stealth.error_simulation.correction_behaviors_executed'] += 1

            # Correction will be handled by main action

        # Other error types would be handled in their respective contexts

    async def _execute_typing_error_simulation(self, page, element_handle, error_sim: Dict[str, Any]) -> None:
        """Execute typing-specific error simulation (wrong focus, premature typing, etc.)."""
        error_type = error_sim['type']

        if error_type == 'wrong_focus':
            # Simulate clicking on wrong input field first
            wrong_element = error_sim.get('wrong_element')
            if wrong_element:
                coords = wrong_element.get('center', {})
                x, y = coords.get('x', 0), coords.get('y', 0)

                # Click wrong element
                await page.mouse.click(x, y)
                await asyncio.sleep(random.uniform(0.3, 0.8))  # Brief confusion

                # Track wrong focus execution
                if hasattr(self, 'session') and self.session:
                    self.session._stealth_counters['stealth.error_simulation.wrong_focus_executions'] += 1

                # Realize mistake and click correct element
                await element_handle.click()
                await asyncio.sleep(random.uniform(0.2, 0.5))  # Brief pause after correction

                # Track correction behavior
                if hasattr(self, 'session') and self.session:
                    self.session._stealth_counters['stealth.error_simulation.correction_behaviors_executed'] += 1

        elif error_type == 'premature_typing':
            # Start typing before fully focusing (simulate impatience)
            premature_text = error_sim.get('premature_text', 'a')
            await page.keyboard.type(premature_text)
            await asyncio.sleep(random.uniform(0.5, 1.0))  # Pause to realize error

            # Track premature typing execution
            if hasattr(self, 'session') and self.session:
                self.session._stealth_counters['stealth.error_simulation.premature_typing_executions'] += 1

            # Clear the premature text
            for _ in range(len(premature_text)):
                await page.keyboard.press('Backspace')
            await asyncio.sleep(random.uniform(0.2, 0.4))  # Brief pause after correction

            # Track correction behavior
            if hasattr(self, 'session') and self.session:
                self.session._stealth_counters['stealth.error_simulation.correction_behaviors_executed'] += 1

    async def _execute_primary_action(self, page, action: Dict[str, Any]) -> None:
        """Execute the primary action of an interaction plan."""
        action_type = action['type']
        element = action['element']

        coords = element.get('center', {})
        x, y = coords.get('x', 0), coords.get('y', 0)

        if action_type == 'click':
            # Use direct mouse operations to avoid recursion
            # Get current mouse position
            current_mouse = await self._get_current_mouse_position(page)
            start_x, start_y = current_mouse

            # Generate movement path
            movement_path = self.motion_engine.generate_movement_path(
                start_x, start_y, x, y
            )

            # Execute movement
            await self._execute_mouse_movement(page, movement_path)

            # Final settling time
            distance = math.sqrt((x - start_x)**2 + (y - start_y)**2)
            settle_time = self.timing_engine.get_mouse_settle_time(distance)
            await asyncio.sleep(settle_time)

            # Execute click
            await page.mouse.click(x, y)

        elif action_type == 'focus_and_type':
            # This would be handled by the typing method
            pass
        # Add other action types as needed

    async def _execute_post_action_behavior(self, page, behavior: Dict[str, Any]) -> None:
        """Execute post-action behavior (observation, micro-adjustments, etc.)."""
        behavior_type = behavior['type']

        if behavior_type == 'observation_pause':
            await asyncio.sleep(behavior['duration'])
        elif behavior_type == 'micro_adjustment':
            # Small random mouse movement
            current_pos = await self._get_current_mouse_position(page)
            radius = behavior['movement_radius']

            angle = random.uniform(0, 2 * math.pi)
            new_x = current_pos[0] + (math.cos(angle) * radius)
            new_y = current_pos[1] + (math.sin(angle) * radius)

            await page.mouse.move(new_x, new_y)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics for monitoring and debugging."""
        return {
            'session_id': self.session_id,
            'action_count': self.action_count,
            'confidence_level': self.behavioral_state.confidence_level,
            'stress_level': self.behavioral_state.stress_level,
            'familiarity_score': self.behavioral_state.familiarity_score,
            'success_rate': (self.behavioral_state.successful_actions /
                           max(1, self.behavioral_state.total_actions)),
            'profile_type': f"Tech-{self.profile.tech_savviness:.1f}_Prec-{self.profile.motor_precision:.1f}"
        }

    # --- Internal: bounded profile drift ---
    def _maybe_profile_drift(self) -> None:
        try:
            self._actions_since_last_drift += 1
            if self._actions_since_last_drift < 3:
                return
            self._actions_since_last_drift = 0

            # Small, bounded drift to avoid static fingerprint
            self.profile.typing_speed_wpm = float(max(25.0, min(120.0, self.profile.typing_speed_wpm * random.uniform(0.98, 1.02))))
            self.profile.motor_precision = float(max(0.4, min(0.99, self.profile.motor_precision + random.uniform(-0.02, 0.02))))
            self.profile.reaction_time_ms = float(max(150.0, min(550.0, self.profile.reaction_time_ms * random.uniform(0.97, 1.03))))

        except Exception:
            # Never fail core flows due to drift
            pass


# Factory functions for easy integration

def create_stealth_manager(profile_type: str = "random") -> StealthManager:
    """
    Factory function to create a StealthManager with predefined profiles.

    Args:
        profile_type: "random", "expert", "novice", or "custom"

    Returns:
        Configured StealthManager instance
    """
    if profile_type == "expert":
        profile = HumanProfile.create_expert_profile()
    elif profile_type == "novice":
        profile = HumanProfile.create_novice_profile()
    elif profile_type == "random":
        profile = HumanProfile.create_random_profile()
    else:
        profile = HumanProfile.create_random_profile()

    return StealthManager(profile)


# Global stealth manager instance for browser session integration
_global_stealth_manager: Optional[StealthManager] = None


def get_stealth_manager() -> StealthManager:
    """Get or create the global stealth manager instance."""
    global _global_stealth_manager
    if _global_stealth_manager is None:
        _global_stealth_manager = create_stealth_manager()
    return _global_stealth_manager


def reset_stealth_manager(profile_type: str = "random") -> None:
    """Reset the global stealth manager with a new profile."""
    global _global_stealth_manager
    _global_stealth_manager = create_stealth_manager(profile_type)
