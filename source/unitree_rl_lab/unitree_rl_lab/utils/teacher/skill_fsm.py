"""Explicit skill routing and conditional success accounting for BoxPickPlace.

The four behavioural-cloning policies remain separate on purpose.  The FSM
routes the same student-only QR/proprio observation to exactly one policy and
advances only after a physical success predicate.  It never teleports the Box
or silently falls back to a previous phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Mapping

import numpy as np

from .student_observation import validate_student_observation


class SkillPhase(IntEnum):
    APPROACH = 0
    GRASP_LIFT = 1
    CARRY = 2
    PLACE = 3
    COMPLETE = 4


SKILL_PHASE_NAMES = tuple(phase.name.lower() for phase in SkillPhase)


def box_reached_ground_outside_bin(
    *,
    box_height_m: float,
    environment_origin_z_m: float,
    dropped_object_height_m: float,
    box_in_bin: bool,
) -> bool:
    """Classify a physical ground drop without rejecting an airborne release.

    ``box_in_bin`` may include a vertical bound.  Consequently, losing hand
    contact above the aperture is not enough to declare a PLACE failure; the
    Box must first descend below the environment-relative ground threshold.
    """

    values = (
        float(box_height_m),
        float(environment_origin_z_m),
        float(dropped_object_height_m),
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Box/ground drop heights must be finite")
    if values[2] < 0.0:
        raise ValueError("dropped_object_height_m must be non-negative")
    return values[0] < values[1] + values[2] and not bool(box_in_bin)


@dataclass(slots=True)
class SkillSignals:
    """Post-physics predicates used for FSM transitions.

    Every field is a one-dimensional per-environment array.  ``hold_steps``
    must be the continuous strict bilateral-contact + lift counter, not a
    cumulative contact count.
    """

    box_visible: np.ndarray
    at_pregrasp: np.ndarray
    left_contact: np.ndarray
    right_contact: np.ndarray
    lift_height_m: np.ndarray
    hold_steps: np.ndarray
    at_staging: np.ndarray
    box_in_bin: np.ndarray
    released: np.ndarray
    place_settle_steps: np.ndarray
    box_dropped: np.ndarray
    bad_orientation: np.ndarray

    def validate(self, num_envs: int) -> "SkillSignals":
        bool_fields = (
            "box_visible",
            "at_pregrasp",
            "left_contact",
            "right_contact",
            "at_staging",
            "box_in_bin",
            "released",
            "box_dropped",
            "bad_orientation",
        )
        for name in bool_fields:
            value = np.asarray(getattr(self, name))
            if value.shape != (num_envs,):
                raise ValueError(
                    f"{name} has shape {value.shape}; expected ({num_envs},)"
                )
            setattr(self, name, value.astype(bool, copy=False))
        self.lift_height_m = np.asarray(self.lift_height_m, dtype=np.float32)
        self.hold_steps = np.asarray(self.hold_steps, dtype=np.int64)
        self.place_settle_steps = np.asarray(
            self.place_settle_steps, dtype=np.int64
        )
        for name in ("lift_height_m", "hold_steps", "place_settle_steps"):
            value = getattr(self, name)
            if value.shape != (num_envs,):
                raise ValueError(
                    f"{name} has shape {value.shape}; expected ({num_envs},)"
                )
        if (
            not np.isfinite(self.lift_height_m).all()
            or np.any(self.hold_steps < 0)
            or np.any(self.place_settle_steps < 0)
        ):
            raise ValueError(
                "lift_height_m must be finite and step counters must be non-negative"
            )
        return self


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    """Masks emitted by one FSM update."""

    approach_succeeded: np.ndarray
    grasp_succeeded: np.ndarray
    carry_succeeded: np.ndarray
    place_succeeded: np.ndarray
    failed: np.ndarray


Policy = Callable[[np.ndarray], np.ndarray]


def bimanual_pregrasp_ready(
    box_position_w: np.ndarray,
    left_fingertip_positions_w: np.ndarray,
    right_fingertip_positions_w: np.ndarray,
    *,
    maximum_tip_distance_m: float = 0.18,
    maximum_cage_midpoint_error_m: float = 0.11,
) -> np.ndarray:
    """Return a per-environment physical hand-approach gate.

    Camera-navigation ``at_pregrasp`` covers base alignment, but the successful
    Arena demonstration can reach the Box with its arms while the pelvis stays
    outside that navigation tolerance.  This gate hands control to BC-2 only
    when one fingertip from each hand is already close and the two closest tips
    form a cage around the Box.
    """

    box = np.asarray(box_position_w, dtype=np.float32)
    left = np.asarray(left_fingertip_positions_w, dtype=np.float32)
    right = np.asarray(right_fingertip_positions_w, dtype=np.float32)
    if box.ndim == 1:
        box = box[None, :]
    if left.ndim == 2:
        left = left[None, :, :]
    if right.ndim == 2:
        right = right[None, :, :]
    if (
        box.ndim != 2
        or box.shape[1:] != (3,)
        or left.ndim != 3
        or right.ndim != 3
        or left.shape[0] != box.shape[0]
        or right.shape[0] != box.shape[0]
        or left.shape[2] != 3
        or right.shape[2] != 3
        or left.shape[1] == 0
        or right.shape[1] == 0
        or not np.isfinite(box).all()
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("Box and fingertip positions have incompatible finite shapes")
    if maximum_tip_distance_m <= 0.0 or maximum_cage_midpoint_error_m <= 0.0:
        raise ValueError("pregrasp distance thresholds must be positive")
    left_distance = np.linalg.norm(left - box[:, None, :], axis=-1)
    right_distance = np.linalg.norm(right - box[:, None, :], axis=-1)
    row = np.arange(box.shape[0])
    left_closest = left[row, np.argmin(left_distance, axis=1)]
    right_closest = right[row, np.argmin(right_distance, axis=1)]
    midpoint_error = np.linalg.norm(
        0.5 * (left_closest + right_closest) - box, axis=-1
    )
    return (
        (left_distance.min(axis=1) <= maximum_tip_distance_m)
        & (right_distance.min(axis=1) <= maximum_tip_distance_m)
        & (midpoint_error <= maximum_cage_midpoint_error_m)
    )


def bimanual_wrists_centered(
    box_position_w: np.ndarray,
    left_wrist_position_w: np.ndarray,
    right_wrist_position_w: np.ndarray,
    *,
    maximum_midpoint_error_m: float = 0.222,
    maximum_distance_asymmetry_m: float = 0.10,
) -> np.ndarray:
    """Require the BC2 handoff to match its nominal wrist geometry boundary."""

    box = np.asarray(box_position_w, dtype=np.float32)
    left = np.asarray(left_wrist_position_w, dtype=np.float32)
    right = np.asarray(right_wrist_position_w, dtype=np.float32)
    if box.ndim == 1:
        box = box[None, :]
    if left.ndim == 1:
        left = left[None, :]
    if right.ndim == 1:
        right = right[None, :]
    if (
        box.ndim != 2
        or box.shape[1:] != (3,)
        or left.shape != box.shape
        or right.shape != box.shape
        or not np.isfinite(box).all()
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("Box and wrist positions require matching finite (N, 3) shapes")
    if maximum_midpoint_error_m <= 0.0 or maximum_distance_asymmetry_m <= 0.0:
        raise ValueError("wrist-centering thresholds must be positive")
    left_distance = np.linalg.norm(left - box, axis=-1)
    right_distance = np.linalg.norm(right - box, axis=-1)
    midpoint_error = np.linalg.norm(0.5 * (left + right) - box, axis=-1)
    return (midpoint_error <= maximum_midpoint_error_m) & (
        np.abs(left_distance - right_distance)
        <= maximum_distance_asymmetry_m
    )


class SkillRouter:
    """Route observations through separate BC policies using physical gates."""

    def __init__(
        self,
        policies: Mapping[SkillPhase, Policy],
        *,
        num_envs: int,
        action_dim: int,
        lift_threshold_m: float = 0.12,
        hold_steps_required: int = 10,
        place_settle_steps_required: int = 10,
    ) -> None:
        required = {
            SkillPhase.APPROACH,
            SkillPhase.GRASP_LIFT,
            SkillPhase.CARRY,
            SkillPhase.PLACE,
        }
        missing = required.difference(policies)
        if missing:
            raise ValueError(
                f"missing skill policies: {sorted(phase.name for phase in missing)}"
            )
        if (
            num_envs <= 0
            or action_dim <= 0
            or hold_steps_required <= 0
            or place_settle_steps_required <= 0
            or lift_threshold_m <= 0.0
        ):
            raise ValueError(
                "num_envs, action_dim, hold/place step requirements and lift_threshold_m must be positive"
            )
        self.policies = dict(policies)
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.lift_threshold_m = float(lift_threshold_m)
        self.hold_steps_required = int(hold_steps_required)
        self.place_settle_steps_required = int(place_settle_steps_required)
        self.phase = np.full(self.num_envs, int(SkillPhase.APPROACH), dtype=np.int8)
        self.failed = np.zeros(self.num_envs, dtype=bool)

    def reset(self, env_ids: np.ndarray | list[int] | None = None) -> None:
        if env_ids is None:
            self.phase.fill(int(SkillPhase.APPROACH))
            self.failed.fill(False)
            return
        ids = np.asarray(env_ids, dtype=np.int64)
        self.phase[ids] = int(SkillPhase.APPROACH)
        self.failed[ids] = False

    def update(self, signals: SkillSignals) -> TransitionBatch:
        """Advance phases from the current physical state and return event masks."""

        signals.validate(self.num_envs)
        active = ~self.failed & (self.phase != int(SkillPhase.COMPLETE))
        fatal = active & signals.bad_orientation
        manipulation = (self.phase == int(SkillPhase.CARRY)) | (
            self.phase == int(SkillPhase.PLACE)
        )
        fatal |= active & manipulation & signals.box_dropped

        approach = active & (self.phase == int(SkillPhase.APPROACH))
        approach_ok = approach & signals.box_visible & signals.at_pregrasp

        # Later gates are evaluated from the phase present at function entry.
        # This prevents a single state from skipping several skills at once.
        grasp = active & (self.phase == int(SkillPhase.GRASP_LIFT))
        bilateral = signals.left_contact & signals.right_contact
        grasp_ok = (
            grasp
            & bilateral
            & (signals.lift_height_m >= self.lift_threshold_m)
            & (signals.hold_steps >= self.hold_steps_required)
        )
        carry = active & (self.phase == int(SkillPhase.CARRY))
        carry_ok = carry & bilateral & signals.at_staging & ~signals.box_dropped
        place = active & (self.phase == int(SkillPhase.PLACE))
        place_ok = (
            place
            & signals.box_in_bin
            & signals.released
            & (signals.place_settle_steps >= self.place_settle_steps_required)
        )

        # A carry snapshot is useful only if the strict bilateral lift/hold
        # predicate is still continuous at staging.  Merely reacquiring one
        # contact frame after losing the Box must not qualify provenance for a
        # PLACE start state.
        carry_ok &= (signals.lift_height_m >= self.lift_threshold_m) & (
            signals.hold_steps >= self.hold_steps_required
        )

        # Report only transitions that were actually committed.  Returning a
        # raw success predicate together with ``fatal=True`` allowed callers
        # to publish a snapshot from the same bad-orientation/drop frame even
        # though the FSM correctly refused to advance its internal phase.
        approach_accepted = approach_ok & ~fatal
        grasp_accepted = grasp_ok & ~fatal
        carry_accepted = carry_ok & ~fatal
        place_accepted = place_ok & ~fatal
        self.failed[fatal] = True
        self.phase[approach_accepted] = int(SkillPhase.GRASP_LIFT)
        self.phase[grasp_accepted] = int(SkillPhase.CARRY)
        self.phase[carry_accepted] = int(SkillPhase.PLACE)
        self.phase[place_accepted] = int(SkillPhase.COMPLETE)
        return TransitionBatch(
            approach_accepted,
            grasp_accepted,
            carry_accepted,
            place_accepted,
            fatal,
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Run only the policy selected for each active environment."""

        observation = validate_student_observation(observation)
        if observation.ndim != 2 or observation.shape[0] != self.num_envs:
            raise ValueError(
                f"batched observation has shape {observation.shape}; expected ({self.num_envs}, 517)"
            )
        actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        for phase in (
            SkillPhase.APPROACH,
            SkillPhase.GRASP_LIFT,
            SkillPhase.CARRY,
            SkillPhase.PLACE,
        ):
            mask = (self.phase == int(phase)) & ~self.failed
            if not np.any(mask):
                continue
            value = np.asarray(
                self.policies[phase](observation[mask]), dtype=np.float32
            )
            expected = (int(mask.sum()), self.action_dim)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError(
                    f"{phase.name} policy returned {value.shape}; expected finite {expected}"
                )
            actions[mask] = value
        return actions

    def step(
        self, observation: np.ndarray, signals: SkillSignals
    ) -> tuple[np.ndarray, TransitionBatch]:
        transitions = self.update(signals)
        return self.act(observation), transitions


class ConditionalSuccessMetrics:
    """Count conditional skill success without denominator leakage."""

    def __init__(self) -> None:
        self.episodes = 0
        self.approach_successes = 0
        self.grasp_successes = 0
        self.carry_successes = 0
        self.place_successes = 0

    def add_episode(
        self, final_phase: SkillPhase | int, *, failed: bool = False
    ) -> None:
        phase = SkillPhase(int(final_phase))
        self.episodes += 1
        reached_grasp = phase >= SkillPhase.GRASP_LIFT
        reached_carry = phase >= SkillPhase.CARRY
        reached_place = phase >= SkillPhase.PLACE
        complete = phase == SkillPhase.COMPLETE and not failed
        self.approach_successes += int(reached_grasp)
        self.grasp_successes += int(reached_carry)
        self.carry_successes += int(reached_place)
        self.place_successes += int(complete)

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "episodes": self.episodes,
            "approach_successes": self.approach_successes,
            "grasp_successes": self.grasp_successes,
            "carry_successes": self.carry_successes,
            "place_successes": self.place_successes,
            "P(approach)": self._ratio(self.approach_successes, self.episodes),
            "P(grasp|approach)": self._ratio(
                self.grasp_successes, self.approach_successes
            ),
            "P(carry|grasp)": self._ratio(self.carry_successes, self.grasp_successes),
            "P(place|carry)": self._ratio(self.place_successes, self.carry_successes),
            "full_task_success": self._ratio(self.place_successes, self.episodes),
        }
