"""DAgger corrective-state detection and compact relabel dataset recording."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from ..teacher_demo_recorder import DEFAULT_HIGH_LEVEL_ACTION_NAMES
from .skill_fsm import SkillPhase
from .student_observation import (
    STUDENT_OBSERVATION_DIM,
    mask_qr_observation,
    validate_student_observation,
)


DAGGER_SCHEMA_NAME = "unitree_box_dagger_corrections"
DAGGER_SCHEMA_VERSION = "1.0.0"


class CorrectionType(IntFlag):
    NONE = 0
    ONE_HAND_FIRST = 1 << 0
    WRIST_OFF_CENTER = 1 << 1
    BOX_ROTATED_IN_HAND = 1 << 2
    LIFT_SLIP = 1 << 3
    QR_TEMPORARILY_MISSING = 1 << 4
    # Teacher-labelled post-close state with an intact QR observation.  This is
    # not one of the five recovery gates; it anchors normal perception while
    # corrective states are oversampled.
    VISIBLE_ANCHOR = 1 << 5


@dataclass(frozen=True, slots=True)
class CorrectiveState:
    left_contact: bool
    right_contact: bool
    left_wrist_position_w: np.ndarray
    right_wrist_position_w: np.ndarray
    box_position_w: np.ndarray
    box_quaternion_wxyz: np.ndarray
    reference_box_quaternion_wxyz: np.ndarray
    box_wrist_relative_speed_mps: float
    lift_height_m: float
    qr_visible: bool
    qr_was_visible: bool


def _unit_quaternion_wxyz(value: Any, name: str) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError(f"{name} must be a finite wxyz quaternion")
    norm = np.linalg.norm(quat)
    if norm < 1.0e-8:
        raise ValueError(f"{name} has zero norm")
    return quat / norm


def quaternion_distance_rad_wxyz(first: Any, second: Any) -> float:
    """Shortest unsigned orientation distance between two wxyz quaternions."""

    first_q = _unit_quaternion_wxyz(first, "first")
    second_q = _unit_quaternion_wxyz(second, "second")
    return float(2.0 * np.arccos(np.clip(abs(np.dot(first_q, second_q)), 0.0, 1.0)))


def classify_corrective_state(
    state: CorrectiveState,
    *,
        wrist_center_tolerance_m: float = 0.222,
    wrist_asymmetry_tolerance_m: float = 0.10,
    box_rotation_tolerance_rad: float = np.deg2rad(15.0),
    slip_speed_threshold_mps: float = 0.15,
    minimum_lift_for_slip_m: float = 0.02,
    qr_dropout_injected: bool = False,
) -> CorrectionType:
    """Tag recovery states without modifying simulator physics."""

    positions = []
    for name in ("left_wrist_position_w", "right_wrist_position_w", "box_position_w"):
        value = np.asarray(getattr(state, name), dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite 3-vector")
        positions.append(value)
    left, right, box = positions
    flags = CorrectionType.NONE
    if bool(state.left_contact) ^ bool(state.right_contact):
        flags |= CorrectionType.ONE_HAND_FIRST

    midpoint_error = np.linalg.norm(0.5 * (left + right) - box)
    left_distance = np.linalg.norm(left - box)
    right_distance = np.linalg.norm(right - box)
    if (
        midpoint_error > wrist_center_tolerance_m
        or abs(left_distance - right_distance) > wrist_asymmetry_tolerance_m
    ):
        flags |= CorrectionType.WRIST_OFF_CENTER

    rotation_error = quaternion_distance_rad_wxyz(
        state.box_quaternion_wxyz, state.reference_box_quaternion_wxyz
    )
    if (
        state.left_contact
        and state.right_contact
        and rotation_error > box_rotation_tolerance_rad
    ):
        flags |= CorrectionType.BOX_ROTATED_IN_HAND

    slipped = state.lift_height_m >= minimum_lift_for_slip_m and (
        state.box_wrist_relative_speed_mps > slip_speed_threshold_mps
        or not (state.left_contact and state.right_contact)
    )
    if slipped:
        flags |= CorrectionType.LIFT_SLIP
    if qr_dropout_injected or (state.qr_was_visible and not state.qr_visible):
        flags |= CorrectionType.QR_TEMPORARILY_MISSING
    return flags


class QrDropoutSchedule:
    """Episode-local, observation-only QR dropout windows for DAgger."""

    def __init__(
        self,
        *,
        probability_per_step: float = 0.005,
        duration_steps: tuple[int, int] = (5, 20),
        seed: int = 0,
    ) -> None:
        if not 0.0 <= probability_per_step <= 1.0:
            raise ValueError("probability_per_step must be in [0, 1]")
        if duration_steps[0] <= 0 or duration_steps[1] < duration_steps[0]:
            raise ValueError("duration_steps must be a positive inclusive range")
        self.probability_per_step = float(probability_per_step)
        self.duration_steps = tuple(int(value) for value in duration_steps)
        self.rng = np.random.default_rng(seed)
        self.remaining = 0

    def reset(self) -> None:
        self.remaining = 0

    def apply(self, observation: Any, *, phase: SkillPhase) -> tuple[np.ndarray, bool]:
        if self.remaining == 0 and self.rng.random() < self.probability_per_step:
            self.remaining = int(
                self.rng.integers(self.duration_steps[0], self.duration_steps[1] + 1)
            )
        active = self.remaining > 0
        if active:
            self.remaining -= 1
            # Grasp/approach depend on Box QR; carry/place depend on Bin QR.
            box = phase in (SkillPhase.APPROACH, SkillPhase.GRASP_LIFT)
            return mask_qr_observation(observation, box=box, bin=not box), True
        return validate_student_observation(observation).copy(), False


class RecoveryDisturbanceSchedule:
    """Seeded, short physical-wrench campaigns for DAgger recovery coverage."""

    TARGETS = (
        CorrectionType.ONE_HAND_FIRST,
        CorrectionType.WRIST_OFF_CENTER,
        CorrectionType.BOX_ROTATED_IN_HAND,
        CorrectionType.LIFT_SLIP,
    )

    def __init__(
        self,
        *,
        seed: int = 0,
        force_n: float = 4.0,
        torque_nm: float = 0.25,
        pulse_steps: int = 6,
    ) -> None:
        if force_n <= 0.0 or torque_nm <= 0.0 or pulse_steps <= 0:
            raise ValueError("disturbance wrench and pulse duration must be positive")
        self.seed = int(seed)
        self.force_n = float(force_n)
        self.torque_nm = float(torque_nm)
        self.pulse_steps = int(pulse_steps)
        self._trigger_step: int | None = None
        self._target = CorrectionType.NONE

    def reset(self, episode: int) -> CorrectionType:
        self._target = self.TARGETS[(int(episode) + self.seed) % len(self.TARGETS)]
        self._trigger_step = None
        return self._target

    def wrench(
        self,
        *,
        step: int,
        close_index: int,
        bilateral_contact: bool,
        lift_height_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        force = np.zeros(3, dtype=np.float32)
        torque = np.zeros(3, dtype=np.float32)
        if self._target in (
            CorrectionType.ONE_HAND_FIRST,
            CorrectionType.WRIST_OFF_CENTER,
        ):
            trigger = close_index + (
                -3 if self._target == CorrectionType.ONE_HAND_FIRST else 5
            )
            active = trigger <= step < trigger + self.pulse_steps
            if active:
                direction = -1.0 if self._target == CorrectionType.ONE_HAND_FIRST else 1.0
                force[1] = direction * self.force_n
        else:
            ready = bilateral_contact and (
                self._target == CorrectionType.BOX_ROTATED_IN_HAND
                or lift_height_m >= 0.02
            )
            if self._trigger_step is None and ready:
                self._trigger_step = step
            active = self._trigger_step is not None and (
                self._trigger_step <= step < self._trigger_step + self.pulse_steps
            )
            if active and self._target == CorrectionType.BOX_ROTATED_IN_HAND:
                torque[2] = self.torque_nm
            elif active and self._target == CorrectionType.LIFT_SLIP:
                force[:] = (0.5 * self.force_n, 0.0, -self.force_n)
        return force, torque


class BalancedCorrectionSelector:
    """Select bounded, campaign-relevant corrections with QR visibility balance.

    DAgger predicates are bit flags and may overlap.  Without a selector, one
    broad predicate can cause every post-close frame to be recorded, drowning
    the short disturbance responses and normal-visible recovery states.  This
    selector keeps only the active physical campaign (plus explicit QR-loss
    frames), caps each episode, and bounds invisible samples relative to the
    visible corrections already accepted in that episode.
    """

    def __init__(
        self,
        *,
        max_samples_per_episode: int = 128,
        max_invisible_to_visible_ratio: float = 3.0,
        invisible_warmup_samples: int = 8,
    ) -> None:
        if max_samples_per_episode <= 0:
            raise ValueError("max_samples_per_episode must be positive")
        if not np.isfinite(max_invisible_to_visible_ratio) or (
            max_invisible_to_visible_ratio <= 0.0
        ):
            raise ValueError("max_invisible_to_visible_ratio must be finite and positive")
        if invisible_warmup_samples < 0:
            raise ValueError("invisible_warmup_samples must be non-negative")
        self.max_samples_per_episode = int(max_samples_per_episode)
        self.max_invisible_to_visible_ratio = float(max_invisible_to_visible_ratio)
        self.invisible_warmup_samples = int(invisible_warmup_samples)
        self.target = CorrectionType.NONE
        self.total = 0
        self.visible = 0
        self.invisible = 0

    def reset(self, target: CorrectionType | int) -> None:
        target = CorrectionType(int(target))
        physical_targets = CorrectionType(0)
        for item in RecoveryDisturbanceSchedule.TARGETS:
            physical_targets |= item
        if target == CorrectionType.NONE or target & ~physical_targets:
            raise ValueError("selector target must be one physical recovery category")
        self.target = target
        self.total = 0
        self.visible = 0
        self.invisible = 0

    def accept(self, flags: CorrectionType | int, *, qr_visible: bool) -> bool:
        flags = CorrectionType(int(flags))
        if self.target == CorrectionType.NONE:
            raise RuntimeError("selector must be reset for an episode before use")
        relevant = (
            bool(flags & self.target)
            or bool(flags & CorrectionType.QR_TEMPORARILY_MISSING)
            or bool(flags & CorrectionType.VISIBLE_ANCHOR)
        )
        if not relevant or self.total >= self.max_samples_per_episode:
            return False
        if not qr_visible:
            visible_allowance = int(
                np.floor(self.visible * self.max_invisible_to_visible_ratio)
            )
            invisible_limit = max(self.invisible_warmup_samples, visible_allowance)
            if self.invisible >= invisible_limit:
                return False
        self.total += 1
        if qr_visible:
            self.visible += 1
        else:
            self.invisible += 1
        return True


@dataclass(slots=True)
class DaggerSample:
    student_observation: Any
    student_action: Any
    teacher_high_level_action: Any
    processed_43d_joint_target: Any
    skill_phase: SkillPhase | int
    correction_type: CorrectionType | int
    qr_visible: bool
    source_episode: int
    source_step: int


class DaggerCorrectionRecorder:
    """Atomically record only states for which a teacher label was queried."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        joint_names: tuple[str, ...] | list[str],
        overwrite: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if len(joint_names) != 43 or len(set(joint_names)) != 43:
            raise ValueError("joint_names must contain 43 unique entries")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not overwrite:
            raise FileExistsError(self.path)
        fd, partial = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".partial", dir=self.path.parent
        )
        os.close(fd)
        self.partial_path = Path(partial)
        self.overwrite = bool(overwrite)
        self.file = h5py.File(self.partial_path, "w", track_order=True)
        self.file.attrs["schema_name"] = DAGGER_SCHEMA_NAME
        self.file.attrs["schema_version"] = DAGGER_SCHEMA_VERSION
        self.file.attrs["metadata_json"] = json.dumps(
            dict(metadata or {}), sort_keys=True
        )
        self.file.create_dataset(
            "joint_names",
            data=np.asarray(joint_names, dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        self.file.create_dataset(
            "teacher_high_level_action_names",
            data=np.asarray(DEFAULT_HIGH_LEVEL_ACTION_NAMES, dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        self.samples = self.file.create_group("samples", track_order=True)
        self._datasets = {
            "student_observation": self._dataset(
                "student_observation", (STUDENT_OBSERVATION_DIM,), np.float32
            ),
            "student_action": self._dataset("student_action", (43,), np.float32),
            "teacher_high_level_action": self._dataset(
                "teacher_high_level_action", (23,), np.float32
            ),
            "processed_43d_joint_target": self._dataset(
                "processed_43d_joint_target", (43,), np.float32
            ),
            "skill_phase": self._dataset("skill_phase", (), np.uint8),
            "correction_type": self._dataset("correction_type", (), np.uint16),
            "qr_visible": self._dataset("qr_visible", (), np.bool_),
            "source_episode": self._dataset("source_episode", (), np.int64),
            "source_step": self._dataset("source_step", (), np.int64),
        }
        self.count = 0
        self.closed = False

    def _dataset(
        self, name: str, trailing: tuple[int, ...], dtype: Any
    ) -> h5py.Dataset:
        return self.samples.create_dataset(
            name,
            shape=(0, *trailing),
            maxshape=(None, *trailing),
            chunks=(128, *trailing),
            compression="gzip",
            compression_opts=1,
            dtype=dtype,
        )

    @staticmethod
    def _vector(value: Any, length: int, name: str) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (length,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite ({length},) vector")
        return result

    def append(self, sample: DaggerSample) -> int:
        correction = CorrectionType(int(sample.correction_type))
        if correction == CorrectionType.NONE:
            raise ValueError("DAgger recorder accepts corrective states only")
        row = {
            "student_observation": validate_student_observation(
                sample.student_observation
            ),
            "student_action": self._vector(sample.student_action, 43, "student_action"),
            "teacher_high_level_action": self._vector(
                sample.teacher_high_level_action, 23, "teacher_high_level_action"
            ),
            "processed_43d_joint_target": self._vector(
                sample.processed_43d_joint_target, 43, "processed_43d_joint_target"
            ),
            "skill_phase": int(SkillPhase(int(sample.skill_phase))),
            "correction_type": int(correction),
            "qr_visible": bool(sample.qr_visible),
            "source_episode": int(sample.source_episode),
            "source_step": int(sample.source_step),
        }
        if row["source_episode"] < 0 or row["source_step"] < 0:
            raise ValueError("source episode and step must be non-negative")
        index = self.count
        for name, value in row.items():
            dataset = self._datasets[name]
            dataset.resize(index + 1, axis=0)
            dataset[index] = value
        self.count += 1
        return index

    def finalize(self) -> Path:
        if self.closed:
            raise RuntimeError("DAgger recorder is closed")
        if self.count == 0:
            raise RuntimeError("cannot publish an empty DAgger dataset")
        self.file.attrs["num_samples"] = self.count
        self.file.flush()
        self.file.close()
        self.closed = True
        if self.overwrite:
            os.replace(self.partial_path, self.path)
        else:
            os.link(self.partial_path, self.path)
            self.partial_path.unlink()
        return self.path

    def abort(self) -> None:
        if not self.closed:
            self.file.close()
            self.closed = True
        self.partial_path.unlink(missing_ok=True)

    def __enter__(self) -> "DaggerCorrectionRecorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is None:
            try:
                self.finalize()
            except BaseException:
                self.abort()
                raise
        else:
            self.abort()
        return False
