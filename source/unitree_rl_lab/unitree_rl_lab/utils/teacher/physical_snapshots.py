"""Physically captured phase-start snapshots for carry and place training.

Snapshots are accepted only when their provenance and physical gates show that
they came from an actual rollout.  Restoring one replays the complete robot and
Box dynamic state; it is not equivalent to placing a Box into the hands with a
synthetic pose.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


SNAPSHOT_SCHEMA_NAME = "unitree_box_physical_phase_snapshot"
SNAPSHOT_SCHEMA_VERSION = "3.0.0"
SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS = ("1.0.0", "2.0.0", SNAPSHOT_SCHEMA_VERSION)
ROLLOUT_PROVENANCE = "physical_rollout_capture"


class SnapshotKind(IntEnum):
    BILATERAL_HOLD = 0
    CARRY_AT_STAGING = 1


@dataclass(frozen=True, slots=True)
class PhysicalSnapshot:
    kind: SnapshotKind
    robot_root_link_state_w: np.ndarray
    robot_joint_position: np.ndarray
    robot_joint_velocity: np.ndarray
    box_root_link_state_w: np.ndarray
    last_action: np.ndarray
    student_observation: np.ndarray
    task_phase: int
    source_episode: str
    source_step: int
    source_env_id: int
    provenance: str
    left_contact_count: int
    right_contact_count: int
    lift_height_m: float
    stable_hold_steps: int
    at_staging: bool
    bad_orientation: bool
    box_dropped: bool
    box_size: np.ndarray
    box_mass_kg: float
    unprocessed_wbc_raw_action: np.ndarray | None = None
    table_center_e: np.ndarray | None = None
    bin_center_e: np.ndarray | None = None
    bin_qr_position_e: np.ndarray | None = None
    table_asset_root_poses_w: np.ndarray | None = None
    bin_asset_root_poses_w: np.ndarray | None = None

    def validate(
        self,
        *,
        joint_count: int = 43,
        action_dim: int = 43,
        observation_dim: int = 517,
        lift_threshold_m: float = 0.12,
        hold_steps_required: int = 100,
    ) -> "PhysicalSnapshot":
        arrays = {
            "robot_root_link_state_w": (self.robot_root_link_state_w, (13,)),
            "robot_joint_position": (self.robot_joint_position, (joint_count,)),
            "robot_joint_velocity": (self.robot_joint_velocity, (joint_count,)),
            "box_root_link_state_w": (self.box_root_link_state_w, (13,)),
            "last_action": (self.last_action, (action_dim,)),
            "student_observation": (self.student_observation, (observation_dim,)),
            "box_size": (self.box_size, (3,)),
        }
        for name, (value, shape) in arrays.items():
            array = np.asarray(value)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(
                    f"{name} must be finite with shape {shape}; got {array.shape}"
                )
        if self.unprocessed_wbc_raw_action is not None:
            baseline = np.asarray(self.unprocessed_wbc_raw_action)
            if baseline.shape != (action_dim,) or not np.isfinite(baseline).all():
                raise ValueError(
                    "unprocessed_wbc_raw_action must be finite with shape "
                    f"({action_dim},); got {baseline.shape}"
                )
        work_area_arrays = {
            "table_center_e": (self.table_center_e, (3,)),
            "bin_center_e": (self.bin_center_e, (3,)),
            "bin_qr_position_e": (self.bin_qr_position_e, (3,)),
            "table_asset_root_poses_w": (self.table_asset_root_poses_w, (2, 7)),
            "bin_asset_root_poses_w": (self.bin_asset_root_poses_w, (5, 7)),
        }
        work_area_present = [value is not None for value, _ in work_area_arrays.values()]
        if any(work_area_present) and not all(work_area_present):
            raise ValueError("snapshot work-area state must be complete or entirely absent")
        if all(work_area_present):
            for name, (value, shape) in work_area_arrays.items():
                array = np.asarray(value)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(
                        f"{name} must be finite with shape {shape}; got {array.shape}"
                    )
        if self.provenance != ROLLOUT_PROVENANCE:
            raise ValueError(
                f"snapshot provenance must be {ROLLOUT_PROVENANCE!r}; arbitrary assembled states are rejected"
            )
        if not self.source_episode or self.source_step < 0 or self.source_env_id < 0:
            raise ValueError(
                "snapshot requires a source episode, non-negative step, and environment id"
            )
        if self.left_contact_count < 1 or self.right_contact_count < 1:
            raise ValueError("snapshot requires Box-filtered contact from both hands")
        if (
            self.lift_height_m < lift_threshold_m
            or self.stable_hold_steps < hold_steps_required
        ):
            raise ValueError(
                "snapshot does not satisfy strict lift and continuous hold gates"
            )
        if self.bad_orientation or self.box_dropped:
            raise ValueError(
                "unsafe or dropped-Box snapshots cannot seed a later skill"
            )
        if np.any(np.asarray(self.box_size) <= 0.0) or self.box_mass_kg <= 0.0:
            raise ValueError("box size and mass must be positive")
        kind = SnapshotKind(int(self.kind))
        if kind == SnapshotKind.CARRY_AT_STAGING and not self.at_staging:
            raise ValueError(
                "place snapshots must come from a real carry rollout at staging"
            )
        return self


class PhysicalSnapshotStore:
    """Atomically write and read validated rollout snapshots."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        joint_names: Iterable[str],
        overwrite: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.joint_names = tuple(str(name) for name in joint_names)
        if len(self.joint_names) != 43 or len(set(self.joint_names)) != 43:
            raise ValueError("joint_names must contain 43 unique names")
        if self.path.exists() and not overwrite:
            raise FileExistsError(self.path)
        fd, partial = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".partial", dir=self.path.parent
        )
        os.close(fd)
        self.partial_path = Path(partial)
        self._overwrite = overwrite
        self._file = h5py.File(self.partial_path, "w", track_order=True)
        self._file.attrs["schema_name"] = SNAPSHOT_SCHEMA_NAME
        self._file.attrs["schema_version"] = SNAPSHOT_SCHEMA_VERSION
        self._file.attrs["metadata_json"] = json.dumps(metadata or {}, sort_keys=True)
        self._file.create_dataset(
            "joint_names",
            data=np.asarray(self.joint_names, dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        self._groups = self._file.create_group("snapshots", track_order=True)
        self._closed = False
        self._count = 0

    @property
    def count(self) -> int:
        """Number of snapshots appended to the still-open store."""

        return self._count

    def append(self, snapshot: PhysicalSnapshot, *, name: str | None = None) -> str:
        snapshot.validate()
        if self._closed:
            raise RuntimeError("snapshot store is closed")
        if name is None:
            name = f"snapshot_{len(self._groups):06d}"
        if not name or "/" in name or name in self._groups:
            raise ValueError(f"invalid or duplicate snapshot name: {name!r}")
        group = self._groups.create_group(name, track_order=True)
        group.attrs["kind"] = int(snapshot.kind)
        group.attrs["kind_name"] = SnapshotKind(int(snapshot.kind)).name.lower()
        scalar_attrs = (
            "task_phase",
            "source_episode",
            "source_step",
            "source_env_id",
            "provenance",
            "left_contact_count",
            "right_contact_count",
            "lift_height_m",
            "stable_hold_steps",
            "at_staging",
            "bad_orientation",
            "box_dropped",
            "box_mass_kg",
        )
        for field in scalar_attrs:
            group.attrs[field] = getattr(snapshot, field)
        for field in (
            "robot_root_link_state_w",
            "robot_joint_position",
            "robot_joint_velocity",
            "box_root_link_state_w",
            "last_action",
            "student_observation",
            "box_size",
        ):
            group.create_dataset(
                field, data=np.asarray(getattr(snapshot, field), dtype=np.float32)
            )
        group.attrs["has_unprocessed_wbc_raw_action"] = (
            snapshot.unprocessed_wbc_raw_action is not None
        )
        if snapshot.unprocessed_wbc_raw_action is not None:
            group.create_dataset(
                "unprocessed_wbc_raw_action",
                data=np.asarray(
                    snapshot.unprocessed_wbc_raw_action, dtype=np.float32
                ),
            )
        has_work_area_state = snapshot.table_center_e is not None
        group.attrs["has_work_area_state"] = has_work_area_state
        if has_work_area_state:
            for field in (
                "table_center_e",
                "bin_center_e",
                "bin_qr_position_e",
                "table_asset_root_poses_w",
                "bin_asset_root_poses_w",
            ):
                group.create_dataset(
                    field, data=np.asarray(getattr(snapshot, field), dtype=np.float32)
                )
        self._count += 1
        self._file.flush()
        return name

    def finalize(self) -> Path:
        if self._closed:
            raise RuntimeError("snapshot store is closed")
        if len(self._groups) == 0:
            raise RuntimeError("cannot publish an empty snapshot store")
        self._file.attrs["num_snapshots"] = self._count
        self._file.flush()
        self._file.close()
        self._closed = True
        if self._overwrite:
            os.replace(self.partial_path, self.path)
        else:
            os.link(self.partial_path, self.path)
            self.partial_path.unlink()
        return self.path

    def abort(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True
        self.partial_path.unlink(missing_ok=True)

    def __enter__(self) -> "PhysicalSnapshotStore":
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


def load_snapshots(
    path: str | os.PathLike[str],
    *,
    kind: SnapshotKind | None = None,
    expected_joint_names: Iterable[str] | None = None,
    lift_threshold_m: float = 0.12,
    hold_steps_required: int = 100,
    require_unprocessed_wbc_raw_action: bool = False,
) -> tuple[PhysicalSnapshot, ...]:
    """Load and revalidate all (or one kind of) physical snapshots."""

    snapshots: list[PhysicalSnapshot] = []
    with h5py.File(Path(path).expanduser(), "r") as h5_file:
        if h5_file.attrs.get("schema_name") != SNAPSHOT_SCHEMA_NAME:
            raise ValueError("not a Unitree physical phase snapshot file")
        if h5_file.attrs.get("schema_version") not in (
            SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS
        ):
            raise ValueError("unsupported physical snapshot schema version")
        stored_joint_names = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in h5_file["joint_names"][:]
        )
        if len(stored_joint_names) != 43 or len(set(stored_joint_names)) != 43:
            raise ValueError("snapshot file must declare 43 unique robot joint names")
        if expected_joint_names is not None:
            expected = tuple(str(name) for name in expected_joint_names)
            if stored_joint_names != expected:
                raise ValueError(
                    "snapshot joint order does not match the current articulation; "
                    "restoring by array index would be unsafe"
                )
        for group in h5_file["snapshots"].values():
            snapshot_kind = SnapshotKind(int(group.attrs["kind"]))
            if kind is not None and snapshot_kind != kind:
                continue
            has_wbc_baseline = bool(
                group.attrs.get("has_unprocessed_wbc_raw_action", False)
            )
            baseline_dataset_present = "unprocessed_wbc_raw_action" in group
            if has_wbc_baseline != baseline_dataset_present:
                raise ValueError(
                    "snapshot Pink/WBC baseline presence metadata is inconsistent"
                )
            has_work_area_state = bool(group.attrs.get("has_work_area_state", False))
            work_area_fields = (
                "table_center_e",
                "bin_center_e",
                "bin_qr_position_e",
                "table_asset_root_poses_w",
                "bin_asset_root_poses_w",
            )
            work_area_datasets_present = [field in group for field in work_area_fields]
            if has_work_area_state != all(work_area_datasets_present) or (
                any(work_area_datasets_present) and not has_work_area_state
            ):
                raise ValueError("snapshot work-area state metadata is inconsistent")
            snapshot = PhysicalSnapshot(
                kind=snapshot_kind,
                robot_root_link_state_w=group["robot_root_link_state_w"][:],
                robot_joint_position=group["robot_joint_position"][:],
                robot_joint_velocity=group["robot_joint_velocity"][:],
                box_root_link_state_w=group["box_root_link_state_w"][:],
                last_action=group["last_action"][:],
                student_observation=group["student_observation"][:],
                task_phase=int(group.attrs["task_phase"]),
                source_episode=str(group.attrs["source_episode"]),
                source_step=int(group.attrs["source_step"]),
                source_env_id=int(group.attrs["source_env_id"]),
                provenance=str(group.attrs["provenance"]),
                left_contact_count=int(group.attrs["left_contact_count"]),
                right_contact_count=int(group.attrs["right_contact_count"]),
                lift_height_m=float(group.attrs["lift_height_m"]),
                stable_hold_steps=int(group.attrs["stable_hold_steps"]),
                at_staging=bool(group.attrs["at_staging"]),
                bad_orientation=bool(group.attrs["bad_orientation"]),
                box_dropped=bool(group.attrs["box_dropped"]),
                box_size=group["box_size"][:],
                box_mass_kg=float(group.attrs["box_mass_kg"]),
                unprocessed_wbc_raw_action=(
                    group["unprocessed_wbc_raw_action"][:]
                    if has_wbc_baseline
                    else None
                ),
                table_center_e=(group["table_center_e"][:] if has_work_area_state else None),
                bin_center_e=(group["bin_center_e"][:] if has_work_area_state else None),
                bin_qr_position_e=(
                    group["bin_qr_position_e"][:] if has_work_area_state else None
                ),
                table_asset_root_poses_w=(
                    group["table_asset_root_poses_w"][:] if has_work_area_state else None
                ),
                bin_asset_root_poses_w=(
                    group["bin_asset_root_poses_w"][:] if has_work_area_state else None
                ),
            )
            if require_unprocessed_wbc_raw_action and (
                snapshot.unprocessed_wbc_raw_action is None
            ):
                raise ValueError(
                    "snapshot has no unprocessed Pink/WBC raw action baseline"
                )
            snapshots.append(
                snapshot.validate(
                    lift_threshold_m=lift_threshold_m,
                    hold_steps_required=hold_steps_required,
                )
            )
    return tuple(snapshots)


def load_snapshot_store_metadata(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Return validated file-level provenance for a snapshot store."""

    snapshot_path = Path(path).expanduser().resolve()
    with h5py.File(snapshot_path, "r") as h5_file:
        if h5_file.attrs.get("schema_name") != SNAPSHOT_SCHEMA_NAME:
            raise ValueError("not a Unitree physical phase snapshot file")
        if h5_file.attrs.get("schema_version") not in (
            SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS
        ):
            raise ValueError("unsupported physical snapshot schema version")
        raw = h5_file.attrs.get("metadata_json", "{}")
        raw_text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    try:
        metadata = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("snapshot store has invalid metadata_json") from exc
    if not isinstance(metadata, dict):
        raise ValueError("snapshot store metadata must be a JSON object")
    return metadata
