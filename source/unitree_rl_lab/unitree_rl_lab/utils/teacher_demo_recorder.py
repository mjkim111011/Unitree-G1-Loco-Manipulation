"""Streaming HDF5 recorder for Box task teacher demonstrations.

This module intentionally has no Isaac Lab or PyTorch dependency.  A rollout
driver can therefore pass NumPy arrays (or CPU/CUDA tensors exposing
``detach().cpu().numpy()``) without coupling the on-disk schema to a simulator
version.

Each row describes one control transition.  ``student_observation`` and
``teacher_high_level_action`` are sampled before stepping the environment,
``processed_43d_joint_target`` is the command applied for that transition, and
poses, contacts, and task flags are sampled after the step.  The alignment is
also persisted in the HDF5 metadata so consumers do not have to infer it.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


SCHEMA_NAME = "unitree_box_teacher_demo"
SCHEMA_VERSION = "1.0.0"
HIGH_LEVEL_ACTION_DIM = 23
PROCESSED_JOINT_TARGET_DIM = 43

# Arena G1 whole-body-controller action ordering.  Wrist quaternions in this
# vector use xyzw even when root/object poses in the same file use wxyz.
DEFAULT_HIGH_LEVEL_ACTION_NAMES = (
    "left_hand_state",
    "right_hand_state",
    "left_wrist_position_x",
    "left_wrist_position_y",
    "left_wrist_position_z",
    "left_wrist_quaternion_x",
    "left_wrist_quaternion_y",
    "left_wrist_quaternion_z",
    "left_wrist_quaternion_w",
    "right_wrist_position_x",
    "right_wrist_position_y",
    "right_wrist_position_z",
    "right_wrist_quaternion_x",
    "right_wrist_quaternion_y",
    "right_wrist_quaternion_z",
    "right_wrist_quaternion_w",
    "base_velocity_x",
    "base_velocity_y",
    "base_angular_velocity_z",
    "base_height",
    "torso_roll",
    "torso_pitch",
    "torso_yaw",
)

DEFAULT_PHASE_NAMES = {
    0: "approach",
    1: "grasp",
    2: "carry",
    3: "place",
}

DEFAULT_STEP_ALIGNMENT = {
    "student_observation": "pre_step_t",
    "teacher_high_level_action": "pre_step_t",
    "processed_43d_joint_target": "command_for_transition_t_to_t+1",
    "phase_id": "pre_step_t",
    "box_pose": "post_step_t+1",
    "bin_pose": "post_step_t+1",
    "root_pose": "post_step_t+1",
    "left_wrist_pose": "post_step_t+1",
    "right_wrist_pose": "post_step_t+1",
    "left_fingertip_positions": "post_step_t+1",
    "right_fingertip_positions": "post_step_t+1",
    "box_size": "episode_constant_repeated_per_step",
    "left_contact": "post_step_t+1",
    "right_contact": "post_step_t+1",
    "left_contact_count": "post_step_t+1",
    "right_contact_count": "post_step_t+1",
    "lift": "post_step_t+1",
    "drop": "post_step_t+1",
    "success": "post_step_t+1",
}


@dataclass(slots=True)
class TeacherDemoStep:
    """Values recorded for one environment control transition.

    Poses have layout ``[x, y, z, q0, q1, q2, q3]``.  The ordering of the
    quaternion entries is selected once for the complete file through
    :class:`TeacherDemoRecorder`'s ``quaternion_convention`` argument.
    """

    student_observation: Any
    teacher_high_level_action: Any
    processed_43d_joint_target: Any
    phase_id: int
    box_pose: Any
    bin_pose: Any
    root_pose: Any
    box_size: Any
    left_contact_count: int = 0
    right_contact_count: int = 0
    left_contact: bool | None = None
    right_contact: bool | None = None
    lift: bool = False
    drop: bool = False
    success: bool = False
    left_wrist_pose: Any | None = None
    right_wrist_pose: Any | None = None
    left_fingertip_positions: Any | None = None
    right_fingertip_positions: Any | None = None


class TeacherDemoRecorder:
    """Write one or more teacher episodes to a publish-on-finalize HDF5 file.

    Data is first written to a uniquely named partial file beside ``path``.
    :meth:`finalize` flushes and closes HDF5 and then publishes the finished
    file atomically on POSIX file systems.  Unless ``overwrite=True``, an
    existing destination is never replaced.

    Args:
        path: Destination HDF5 path.  It is not visible until finalization.
        control_frequency_hz: Environment action/control frequency.
        joint_names: Exact 43-element order used by processed joint targets.
        high_level_action_names: Exact 23-element teacher action order.
        phase_names: Integer phase identifier to readable name mapping.
        quaternion_convention: Quaternion order for ``box_pose``, ``bin_pose``,
            ``root_pose``, and both wrist poses.  Must be ``"wxyz"`` or
            ``"xyzw"``.
        metadata: Optional JSON-serializable run metadata.
        chunk_steps: Number of rows buffered and used as the HDF5 chunk length.
        compression: HDF5 compression filter (``"gzip"``, ``"lzf"``, or None).
        gzip_compression_level: Gzip level when ``compression="gzip"``.
        overwrite: Replace an existing complete destination on finalization.
        keep_partial_on_error: Preserve the partial file when a context exits
            due to an exception.  It is removed by default.
    """

    _FIXED_DATASETS = {
        "teacher_high_level_action": ((HIGH_LEVEL_ACTION_DIM,), np.float32),
        "processed_43d_joint_target": ((PROCESSED_JOINT_TARGET_DIM,), np.float32),
        "phase_id": ((), np.int16),
        "box_pose": ((7,), np.float32),
        "bin_pose": ((7,), np.float32),
        "root_pose": ((7,), np.float32),
        "left_wrist_pose": ((7,), np.float32),
        "right_wrist_pose": ((7,), np.float32),
        "left_fingertip_positions": ((3, 3), np.float32),
        "right_fingertip_positions": ((3, 3), np.float32),
        "box_size": ((3,), np.float32),
        "left_contact": ((), np.bool_),
        "right_contact": ((), np.bool_),
        "left_contact_count": ((), np.uint16),
        "right_contact_count": ((), np.uint16),
        "lift": ((), np.bool_),
        "drop": ((), np.bool_),
        "success": ((), np.bool_),
        "step_index": ((), np.int64),
        "time_s": ((), np.float64),
    }

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        control_frequency_hz: float,
        joint_names: Sequence[str],
        high_level_action_names: Sequence[str] = DEFAULT_HIGH_LEVEL_ACTION_NAMES,
        phase_names: Mapping[int, str] = DEFAULT_PHASE_NAMES,
        quaternion_convention: str = "wxyz",
        metadata: Mapping[str, Any] | None = None,
        chunk_steps: int = 128,
        compression: str | None = "gzip",
        gzip_compression_level: int = 1,
        overwrite: bool = False,
        keep_partial_on_error: bool = False,
    ) -> None:
        self._path = Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._overwrite = bool(overwrite)
        self._keep_partial_on_error = bool(keep_partial_on_error)

        if self._path.exists() and not self._overwrite:
            raise FileExistsError(f"Teacher demonstration already exists: {self._path}")
        if not np.isfinite(control_frequency_hz) or control_frequency_hz <= 0.0:
            raise ValueError("control_frequency_hz must be finite and greater than zero")
        if not isinstance(chunk_steps, int) or isinstance(chunk_steps, bool) or chunk_steps <= 0:
            raise ValueError("chunk_steps must be a positive integer")
        if compression not in (None, "gzip", "lzf"):
            raise ValueError("compression must be one of None, 'gzip', or 'lzf'")
        if not 0 <= gzip_compression_level <= 9:
            raise ValueError("gzip_compression_level must be between 0 and 9")
        if quaternion_convention not in ("wxyz", "xyzw"):
            raise ValueError("quaternion_convention must be 'wxyz' or 'xyzw'")

        self._joint_names = self._validate_names(
            joint_names, PROCESSED_JOINT_TARGET_DIM, "joint_names"
        )
        self._high_level_action_names = self._validate_names(
            high_level_action_names, HIGH_LEVEL_ACTION_DIM, "high_level_action_names"
        )
        self._phase_names = self._validate_phase_names(phase_names)
        self._control_frequency_hz = float(control_frequency_hz)
        self._chunk_steps = chunk_steps
        self._compression = compression
        self._gzip_compression_level = gzip_compression_level

        fd, partial_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.", suffix=".partial", dir=self._path.parent
        )
        os.close(fd)
        self._partial_path = Path(partial_name)
        try:
            self._file = h5py.File(self._partial_path, "w", libver="latest", track_order=True)
            self._initialize_file(metadata or {}, quaternion_convention)
        except BaseException:
            if hasattr(self, "_file"):
                self._file.close()
            self._partial_path.unlink(missing_ok=True)
            raise

        self._active_group: h5py.Group | None = None
        self._active_name: str | None = None
        self._buffer: dict[str, list[Any]] = {}
        self._phase_history: list[int] = []
        self._episode_step_count = 0
        self._episode_any_success = False
        self._total_steps = 0
        self._episode_count = 0
        self._closed = False
        self._finalized = False

    @property
    def path(self) -> Path:
        """Final destination path."""

        return self._path

    @property
    def partial_path(self) -> Path:
        """Current unpublished file path."""

        return self._partial_path

    @property
    def active_episode(self) -> str | None:
        """Name of the episode currently accepting steps."""

        return self._active_name

    def start_episode(
        self, name: str | None = None, *, metadata: Mapping[str, Any] | None = None
    ) -> str:
        """Start an episode and return its HDF5 group name."""

        self._require_open()
        if self._active_group is not None:
            raise RuntimeError(f"Episode {self._active_name!r} is still active")

        if name is None:
            next_index = self._episode_count
            name = f"demo_{next_index}"
            while name in self._file["data"]:
                next_index += 1
                name = f"demo_{next_index}"
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError("Episode name must be a non-empty HDF5 group name without '/'")
        if name in self._file["data"]:
            raise ValueError(f"Episode already exists: {name}")

        group = self._file["data"].create_group(name, track_order=True)
        group.attrs["complete"] = False
        group.attrs["num_samples"] = 0
        group.attrs["metadata_json"] = self._json_dumps(metadata or {})
        for dataset_name, (trailing_shape, dtype) in self._FIXED_DATASETS.items():
            alignment = DEFAULT_STEP_ALIGNMENT.get(dataset_name, "derived_from_step_index")
            self._create_dataset(group, dataset_name, trailing_shape, dtype, alignment)

        self._active_group = group
        self._active_name = name
        self._buffer = {dataset_name: [] for dataset_name in self._FIXED_DATASETS}
        self._buffer["student_observation"] = []
        self._phase_history = []
        self._episode_step_count = 0
        self._episode_any_success = False
        return name

    def append(self, step: TeacherDemoStep) -> int:
        """Validate and queue one control transition, returning its step index."""

        self._require_open()
        if self._active_group is None:
            raise RuntimeError("start_episode() must be called before append()")
        if not isinstance(step, TeacherDemoStep):
            raise TypeError("append() expects a TeacherDemoStep instance")

        step_index = self._episode_step_count
        left_count = self._contact_count(step.left_contact_count, "left_contact_count")
        right_count = self._contact_count(step.right_contact_count, "right_contact_count")
        left_contact = bool(left_count > 0) if step.left_contact is None else bool(step.left_contact)
        right_contact = bool(right_count > 0) if step.right_contact is None else bool(step.right_contact)
        phase_id = self._phase_id(step.phase_id)

        row = {
            "student_observation": self._float_array(
                step.student_observation, None, "student_observation", require_nonempty=True
            ),
            "teacher_high_level_action": self._float_array(
                step.teacher_high_level_action,
                (HIGH_LEVEL_ACTION_DIM,),
                "teacher_high_level_action",
            ),
            "processed_43d_joint_target": self._float_array(
                step.processed_43d_joint_target,
                (PROCESSED_JOINT_TARGET_DIM,),
                "processed_43d_joint_target",
            ),
            "phase_id": phase_id,
            "box_pose": self._float_array(step.box_pose, (7,), "box_pose"),
            "bin_pose": self._float_array(step.bin_pose, (7,), "bin_pose"),
            "root_pose": self._float_array(step.root_pose, (7,), "root_pose"),
            "left_wrist_pose": self._optional_float_array(
                step.left_wrist_pose, (7,), "left_wrist_pose"
            ),
            "right_wrist_pose": self._optional_float_array(
                step.right_wrist_pose, (7,), "right_wrist_pose"
            ),
            "left_fingertip_positions": self._optional_float_array(
                step.left_fingertip_positions, (3, 3), "left_fingertip_positions"
            ),
            "right_fingertip_positions": self._optional_float_array(
                step.right_fingertip_positions, (3, 3), "right_fingertip_positions"
            ),
            "box_size": self._box_size(step.box_size),
            "left_contact": left_contact,
            "right_contact": right_contact,
            "left_contact_count": left_count,
            "right_contact_count": right_count,
            "lift": bool(step.lift),
            "drop": bool(step.drop),
            "success": bool(step.success),
            "step_index": step_index,
            "time_s": step_index / self._control_frequency_hz,
        }

        observation = row["student_observation"]
        if "student_observation" not in self._active_group:
            self._create_dataset(
                self._active_group,
                "student_observation",
                observation.shape,
                np.float32,
                DEFAULT_STEP_ALIGNMENT["student_observation"],
            )
            self._active_group["student_observation"].attrs["feature_shape_json"] = self._json_dumps(
                list(observation.shape)
            )
        elif observation.shape != self._active_group["student_observation"].shape[1:]:
            expected = self._active_group["student_observation"].shape[1:]
            raise ValueError(
                f"student_observation has shape {observation.shape}; expected {expected} for this episode"
            )

        for key, value in row.items():
            self._buffer[key].append(value)
        self._phase_history.append(phase_id)
        self._episode_step_count += 1
        self._episode_any_success |= bool(step.success)

        if len(self._buffer["phase_id"]) >= self._chunk_steps:
            self.flush()
        return step_index

    def append_step(self, **kwargs: Any) -> int:
        """Construct :class:`TeacherDemoStep` from keyword arguments and append it."""

        return self.append(TeacherDemoStep(**kwargs))

    def flush(self) -> None:
        """Write buffered rows to resizable datasets and flush HDF5 buffers."""

        self._require_open()
        if self._active_group is None or not self._buffer.get("phase_id"):
            self._file.flush()
            return

        count = len(self._buffer["phase_id"])
        old_size = int(self._active_group["phase_id"].shape[0])
        new_size = old_size + count
        for dataset_name, values in self._buffer.items():
            dataset = self._active_group[dataset_name]
            dataset.resize(new_size, axis=0)
            dataset[old_size:new_size] = np.asarray(values, dtype=dataset.dtype)
            values.clear()
        self._active_group.attrs.modify("num_samples", self._episode_step_count)
        self._file.flush()

    def end_episode(
        self,
        *,
        success: bool | None = None,
        termination_reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Finish the active episode and persist phase segment boundaries."""

        self._require_open()
        if self._active_group is None or self._active_name is None:
            raise RuntimeError("No active episode to end")
        if self._episode_step_count == 0:
            raise RuntimeError("Cannot finish an episode containing no steps")

        self.flush()
        group = self._active_group
        episode_name = self._active_name
        episode_success = self._episode_any_success if success is None else bool(success)
        group.attrs.modify("num_samples", self._episode_step_count)
        group.attrs["success"] = episode_success
        group.attrs["complete"] = True
        group.attrs["phase_boundaries_json"] = self._json_dumps(self._phase_boundaries())
        if termination_reason is not None:
            group.attrs["termination_reason"] = str(termination_reason)
        if metadata is not None:
            group.attrs["final_metadata_json"] = self._json_dumps(metadata)

        self._total_steps += self._episode_step_count
        self._episode_count += 1
        self._file["data"].attrs.modify("total", self._total_steps)
        self._file["data"].attrs.modify("num_episodes", self._episode_count)
        self._file.flush()

        self._active_group = None
        self._active_name = None
        self._buffer = {}
        self._phase_history = []
        self._episode_step_count = 0
        self._episode_any_success = False
        return episode_name

    def discard_episode(
        self,
        *,
        reason: str = "failed_success_filter",
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Remove the active episode without advancing published counters.

        Randomized teacher generation can require several physics attempts for
        each accepted demonstration.  Failed attempts must never leak into the
        behavior-cloning dataset.  This method flushes nothing, deletes any
        chunks already written for the active group, and leaves automatic demo
        naming and ``data.num_episodes`` based only on accepted episodes.

        The rejection reason and optional metadata are intentionally returned
        to the caller rather than stored in the output dataset; batch drivers
        persist them in a separate metrics/manifest JSON file.
        """

        self._require_open()
        if self._active_group is None or self._active_name is None:
            raise RuntimeError("No active episode to discard")
        episode_name = self._active_name
        # HDF5 permits removing an open group's link.  Clear our reference
        # immediately afterwards so later flush/finalize calls cannot reach it.
        del self._file["data"][episode_name]
        self._file.flush()
        self._active_group = None
        self._active_name = None
        self._buffer = {}
        self._phase_history = []
        self._episode_step_count = 0
        self._episode_any_success = False
        # Validate JSON serializability now, even though rejection details are
        # external, so a metrics writer receives only recorder-compatible data.
        self._json_dumps({"reason": str(reason), "metadata": metadata or {}})
        return episode_name

    def finalize(self) -> Path:
        """Close and atomically publish the completed HDF5 file."""

        self._require_open()
        if self._active_group is not None:
            raise RuntimeError(
                f"Episode {self._active_name!r} is active; call end_episode() before finalize()"
            )
        if self._episode_count == 0:
            raise RuntimeError("Cannot finalize a demonstration file with no episodes")

        self._file.attrs.modify("complete", True)
        self._file.attrs["completed_utc"] = self._utc_now()
        self._file.flush()
        self._fsync_hdf5_file()
        self._file.close()
        self._closed = True

        if self._overwrite:
            os.replace(self._partial_path, self._path)
        else:
            # A hard link publishes without a check-then-rename race and cannot
            # overwrite a complete file created by another process.
            os.link(self._partial_path, self._path)
            self._partial_path.unlink()
        self._fsync_directory(self._path.parent)
        self._finalized = True
        return self._path

    def abort(self, *, remove_partial: bool = True) -> None:
        """Close without publishing, optionally retaining the partial file."""

        if not getattr(self, "_closed", True):
            try:
                self._file.close()
            finally:
                self._closed = True
        # finalize() closes HDF5 before publishing.  If publication then fails,
        # context-manager cleanup must still honor remove_partial.
        if remove_partial and not getattr(self, "_finalized", False):
            self._partial_path.unlink(missing_ok=True)

    def __enter__(self) -> "TeacherDemoRecorder":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if exc_type is None:
            try:
                self.finalize()
            except BaseException:
                self.abort(remove_partial=not self._keep_partial_on_error)
                raise
        else:
            self.abort(remove_partial=not self._keep_partial_on_error)
        return False

    def __del__(self) -> None:
        # Best-effort cleanup only; never make interpreter shutdown noisy.
        try:
            self.abort(remove_partial=not getattr(self, "_keep_partial_on_error", False))
        except Exception:
            pass

    def _initialize_file(self, metadata: Mapping[str, Any], quaternion_convention: str) -> None:
        self._file.attrs["schema_name"] = SCHEMA_NAME
        self._file.attrs["schema_version"] = SCHEMA_VERSION
        self._file.attrs["complete"] = False
        self._file.attrs["created_utc"] = self._utc_now()
        self._file.attrs["control_frequency_hz"] = self._control_frequency_hz
        self._file.attrs["control_dt_s"] = 1.0 / self._control_frequency_hz
        self._file.attrs["pose_layout"] = "xyz+quaternion"
        self._file.attrs["quaternion_convention"] = quaternion_convention
        self._file.attrs["high_level_wrist_quaternion_convention"] = "xyzw"
        self._file.attrs["step_alignment_json"] = self._json_dumps(DEFAULT_STEP_ALIGNMENT)
        self._file.attrs["phase_names_json"] = self._json_dumps(self._phase_names)
        self._file.attrs["metadata_json"] = self._json_dumps(metadata)

        data_group = self._file.create_group("data", track_order=True)
        data_group.attrs["total"] = 0
        data_group.attrs["num_episodes"] = 0
        metadata_group = self._file.create_group("metadata", track_order=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        metadata_group.create_dataset(
            "teacher_high_level_action_names",
            data=np.asarray(self._high_level_action_names, dtype=object),
            dtype=string_dtype,
        )
        metadata_group.create_dataset(
            "processed_joint_names",
            data=np.asarray(self._joint_names, dtype=object),
            dtype=string_dtype,
        )

    def _create_dataset(
        self,
        group: h5py.Group,
        name: str,
        trailing_shape: tuple[int, ...],
        dtype: Any,
        alignment: str,
    ) -> h5py.Dataset:
        kwargs: dict[str, Any] = {
            "shape": (0, *trailing_shape),
            "maxshape": (None, *trailing_shape),
            "chunks": (self._chunk_steps, *trailing_shape),
            "dtype": dtype,
        }
        if self._compression is not None:
            kwargs["compression"] = self._compression
            if self._compression == "gzip":
                kwargs["compression_opts"] = self._gzip_compression_level
        dataset = group.create_dataset(name, **kwargs)
        dataset.attrs["alignment"] = alignment
        if name in {
            "left_wrist_pose",
            "right_wrist_pose",
            "left_fingertip_positions",
            "right_fingertip_positions",
        }:
            dataset.attrs["missing_value"] = "NaN"
        return dataset

    def _phase_boundaries(self) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        start = 0
        for index in range(1, len(self._phase_history) + 1):
            if index == len(self._phase_history) or self._phase_history[index] != self._phase_history[start]:
                phase_id = self._phase_history[start]
                segments.append(
                    {
                        "phase_id": phase_id,
                        "phase_name": self._phase_names.get(phase_id, "unknown"),
                        "start_step": start,
                        "end_step_exclusive": index,
                    }
                )
                start = index
        return segments

    def _require_open(self) -> None:
        if getattr(self, "_closed", False):
            state = "finalized" if getattr(self, "_finalized", False) else "closed"
            raise RuntimeError(f"TeacherDemoRecorder is already {state}")

    def _fsync_hdf5_file(self) -> None:
        try:
            handle = self._file.id.get_vfd_handle()
            if isinstance(handle, tuple):
                handle = handle[0]
            if isinstance(handle, int):
                os.fsync(handle)
        except (AttributeError, OSError, TypeError, ValueError):
            # Some HDF5 virtual file drivers do not expose an OS descriptor.
            pass

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @classmethod
    def _float_array(
        cls,
        value: Any,
        expected_shape: tuple[int, ...] | None,
        field_name: str,
        *,
        require_nonempty: bool = False,
    ) -> np.ndarray:
        array = np.asarray(cls._to_numpy(value), dtype=np.float32)
        if expected_shape is not None and array.shape != expected_shape:
            raise ValueError(f"{field_name} has shape {array.shape}; expected {expected_shape}")
        if require_nonempty and (array.ndim != 1 or array.size == 0):
            raise ValueError(f"{field_name} must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{field_name} contains NaN or infinity")
        # Snapshot mutable simulator buffers; rollout code commonly reuses the
        # same NumPy/Torch storage on every control step.
        return np.array(array, dtype=np.float32, copy=True, order="C")

    @classmethod
    def _box_size(cls, value: Any) -> np.ndarray:
        size = cls._float_array(value, (3,), "box_size")
        if np.any(size <= 0.0):
            raise ValueError("box_size entries must be greater than zero")
        return size

    @classmethod
    def _optional_float_array(
        cls, value: Any | None, expected_shape: tuple[int, ...], field_name: str
    ) -> np.ndarray:
        if value is None:
            return np.full(expected_shape, np.nan, dtype=np.float32)
        return cls._float_array(value, expected_shape, field_name)

    @staticmethod
    def _contact_count(value: Any, field_name: str) -> int:
        array = np.asarray(value)
        if array.shape != ():
            raise ValueError(f"{field_name} must be a scalar integer")
        scalar = array.item()
        if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, (int, np.integer)):
            raise ValueError(f"{field_name} must be a scalar integer")
        count = int(scalar)
        if not 0 <= count <= np.iinfo(np.uint16).max:
            raise ValueError(f"{field_name} must be between 0 and 65535")
        return count

    @staticmethod
    def _phase_id(value: Any) -> int:
        array = np.asarray(value)
        if array.shape != ():
            raise ValueError("phase_id must be a scalar integer")
        scalar = array.item()
        if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, (int, np.integer)):
            raise ValueError("phase_id must be a scalar integer")
        phase_id = int(scalar)
        limits = np.iinfo(np.int16)
        if not limits.min <= phase_id <= limits.max:
            raise ValueError(f"phase_id must fit int16 ({limits.min} to {limits.max})")
        return phase_id

    @staticmethod
    def _validate_names(names: Sequence[str], expected_count: int, field_name: str) -> tuple[str, ...]:
        result = tuple(names)
        if len(result) != expected_count:
            raise ValueError(f"{field_name} must contain {expected_count} entries; got {len(result)}")
        if any(not isinstance(name, str) or not name for name in result):
            raise ValueError(f"Every entry in {field_name} must be a non-empty string")
        if len(set(result)) != len(result):
            raise ValueError(f"Entries in {field_name} must be unique")
        return result

    @staticmethod
    def _validate_phase_names(phase_names: Mapping[int, str]) -> dict[int, str]:
        result: dict[int, str] = {}
        for phase_id, phase_name in phase_names.items():
            if isinstance(phase_id, bool) or not isinstance(phase_id, int):
                raise ValueError("phase_names keys must be integers")
            if not isinstance(phase_name, str) or not phase_name:
                raise ValueError("phase_names values must be non-empty strings")
            result[int(phase_id)] = phase_name
        return result

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=TeacherDemoRecorder._json_default)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()


__all__ = [
    "DEFAULT_HIGH_LEVEL_ACTION_NAMES",
    "DEFAULT_PHASE_NAMES",
    "DEFAULT_STEP_ALIGNMENT",
    "HIGH_LEVEL_ACTION_DIM",
    "PROCESSED_JOINT_TARGET_DIM",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "TeacherDemoRecorder",
    "TeacherDemoStep",
]
