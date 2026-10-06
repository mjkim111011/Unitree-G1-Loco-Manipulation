"""Phase-wise behavior cloning for the Box pick-and-place student.

Only the recorded 517-D ``student_observation`` is ever returned as a model
input.  Privileged object/root/wrist poses may coexist in a teacher HDF5 file,
but this module does not expose a generic feature-concatenation path.  Contact
and lift datasets are read solely to derive supervision segment boundaries.

The four policies intentionally remain independent so that the runtime FSM can
route ``APPROACH -> GRASP_LIFT -> CARRY -> PLACE`` explicitly.
"""

from __future__ import annotations

import hashlib
import glob
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .carry_runtime_contract import (
    carry_runtime_contract_sha256,
    validate_carry_runtime_contract,
)
from .place_runtime_contract import (
    place_runtime_contract_sha256,
    validate_place_runtime_contract,
)
from .physical_snapshots import (
    SNAPSHOT_SCHEMA_NAME,
    SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS,
    SnapshotKind,
    load_snapshot_store_metadata,
    load_snapshots,
)
from .skill_fsm import SkillPhase
from .student_observation import (
    STUDENT_OBSERVATION_DIM,
    STUDENT_OBSERVATION_SCHEMA,
    validate_dataset_student_boundary,
    validate_student_observation,
)


BC_CHECKPOINT_FORMAT = "unitree_box_phase_bc_v1"
BC2_GATE_FORMAT = "unitree_box_bc2_gate_v1"
BC2_SNAPSHOT_AUDIT_FORMAT = "unitree_box_bc2_snapshot_audit_v1"
ACTION_DIMS: Mapping[str, int] = {
    "teacher_high_level_action": 23,
    "processed_43d_joint_target": 43,
}
TRAINABLE_SKILLS = (
    SkillPhase.APPROACH,
    SkillPhase.GRASP_LIFT,
    SkillPhase.CARRY,
    SkillPhase.PLACE,
)
SKILL_NAMES: Mapping[SkillPhase, str] = {
    SkillPhase.APPROACH: "approach",
    SkillPhase.GRASP_LIFT: "grasp_lift",
    SkillPhase.CARRY: "carry",
    SkillPhase.PLACE: "place",
}
SKILL_BY_NAME = {name: skill for skill, name in SKILL_NAMES.items()}
TEACHER_SHARD_MANIFEST_SCHEMA = "unitree_box_xy_randomized_teacher_shards_v1"
DIVERSE_TEACHER_SHARD_MANIFEST_SCHEMA = "unitree_spatially_balanced_teacher_shards_v1"
MINIMUM_PRODUCTION_DEMOS = 500
MAXIMUM_PRODUCTION_DEMOS = 1000


def action_contract_sha256(action_key: str, ordered_names: Sequence[str]) -> str:
    """Hash the semantic action representation and its exact ordered channels."""

    action_dim = validate_action_key(action_key)
    names = tuple(str(name) for name in ordered_names)
    if (
        len(names) != action_dim
        or len(set(names)) != action_dim
        or any(not name for name in names)
    ):
        raise ValueError(
            f"{action_key} requires {action_dim} unique non-empty ordered action names"
        )
    payload = json.dumps(
        {"action_key": action_key, "ordered_names": names},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decode_string_dataset(dataset: h5py.Dataset) -> tuple[str, ...]:
    values = dataset.asstr()[:]
    return tuple(str(value) for value in np.asarray(values).tolist())


def teacher_dataset_action_names(
    path: str | os.PathLike[str], action_key: str
) -> tuple[str, ...]:
    """Read the ordered action contract from a nominal or DAgger HDF5 file."""

    validate_action_key(action_key)
    resolved = Path(path).expanduser().resolve()
    with h5py.File(resolved, "r") as source:
        if action_key == "processed_43d_joint_target":
            candidates = ("metadata/processed_joint_names", "joint_names")
        else:
            candidates = (
                "metadata/teacher_high_level_action_names",
                "teacher_high_level_action_names",
            )
        for candidate in candidates:
            if candidate in source:
                names = _decode_string_dataset(source[candidate])
                action_contract_sha256(action_key, names)
                return names
    raise ValueError(
        f"{resolved} does not declare ordered names for {action_key}; refusing an unbound action target"
    )


def validate_teacher_action_contract(
    paths: Sequence[str | os.PathLike[str]],
    action_key: str,
    *,
    expected_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Require exact channel order agreement across all teacher data sources."""

    concrete = expand_teacher_dataset_paths(paths)
    reference = (
        None if expected_names is None else tuple(str(v) for v in expected_names)
    )
    if reference is not None:
        action_contract_sha256(action_key, reference)
    for path in concrete:
        names = teacher_dataset_action_names(path, action_key)
        if reference is None:
            reference = names
        elif names != reference:
            raise ValueError(
                f"action order mismatch in {path}; expected exact {action_key} channel order"
            )
    if reference is None:
        raise ValueError("at least one action-bound teacher dataset is required")
    return reference


def parse_skill(value: SkillPhase | int | str) -> SkillPhase:
    """Parse one of the four trainable BC skills."""

    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized.startswith("bc_"):
            normalized = normalized[3:]
        if normalized not in SKILL_BY_NAME:
            raise ValueError(
                f"unknown BC skill {value!r}; choose from {sorted(SKILL_BY_NAME)}"
            )
        return SKILL_BY_NAME[normalized]
    skill = SkillPhase(int(value))
    if skill not in TRAINABLE_SKILLS:
        raise ValueError(f"{skill.name} is not a trainable BC skill")
    return skill


def validate_action_key(action_key: str) -> int:
    """Return the action width for a supported teacher target."""

    if action_key not in ACTION_DIMS:
        raise ValueError(
            f"unsupported action_key {action_key!r}; choose from {sorted(ACTION_DIMS)}"
        )
    return ACTION_DIMS[action_key]


@dataclass(frozen=True, slots=True)
class PhaseInferenceConfig:
    """Signal thresholds for splitting task phase 0 into BC-1 and BC-2."""

    hand_close_threshold: float = 0.5
    grasp_lead_steps: int = 0

    def validate(self) -> "PhaseInferenceConfig":
        if not math.isfinite(self.hand_close_threshold):
            raise ValueError("hand_close_threshold must be finite")
        if self.grasp_lead_steps < 0:
            raise ValueError("grasp_lead_steps must be non-negative")
        return self


def _one_dimensional_signal(
    group: h5py.Group, name: str, length: int, *, default: bool = False
) -> np.ndarray:
    if name not in group:
        return np.full(length, default, dtype=bool)
    value = np.asarray(group[name][:])
    if value.shape != (length,):
        raise ValueError(
            f"{group.name}/{name} has shape {value.shape}; expected ({length},)"
        )
    return value.astype(bool, copy=False)


def infer_bc_skill_labels(
    episode: h5py.Group,
    *,
    config: PhaseInferenceConfig = PhaseInferenceConfig(),
) -> np.ndarray:
    """Infer a per-step :class:`SkillPhase` label without using pose inputs.

    The task's phase 0 contains both approach and grasp/lift.  Its first close
    command, physical hand contact, or lift flag starts BC-2.  This makes the
    split work for both nominal teacher data (which has an injected close
    command) and later corrective data where contact can precede a clean close
    command.  Task phases 1 and 2 map to carry and place.  Complete/unknown
    steps receive ``-1`` and are excluded.
    """

    config.validate()
    if "phase_id" not in episode:
        raise KeyError(f"{episode.name} has no phase_id dataset")
    task_phase = np.asarray(episode["phase_id"][:])
    if task_phase.ndim != 1:
        raise ValueError(f"{episode.name}/phase_id must be one-dimensional")
    length = int(task_phase.shape[0])
    labels = np.full(length, -1, dtype=np.int8)

    phase_zero = task_phase == 0
    labels[phase_zero] = int(SkillPhase.APPROACH)
    labels[task_phase == 1] = int(SkillPhase.CARRY)
    labels[task_phase == 2] = int(SkillPhase.PLACE)

    if not np.any(phase_zero):
        return labels
    if "teacher_high_level_action" not in episode:
        raise KeyError(
            f"{episode.name} has no teacher_high_level_action dataset for phase-0 inference"
        )
    high_level = np.asarray(
        episode["teacher_high_level_action"][:, :2], dtype=np.float32
    )
    if high_level.shape != (length, 2) or not np.isfinite(high_level).all():
        raise ValueError(
            f"{episode.name}/teacher_high_level_action hand channels must have shape ({length}, 2) and be finite"
        )
    close_intent = np.any(high_level > config.hand_close_threshold, axis=1)
    left_contact = _one_dimensional_signal(episode, "left_contact", length)
    right_contact = _one_dimensional_signal(episode, "right_contact", length)
    lift = _one_dimensional_signal(episode, "lift", length)
    trigger = phase_zero & (close_intent | left_contact | right_contact | lift)
    trigger_indices = np.flatnonzero(trigger)
    if trigger_indices.size:
        phase_zero_indices = np.flatnonzero(phase_zero)
        first_phase_zero = int(phase_zero_indices[0])
        grasp_start = max(
            first_phase_zero, int(trigger_indices[0]) - config.grasp_lead_steps
        )
        labels[phase_zero & (np.arange(length) >= grasp_start)] = int(
            SkillPhase.GRASP_LIFT
        )
    return labels


@dataclass(frozen=True, slots=True)
class EpisodeRef:
    path: str
    episode: str
    num_steps: int

    @property
    def uid(self) -> str:
        return f"{self.path}::{self.episode}"


@dataclass(frozen=True, slots=True)
class DaggerEpisodeRef:
    """One physical DAgger rollout represented in a flat correction file."""

    path: str
    source_episode: int
    num_steps: int

    @property
    def uid(self) -> str:
        return f"{self.path}::dagger_source_episode_{self.source_episode}"


def _episode_succeeded(group: h5py.Group) -> bool:
    if "success" in group.attrs:
        return bool(group.attrs["success"])
    if "success" in group:
        return bool(np.any(group["success"][:]))
    return False


def expand_teacher_dataset_paths(
    paths: Sequence[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    """Expand files, shell-style globs, shard directories, or manifests.

    VS Code's Python launcher does not perform shell glob expansion.  Accepting
    the production shard directory directly keeps CLI and launch workflows
    equivalent while still resolving every concrete HDF5 path into checkpoint
    episode identifiers.
    """

    expanded: list[Path] = []
    for raw_path in paths:
        text = os.fspath(raw_path)
        candidates: list[Path]
        if glob.has_magic(text):
            candidates = [
                Path(value) for value in sorted(glob.glob(os.path.expanduser(text)))
            ]
        else:
            path = Path(text).expanduser()
            if path.is_dir():
                candidates = sorted(path.glob("shard_*.hdf5"))
                candidates = [
                    value
                    for value in candidates
                    if not value.name.endswith(".snapshots.hdf5")
                ]
            elif path.suffix.lower() == ".json" and path.is_file():
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest.get("schema") not in (TEACHER_SHARD_MANIFEST_SCHEMA, DIVERSE_TEACHER_SHARD_MANIFEST_SCHEMA):
                    raise ValueError(f"unsupported teacher manifest: {path}")
                candidates = [
                    Path(record["path"]).expanduser()
                    for record in manifest.get("shards", [])
                ]
            else:
                candidates = [path]
        if not candidates:
            raise FileNotFoundError(
                f"teacher dataset path matched no HDF5 files: {text}"
            )
        expanded.extend(candidates)

    result: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        resolved = path.resolve()
        if not resolved.is_file() or resolved.suffix.lower() not in {".h5", ".hdf5"}:
            raise FileNotFoundError(f"teacher dataset is not an HDF5 file: {resolved}")
        if resolved in seen:
            raise ValueError(f"teacher HDF5 path was supplied twice: {resolved}")
        seen.add(resolved)
        result.append(resolved)
    return tuple(result)


def teacher_dataset_carry_runtime_contract(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read and bind one CARRY corpus contract to its exact 43-D joint order."""

    resolved = Path(path).expanduser().resolve()
    with h5py.File(resolved, "r") as source:
        raw_metadata = source.attrs.get("metadata_json", "{}")
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        try:
            metadata = json.loads(str(raw_metadata))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid metadata_json in carry teacher dataset: {resolved}"
            ) from exc
        if not isinstance(metadata, dict):
            raise ValueError(
                f"carry teacher metadata_json must be an object: {resolved}"
            )
        if "carry_runtime_contract" not in metadata:
            raise ValueError(
                f"carry teacher dataset has no carry_runtime_contract: {resolved}"
            )
        if "metadata/processed_joint_names" in source:
            joint_names = _decode_string_dataset(
                source["metadata/processed_joint_names"]
            )
        elif "joint_names" in source:
            joint_names = _decode_string_dataset(source["joint_names"])
        else:
            raise ValueError(
                f"carry teacher dataset has no ordered 43-D joint names: {resolved}"
            )
        contract = validate_carry_runtime_contract(
            metadata["carry_runtime_contract"],
            expected_ordered_joint_names=joint_names,
        )
        if metadata.get("carry_runtime_contract_sha256") != contract["contract_sha256"]:
            raise ValueError(
                f"carry teacher root contract hash is missing or invalid: {resolved}"
            )
        return contract


def validate_carry_teacher_runtime_contract(
    paths: Sequence[str | os.PathLike[str]],
) -> dict[str, Any]:
    """Require exact CARRY semantics across every contributing teacher source."""

    concrete = expand_teacher_dataset_paths(paths)
    reference: dict[str, Any] | None = None
    reference_path: Path | None = None
    for path in concrete:
        contract = teacher_dataset_carry_runtime_contract(path)
        if reference is None:
            reference = contract
            reference_path = path
        elif contract != reference:
            raise ValueError(
                "carry runtime contract mismatch between teacher sources "
                f"{reference_path} and {path}"
            )
    if reference is None:
        raise ValueError("at least one carry-contract teacher dataset is required")
    return reference


def teacher_dataset_place_runtime_contract(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read and bind one PLACE corpus contract to its exact 43-D joint order."""

    resolved = Path(path).expanduser().resolve()
    with h5py.File(resolved, "r") as source:
        raw_metadata = source.attrs.get("metadata_json", "{}")
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        try:
            metadata = json.loads(str(raw_metadata))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid metadata_json in place teacher dataset: {resolved}"
            ) from exc
        if not isinstance(metadata, dict):
            raise ValueError(
                f"place teacher metadata_json must be an object: {resolved}"
            )
        if "place_runtime_contract" not in metadata:
            raise ValueError(
                f"place teacher dataset has no place_runtime_contract: {resolved}"
            )
        if "metadata/processed_joint_names" in source:
            joint_names = _decode_string_dataset(
                source["metadata/processed_joint_names"]
            )
        elif "joint_names" in source:
            joint_names = _decode_string_dataset(source["joint_names"])
        else:
            raise ValueError(
                f"place teacher dataset has no ordered 43-D joint names: {resolved}"
            )
        contract = validate_place_runtime_contract(
            metadata["place_runtime_contract"],
            expected_ordered_joint_names=joint_names,
        )
        if metadata.get("place_runtime_contract_sha256") != contract["contract_sha256"]:
            raise ValueError(
                f"place teacher root contract hash is missing or invalid: {resolved}"
            )
        return contract


def validate_place_teacher_runtime_contract(
    paths: Sequence[str | os.PathLike[str]],
) -> dict[str, Any]:
    """Require exact PLACE semantics across every contributing teacher source."""

    concrete = expand_teacher_dataset_paths(paths)
    reference: dict[str, Any] | None = None
    reference_path: Path | None = None
    for path in concrete:
        contract = teacher_dataset_place_runtime_contract(path)
        if reference is None:
            reference = contract
            reference_path = path
        elif contract != reference:
            raise ValueError(
                "place runtime contract mismatch between teacher sources "
                f"{reference_path} and {path}"
            )
    if reference is None:
        raise ValueError("at least one place-contract teacher dataset is required")
    return reference


def validate_complete_teacher_corpus(
    paths: Sequence[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    """Validate every production manifest explicitly supplied or found in a directory.

    Individual HDF5 files and globs intentionally remain usable for diagnostics.
    A production shard directory (or its manifest), however, must not silently
    start BC before the requested 500--1000-demo, 80%-success gate is complete.
    The returned paths are the manifests that were checked.
    """

    manifests: list[Path] = []
    for raw_path in paths:
        text = os.fspath(raw_path)
        if glob.has_magic(text):
            continue
        path = Path(text).expanduser()
        if path.is_dir():
            candidate = path / "manifest.json"
            if candidate.is_file():
                manifests.append(candidate.resolve())
        elif path.suffix.lower() == ".json" and path.is_file():
            manifests.append(path.resolve())

    checked: list[Path] = []
    seen: set[Path] = set()
    for path in manifests:
        if path in seen:
            continue
        seen.add(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") not in (TEACHER_SHARD_MANIFEST_SCHEMA, DIVERSE_TEACHER_SHARD_MANIFEST_SCHEMA):
            raise ValueError(f"unsupported teacher manifest: {path}")
        accepted = int(manifest.get("accepted_demos", -1))
        target = int(manifest.get("target_demos", -1))
        attempts = int(manifest.get("attempts", -1))
        rate = float(manifest.get("aggregate_teacher_success_rate", float("nan")))
        gates = manifest.get("gates", {})
        minimum_rate = float(gates.get("minimum_success_rate", 0.80))
        if target < MINIMUM_PRODUCTION_DEMOS or target > MAXIMUM_PRODUCTION_DEMOS:
            raise RuntimeError(
                f"teacher manifest target must be in "
                f"[{MINIMUM_PRODUCTION_DEMOS}, {MAXIMUM_PRODUCTION_DEMOS}], got {target}: {path}"
            )
        complete = (
            bool(manifest.get("complete", False))
            and accepted >= target
            and attempts >= accepted
            and np.isfinite(rate)
            and rate >= minimum_rate
            and bool(gates.get("demo_count_reached", False))
            and bool(gates.get("success_rate_at_least_minimum", False))
            and bool(gates.get("all_demos_strict_bilateral_lift_hold", False))
        )
        shard_count = sum(
            int(record.get("accepted", 0)) for record in manifest.get("shards", [])
        )
        if shard_count != accepted:
            raise RuntimeError(
                f"teacher manifest shard count {shard_count} does not match accepted_demos {accepted}: {path}"
            )
        if not complete:
            raise RuntimeError(
                f"teacher corpus is incomplete or below its physical gate "
                f"(accepted={accepted}/{target}, success_rate={rate:.1%}): {path}"
            )
        if manifest.get("schema") == DIVERSE_TEACHER_SHARD_MANIFEST_SCHEMA:
            quotas = manifest.get("target_successes_per_cell", [])
            counts = [sum(int(record.get("accepted", 0)) for record in manifest["shards"] if record.get("cell") == cell) for cell in range(16)]
            if minimum_rate < 0.8 or len(quotas) != 16 or sum(quotas) != target or counts != quotas or not gates.get("balanced_spatial_coverage_reached"):
                raise RuntimeError(f"diverse teacher spatial coverage/physical reliability gate failed: {path}")
            if len({record["path"] for record in manifest["shards"]}) != len(manifest["shards"]):
                raise RuntimeError(f"diverse teacher manifest contains duplicated shard paths: {path}")
            for record in manifest["shards"]:
                shard_path = Path(record["path"])
                if not shard_path.is_file() or file_sha256(shard_path) != record.get("sha256"):
                    raise RuntimeError(f"diverse teacher shard missing or SHA mismatch: {shard_path}")
        checked.append(path)
    return tuple(checked)


def _validate_phase_teacher_publication(path: Path, source: h5py.File) -> None:
    """Reject stale carry/place files left behind by a failed overwrite run.

    ``TeacherDemoRecorder`` publishes atomically, so aborting a failed run
    intentionally preserves any older file at the requested output path.  A
    carry/place corpus is therefore trainable only when its sidecar metrics
    binds the current HDF5 bytes and collection run id to a successful publish.
    Other recorder corpora keep their existing validation contracts.
    """

    raw_metadata = source.attrs.get("metadata_json", "{}")
    if isinstance(raw_metadata, bytes):
        raw_metadata = raw_metadata.decode("utf-8")
    try:
        metadata = json.loads(str(raw_metadata))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid metadata_json in teacher dataset: {path}") from exc
    teacher_phase = metadata.get("teacher_phase")
    if teacher_phase not in {"carry", "place"}:
        return

    collection_run_id = metadata.get("collection_run_id")
    if not isinstance(collection_run_id, str) or not collection_run_id:
        raise ValueError(
            f"carry/place teacher dataset has no collection_run_id binding: {path}"
        )
    metrics_path = path.with_suffix(".metrics.json")
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"carry/place teacher publication metrics are missing: {metrics_path}"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("collection_run_id") != collection_run_id:
        raise ValueError(f"carry/place teacher run id mismatch: {path}")
    if metrics.get("phase") != teacher_phase:
        raise ValueError(f"carry/place teacher phase mismatch: {path}")
    if metrics.get("dataset_published") is not True:
        raise ValueError(
            f"carry/place teacher was not published by its latest run: {path}"
        )
    if Path(str(metrics.get("published_output", ""))).resolve() != path:
        raise ValueError(f"carry/place teacher publication path mismatch: {path}")
    if metrics.get("dataset_sha256") != file_sha256(path):
        raise ValueError(f"carry/place teacher publication SHA-256 mismatch: {path}")
    if metadata.get("action_source") == "teacher_relabels_on_student_states":
        if metrics.get("evaluation_mode") != "student_teacher_relabel":
            raise ValueError(f"carry/place Student relabel mode mismatch: {path}")
        if int(metrics.get("trials", 0)) <= 0:
            raise ValueError(f"carry/place Student relabel corpus is empty: {path}")
        if metrics.get("student_checkpoint_sha256") != metadata.get(
            "student_checkpoint_sha256"
        ):
            raise ValueError(
                f"carry/place Student relabel checkpoint SHA-256 mismatch: {path}"
            )
        return
    if int(metrics.get("successes", 0)) <= 0:
        raise ValueError(
            f"carry/place teacher has no successful physical rollout: {path}"
        )


def discover_episodes(
    paths: Sequence[str | os.PathLike[str]],
    *,
    success_only: bool = True,
) -> list[EpisodeRef]:
    """Discover complete recorder episodes in deterministic order."""

    if not paths:
        raise ValueError("at least one teacher HDF5 path is required")
    refs: list[EpisodeRef] = []
    seen: set[str] = set()
    for resolved_path in expand_teacher_dataset_paths(paths):
        path = str(resolved_path)
        if path in seen:
            raise ValueError(f"teacher HDF5 path was supplied twice: {path}")
        seen.add(path)
        with h5py.File(path, "r") as source:
            _validate_phase_teacher_publication(resolved_path, source)
            if "data" not in source:
                raise ValueError(f"{path} has no /data group")
            for episode_name in sorted(source["data"], key=_natural_episode_key):
                group = source["data"][episode_name]
                if not isinstance(group, h5py.Group):
                    continue
                complete = bool(group.attrs.get("complete", True))
                if not complete or (success_only and not _episode_succeeded(group)):
                    continue
                validate_dataset_student_boundary(group)
                num_steps = int(group["student_observation"].shape[0])
                if num_steps <= 0:
                    continue
                refs.append(EpisodeRef(path, episode_name, num_steps))
    if not refs:
        qualifier = "successful " if success_only else ""
        raise ValueError(f"no complete {qualifier}teacher episodes were found")
    return refs


def validate_strict_grasp_corpus(
    episodes: Sequence[EpisodeRef], *, required_hold_steps: int = 100
) -> None:
    """Independently reject nominal BC-2 episodes without strict bilateral lift."""

    if required_hold_steps <= 0:
        raise ValueError("required_hold_steps must be positive")
    for episode in episodes:
        with h5py.File(episode.path, "r") as source:
            group = source[f"data/{episode.episode}"]
            required = ("lift", "left_contact", "right_contact", "drop")
            missing = [name for name in required if name not in group]
            if missing:
                raise ValueError(
                    f"{group.name} cannot prove strict BC-2 success; missing {missing}"
                )
            strict = (
                np.asarray(group["lift"][:], dtype=bool)
                & np.asarray(group["left_contact"][:], dtype=bool)
                & np.asarray(group["right_contact"][:], dtype=bool)
                & ~np.asarray(group["drop"][:], dtype=bool)
            )
            padded = np.pad(strict.astype(np.int8), (1, 1))
            edges = np.flatnonzero(np.diff(padded))
            longest = 0 if edges.size == 0 else int(np.max(edges[1::2] - edges[::2]))
            if longest < required_hold_steps:
                raise ValueError(
                    f"{group.name} has only {longest} strict bilateral lift steps; "
                    f"BC-2 requires {required_hold_steps}"
                )


def _validate_dagger_publication(path: Path, source: h5py.File) -> None:
    """Require a SHA-bound sidecar from the run that published ``path``.

    A failed overwrite intentionally leaves the previous HDF5 file in place.
    The sidecar is written before collection starts, however, so an interrupted
    or failed run invalidates that stale file until a new run publishes both
    artifacts successfully.
    """

    raw_metadata = source.attrs.get("metadata_json", "{}")
    if isinstance(raw_metadata, bytes):
        raw_metadata = raw_metadata.decode("utf-8")
    try:
        metadata = json.loads(str(raw_metadata))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid metadata_json in DAgger dataset: {path}") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError(f"DAgger metadata_json must contain an object: {path}")

    collection_run_id = metadata.get("collection_run_id")
    if not isinstance(collection_run_id, str) or not collection_run_id:
        raise ValueError(f"DAgger dataset has no collection_run_id binding: {path}")

    metrics_path = path.with_suffix(".metrics.json")
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"DAgger publication metrics are missing: {metrics_path}"
        )
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid DAgger publication metrics JSON: {metrics_path}"
        ) from exc
    if not isinstance(metrics, Mapping):
        raise ValueError(
            f"DAgger publication metrics must contain an object: {metrics_path}"
        )
    if metrics.get("collection_run_id") != collection_run_id:
        raise ValueError(f"DAgger publication run id mismatch: {path}")
    if (
        metrics.get("status") != "completed"
        or metrics.get("dataset_published") is not True
    ):
        raise ValueError(f"DAgger dataset was not published by its latest run: {path}")
    if metrics.get("requested_output") != str(path):
        raise ValueError(f"DAgger requested output path mismatch: {path}")
    if metrics.get("published_output") != str(path):
        raise ValueError(f"DAgger publication path mismatch: {path}")
    if metrics.get("dataset_sha256") != file_sha256(path):
        raise ValueError(f"DAgger publication SHA-256 mismatch: {path}")

    # Balanced production collections bind the student and its certification,
    # not merely their mutable paths.  Legacy smoke datasets without these
    # fields remain readable, while a partially specified binding fails closed.
    checkpoint_fields = ("student_checkpoint", "student_checkpoint_sha256")
    gate_fields = ("bc2_gate", "bc2_gate_sha256")
    provenance_fields = (*checkpoint_fields, *gate_fields)
    if any(field in metadata for field in provenance_fields):
        expected_fields = (
            checkpoint_fields
            if metadata.get("skill_phase") == "approach"
            else provenance_fields
        )
        missing = [field for field in expected_fields if field not in metadata]
        if missing:
            raise ValueError(
                f"DAgger provenance binding is incomplete {missing}: {path}"
            )
        checkpoint_path = Path(str(metadata["student_checkpoint"])).expanduser()
        if (
            not checkpoint_path.is_absolute()
            or str(checkpoint_path.resolve()) != str(metadata["student_checkpoint"])
            or not checkpoint_path.is_file()
            or file_sha256(checkpoint_path) != metadata["student_checkpoint_sha256"]
        ):
            raise ValueError(
                f"DAgger source checkpoint path/SHA-256 is invalid: {path}"
            )
        if expected_fields == provenance_fields:
            gate_path = Path(str(metadata["bc2_gate"])).expanduser()
            if (
                not gate_path.is_absolute()
                or str(gate_path.resolve()) != str(metadata["bc2_gate"])
                or not gate_path.is_file()
                or file_sha256(gate_path) != metadata["bc2_gate_sha256"]
            ):
                raise ValueError(f"DAgger BC-2 gate path/SHA-256 is invalid: {path}")
            gate = validate_bc2_gate(gate_path)
            if Path(gate["checkpoint_path"]).resolve() != checkpoint_path.resolve():
                raise ValueError(
                    f"DAgger gate certifies a different checkpoint: {path}"
                )
        for field in expected_fields:
            if metrics.get(field) != metadata[field]:
                raise ValueError(
                    f"DAgger metrics changed provenance field {field!r}: {path}"
                )


def discover_dagger_episodes(
    paths: Sequence[str | os.PathLike[str]],
) -> list[DaggerEpisodeRef]:
    """Discover DAgger groups, using ``source_episode`` as the split unit."""

    refs: list[DaggerEpisodeRef] = []
    seen: set[str] = set()
    for raw_path in paths:
        resolved_path = Path(raw_path).expanduser().resolve()
        path = str(resolved_path)
        if path in seen:
            raise ValueError(f"DAgger HDF5 path was supplied twice: {path}")
        seen.add(path)
        with h5py.File(resolved_path, "r") as source:
            _validate_dagger_publication(resolved_path, source)
            if "samples" not in source:
                raise ValueError(f"{path} has no /samples group")
            samples = source["samples"]
            required = {
                "student_observation": (STUDENT_OBSERVATION_DIM,),
                "teacher_high_level_action": (23,),
                "processed_43d_joint_target": (43,),
                "skill_phase": (),
                "correction_type": (),
                "source_episode": (),
            }
            lengths: set[int] = set()
            for name, trailing in required.items():
                if name not in samples:
                    raise KeyError(f"{samples.name} has no {name} dataset")
                dataset = samples[name]
                if dataset.shape[1:] != trailing:
                    raise ValueError(
                        f"{dataset.name} has trailing shape {dataset.shape[1:]}; expected {trailing}"
                    )
                lengths.add(int(dataset.shape[0]))
            if len(lengths) != 1:
                raise ValueError(
                    f"{samples.name} datasets have inconsistent sample counts"
                )
            source_episode = np.asarray(samples["source_episode"][:], dtype=np.int64)
            if source_episode.ndim != 1 or np.any(source_episode < 0):
                raise ValueError(
                    "DAgger source_episode must be a non-negative one-dimensional array"
                )
            for episode_id in np.unique(source_episode):
                count = int(np.count_nonzero(source_episode == episode_id))
                refs.append(DaggerEpisodeRef(path, int(episode_id), count))
    return refs


def _natural_episode_key(name: str) -> tuple[str, int, str]:
    prefix, separator, suffix = name.rpartition("_")
    if separator and suffix.isdigit():
        return prefix, int(suffix), name
    return name, -1, name


@dataclass(frozen=True, slots=True)
class EpisodeSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed: int
    ratios: tuple[float, float, float]

    def ids(self, subset: str) -> tuple[str, ...]:
        if subset not in {"train", "validation", "test"}:
            raise ValueError("subset must be train, validation, or test")
        return getattr(self, subset)

    def as_dict(self) -> dict[str, Any]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "seed": self.seed,
            "ratios": list(self.ratios),
        }


def split_episodes(
    episodes: Sequence[EpisodeRef | DaggerEpisodeRef],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> EpisodeSplit:
    """Split whole episodes, guaranteeing no transition leakage."""

    if len(ratios) != 3 or any(
        not math.isfinite(value) or value < 0.0 for value in ratios
    ):
        raise ValueError("ratios must contain three finite non-negative values")
    total_ratio = sum(ratios)
    if total_ratio <= 0.0:
        raise ValueError("at least one split ratio must be positive")
    normalized = np.asarray(ratios, dtype=np.float64) / total_ratio
    uids = [episode.uid for episode in episodes]
    if len(set(uids)) != len(uids):
        raise ValueError("episode references must be unique")
    rng = np.random.default_rng(seed)
    order = np.asarray(sorted(uids), dtype=object)
    rng.shuffle(order)
    count = len(order)
    raw = normalized * count
    allocation = np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - allocation))[: count - int(allocation.sum())]:
        allocation[index] += 1

    # Train must exist.  For >=3 episodes, also keep validation/test nonempty
    # whenever their requested ratios are nonzero.
    required = [0]
    if count >= 3:
        required.extend(index for index in (1, 2) if normalized[index] > 0.0)
    for target in required:
        if allocation[target] > 0:
            continue
        donors = [
            index
            for index in range(3)
            if allocation[index] > 1 and index not in required[:1]
        ]
        if not donors:
            donors = [index for index in range(3) if allocation[index] > 1]
        if not donors:
            break
        donor = max(donors, key=lambda index: allocation[index])
        allocation[donor] -= 1
        allocation[target] += 1
    if allocation[0] == 0:
        donor = int(np.argmax(allocation))
        allocation[donor] -= 1
        allocation[0] += 1

    train_stop = int(allocation[0])
    validation_stop = train_stop + int(allocation[1])
    return EpisodeSplit(
        train=tuple(str(value) for value in order[:train_stop]),
        validation=tuple(str(value) for value in order[train_stop:validation_stop]),
        test=tuple(str(value) for value in order[validation_stop:]),
        seed=int(seed),
        ratios=tuple(float(value) for value in normalized),
    )


@dataclass(frozen=True, slots=True)
class PhaseArrays:
    observations: np.ndarray
    actions: np.ndarray
    episode_ids: tuple[str, ...]
    sample_episode_index: np.ndarray
    skill: SkillPhase
    action_key: str

    @property
    def num_samples(self) -> int:
        return int(self.observations.shape[0])

    @property
    def num_episodes(self) -> int:
        return len(self.episode_ids)


def load_phase_arrays(
    episodes: Sequence[EpisodeRef],
    episode_ids: Iterable[str],
    *,
    skill: SkillPhase | int | str,
    action_key: str,
    inference_config: PhaseInferenceConfig = PhaseInferenceConfig(),
) -> PhaseArrays:
    """Load one skill using only its explicit student observation dataset."""

    skill = parse_skill(skill)
    action_dim = validate_action_key(action_key)
    selected = set(episode_ids)
    by_path: dict[str, list[EpisodeRef]] = {}
    for episode in episodes:
        if episode.uid in selected:
            by_path.setdefault(episode.path, []).append(episode)
    missing = selected.difference(
        episode.uid for values in by_path.values() for episode in values
    )
    if missing:
        raise ValueError(f"split references unknown episodes: {sorted(missing)[:3]}")

    observation_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    sample_episode_chunks: list[np.ndarray] = []
    contributing: list[str] = []
    for path in sorted(by_path):
        with h5py.File(path, "r") as source:
            for episode in sorted(
                by_path[path], key=lambda value: _natural_episode_key(value.episode)
            ):
                group = source["data"][episode.episode]
                validate_dataset_student_boundary(group)
                if action_key not in group:
                    raise KeyError(f"{group.name} has no {action_key} teacher target")
                labels = infer_bc_skill_labels(group, config=inference_config)
                mask = labels == int(skill)
                if not np.any(mask):
                    continue

                # This is deliberately the only model-input read in this module.
                observations = validate_student_observation(
                    group["student_observation"][:]
                )
                actions = np.asarray(group[action_key][:], dtype=np.float32)
                expected_actions = (len(labels), action_dim)
                if actions.shape != expected_actions or not np.isfinite(actions).all():
                    raise ValueError(
                        f"{group.name}/{action_key} has shape {actions.shape}; expected finite {expected_actions}"
                    )
                if observations.shape[0] != len(labels):
                    raise ValueError(
                        f"{group.name} datasets have inconsistent step counts"
                    )
                contributing_index = len(contributing)
                contributing.append(episode.uid)
                observation_chunks.append(np.ascontiguousarray(observations[mask]))
                action_chunks.append(np.ascontiguousarray(actions[mask]))
                sample_episode_chunks.append(
                    np.full(int(mask.sum()), contributing_index, dtype=np.int32)
                )
    if not observation_chunks:
        return PhaseArrays(
            observations=np.empty((0, STUDENT_OBSERVATION_DIM), dtype=np.float32),
            actions=np.empty((0, action_dim), dtype=np.float32),
            episode_ids=(),
            sample_episode_index=np.empty((0,), dtype=np.int32),
            skill=skill,
            action_key=action_key,
        )
    return PhaseArrays(
        observations=np.concatenate(observation_chunks, axis=0),
        actions=np.concatenate(action_chunks, axis=0),
        episode_ids=tuple(contributing),
        sample_episode_index=np.concatenate(sample_episode_chunks, axis=0),
        skill=skill,
        action_key=action_key,
    )


def load_dagger_phase_arrays(
    episodes: Sequence[DaggerEpisodeRef],
    episode_ids: Iterable[str],
    *,
    skill: SkillPhase | int | str,
    action_key: str,
) -> PhaseArrays:
    """Load corrective labels while preserving DAgger rollout boundaries."""

    skill = parse_skill(skill)
    action_dim = validate_action_key(action_key)
    selected = set(episode_ids)
    by_path: dict[str, list[DaggerEpisodeRef]] = {}
    for episode in episodes:
        if episode.uid in selected:
            by_path.setdefault(episode.path, []).append(episode)
    known = {episode.uid for values in by_path.values() for episode in values}
    missing = selected.difference(known)
    if missing:
        raise ValueError(
            f"split references unknown DAgger episodes: {sorted(missing)[:3]}"
        )

    observation_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    sample_episode_chunks: list[np.ndarray] = []
    contributing: list[str] = []
    for path in sorted(by_path):
        with h5py.File(path, "r") as source:
            samples = source["samples"]
            observations = validate_student_observation(
                samples["student_observation"][:]
            )
            actions = np.asarray(samples[action_key][:], dtype=np.float32)
            phases = np.asarray(samples["skill_phase"][:], dtype=np.int64)
            source_episode = np.asarray(samples["source_episode"][:], dtype=np.int64)
            correction_type = np.asarray(samples["correction_type"][:], dtype=np.int64)
            length = int(observations.shape[0])
            if actions.shape != (length, action_dim) or not np.isfinite(actions).all():
                raise ValueError(
                    f"{samples.name}/{action_key} has shape {actions.shape}; expected finite ({length}, {action_dim})"
                )
            for name, value in (
                ("skill_phase", phases),
                ("source_episode", source_episode),
            ):
                if value.shape != (length,):
                    raise ValueError(
                        f"{samples.name}/{name} must have shape ({length},)"
                    )
            if correction_type.shape != (length,) or np.any(correction_type <= 0):
                raise ValueError(
                    "DAgger correction_type must be nonzero for every stored sample"
                )

            for episode in sorted(
                by_path[path], key=lambda value: value.source_episode
            ):
                mask = (source_episode == episode.source_episode) & (
                    phases == int(skill)
                )
                if not np.any(mask):
                    continue
                contributing_index = len(contributing)
                contributing.append(episode.uid)
                observation_chunks.append(np.ascontiguousarray(observations[mask]))
                action_chunks.append(np.ascontiguousarray(actions[mask]))
                sample_episode_chunks.append(
                    np.full(int(mask.sum()), contributing_index, dtype=np.int32)
                )
    if not observation_chunks:
        return PhaseArrays(
            observations=np.empty((0, STUDENT_OBSERVATION_DIM), dtype=np.float32),
            actions=np.empty((0, action_dim), dtype=np.float32),
            episode_ids=(),
            sample_episode_index=np.empty((0,), dtype=np.int32),
            skill=skill,
            action_key=action_key,
        )
    return PhaseArrays(
        observations=np.concatenate(observation_chunks, axis=0),
        actions=np.concatenate(action_chunks, axis=0),
        episode_ids=tuple(contributing),
        sample_episode_index=np.concatenate(sample_episode_chunks, axis=0),
        skill=skill,
        action_key=action_key,
    )


def merge_phase_arrays(*arrays: PhaseArrays) -> PhaseArrays:
    """Merge nominal and corrective samples, remapping episode indices."""

    if not arrays:
        raise ValueError("at least one PhaseArrays value is required")
    skill = arrays[0].skill
    action_key = arrays[0].action_key
    action_dim = validate_action_key(action_key)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    episode_ids: list[str] = []
    for value in arrays:
        if value.skill != skill or value.action_key != action_key:
            raise ValueError("all arrays must have the same skill and action key")
        if value.observations.shape != (value.num_samples, STUDENT_OBSERVATION_DIM):
            raise ValueError("invalid student observation shape while merging")
        if value.actions.shape != (value.num_samples, action_dim):
            raise ValueError("invalid teacher action shape while merging")
        duplicate = set(episode_ids).intersection(value.episode_ids)
        if duplicate:
            raise ValueError(
                f"episode appears in multiple merged inputs: {sorted(duplicate)[:3]}"
            )
        offset = len(episode_ids)
        episode_ids.extend(value.episode_ids)
        if value.num_samples:
            observations.append(value.observations)
            actions.append(value.actions)
            sample_indices.append(value.sample_episode_index + offset)
    if not observations:
        return PhaseArrays(
            observations=np.empty((0, STUDENT_OBSERVATION_DIM), dtype=np.float32),
            actions=np.empty((0, action_dim), dtype=np.float32),
            episode_ids=(),
            sample_episode_index=np.empty((0,), dtype=np.int32),
            skill=skill,
            action_key=action_key,
        )
    return PhaseArrays(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        episode_ids=tuple(episode_ids),
        sample_episode_index=np.concatenate(sample_indices, axis=0),
        skill=skill,
        action_key=action_key,
    )


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls, value: np.ndarray, *, minimum_scale: float = 1.0e-6
    ) -> "NormalizationStats":
        value = np.asarray(value)
        if (
            value.ndim != 2
            or value.shape[0] == 0
            or not np.issubdtype(value.dtype, np.number)
            or not np.isfinite(value).all()
        ):
            raise ValueError("normalization input must be a nonempty finite matrix")
        # Accumulate in float64 without first duplicating the complete 500--1000
        # episode corpus as a float64 matrix.
        mean = value.mean(axis=0, dtype=np.float64)
        scale = value.std(axis=0, dtype=np.float64)
        scale[scale < minimum_scale] = 1.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def normalize(self, value: np.ndarray) -> np.ndarray:
        return (np.asarray(value, dtype=np.float32) - self.mean) / self.scale

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) * self.scale + self.mean

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "NormalizationStats":
        mean = np.asarray(state["mean"], dtype=np.float32)
        scale = np.asarray(state["scale"], dtype=np.float32)
        if mean.ndim != 1 or scale.shape != mean.shape or not np.isfinite(mean).all():
            raise ValueError("invalid normalization state")
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError("normalization scales must be finite and positive")
        return cls(mean, scale)


class _NumpyPhaseDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        observations: np.ndarray,
        normalized_actions: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        self.observations = observations
        self.actions = normalized_actions
        if sample_weights is None:
            sample_weights = np.ones((len(observations),), dtype=np.float32)
        sample_weights = np.asarray(sample_weights, dtype=np.float32)
        if (
            sample_weights.shape != (len(observations),)
            or not np.isfinite(sample_weights).all()
            or np.any(sample_weights <= 0.0)
        ):
            raise ValueError("sample weights must be finite positive values")
        self.sample_weights = sample_weights

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.observations[index]),
            torch.from_numpy(self.actions[index]),
            torch.as_tensor(self.sample_weights[index]),
        )


class PhaseBCPolicy(nn.Module):
    """MLP with deployable input/output normalization embedded as buffers."""

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dims: Sequence[int],
        observation_stats: NormalizationStats,
        action_stats: NormalizationStats,
    ) -> None:
        super().__init__()
        if (
            action_dim <= 0
            or not hidden_dims
            or any(int(width) <= 0 for width in hidden_dims)
        ):
            raise ValueError("action_dim and all hidden_dims must be positive")
        if observation_stats.mean.shape != (STUDENT_OBSERVATION_DIM,):
            raise ValueError(
                "observation normalizer does not match the 517-D student contract"
            )
        if action_stats.mean.shape != (action_dim,):
            raise ValueError("action normalizer does not match action_dim")
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.register_buffer(
            "observation_mean", torch.from_numpy(observation_stats.mean.copy())
        )
        self.register_buffer(
            "observation_scale", torch.from_numpy(observation_stats.scale.copy())
        )
        self.register_buffer("action_mean", torch.from_numpy(action_stats.mean.copy()))
        self.register_buffer(
            "action_scale", torch.from_numpy(action_stats.scale.copy())
        )
        layers: list[nn.Module] = []
        input_dim = STUDENT_OBSERVATION_DIM
        for width in self.hidden_dims:
            layers.extend((nn.Linear(input_dim, width), nn.ELU()))
            input_dim = width
        layers.append(nn.Linear(input_dim, self.action_dim))
        self.network = nn.Sequential(*layers)

    def forward_normalized(self, observation: torch.Tensor) -> torch.Tensor:
        normalized_observation = (
            observation - self.observation_mean
        ) / self.observation_scale
        return self.network(normalized_observation)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized_action = self.forward_normalized(observation)
        return normalized_action * self.action_scale + self.action_mean


@dataclass(frozen=True, slots=True)
class TrainConfig:
    hidden_dims: tuple[int, ...] = (512, 512, 256)
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    batch_size: int = 1024
    epochs: int = 100
    patience: int = 15
    num_workers: int = 0
    seed: int = 0

    def validate(self) -> "TrainConfig":
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError(
                "learning_rate must be positive and weight_decay non-negative"
            )
        if (
            self.batch_size <= 0
            or self.epochs <= 0
            or self.patience <= 0
            or self.num_workers < 0
        ):
            raise ValueError(
                "batch_size, epochs, patience must be positive; num_workers non-negative"
            )
        return self


def _evaluate_arrays(
    policy: PhaseBCPolicy,
    arrays: PhaseArrays,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any] | None:
    if arrays.num_samples == 0:
        return None
    policy.eval()
    sum_squared = 0.0
    sum_absolute = 0.0
    normalized_sum_squared = 0.0
    per_dim_squared = np.zeros(policy.action_dim, dtype=np.float64)
    count = 0
    with torch.inference_mode():
        for start in range(0, arrays.num_samples, batch_size):
            stop = min(start + batch_size, arrays.num_samples)
            observation = torch.from_numpy(arrays.observations[start:stop]).to(device)
            target = torch.from_numpy(arrays.actions[start:stop]).to(device)
            prediction = policy(observation)
            error = prediction - target
            sum_squared += float(torch.square(error).sum().cpu())
            sum_absolute += float(torch.abs(error).sum().cpu())
            per_dim_squared += torch.square(error).sum(dim=0).cpu().numpy()
            normalized_target = (target - policy.action_mean) / policy.action_scale
            normalized_error = (
                policy.forward_normalized(observation) - normalized_target
            )
            normalized_sum_squared += float(torch.square(normalized_error).sum().cpu())
            count += int(error.numel())
    per_dim_rmse = np.sqrt(per_dim_squared / arrays.num_samples)
    return {
        "num_samples": arrays.num_samples,
        "num_episodes": arrays.num_episodes,
        "native_mse": sum_squared / count,
        "native_rmse": math.sqrt(sum_squared / count),
        "native_mae": sum_absolute / count,
        "normalized_mse": normalized_sum_squared / count,
        "per_action_rmse": per_dim_rmse.tolist(),
        "mean_per_action_rmse": float(per_dim_rmse.mean()),
        "max_per_action_rmse": float(per_dim_rmse.max()),
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_phase_policy(
    *,
    train_arrays: PhaseArrays,
    validation_arrays: PhaseArrays,
    test_arrays: PhaseArrays,
    split: EpisodeSplit,
    output_path: str | os.PathLike[str],
    config: TrainConfig = TrainConfig(),
    device: str | torch.device = "cpu",
    source_paths: Sequence[str | os.PathLike[str]] = (),
    initial_checkpoint: str | os.PathLike[str] | None = None,
    train_sample_weights: np.ndarray | None = None,
    selection_validation_arrays: PhaseArrays | None = None,
    auxiliary_evaluation_arrays: Mapping[str, PhaseArrays] | None = None,
    select_last_epoch: bool = False,
    checkpoint_metadata_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one phase policy and publish its best validation checkpoint."""

    config.validate()
    ordered_action_names = validate_teacher_action_contract(
        source_paths, train_arrays.action_key
    )
    carry_runtime_contract: dict[str, Any] | None = None
    place_runtime_contract: dict[str, Any] | None = None
    if train_arrays.skill in (SkillPhase.CARRY, SkillPhase.PLACE):
        # Validate exactly the HDF5 files that contributed phase samples.  A
        # command-line source may also contain unrelated phases; empty sources
        # are not allowed to dictate runtime semantics for this checkpoint.
        contributing_paths = sorted(
            {
                uid.rsplit("::", 1)[0]
                for arrays in (train_arrays, validation_arrays, test_arrays)
                for uid in arrays.episode_ids
            }
        )
        if not contributing_paths:
            raise ValueError(
                f"{train_arrays.skill.name} training has no contributing teacher sources"
            )
        if train_arrays.skill == SkillPhase.CARRY:
            carry_runtime_contract = validate_carry_teacher_runtime_contract(
                contributing_paths
            )
        else:
            place_runtime_contract = validate_place_teacher_runtime_contract(
                contributing_paths
            )
    if train_arrays.num_samples == 0:
        raise ValueError(
            f"training split has no {SKILL_NAMES[train_arrays.skill]} samples"
        )
    for arrays in (validation_arrays, test_arrays):
        if (
            arrays.skill != train_arrays.skill
            or arrays.action_key != train_arrays.action_key
        ):
            raise ValueError(
                "all PhaseArrays must describe the same skill and action key"
            )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(device)
    initial_checkpoint_path: Path | None = None
    initial_checkpoint_sha256: str | None = None
    initial_state: dict[str, torch.Tensor] | None = None
    if initial_checkpoint is None:
        observation_stats = NormalizationStats.fit(train_arrays.observations)
        action_stats = NormalizationStats.fit(train_arrays.actions)
    else:
        initial_checkpoint_path = Path(initial_checkpoint).expanduser().resolve()
        loaded = _load_checkpoint(initial_checkpoint_path, "cpu")
        initial_metadata = loaded["metadata"]
        if parse_skill(initial_metadata["skill_phase"]) != train_arrays.skill:
            raise ValueError("initial checkpoint skill does not match training skill")
        if initial_metadata["action_key"] != train_arrays.action_key:
            raise ValueError(
                "initial checkpoint action key does not match training action key"
            )
        if tuple(initial_metadata["hidden_dims"]) != tuple(config.hidden_dims):
            raise ValueError(
                "initial checkpoint hidden dims do not match training configuration"
            )
        observation_stats = NormalizationStats.from_dict(
            loaded["observation_normalization"]
        )
        action_stats = NormalizationStats.from_dict(loaded["action_normalization"])
        initial_state = {
            name: value.detach().cpu().clone()
            for name, value in loaded["model_state_dict"].items()
        }
        initial_checkpoint_sha256 = file_sha256(initial_checkpoint_path)
    policy = PhaseBCPolicy(
        action_dim=train_arrays.actions.shape[1],
        hidden_dims=config.hidden_dims,
        observation_stats=observation_stats,
        action_stats=action_stats,
    ).to(device)
    if initial_state is not None:
        policy.load_state_dict(initial_state)
    normalized_actions = np.ascontiguousarray(
        action_stats.normalize(train_arrays.actions)
    )
    dataset = _NumpyPhaseDataset(
        train_arrays.observations, normalized_actions, train_sample_weights
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.MSELoss(reduction="none")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_score = math.inf
    if initial_state is not None:
        selection_arrays = (
            validation_arrays
            if selection_validation_arrays is None
            else selection_validation_arrays
        )
        initial_metrics = _evaluate_arrays(
            policy, selection_arrays, batch_size=config.batch_size, device=device
        )
        if initial_metrics is not None:
            best_score = float(initial_metrics["normalized_mse"])
            best_state = initial_state
    epochs_without_improvement = 0
    history: list[dict[str, float | int | None]] = []
    for epoch in range(config.epochs):
        policy.train()
        total_loss = 0.0
        seen = 0
        for observation, normalized_target, sample_weight in loader:
            observation = observation.to(device, non_blocking=True)
            normalized_target = normalized_target.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = policy.forward_normalized(observation)
            per_sample_loss = loss_function(prediction, normalized_target).mean(dim=1)
            loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum()
            loss.backward()
            optimizer.step()
            batch_count = int(observation.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_count
            seen += batch_count
        train_loss = total_loss / seen
        selection_arrays = (
            validation_arrays
            if selection_validation_arrays is None
            else selection_validation_arrays
        )
        validation_metrics = _evaluate_arrays(
            policy, selection_arrays, batch_size=config.batch_size, device=device
        )
        score = (
            train_loss
            if validation_metrics is None
            else float(validation_metrics["normalized_mse"])
        )
        history.append(
            {
                "epoch": epoch,
                "train_normalized_mse": train_loss,
                "validation_normalized_mse": None
                if validation_metrics is None
                else score,
            }
        )
        if select_last_epoch:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in policy.state_dict().items()
            }
            epochs_without_improvement = 0
        elif score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in policy.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a finite checkpoint")
    policy.load_state_dict(best_state)
    policy.to(device)
    metrics = {
        "train": _evaluate_arrays(
            policy, train_arrays, batch_size=config.batch_size, device=device
        ),
        "validation": _evaluate_arrays(
            policy, validation_arrays, batch_size=config.batch_size, device=device
        ),
        "test": _evaluate_arrays(
            policy, test_arrays, batch_size=config.batch_size, device=device
        ),
    }
    if selection_validation_arrays is not None:
        metrics["selection_validation"] = _evaluate_arrays(
            policy,
            selection_validation_arrays,
            batch_size=config.batch_size,
            device=device,
        )
    for name, arrays in (auxiliary_evaluation_arrays or {}).items():
        if not name or name in metrics:
            raise ValueError(f"invalid or duplicate auxiliary metric name: {name!r}")
        metrics[name] = _evaluate_arrays(
            policy, arrays, batch_size=config.batch_size, device=device
        )
    metadata = {
        "format": BC_CHECKPOINT_FORMAT,
        "skill_phase": SKILL_NAMES[train_arrays.skill],
        "skill_phase_id": int(train_arrays.skill),
        "action_key": train_arrays.action_key,
        "action_dim": int(train_arrays.actions.shape[1]),
        "ordered_action_names": list(ordered_action_names),
        "action_contract_sha256": action_contract_sha256(
            train_arrays.action_key, ordered_action_names
        ),
        "student_observation_schema": STUDENT_OBSERVATION_SCHEMA,
        "student_observation_dim": STUDENT_OBSERVATION_DIM,
        "student_input_datasets": ["student_observation"],
        "privileged_student_inputs": [],
        "hidden_dims": list(config.hidden_dims),
        "source_paths": [
            str(Path(path).expanduser().resolve()) for path in source_paths
        ],
        "episode_split": split.as_dict(),
        "sample_counts": {
            "train": train_arrays.num_samples,
            "validation": validation_arrays.num_samples,
            "test": test_arrays.num_samples,
        },
        "episode_counts": {
            "train": train_arrays.num_episodes,
            "validation": validation_arrays.num_episodes,
            "test": test_arrays.num_episodes,
        },
        "train_config": asdict(config),
        "sample_weighting": {
            "enabled": train_sample_weights is not None,
            "minimum": 1.0
            if train_sample_weights is None
            else float(np.min(train_sample_weights)),
            "maximum": 1.0
            if train_sample_weights is None
            else float(np.max(train_sample_weights)),
            "mean": 1.0
            if train_sample_weights is None
            else float(np.mean(train_sample_weights)),
        },
        "initial_checkpoint": None
        if initial_checkpoint_path is None
        else str(initial_checkpoint_path),
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "best_epoch": best_epoch,
        "checkpoint_selection": "last" if select_last_epoch else "validation",
        "metrics": metrics,
    }
    for key, value in (checkpoint_metadata_extra or {}).items():
        if key in metadata:
            raise ValueError(f"checkpoint metadata extra collides with {key!r}")
        metadata[key] = value
    if carry_runtime_contract is not None:
        metadata["carry_runtime_contract"] = carry_runtime_contract
        metadata["carry_runtime_contract_sha256"] = carry_runtime_contract_sha256(
            carry_runtime_contract
        )
    if place_runtime_contract is not None:
        metadata["place_runtime_contract"] = place_runtime_contract
        metadata["place_runtime_contract_sha256"] = place_runtime_contract_sha256(
            place_runtime_contract
        )
    checkpoint = {
        "metadata": metadata,
        "model_state_dict": best_state,
        "observation_normalization": observation_stats.as_dict(),
        "action_normalization": action_stats.as_dict(),
        "history": history,
    }
    output = Path(output_path).expanduser().resolve()
    _atomic_torch_save(checkpoint, output)
    return {"checkpoint": str(output), **metadata}


def _load_checkpoint(
    path: str | os.PathLike[str], device: str | torch.device
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("metadata"), dict
    ):
        raise ValueError("not a phase-BC checkpoint")
    metadata = checkpoint["metadata"]
    if metadata.get("format") != BC_CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported BC checkpoint format: {metadata.get('format')!r}"
        )
    if metadata.get("student_observation_dim") != STUDENT_OBSERVATION_DIM:
        raise ValueError(
            "checkpoint does not use the deployed 517-D student observation"
        )
    if metadata.get("student_input_datasets") != ["student_observation"]:
        raise ValueError("checkpoint declares an unsupported student input boundary")
    if metadata.get("privileged_student_inputs") not in ([], None):
        raise ValueError("checkpoint declares privileged student inputs")
    skill = parse_skill(metadata["skill_phase"])
    validate_action_key(metadata["action_key"])
    if int(metadata.get("action_dim", -1)) != validate_action_key(
        metadata["action_key"]
    ):
        raise ValueError("checkpoint action dimension does not match its action key")
    ordered_action_names = tuple(metadata.get("ordered_action_names", ()))
    expected_contract = action_contract_sha256(
        metadata["action_key"], ordered_action_names
    )
    if metadata.get("action_contract_sha256") != expected_contract:
        raise ValueError("checkpoint action contract hash is missing or invalid")
    if skill == SkillPhase.CARRY:
        carry_contract = validate_carry_runtime_contract(
            metadata.get("carry_runtime_contract", {})
        )
        if (
            metadata.get("carry_runtime_contract_sha256")
            != carry_contract["contract_sha256"]
        ):
            raise ValueError(
                "checkpoint carry runtime contract hash is missing or invalid"
            )
        if (
            metadata["action_key"] == "processed_43d_joint_target"
            and tuple(carry_contract["ordered_raw_joint_names"]) != ordered_action_names
        ):
            raise ValueError(
                "processed 43-D checkpoint action order does not match its carry runtime contract"
            )
    if skill == SkillPhase.PLACE:
        place_contract = validate_place_runtime_contract(
            metadata.get("place_runtime_contract", {})
        )
        if (
            metadata.get("place_runtime_contract_sha256")
            != place_contract["contract_sha256"]
        ):
            raise ValueError(
                "checkpoint place runtime contract hash is missing or invalid"
            )
        if (
            metadata["action_key"] == "processed_43d_joint_target"
            and tuple(place_contract["ordered_raw_joint_names"]) != ordered_action_names
        ):
            raise ValueError(
                "processed 43-D checkpoint action order does not match its place runtime contract"
            )
    return checkpoint


class LoadedBCPolicy:
    """Callable NumPy runtime wrapper with immutable checkpoint metadata."""

    def __init__(
        self, model: PhaseBCPolicy, metadata: Mapping[str, Any], device: torch.device
    ) -> None:
        self.model = model.eval()
        self.metadata = dict(metadata)
        self.device = device
        self.skill_phase = parse_skill(self.metadata["skill_phase"])
        self.action_key = str(self.metadata["action_key"])
        self.action_dim = int(self.metadata["action_dim"])
        self.ordered_action_names = tuple(self.metadata["ordered_action_names"])
        self.action_contract_sha256 = str(self.metadata["action_contract_sha256"])
        self.carry_runtime_contract = (
            validate_carry_runtime_contract(self.metadata["carry_runtime_contract"])
            if self.skill_phase == SkillPhase.CARRY
            else None
        )
        self.carry_runtime_contract_sha256 = (
            None
            if self.carry_runtime_contract is None
            else str(self.carry_runtime_contract["contract_sha256"])
        )
        self.place_runtime_contract = (
            validate_place_runtime_contract(self.metadata["place_runtime_contract"])
            if self.skill_phase == SkillPhase.PLACE
            else None
        )
        self.place_runtime_contract_sha256 = (
            None
            if self.place_runtime_contract is None
            else str(self.place_runtime_contract["contract_sha256"])
        )

    def __call__(self, observation: Any) -> np.ndarray:
        value = validate_student_observation(observation)
        if value.ndim == 1:
            value = value[None, :]
        with torch.inference_mode():
            prediction = self.model(
                torch.from_numpy(np.ascontiguousarray(value)).to(self.device)
            )
        result = prediction.cpu().numpy().astype(np.float32, copy=False)
        expected = (value.shape[0], self.action_dim)
        if result.shape != expected or not np.isfinite(result).all():
            raise RuntimeError(
                f"BC policy returned {result.shape}; expected finite {expected}"
            )
        return result


def validate_runtime_action_contract(
    policy: LoadedBCPolicy, ordered_action_names: Sequence[str]
) -> None:
    """Fail before applying a BC action to a differently ordered runtime."""

    names = tuple(str(name) for name in ordered_action_names)
    if names != policy.ordered_action_names:
        raise ValueError(
            f"runtime {policy.action_key} order does not match checkpoint action contract"
        )
    if (
        action_contract_sha256(policy.action_key, names)
        != policy.action_contract_sha256
    ):
        raise ValueError("runtime action contract hash does not match checkpoint")


def load_bc_policy(
    checkpoint_path: str | os.PathLike[str],
    device: str | torch.device = "cpu",
) -> LoadedBCPolicy:
    """Load a deployable callable accepting ``(N, 517)`` student inputs.

    The returned object's ``metadata``, ``skill_phase``, ``action_key``, and
    ``action_dim`` attributes let the FSM verify routing compatibility.
    """

    device = torch.device(device)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    metadata = checkpoint["metadata"]
    observation_stats = NormalizationStats.from_dict(
        checkpoint["observation_normalization"]
    )
    action_stats = NormalizationStats.from_dict(checkpoint["action_normalization"])
    policy = PhaseBCPolicy(
        action_dim=int(metadata["action_dim"]),
        hidden_dims=tuple(int(value) for value in metadata["hidden_dims"]),
        observation_stats=observation_stats,
        action_stats=action_stats,
    )
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.to(device)
    return LoadedBCPolicy(policy, metadata, device)


def evaluate_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    arrays: PhaseArrays,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Evaluate a checkpoint in native and normalized action units."""

    loaded = load_bc_policy(checkpoint_path, device)
    if loaded.skill_phase != arrays.skill or loaded.action_key != arrays.action_key:
        raise ValueError("checkpoint skill/action key does not match evaluation data")
    metrics = _evaluate_arrays(
        loaded.model, arrays, batch_size=batch_size, device=loaded.device
    )
    if metrics is None:
        raise ValueError("evaluation subset has no samples")
    return metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    return _sha256(Path(path).expanduser().resolve())


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _canonical_sequence_sha256(value: Sequence[Any]) -> str:
    payload = json.dumps(list(value), separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _audit_bc2_snapshot_store(
    snapshot_path: Path,
    *,
    rollout_metrics: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    action_key: str,
    action_contract_sha256: str,
    required_hold_steps: int,
    strict_hold_seconds: float,
    strict_lift_m: float,
    min_bimanual_lift_success_rate: float,
) -> dict[str, Any]:
    """Reopen and semantically audit the complete BC-2 success snapshot store."""

    episode_results = rollout_metrics.get("episode_results")
    episodes = int(rollout_metrics.get("episodes", 0))
    if not isinstance(episode_results, list) or len(episode_results) != episodes:
        raise ValueError(
            "BC-2 snapshot audit requires one rollout result per physical episode"
        )
    successful_episode_ids: list[int] = []
    for expected_episode, result in enumerate(episode_results):
        if (
            not isinstance(result, Mapping)
            or type(result.get("episode")) is not int
            or result["episode"] != expected_episode
            or type(result.get("success")) is not bool
        ):
            raise ValueError(
                "BC-2 snapshot audit found invalid rollout episode results"
            )
        if result["success"]:
            successful_episode_ids.append(expected_episode)
    expected_sources = [f"bc2_episode_{episode}" for episode in successful_episode_ids]
    minimum_required_snapshots = int(
        math.ceil(min_bimanual_lift_success_rate * episodes - 1.0e-12)
    )
    if len(expected_sources) < minimum_required_snapshots:
        raise ValueError(
            "BC-2 snapshot count is below the certification success threshold"
        )

    try:
        metadata = load_snapshot_store_metadata(snapshot_path)
        with h5py.File(snapshot_path, "r") as h5_file:
            schema_name = h5_file.attrs.get("schema_name")
            schema_version = h5_file.attrs.get("schema_version")
            if isinstance(schema_name, bytes):
                schema_name = schema_name.decode("utf-8")
            if isinstance(schema_version, bytes):
                schema_version = schema_version.decode("utf-8")
            if (
                schema_name != SNAPSHOT_SCHEMA_NAME
                or schema_version not in SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS
            ):
                raise ValueError("BC-2 snapshot store has an unsupported schema")
            if "snapshots" not in h5_file or not isinstance(
                h5_file["snapshots"], h5py.Group
            ):
                raise ValueError("BC-2 snapshot store has no snapshots group")
            snapshot_group = h5_file["snapshots"]
            snapshot_names = list(snapshot_group.keys())
            declared_count = h5_file.attrs.get("num_snapshots")
            if not isinstance(declared_count, (int, np.integer)) or int(
                declared_count
            ) != len(snapshot_names):
                raise ValueError(
                    "BC-2 snapshot store is incomplete: finalized count does not match contents"
                )
        snapshots = load_snapshots(
            snapshot_path,
            lift_threshold_m=strict_lift_m,
            hold_steps_required=required_hold_steps,
        )
    except (KeyError, OSError) as exc:
        raise ValueError("BC-2 snapshot store cannot be read completely") from exc

    actual_sources = [snapshot.source_episode for snapshot in snapshots]
    if len(snapshots) != len(expected_sources):
        raise ValueError(
            "BC-2 snapshot count does not equal the successful rollout episode count"
        )
    if len(snapshots) < minimum_required_snapshots:
        raise ValueError(
            "BC-2 snapshot count is below the certification success threshold"
        )
    if actual_sources != expected_sources:
        raise ValueError(
            "BC-2 snapshots do not map in order to the successful rollout episodes"
        )
    for snapshot in snapshots:
        if SnapshotKind(int(snapshot.kind)) != SnapshotKind.BILATERAL_HOLD:
            raise ValueError(
                "BC-2 snapshot store contains a non-bilateral-hold snapshot"
            )
        if (
            snapshot.stable_hold_steps < required_hold_steps
            or snapshot.lift_height_m < strict_lift_m
            or snapshot.left_contact_count < 1
            or snapshot.right_contact_count < 1
            or snapshot.bad_orientation
            or snapshot.box_dropped
        ):
            raise ValueError(
                "BC-2 snapshot does not satisfy the strict bilateral lift/hold contract"
            )

    expected_metadata: dict[str, Any] = {
        "producer": "evaluate_bc2_rollout",
        "source": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "action_key": action_key,
        "action_contract_sha256": action_contract_sha256,
        "required_hold_steps": required_hold_steps,
        "box_teleport_into_hands": False,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"BC-2 snapshot store metadata {key!r} does not match the rollout/checkpoint"
            )
    if not math.isclose(
        float(metadata.get("strict_hold_seconds", math.nan)),
        strict_hold_seconds,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        float(metadata.get("strict_lift_m", math.nan)),
        strict_lift_m,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "BC-2 snapshot store metadata weakened the strict hold/lift contract"
        )

    return {
        "format": BC2_SNAPSHOT_AUDIT_FORMAT,
        "schema_name": SNAPSHOT_SCHEMA_NAME,
        "schema_version": schema_version,
        "num_snapshots": len(snapshots),
        "successful_rollout_episodes": len(expected_sources),
        "minimum_required_snapshots": minimum_required_snapshots,
        "snapshot_names_sha256": _canonical_sequence_sha256(snapshot_names),
        "source_episodes_sha256": _canonical_sequence_sha256(actual_sources),
        "store_metadata_sha256": _canonical_mapping_sha256(metadata),
    }


def certify_bc2_gate(
    checkpoint_path: str | os.PathLike[str],
    rollout_metrics: Mapping[str, Any],
    output_path: str | os.PathLike[str],
    *,
    max_validation_normalized_mse: float = 0.25,
    min_bimanual_lift_success_rate: float = 0.80,
    max_bad_orientation_rate: float = 0.05,
    min_rollout_episodes: int = 20,
    rollout_metrics_sha256: str | None = None,
    rollout_metrics_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Certify BC-2 before any other phase is trained.

    Offline action error alone cannot prove a physically stable lift.  The
    gate therefore requires both held-out normalized MSE and rollout success /
    bad-orientation rates.  The marker is bound to the checkpoint SHA-256.
    """

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path, "cpu")
    metadata = checkpoint["metadata"]
    if parse_skill(metadata["skill_phase"]) != SkillPhase.GRASP_LIFT:
        raise ValueError("BC-2 gate can certify only a grasp_lift checkpoint")
    validation = metadata.get("metrics", {}).get("validation")
    if not validation:
        raise ValueError("BC-2 checkpoint has no held-out validation metrics")
    checkpoint_sha256 = _sha256(checkpoint_path)
    required_binding = {
        "format": "unitree_box_bc2_physical_rollout_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "action_key": metadata["action_key"],
        "action_contract_sha256": metadata["action_contract_sha256"],
    }
    for key, expected in required_binding.items():
        if rollout_metrics.get(key) != expected:
            raise ValueError(
                f"rollout metrics {key!r} is not bound to the candidate checkpoint"
            )
    evaluator_config = rollout_metrics.get("evaluator_config")
    if not isinstance(evaluator_config, Mapping):
        raise ValueError("rollout metrics require evaluator_config")
    evaluator_hash = _canonical_mapping_sha256(evaluator_config)
    if rollout_metrics.get("evaluator_config_sha256") != evaluator_hash:
        raise ValueError("rollout evaluator config hash is invalid")
    episodes = int(rollout_metrics.get("episodes", 0))
    strict_hold_seconds = float(rollout_metrics.get("strict_hold_seconds", 0.0))
    strict_lift_m = float(rollout_metrics.get("strict_lift_m", 0.0))
    config_episodes = int(evaluator_config.get("episodes", 0))
    config_hold_seconds = float(evaluator_config.get("strict_hold_seconds", 0.0))
    config_lift_m = float(evaluator_config.get("strict_lift_m", 0.0))
    control_frequency_hz = float(evaluator_config.get("control_frequency_hz", 0.0))
    episode_seconds = float(evaluator_config.get("episode_seconds", 0.0))
    if (
        config_episodes != episodes
        or not math.isclose(config_hold_seconds, strict_hold_seconds, abs_tol=1.0e-12)
        or not math.isclose(config_lift_m, strict_lift_m, abs_tol=1.0e-12)
        or not math.isfinite(control_frequency_hz)
        or control_frequency_hz <= 0.0
        or not math.isfinite(episode_seconds)
        or episode_seconds < strict_hold_seconds
    ):
        raise ValueError("BC-2 top-level metrics disagree with evaluator_config")
    if episodes < min_rollout_episodes:
        raise ValueError(
            f"BC-2 certification requires at least {min_rollout_episodes} physical episodes"
        )
    if strict_hold_seconds < 2.0 or strict_lift_m < 0.12:
        raise ValueError("BC-2 rollout weakened the strict lift/hold contract")
    episode_results = rollout_metrics.get("episode_results")
    if not isinstance(episode_results, list) or len(episode_results) != episodes:
        raise ValueError("BC-2 rollout metrics require one result per physical episode")
    required_hold_steps = int(
        math.ceil(strict_hold_seconds * control_frequency_hz - 1.0e-9)
    )
    x_range = np.asarray(evaluator_config.get("box_x_range_m", ()), dtype=np.float64)
    y_range = np.asarray(evaluator_config.get("box_y_range_m", ()), dtype=np.float64)
    validate_xy = x_range.shape == (2,) and y_range.shape == (2,)
    for expected_episode, result in enumerate(episode_results):
        if not isinstance(result, Mapping):
            raise ValueError("BC-2 episode result must be an object")
        success_value = result.get("success")
        bad_value = result.get("bad_orientation")
        steps_value = result.get("steps")
        hold_value = result.get("max_continuous_bimanual_lift_hold_steps")
        if type(success_value) is not bool or type(bad_value) is not bool:
            raise ValueError("BC-2 episode success/safety fields must be booleans")
        if (
            type(result.get("episode")) is not int
            or result["episode"] != expected_episode
            or type(steps_value) is not int
            or steps_value <= 0
            or type(hold_value) is not int
            or hold_value < 0
        ):
            raise ValueError("BC-2 episode ids/steps/hold counts are invalid")
        if validate_xy:
            xy = np.asarray(result.get("initial_box_xy_m", ()), dtype=np.float64)
            if (
                xy.shape != (2,)
                or not np.isfinite(xy).all()
                or not (x_range[0] <= xy[0] <= x_range[1])
                or not (y_range[0] <= xy[1] <= y_range[1])
            ):
                raise ValueError("BC-2 episode Box x/y is outside evaluator bounds")
        if success_value and (
            bad_value
            or result.get("reason") != "strict_bimanual_lift_hold_success"
            or hold_value < required_hold_steps
        ):
            raise ValueError(
                "BC-2 success episode does not satisfy the strict physical hold contract"
            )
        if bad_value and result.get("reason") != "bad_orientation":
            raise ValueError("BC-2 bad-orientation result has an inconsistent reason")
    validation_mse = float(validation["normalized_mse"])
    success_key = next(
        (
            key
            for key in ("bimanual_grasp_lift_success_rate", "grasp_lift_success_rate")
            if key in rollout_metrics
        ),
        None,
    )
    if success_key is None or "bad_orientation_rate" not in rollout_metrics:
        raise ValueError(
            "rollout metrics require bimanual_grasp_lift_success_rate (or grasp_lift_success_rate) "
            "and bad_orientation_rate"
        )
    success_rate = float(rollout_metrics[success_key])
    bad_orientation_rate = float(rollout_metrics["bad_orientation_rate"])
    recomputed_success_rate = (
        sum(bool(result.get("success", False)) for result in episode_results) / episodes
    )
    recomputed_bad_orientation_rate = (
        sum(bool(result.get("bad_orientation", False)) for result in episode_results)
        / episodes
    )
    if not math.isclose(success_rate, recomputed_success_rate, abs_tol=1.0e-12):
        raise ValueError("BC-2 success scalar disagrees with episode results")
    if not math.isclose(
        bad_orientation_rate, recomputed_bad_orientation_rate, abs_tol=1.0e-12
    ):
        raise ValueError("BC-2 bad-orientation scalar disagrees with episode results")
    values = (
        validation_mse,
        success_rate,
        bad_orientation_rate,
        strict_hold_seconds,
        strict_lift_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("BC-2 gate metrics must be finite")
    checks = {
        "validation_normalized_mse": validation_mse <= max_validation_normalized_mse,
        "bimanual_grasp_lift_success_rate": success_rate
        >= min_bimanual_lift_success_rate,
        "bad_orientation_rate": bad_orientation_rate <= max_bad_orientation_rate,
    }
    if not all(checks.values()):
        raise RuntimeError(f"BC-2 gate failed: {checks}")
    if rollout_metrics_path is None or rollout_metrics_sha256 is None:
        raise ValueError(
            "BC-2 gate requires the physical rollout metrics file and SHA-256"
        )
    rollout_path = Path(rollout_metrics_path).expanduser().resolve()
    if not rollout_path.is_file() or _sha256(rollout_path) != rollout_metrics_sha256:
        raise ValueError("BC-2 rollout metrics path/SHA-256 binding is invalid")
    with rollout_path.open("r", encoding="utf-8") as stream:
        bound_rollout_metrics = json.load(stream)
    if not isinstance(bound_rollout_metrics, Mapping) or dict(
        bound_rollout_metrics
    ) != dict(rollout_metrics):
        raise ValueError("BC-2 audited metrics mapping differs from its bound file")
    snapshot_path_value = rollout_metrics.get("snapshot_output_path")
    snapshot_sha256 = rollout_metrics.get("snapshot_output_sha256")
    if not isinstance(snapshot_path_value, str) or not snapshot_path_value:
        raise ValueError(
            "BC-2 rollout metrics require an absolute snapshot output path"
        )
    snapshot_path = Path(snapshot_path_value).expanduser()
    if (
        not snapshot_path.is_absolute()
        or str(snapshot_path.resolve()) != snapshot_path_value
    ):
        raise ValueError(
            "BC-2 rollout snapshot output path must be absolute and canonical"
        )
    if (
        not isinstance(snapshot_sha256, str)
        or len(snapshot_sha256) != 64
        or not snapshot_path.is_file()
        or _sha256(snapshot_path) != snapshot_sha256
    ):
        raise ValueError("BC-2 rollout snapshot path/SHA-256 binding is invalid")
    snapshot_audit = _audit_bc2_snapshot_store(
        snapshot_path,
        rollout_metrics=rollout_metrics,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        action_key=str(metadata["action_key"]),
        action_contract_sha256=str(metadata["action_contract_sha256"]),
        required_hold_steps=required_hold_steps,
        strict_hold_seconds=strict_hold_seconds,
        strict_lift_m=strict_lift_m,
        min_bimanual_lift_success_rate=min_bimanual_lift_success_rate,
    )
    gate = {
        "format": BC2_GATE_FORMAT,
        "passed": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "action_key": metadata["action_key"],
        "action_contract_sha256": metadata["action_contract_sha256"],
        "rollout_metrics_sha256": rollout_metrics_sha256,
        "rollout_metrics_path": str(rollout_path),
        "snapshot_output_path": str(snapshot_path),
        "snapshot_output_sha256": snapshot_sha256,
        "snapshot_audit": snapshot_audit,
        "evaluator_config_sha256": evaluator_hash,
        "criteria": {
            "max_validation_normalized_mse": max_validation_normalized_mse,
            "min_bimanual_lift_success_rate": min_bimanual_lift_success_rate,
            "max_bad_orientation_rate": max_bad_orientation_rate,
            "min_rollout_episodes": min_rollout_episodes,
        },
        "measured": {
            "validation_normalized_mse": validation_mse,
            "bimanual_grasp_lift_success_rate": success_rate,
            "bad_orientation_rate": bad_orientation_rate,
            "rollout_episodes": episodes,
            "strict_hold_seconds": strict_hold_seconds,
            "strict_lift_m": strict_lift_m,
            "required_hold_steps": required_hold_steps,
            "strict_episode_audit_passed": True,
        },
        "checks": checks,
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(gate, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return gate


def validate_bc2_gate(
    gate_path: str | os.PathLike[str],
    *,
    required_action_key: str | None = None,
) -> dict[str, Any]:
    """Validate a BC-2 certification marker and its bound checkpoint."""

    gate_path = Path(gate_path).expanduser().resolve()
    with gate_path.open("r", encoding="utf-8") as stream:
        gate = json.load(stream)
    if gate.get("format") != BC2_GATE_FORMAT or gate.get("passed") is not True:
        raise ValueError("not a passing BC-2 gate marker")
    checkpoint_path = Path(gate["checkpoint_path"])
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != gate.get(
        "checkpoint_sha256"
    ):
        raise ValueError("BC-2 gate checkpoint is missing or its SHA-256 changed")
    checkpoint = _load_checkpoint(checkpoint_path, "cpu")
    metadata = checkpoint["metadata"]
    if parse_skill(metadata["skill_phase"]) != SkillPhase.GRASP_LIFT:
        raise ValueError("gate checkpoint is not BC-2")
    if gate.get("action_contract_sha256") != metadata.get("action_contract_sha256"):
        raise ValueError("BC-2 gate action contract does not match its checkpoint")
    if gate.get("action_key") != metadata.get("action_key"):
        raise ValueError("BC-2 gate action key does not match its checkpoint")
    rollout_path = Path(gate.get("rollout_metrics_path", "")).expanduser().resolve()
    if not rollout_path.is_file() or _sha256(rollout_path) != gate.get(
        "rollout_metrics_sha256"
    ):
        raise ValueError("BC-2 gate rollout metrics are missing or changed")
    with rollout_path.open("r", encoding="utf-8") as stream:
        rollout_metrics = json.load(stream)
    snapshot_path_value = gate.get("snapshot_output_path")
    snapshot_sha256 = gate.get("snapshot_output_sha256")
    if not isinstance(snapshot_path_value, str) or not snapshot_path_value:
        raise ValueError("BC-2 gate has no bound snapshot output path")
    snapshot_path = Path(snapshot_path_value).expanduser()
    if (
        not snapshot_path.is_absolute()
        or str(snapshot_path.resolve()) != snapshot_path_value
        or not isinstance(snapshot_sha256, str)
        or len(snapshot_sha256) != 64
        or not snapshot_path.is_file()
        or _sha256(snapshot_path) != snapshot_sha256
    ):
        raise ValueError("BC-2 gate snapshots are missing or changed")
    metric_bindings = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": gate["checkpoint_sha256"],
        "action_key": gate["action_key"],
        "action_contract_sha256": gate["action_contract_sha256"],
        "evaluator_config_sha256": gate["evaluator_config_sha256"],
        "snapshot_output_path": str(snapshot_path),
        "snapshot_output_sha256": snapshot_sha256,
    }
    for key, expected in metric_bindings.items():
        if rollout_metrics.get(key) != expected:
            raise ValueError(f"BC-2 gate rollout metrics changed binding {key!r}")
    checks = gate.get("checks")
    criteria = gate.get("criteria")
    measured = gate.get("measured")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
        or not isinstance(criteria, Mapping)
        or not isinstance(measured, Mapping)
    ):
        raise ValueError("BC-2 gate decision fields are invalid")
    if (
        float(measured.get("bimanual_grasp_lift_success_rate", -1.0))
        != float(rollout_metrics.get("bimanual_grasp_lift_success_rate", -2.0))
        or float(measured.get("bad_orientation_rate", -1.0))
        != float(rollout_metrics.get("bad_orientation_rate", -2.0))
        or int(measured.get("rollout_episodes", -1))
        != int(rollout_metrics.get("episodes", -2))
    ):
        raise ValueError(
            "BC-2 gate measured values disagree with bound rollout metrics"
        )
    validation = metadata.get("metrics", {}).get("validation", {})
    if not math.isclose(
        float(measured.get("validation_normalized_mse", math.inf)),
        float(validation.get("normalized_mse", -math.inf)),
        abs_tol=1.0e-12,
    ):
        raise ValueError("BC-2 gate validation metric disagrees with its checkpoint")
    config = rollout_metrics.get("evaluator_config", {})
    required_hold_steps = int(
        math.ceil(
            float(rollout_metrics.get("strict_hold_seconds", 0.0))
            * float(config.get("control_frequency_hz", 0.0))
            - 1.0e-9
        )
    )
    if (
        measured.get("strict_episode_audit_passed") is not True
        or not math.isclose(
            float(measured.get("strict_hold_seconds", 0.0)),
            float(rollout_metrics.get("strict_hold_seconds", -1.0)),
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(measured.get("strict_lift_m", 0.0)),
            float(rollout_metrics.get("strict_lift_m", -1.0)),
            abs_tol=1.0e-12,
        )
        or float(measured.get("strict_hold_seconds", 0.0)) < 2.0
        or float(measured.get("strict_lift_m", 0.0)) < 0.12
        or int(measured.get("required_hold_steps", -1)) != required_hold_steps
    ):
        raise ValueError("BC-2 gate strict episode audit fields are invalid")
    snapshot_audit = _audit_bc2_snapshot_store(
        snapshot_path,
        rollout_metrics=rollout_metrics,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=str(gate["checkpoint_sha256"]),
        action_key=str(gate["action_key"]),
        action_contract_sha256=str(gate["action_contract_sha256"]),
        required_hold_steps=required_hold_steps,
        strict_hold_seconds=float(measured["strict_hold_seconds"]),
        strict_lift_m=float(measured["strict_lift_m"]),
        min_bimanual_lift_success_rate=float(
            criteria["min_bimanual_lift_success_rate"]
        ),
    )
    if gate.get("snapshot_audit") != snapshot_audit:
        raise ValueError(
            "BC-2 gate snapshot audit does not match the bound snapshot store"
        )
    if (
        float(measured["bimanual_grasp_lift_success_rate"])
        < float(criteria["min_bimanual_lift_success_rate"])
        or float(measured["bad_orientation_rate"])
        > float(criteria["max_bad_orientation_rate"])
        or int(measured["rollout_episodes"]) < int(criteria["min_rollout_episodes"])
        or float(measured["validation_normalized_mse"])
        > float(criteria["max_validation_normalized_mse"])
    ):
        raise ValueError("BC-2 gate no longer satisfies its recorded criteria")
    if (
        required_action_key is not None
        and metadata["action_key"] != required_action_key
    ):
        raise ValueError(
            f"BC-2 gate action key {metadata['action_key']!r} does not match {required_action_key!r}"
        )
    return gate


def write_metrics_json(
    metrics: Mapping[str, Any], path: str | os.PathLike[str]
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(dict(metrics), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return output
