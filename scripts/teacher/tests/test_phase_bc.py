from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "source" / "unitree_rl_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from unitree_rl_lab.utils.teacher.phase_bc import (  # noqa: E402
    EpisodeSplit,
    PhaseInferenceConfig,
    TrainConfig,
    certify_bc2_gate,
    discover_dagger_episodes,
    discover_episodes,
    expand_teacher_dataset_paths,
    infer_bc_skill_labels,
    load_bc_policy,
    load_dagger_phase_arrays,
    load_phase_arrays,
    merge_phase_arrays,
    split_episodes,
    train_phase_policy,
    validate_bc2_gate,
    validate_complete_teacher_corpus,
    file_sha256,
)
from unitree_rl_lab.utils.teacher.physical_snapshots import (  # noqa: E402
    PhysicalSnapshot,
    PhysicalSnapshotStore,
    ROLLOUT_PROVENANCE,
    SnapshotKind,
)
from unitree_rl_lab.utils.teacher.skill_fsm import SkillPhase  # noqa: E402
from unitree_rl_lab.utils.teacher.student_observation import STUDENT_OBSERVATION_DIM  # noqa: E402


def _write_teacher_fixture(
    path: Path, *, num_episodes: int = 8, observation_dim: int = 517
) -> None:
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as output:
        metadata = output.create_group("metadata")
        metadata.create_dataset(
            "processed_joint_names",
            data=np.asarray([f"joint_{index}" for index in range(43)], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        data = output.create_group("data")
        for episode_index in range(num_episodes):
            group = data.create_group(f"demo_{episode_index}")
            group.attrs["complete"] = True
            group.attrs["success"] = True
            steps = 18
            observations = rng.normal(size=(steps, observation_dim)).astype(np.float32)
            group.create_dataset("student_observation", data=observations)
            high_level = np.zeros((steps, 23), dtype=np.float32)
            high_level[4:, :2] = 1.0
            high_level[:, 2:] = rng.normal(scale=0.1, size=(steps, 21))
            group.create_dataset("teacher_high_level_action", data=high_level)
            # A learnable low-amplitude target; exact convergence is not part of this unit test.
            processed = np.zeros((steps, 43), dtype=np.float32)
            common = min(43, observation_dim)
            processed[:, :common] = 0.1 * observations[:, :common]
            group.create_dataset("processed_43d_joint_target", data=processed)
            group.create_dataset(
                "phase_id",
                data=np.asarray([0] * 8 + [1] * 4 + [2] * 4 + [3] * 2, dtype=np.int16),
            )
            left_contact = np.zeros(steps, dtype=bool)
            right_contact = np.zeros(steps, dtype=bool)
            lift = np.zeros(steps, dtype=bool)
            left_contact[5:] = True
            right_contact[6:] = True
            lift[7:] = True
            group.create_dataset("left_contact", data=left_contact)
            group.create_dataset("right_contact", data=right_contact)
            group.create_dataset("lift", data=lift)
            # Deliberately huge privileged values catch accidental concatenation.
            group.create_dataset(
                "box_pose", data=np.full((steps, 7), 1.0e6, dtype=np.float32)
            )


def _write_bc2_snapshot_fixture(
    path: Path,
    *,
    checkpoint: Path,
    action_key: str,
    action_contract_sha256: str,
    successful_episode_ids: tuple[int, ...],
    strict_hold_seconds: float = 2.0,
    strict_lift_m: float = 0.12,
    control_frequency_hz: float = 50.0,
    overwrite: bool = False,
) -> None:
    required_hold_steps = int(
        np.ceil(strict_hold_seconds * control_frequency_hz - 1.0e-9)
    )
    with PhysicalSnapshotStore(
        path,
        joint_names=[f"joint_{index}" for index in range(43)],
        overwrite=overwrite,
        metadata={
            "producer": "evaluate_bc2_rollout",
            "source": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "action_key": action_key,
            "action_contract_sha256": action_contract_sha256,
            "strict_lift_m": strict_lift_m,
            "strict_hold_seconds": strict_hold_seconds,
            "required_hold_steps": required_hold_steps,
            "gate": "BC-2 strict bilateral 12cm lift held continuously for 2s",
            "box_teleport_into_hands": False,
        },
    ) as store:
        for episode in successful_episode_ids:
            store.append(
                PhysicalSnapshot(
                    kind=SnapshotKind.BILATERAL_HOLD,
                    robot_root_link_state_w=np.r_[
                        np.zeros(3), 1.0, np.zeros(9)
                    ].astype(np.float32),
                    robot_joint_position=np.zeros(43, dtype=np.float32),
                    robot_joint_velocity=np.zeros(43, dtype=np.float32),
                    box_root_link_state_w=np.r_[
                        np.zeros(3), 1.0, np.zeros(9)
                    ].astype(np.float32),
                    last_action=np.zeros(43, dtype=np.float32),
                    student_observation=np.zeros(
                        STUDENT_OBSERVATION_DIM, dtype=np.float32
                    ),
                    task_phase=0,
                    source_episode=f"bc2_episode_{episode}",
                    source_step=249,
                    source_env_id=0,
                    provenance=ROLLOUT_PROVENANCE,
                    left_contact_count=2,
                    right_contact_count=2,
                    lift_height_m=strict_lift_m,
                    stable_hold_steps=required_hold_steps,
                    at_staging=False,
                    bad_orientation=False,
                    box_dropped=False,
                    box_size=np.full(3, 0.2, dtype=np.float32),
                    box_mass_kg=0.2,
                )
            )


def _phase_arrays(path: Path, skill: SkillPhase):
    episodes = discover_episodes([path])
    split = split_episodes(episodes, ratios=(0.5, 0.25, 0.25), seed=3)
    values = {
        subset: load_phase_arrays(
            episodes,
            split.ids(subset),
            skill=skill,
            action_key="processed_43d_joint_target",
        )
        for subset in ("train", "validation", "test")
    }
    return episodes, split, values


def _write_dagger_fixture(path: Path, *, num_source_episodes: int = 4) -> None:
    path = path.expanduser().resolve()
    collection_run_id = "dagger-fixture-run"
    samples_per_episode = 3
    count = num_source_episodes * samples_per_episode
    source_episode = np.repeat(np.arange(num_source_episodes), samples_per_episode)
    observations = np.repeat(
        source_episode[:, None], STUDENT_OBSERVATION_DIM, axis=1
    ).astype(np.float32)
    observations += 50.0
    with h5py.File(path, "w") as output:
        output.attrs["schema_name"] = "unitree_box_dagger_corrections"
        output.attrs["metadata_json"] = json.dumps(
            {"collection_run_id": collection_run_id}
        )
        output.create_dataset(
            "joint_names",
            data=np.asarray([f"joint_{index}" for index in range(43)], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        output.create_dataset(
            "teacher_high_level_action_names",
            data=np.asarray([f"high_{index}" for index in range(23)], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        samples = output.create_group("samples")
        samples.create_dataset("student_observation", data=observations)
        samples.create_dataset(
            "teacher_high_level_action", data=np.ones((count, 23), dtype=np.float32)
        )
        samples.create_dataset(
            "processed_43d_joint_target",
            data=np.full((count, 43), 0.5, dtype=np.float32),
        )
        samples.create_dataset(
            "skill_phase",
            data=np.full(count, int(SkillPhase.GRASP_LIFT), dtype=np.uint8),
        )
        samples.create_dataset("correction_type", data=np.ones(count, dtype=np.uint16))
        samples.create_dataset("source_episode", data=source_episode.astype(np.int64))
    path.with_suffix(".metrics.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "collection_run_id": collection_run_id,
                "requested_output": str(path),
                "dataset_published": True,
                "published_output": str(path),
                "dataset_sha256": file_sha256(path),
            }
        ),
        encoding="utf-8",
    )


def test_phase_zero_is_split_at_first_close_or_contact(tmp_path: Path) -> None:
    path = tmp_path / "teacher.hdf5"
    _write_teacher_fixture(path, num_episodes=1)
    with h5py.File(path, "r") as source:
        labels = infer_bc_skill_labels(source["data/demo_0"])
        labels_with_lead = infer_bc_skill_labels(
            source["data/demo_0"], config=PhaseInferenceConfig(grasp_lead_steps=2)
        )
    assert labels.tolist() == (
        [int(SkillPhase.APPROACH)] * 4
        + [int(SkillPhase.GRASP_LIFT)] * 4
        + [int(SkillPhase.CARRY)] * 4
        + [int(SkillPhase.PLACE)] * 4
        + [-1] * 2
    )
    assert labels_with_lead[:8].tolist() == [0, 0, 1, 1, 1, 1, 1, 1]


def test_shard_directory_and_manifest_expand_without_snapshot_files(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shard_00000.hdf5"
    snapshot = tmp_path / "shard_00000.snapshots.hdf5"
    _write_teacher_fixture(shard, num_episodes=1)
    snapshot.touch()
    assert expand_teacher_dataset_paths([tmp_path]) == (shard.resolve(),)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "unitree_box_xy_randomized_teacher_shards_v1",
                "shards": [{"path": str(shard)}],
            }
        ),
        encoding="utf-8",
    )
    assert expand_teacher_dataset_paths([manifest]) == (shard.resolve(),)
    assert len(discover_episodes([manifest])) == 1


def test_production_manifest_must_be_complete_before_bc(tmp_path: Path) -> None:
    shard = tmp_path / "shard_00000.hdf5"
    _write_teacher_fixture(shard, num_episodes=1)
    manifest = tmp_path / "manifest.json"
    payload = {
        "schema": "unitree_box_xy_randomized_teacher_shards_v1",
        "complete": False,
        "target_demos": 500,
        "accepted_demos": 25,
        "attempts": 30,
        "aggregate_teacher_success_rate": 25 / 30,
        "gates": {
            "demo_count_reached": False,
            "success_rate_at_least_minimum": True,
            "minimum_success_rate": 0.8,
            "all_demos_strict_bilateral_lift_hold": True,
        },
        "shards": [{"path": str(shard), "accepted": 25}],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_complete_teacher_corpus([tmp_path])

    payload["complete"] = True
    payload["accepted_demos"] = 500
    payload["attempts"] = 600
    payload["aggregate_teacher_success_rate"] = 500 / 600
    payload["gates"]["demo_count_reached"] = True
    payload["shards"][0]["accepted"] = 500
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_complete_teacher_corpus([manifest]) == (manifest.resolve(),)


def test_episode_split_has_no_leakage_and_student_input_stays_517d(
    tmp_path: Path,
) -> None:
    path = tmp_path / "teacher.hdf5"
    _write_teacher_fixture(path)
    episodes, split, values = _phase_arrays(path, SkillPhase.GRASP_LIFT)
    assert len(episodes) == 8
    assert not set(split.train) & set(split.validation)
    assert not set(split.train) & set(split.test)
    assert not set(split.validation) & set(split.test)
    assert set(split.train) | set(split.validation) | set(split.test) == {
        episode.uid for episode in episodes
    }
    assert values["train"].observations.shape[1] == STUDENT_OBSERVATION_DIM
    assert values["train"].actions.shape[1] == 43
    assert values["train"].num_samples == 4 * len(split.train)
    assert np.max(np.abs(values["train"].observations)) < 1.0e6


def test_wrong_student_width_is_rejected_even_when_privileged_pose_exists(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong.hdf5"
    _write_teacher_fixture(path, num_episodes=1, observation_dim=524)
    with pytest.raises(ValueError, match=r"expected \(N, 517\)"):
        discover_episodes([path])


def test_carry_dataset_requires_latest_sha_bound_publication_sidecar(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "carry_teacher.hdf5").resolve()
    metrics_path = path.with_suffix(".metrics.json")
    _write_teacher_fixture(path, num_episodes=1)
    with h5py.File(path, "a") as output:
        output.attrs["metadata_json"] = json.dumps(
            {"teacher_phase": "carry", "collection_run_id": "run-1"}
        )

    with pytest.raises(FileNotFoundError, match="publication metrics"):
        discover_episodes([path])

    metrics = {
        "collection_run_id": "run-1",
        "phase": "carry",
        "successes": 1,
        "dataset_published": False,
        "published_output": None,
        "dataset_sha256": None,
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="was not published"):
        discover_episodes([path])

    metrics.update(
        dataset_published=True,
        published_output=str(path),
        dataset_sha256=file_sha256(path),
    )
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    assert len(discover_episodes([path])) == 1

    metrics["dataset_sha256"] = "0" * 64
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        discover_episodes([path])


def test_dagger_dataset_requires_latest_sha_bound_publication_sidecar(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "dagger.hdf5").resolve()
    metrics_path = path.with_suffix(".metrics.json")
    _write_dagger_fixture(path)
    valid_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    metrics_path.unlink()
    with pytest.raises(FileNotFoundError, match="DAgger publication metrics"):
        discover_dagger_episodes([path])

    unpublished_metrics = dict(valid_metrics)
    unpublished_metrics.update(
        status="aborted",
        dataset_published=False,
        published_output=None,
        dataset_sha256=None,
    )
    metrics_path.write_text(json.dumps(unpublished_metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="was not published"):
        discover_dagger_episodes([path])

    metrics_path.write_text(json.dumps(valid_metrics), encoding="utf-8")
    assert len(discover_dagger_episodes([path])) == 4

    mismatched_run = dict(valid_metrics, collection_run_id="stale-run")
    metrics_path.write_text(json.dumps(mismatched_run), encoding="utf-8")
    with pytest.raises(ValueError, match="run id mismatch"):
        discover_dagger_episodes([path])

    mismatched_path = dict(
        valid_metrics,
        published_output=str((tmp_path / "another.hdf5").resolve()),
    )
    metrics_path.write_text(json.dumps(mismatched_path), encoding="utf-8")
    with pytest.raises(ValueError, match="publication path mismatch"):
        discover_dagger_episodes([path])

    mismatched_sha = dict(valid_metrics, dataset_sha256="0" * 64)
    metrics_path.write_text(json.dumps(mismatched_sha), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        discover_dagger_episodes([path])

    metrics_path.write_text(json.dumps(valid_metrics), encoding="utf-8")
    with h5py.File(path, "a") as output:
        output.attrs["metadata_json"] = json.dumps({})
    with pytest.raises(ValueError, match="no collection_run_id"):
        discover_dagger_episodes([path])


def test_train_checkpoint_runtime_loader_and_bc2_gate(tmp_path: Path) -> None:
    path = tmp_path / "teacher.hdf5"
    checkpoint = tmp_path / "bc2.pt"
    gate_path = tmp_path / "bc2_gate.json"
    rollout_path = tmp_path / "bc2_rollout.json"
    snapshot_path = (tmp_path / "bc2_success.snapshots.hdf5").resolve()
    _write_teacher_fixture(path)
    _, split, values = _phase_arrays(path, SkillPhase.GRASP_LIFT)
    report = train_phase_policy(
        train_arrays=values["train"],
        validation_arrays=values["validation"],
        test_arrays=values["test"],
        split=split,
        output_path=checkpoint,
        config=TrainConfig(
            hidden_dims=(16,), batch_size=8, epochs=2, patience=2, seed=4
        ),
        device="cpu",
        source_paths=[path],
    )
    assert report["skill_phase"] == "grasp_lift"
    assert report["privileged_student_inputs"] == []
    loaded = load_bc_policy(checkpoint)
    prediction = loaded(np.zeros((3, STUDENT_OBSERVATION_DIM), dtype=np.float32))
    assert prediction.shape == (3, 43)
    assert loaded.skill_phase == SkillPhase.GRASP_LIFT
    assert loaded.action_key == "processed_43d_joint_target"
    _write_bc2_snapshot_fixture(
        snapshot_path,
        checkpoint=checkpoint,
        action_key=loaded.action_key,
        action_contract_sha256=loaded.action_contract_sha256,
        successful_episode_ids=tuple(range(16)),
    )

    evaluator_config = {
        "episodes": 20,
        "episode_seconds": 10.0,
        "strict_hold_seconds": 2.0,
        "strict_lift_m": 0.12,
        "control_frequency_hz": 50.0,
    }
    rollout_metrics = {
        "format": "unitree_box_bc2_physical_rollout_v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "action_key": loaded.action_key,
        "action_contract_sha256": loaded.action_contract_sha256,
        "snapshot_output_path": str(snapshot_path),
        "snapshot_output_sha256": file_sha256(snapshot_path),
        "episodes": 20,
        "strict_hold_seconds": 2.0,
        "strict_lift_m": 0.12,
        "evaluator_config": evaluator_config,
        "evaluator_config_sha256": hashlib.sha256(
            json.dumps(
                evaluator_config, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
        "bimanual_grasp_lift_success_rate": 0.85,
        "bad_orientation_rate": 0.0,
        "episode_results": [
            {
                "episode": index,
                "success": index < 17,
                "bad_orientation": False,
                "reason": (
                    "strict_bimanual_lift_hold_success"
                    if index < 17
                    else "horizon"
                ),
                "steps": 250,
                "max_continuous_bimanual_lift_hold_steps": (
                    100 if index < 17 else 0
                ),
            }
            for index in range(20)
        ],
    }
    rollout_path.write_text(json.dumps(rollout_metrics), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="snapshot count does not equal the successful rollout episode count",
    ):
        certify_bc2_gate(
            checkpoint,
            rollout_metrics,
            gate_path,
            max_validation_normalized_mse=1.0e9,
            rollout_metrics_sha256=file_sha256(rollout_path),
            rollout_metrics_path=rollout_path,
        )

    _write_bc2_snapshot_fixture(
        snapshot_path,
        checkpoint=checkpoint,
        action_key=loaded.action_key,
        action_contract_sha256=loaded.action_contract_sha256,
        successful_episode_ids=tuple(range(17)),
        overwrite=True,
    )
    rollout_metrics["snapshot_output_sha256"] = file_sha256(snapshot_path)
    rollout_path.write_text(json.dumps(rollout_metrics), encoding="utf-8")
    gate = certify_bc2_gate(
        checkpoint,
        rollout_metrics,
        gate_path,
        max_validation_normalized_mse=1.0e9,
        rollout_metrics_sha256=file_sha256(rollout_path),
        rollout_metrics_path=rollout_path,
    )
    assert gate["passed"]
    assert validate_bc2_gate(
        gate_path, required_action_key="processed_43d_joint_target"
    )["passed"]
    with gate_path.open("r", encoding="utf-8") as stream:
        stored_gate = json.load(stream)
    assert stored_gate["checkpoint_sha256"]
    assert stored_gate["snapshot_output_path"] == str(snapshot_path)
    assert stored_gate["snapshot_output_sha256"] == file_sha256(snapshot_path)
    assert stored_gate["snapshot_audit"]["num_snapshots"] == 17
    assert stored_gate["snapshot_audit"]["successful_rollout_episodes"] == 17
    assert stored_gate["snapshot_audit"]["minimum_required_snapshots"] == 16

    tampered_gate = dict(stored_gate)
    tampered_gate["snapshot_audit"] = dict(stored_gate["snapshot_audit"])
    tampered_gate["snapshot_audit"]["num_snapshots"] = 16
    gate_path.write_text(json.dumps(tampered_gate), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot audit does not match"):
        validate_bc2_gate(gate_path)
    gate_path.write_text(json.dumps(stored_gate), encoding="utf-8")

    snapshot_path.write_bytes(b"tampered evaluator snapshot fixture")
    with pytest.raises(ValueError, match="snapshots are missing or changed"):
        validate_bc2_gate(gate_path)


def test_loader_rejects_non_student_observation_at_runtime(tmp_path: Path) -> None:
    path = tmp_path / "teacher.hdf5"
    checkpoint = tmp_path / "bc2.pt"
    _write_teacher_fixture(path)
    _, split, values = _phase_arrays(path, SkillPhase.GRASP_LIFT)
    train_phase_policy(
        train_arrays=values["train"],
        validation_arrays=values["validation"],
        test_arrays=values["test"],
        split=EpisodeSplit(
            split.train, split.validation, split.test, split.seed, split.ratios
        ),
        output_path=checkpoint,
        config=TrainConfig(hidden_dims=(8,), batch_size=16, epochs=1, patience=1),
        device="cpu",
        source_paths=[path],
    )
    loaded = load_bc_policy(checkpoint)
    with pytest.raises(ValueError, match="expected"):
        loaded(torch.zeros(1, STUDENT_OBSERVATION_DIM + 7))


def test_dagger_is_grouped_by_source_episode_and_normalizer_uses_combined_train(
    tmp_path: Path,
) -> None:
    teacher_path = tmp_path / "teacher.hdf5"
    dagger_path = tmp_path / "dagger.hdf5"
    checkpoint = tmp_path / "combined_bc2.pt"
    _write_teacher_fixture(teacher_path, num_episodes=6)
    _write_dagger_fixture(dagger_path)
    teacher = discover_episodes([teacher_path])
    dagger = discover_dagger_episodes([dagger_path])
    assert len(dagger) == 4
    split = split_episodes([*teacher, *dagger], ratios=(0.6, 0.2, 0.2), seed=11)
    teacher_uids = {episode.uid for episode in teacher}
    dagger_uids = {episode.uid for episode in dagger}
    values = {}
    for subset in ("train", "validation", "test"):
        selected = set(split.ids(subset))
        nominal = load_phase_arrays(
            teacher,
            selected & teacher_uids,
            skill=SkillPhase.GRASP_LIFT,
            action_key="processed_43d_joint_target",
        )
        corrective = load_dagger_phase_arrays(
            dagger,
            selected & dagger_uids,
            skill=SkillPhase.GRASP_LIFT,
            action_key="processed_43d_joint_target",
        )
        values[subset] = merge_phase_arrays(nominal, corrective)

    # Every source_episode produces one UID and therefore belongs to one split.
    assigned = [
        uid for subset in (split.train, split.validation, split.test) for uid in subset
    ]
    assert len(assigned) == len(set(assigned)) == 10
    train_phase_policy(
        train_arrays=values["train"],
        validation_arrays=values["validation"],
        test_arrays=values["test"],
        split=split,
        output_path=checkpoint,
        config=TrainConfig(hidden_dims=(8,), batch_size=32, epochs=1, patience=1),
        device="cpu",
        source_paths=[teacher_path, dagger_path],
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_mean = values["train"].observations.mean(axis=0)
    np.testing.assert_allclose(
        saved["observation_normalization"]["mean"], expected_mean, atol=1.0e-5
    )
