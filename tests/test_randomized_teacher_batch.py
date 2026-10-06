from __future__ import annotations

import h5py
import numpy as np

from scripts.teacher.generate_randomized_box_teacher import (
    _snapshot_path,
    audit_success_only_shard,
)


def _make_shard(path, run_length: int) -> None:
    with h5py.File(path, "w") as dataset:
        dataset.attrs["complete"] = True
        metadata = dataset.create_group("metadata")
        metadata.create_dataset(
            "processed_joint_names",
            data=np.asarray([f"joint_{index}" for index in range(43)], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        data = dataset.create_group("data")
        episode = data.create_group("demo_0")
        episode.attrs["num_samples"] = 120
        episode.attrs["success"] = run_length >= 100
        episode.attrs["metadata_json"] = (
            '{"initial_box_pose_wxyz":[0.62,0.0,0.84,1.0,0.0,0.0,0.0]}'
        )
        lift = np.zeros(120, dtype=bool)
        left = np.zeros(120, dtype=bool)
        right = np.zeros(120, dtype=bool)
        lift[10 : 10 + run_length] = True
        left[10 : 10 + run_length] = True
        right[10 : 10 + run_length] = True
        episode.create_dataset("lift", data=lift)
        episode.create_dataset("left_contact", data=left)
        episode.create_dataset("right_contact", data=right)
        episode.create_dataset("drop", data=np.zeros(120, dtype=bool))
        root_pose = np.zeros((120, 7), dtype=np.float32)
        root_pose[:, 3] = 1.0
        episode.create_dataset("root_pose", data=root_pose)
        episode.create_dataset("box_size", data=np.full((120, 3), 0.2, np.float32))
        episode.create_dataset(
            "student_observation", data=np.zeros((120, 517), np.float32)
        )
        high_level = np.zeros((120, 23), np.float32)
        high_level[10:, :2] = 1.0
        episode.create_dataset("teacher_high_level_action", data=high_level)


def test_shard_audit_accepts_only_continuous_strict_hold(tmp_path):
    valid_path = tmp_path / "valid.hdf5"
    invalid_path = tmp_path / "invalid.hdf5"
    _make_shard(valid_path, 100)
    _make_shard(invalid_path, 99)

    valid = audit_success_only_shard(valid_path, required_hold_steps=100)
    invalid = audit_success_only_shard(invalid_path, required_hold_steps=100)
    assert valid.valid
    assert valid.episodes == 1
    assert valid.total_steps == 120
    assert invalid.invalid_episodes == ("demo_0",)
    assert not invalid.valid


def test_snapshot_companion_name_is_not_a_demo_shard(tmp_path):
    shard = tmp_path / "shard_00000.hdf5"
    assert _snapshot_path(shard).name == "shard_00000.snapshots.hdf5"
    shard.touch()
    _snapshot_path(shard).touch()
    demo_shards = [
        path
        for path in tmp_path.glob("shard_*.hdf5")
        if not path.name.endswith(".snapshots.hdf5")
    ]
    assert demo_shards == [shard]
