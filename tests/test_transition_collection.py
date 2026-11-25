from dataclasses import replace

import h5py
import numpy as np
import pytest

from unitree_rl_lab.utils.teacher.physical_snapshots import (
    PhysicalSnapshot,
    PhysicalSnapshotStore,
    ROLLOUT_PROVENANCE,
    SnapshotGateError,
    SnapshotKind,
    load_snapshots,
)


@pytest.fixture
def snapshot():
    return PhysicalSnapshot(
        kind=SnapshotKind.LOADED_STAND_END,
        robot_root_link_state_w=np.r_[np.zeros(3), 1.0, np.zeros(9)],
        robot_joint_position=np.zeros(43), robot_joint_velocity=np.zeros(43),
        box_root_link_state_w=np.r_[np.zeros(3), 1.0, np.zeros(9)],
        last_action=np.full(43, 0.2), student_observation=np.zeros(517),
        task_phase=1, source_episode="live_episode_0", source_step=350,
        source_env_id=0, provenance=ROLLOUT_PROVENANCE,
        left_contact_count=2, right_contact_count=2, lift_height_m=0.13,
        stable_hold_steps=350, at_staging=False, bad_orientation=False,
        box_dropped=False, box_size=np.full(3, 0.2), box_mass_kg=0.2,
    )


def store_at(path):
    return PhysicalSnapshotStore(path, joint_names=[f"joint_{i}" for i in range(43)])


def test_checkpoint_readable_during_collection_and_finalize(tmp_path, snapshot):
    path = tmp_path / "train.hdf5"
    store = store_at(path)
    try:
        store.append(snapshot)
        store.checkpoint()
        loaded = load_snapshots(path, kind=SnapshotKind.LOADED_STAND_END)
        assert len(loaded) == 1
        np.testing.assert_array_equal(loaded[0].last_action, snapshot.last_action.astype(np.float32))
        with h5py.File(path) as saved:
            assert saved.attrs["collection_status"] == "in_progress"
        store.append(replace(snapshot, source_episode="live_episode_1"))
        store.checkpoint()
        assert len(load_snapshots(path)) == 2
        store.finalize()
        with h5py.File(path) as saved:
            assert saved.attrs["collection_status"] == "complete"
            assert saved.attrs["num_snapshots"] == 2
    finally:
        store.abort()
    assert not list(tmp_path.glob("*.partial"))


def test_abort_keeps_last_checkpoint_after_invalid_state(tmp_path, snapshot):
    path = tmp_path / "train.hdf5"
    store = store_at(path)
    store.append(snapshot)
    store.checkpoint()
    with pytest.raises(SnapshotGateError):
        store.append(replace(snapshot, stable_hold_steps=12))
    store.abort()
    assert len(load_snapshots(path)) == 1
    assert not store.partial_path.exists()


def test_checkpoint_write_failure_keeps_previous_file(tmp_path, snapshot, monkeypatch):
    path = tmp_path / "train.hdf5"
    store = store_at(path)
    store.append(snapshot)
    store.checkpoint()
    store.append(replace(snapshot, source_episode="live_episode_1"))

    def fail_replace(*args):
        raise OSError("disk failure")

    monkeypatch.setattr("unitree_rl_lab.utils.teacher.physical_snapshots.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        store.checkpoint()
    assert store.checkpoint_count == 1
    store.abort()
    assert len(load_snapshots(path)) == 1
    assert not list(tmp_path.glob("*.partial"))


def test_keyboard_interrupt_keeps_collected_snapshot(tmp_path, snapshot):
    path = tmp_path / "train.hdf5"
    with pytest.raises(KeyboardInterrupt):
        with store_at(path) as store:
            store.append(snapshot)
            store.checkpoint()
            raise KeyboardInterrupt
    assert len(load_snapshots(path)) == 1
    assert not store.partial_path.exists()


def test_checkpoint_does_not_overwrite_foreign_output(tmp_path, snapshot):
    path = tmp_path / "train.hdf5"
    store = store_at(path)
    store.append(snapshot)
    path.write_bytes(b"another collector")
    with pytest.raises(FileExistsError):
        store.checkpoint()
    store.abort()
    assert path.read_bytes() == b"another collector"


@pytest.mark.parametrize("changes", [
    {"stable_hold_steps": 99}, {"left_contact_count": 0}, {"lift_height_m": 0.11},
])
def test_physical_gate_failures_are_skippable(snapshot, changes):
    with pytest.raises(SnapshotGateError):
        replace(snapshot, **changes).validate()


def test_malformed_snapshot_is_not_a_skippable_gate_failure(snapshot):
    with pytest.raises(ValueError) as error:
        replace(snapshot, last_action=np.zeros(3)).validate()
    assert not isinstance(error.value, SnapshotGateError)


def test_empty_collection_does_not_publish(tmp_path):
    path = tmp_path / "train.hdf5"
    store = store_at(path)
    with pytest.raises(RuntimeError, match="empty"):
        store.checkpoint()
    store.abort()
    assert not path.exists()
