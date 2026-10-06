from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from unitree_rl_lab.utils.teacher_demo_recorder import (
    DEFAULT_HIGH_LEVEL_ACTION_NAMES,
    PROCESSED_JOINT_TARGET_DIM,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TeacherDemoRecorder,
    TeacherDemoStep,
)


JOINT_NAMES = tuple(f"joint_{index:02d}" for index in range(PROCESSED_JOINT_TARGET_DIM))


def _step(index: int, phase_id: int, *, success: bool = False) -> TeacherDemoStep:
    return TeacherDemoStep(
        student_observation=np.arange(12, dtype=np.float32) + index,
        teacher_high_level_action=np.arange(23, dtype=np.float32) + index,
        processed_43d_joint_target=np.arange(43, dtype=np.float32) / 10.0 + index,
        phase_id=phase_id,
        box_pose=np.array([0.5, 0.1, 0.8, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        bin_pose=np.array([1.5, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        root_pose=np.array([0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        box_size=np.array([0.2, 0.18, 0.16], dtype=np.float32),
        left_contact_count=2 if index >= 1 else 0,
        right_contact_count=3 if index >= 1 else 0,
        lift=index >= 2,
        drop=False,
        success=success,
    )


def test_streaming_schema_metadata_and_atomic_finalize(tmp_path):
    output_path = tmp_path / "teacher_demo.hdf5"

    with TeacherDemoRecorder(
        output_path,
        control_frequency_hz=50.0,
        joint_names=JOINT_NAMES,
        metadata={"task": "nominal_box_lift", "seed": np.int64(7)},
        chunk_steps=2,
    ) as recorder:
        partial_path = recorder.partial_path
        assert partial_path.exists()
        assert not output_path.exists()

        assert recorder.start_episode(metadata={"box_fixed": True}) == "demo_0"
        for index, phase_id in enumerate((0, 0, 1, 2, 2, 3)):
            assert recorder.append(_step(index, phase_id, success=index == 5)) == index
        recorder.end_episode(termination_reason="task_success")

    assert output_path.exists()
    assert not partial_path.exists()

    with h5py.File(output_path, "r") as h5_file:
        assert h5_file.attrs["schema_name"] == SCHEMA_NAME
        assert h5_file.attrs["schema_version"] == SCHEMA_VERSION
        assert bool(h5_file.attrs["complete"])
        assert h5_file.attrs["control_frequency_hz"] == pytest.approx(50.0)
        assert h5_file.attrs["control_dt_s"] == pytest.approx(0.02)
        assert h5_file.attrs["quaternion_convention"] == "wxyz"
        assert h5_file.attrs["high_level_wrist_quaternion_convention"] == "xyzw"
        assert json.loads(h5_file.attrs["metadata_json"])["seed"] == 7

        np.testing.assert_array_equal(
            h5_file["metadata/teacher_high_level_action_names"].asstr()[:],
            DEFAULT_HIGH_LEVEL_ACTION_NAMES,
        )
        np.testing.assert_array_equal(
            h5_file["metadata/processed_joint_names"].asstr()[:], JOINT_NAMES
        )

        assert h5_file["data"].attrs["total"] == 6
        assert h5_file["data"].attrs["num_episodes"] == 1
        episode = h5_file["data/demo_0"]
        assert episode.attrs["num_samples"] == 6
        assert bool(episode.attrs["complete"])
        assert bool(episode.attrs["success"])
        assert episode.attrs["termination_reason"] == "task_success"

        expected_fields = {
            "student_observation",
            "teacher_high_level_action",
            "processed_43d_joint_target",
            "phase_id",
            "box_pose",
            "bin_pose",
            "root_pose",
            "left_wrist_pose",
            "right_wrist_pose",
            "left_fingertip_positions",
            "right_fingertip_positions",
            "box_size",
            "left_contact",
            "right_contact",
            "left_contact_count",
            "right_contact_count",
            "lift",
            "drop",
            "success",
            "step_index",
            "time_s",
        }
        assert set(episode) == expected_fields
        assert episode["student_observation"].shape == (6, 12)
        assert episode["teacher_high_level_action"].shape == (6, 23)
        assert episode["processed_43d_joint_target"].shape == (6, 43)
        assert episode["box_pose"].shape == (6, 7)
        assert episode["left_wrist_pose"].shape == (6, 7)
        assert episode["right_wrist_pose"].shape == (6, 7)
        assert episode["left_fingertip_positions"].shape == (6, 3, 3)
        assert episode["right_fingertip_positions"].shape == (6, 3, 3)
        assert episode["box_size"].shape == (6, 3)
        assert episode["phase_id"].maxshape == (None,)
        assert episode["student_observation"].maxshape == (None, 12)
        assert episode["phase_id"].chunks == (2,)
        assert episode["student_observation"].attrs["alignment"] == "pre_step_t"
        assert episode["box_pose"].attrs["alignment"] == "post_step_t+1"
        np.testing.assert_array_equal(episode["step_index"][:], np.arange(6))
        np.testing.assert_allclose(episode["time_s"][:], np.arange(6) / 50.0)
        np.testing.assert_array_equal(
            episode["left_contact"][:], [False, True, True, True, True, True]
        )
        np.testing.assert_array_equal(
            episode["right_contact_count"][:], [0, 3, 3, 3, 3, 3]
        )
        assert np.isnan(episode["left_wrist_pose"][:]).all()
        assert np.isnan(episode["right_wrist_pose"][:]).all()
        assert np.isnan(episode["left_fingertip_positions"][:]).all()
        assert np.isnan(episode["right_fingertip_positions"][:]).all()
        assert episode["left_wrist_pose"].attrs["missing_value"] == "NaN"

        boundaries = json.loads(episode.attrs["phase_boundaries_json"])
        assert boundaries == [
            {
                "end_step_exclusive": 2,
                "phase_id": 0,
                "phase_name": "approach",
                "start_step": 0,
            },
            {
                "end_step_exclusive": 3,
                "phase_id": 1,
                "phase_name": "grasp",
                "start_step": 2,
            },
            {
                "end_step_exclusive": 5,
                "phase_id": 2,
                "phase_name": "carry",
                "start_step": 3,
            },
            {
                "end_step_exclusive": 6,
                "phase_id": 3,
                "phase_name": "place",
                "start_step": 5,
            },
        ]


def test_multiple_episodes_and_explicit_contact_flags(tmp_path):
    output_path = tmp_path / "two_episodes.hdf5"
    recorder = TeacherDemoRecorder(
        output_path,
        control_frequency_hz=25.0,
        joint_names=JOINT_NAMES,
        compression=None,
    )
    recorder.start_episode("demo_custom")
    step = _step(0, 1)
    step.left_contact_count = 4
    step.left_contact = False  # A force threshold may reject geometrical contacts.
    recorder.append(step)
    recorder.end_episode(success=False)
    recorder.start_episode()
    recorder.append(_step(0, 0, success=True))
    recorder.end_episode()
    recorder.finalize()

    with h5py.File(output_path, "r") as h5_file:
        assert set(h5_file["data"]) == {"demo_custom", "demo_1"}
        assert h5_file["data"].attrs["total"] == 2
        assert not bool(h5_file["data/demo_custom/left_contact"][0])
        assert h5_file["data/demo_custom/left_contact_count"][0] == 4
        assert not bool(h5_file["data/demo_custom"].attrs["success"])
        assert bool(h5_file["data/demo_1"].attrs["success"])


def test_discard_episode_removes_failed_attempt_and_reuses_demo_index(tmp_path):
    output_path = tmp_path / "success_only.hdf5"
    recorder = TeacherDemoRecorder(
        output_path,
        control_frequency_hz=50.0,
        joint_names=JOINT_NAMES,
        chunk_steps=1,
    )
    recorder.start_episode(metadata={"attempt": 0})
    recorder.append(_step(0, 0))  # flushes a real HDF5 chunk
    assert recorder.discard_episode(reason="no_bilateral_hold") == "demo_0"

    # Failed attempts do not consume demo indices or affect totals.
    assert recorder.start_episode(metadata={"attempt": 1}) == "demo_0"
    recorder.append(_step(0, 1, success=True))
    recorder.end_episode(success=True)
    recorder.finalize()

    with h5py.File(output_path, "r") as h5_file:
        assert set(h5_file["data"]) == {"demo_0"}
        assert h5_file["data"].attrs["num_episodes"] == 1
        assert h5_file["data"].attrs["total"] == 1
        assert bool(h5_file["data/demo_0"].attrs["success"])


def test_validation_error_aborts_context_without_publishing(tmp_path):
    output_path = tmp_path / "invalid.hdf5"

    with pytest.raises(ValueError, match="teacher_high_level_action has shape"):
        with TeacherDemoRecorder(
            output_path,
            control_frequency_hz=50.0,
            joint_names=JOINT_NAMES,
        ) as recorder:
            partial_path = recorder.partial_path
            recorder.start_episode()
            invalid_step = _step(0, 0)
            invalid_step.teacher_high_level_action = np.zeros(22, dtype=np.float32)
            recorder.append(invalid_step)

    assert not output_path.exists()
    assert not partial_path.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("box_size", [0.2, 0.0, 0.1], "box_size entries"),
        ("phase_id", 1.5, "phase_id must be a scalar integer"),
        ("left_contact_count", -1, "left_contact_count must be between"),
        ("student_observation", [1.0, np.nan], "student_observation contains"),
    ],
)
def test_step_value_validation(tmp_path, field, value, message):
    output_path = tmp_path / f"invalid_{field}.hdf5"
    recorder = TeacherDemoRecorder(
        output_path,
        control_frequency_hz=50.0,
        joint_names=JOINT_NAMES,
    )
    recorder.start_episode()
    step = _step(0, 0)
    setattr(step, field, value)
    with pytest.raises(ValueError, match=message):
        recorder.append(step)
    recorder.abort()


def test_existing_destination_is_never_overwritten_by_default(tmp_path):
    output_path = tmp_path / "existing.hdf5"
    output_path.write_bytes(b"keep me")

    with pytest.raises(FileExistsError):
        TeacherDemoRecorder(
            output_path,
            control_frequency_hz=50.0,
            joint_names=JOINT_NAMES,
        )

    assert output_path.read_bytes() == b"keep me"


def test_append_snapshots_mutable_input_buffers(tmp_path):
    output_path = tmp_path / "snapshot.hdf5"
    recorder = TeacherDemoRecorder(
        output_path,
        control_frequency_hz=50.0,
        joint_names=JOINT_NAMES,
        chunk_steps=8,
    )
    recorder.start_episode()
    step = _step(0, 0)
    expected_observation = step.student_observation.copy()
    expected_targets = step.processed_43d_joint_target.copy()
    recorder.append(step)
    step.student_observation[:] = -100.0
    step.processed_43d_joint_target[:] = -100.0
    recorder.end_episode()
    recorder.finalize()

    with h5py.File(output_path, "r") as h5_file:
        np.testing.assert_array_equal(
            h5_file["data/demo_0/student_observation"][0], expected_observation
        )
        np.testing.assert_array_equal(
            h5_file["data/demo_0/processed_43d_joint_target"][0], expected_targets
        )


def test_optional_wrist_and_fingertip_values(tmp_path):
    output_path = tmp_path / "kinematics.hdf5"
    left_wrist_pose = np.array([0.4, 0.2, 0.9, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    right_wrist_pose = np.array([0.4, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    left_tips = np.arange(9, dtype=np.float32).reshape(3, 3) / 10.0
    right_tips = -left_tips

    recorder = TeacherDemoRecorder(
        output_path,
        control_frequency_hz=50.0,
        joint_names=JOINT_NAMES,
    )
    recorder.start_episode()
    step = _step(0, 1)
    step.left_wrist_pose = left_wrist_pose
    step.right_wrist_pose = right_wrist_pose
    step.left_fingertip_positions = left_tips
    step.right_fingertip_positions = right_tips
    recorder.append(step)
    recorder.end_episode()
    recorder.finalize()

    with h5py.File(output_path, "r") as h5_file:
        episode = h5_file["data/demo_0"]
        np.testing.assert_array_equal(episode["left_wrist_pose"][0], left_wrist_pose)
        np.testing.assert_array_equal(episode["right_wrist_pose"][0], right_wrist_pose)
        np.testing.assert_array_equal(episode["left_fingertip_positions"][0], left_tips)
        np.testing.assert_array_equal(
            episode["right_fingertip_positions"][0], right_tips
        )


@pytest.mark.parametrize(
    ("field", "value", "expected_shape"),
    [
        ("left_wrist_pose", np.zeros(6, dtype=np.float32), "\\(7,\\)"),
        ("right_wrist_pose", np.zeros(8, dtype=np.float32), "\\(7,\\)"),
        ("left_fingertip_positions", np.zeros((2, 3), dtype=np.float32), "\\(3, 3\\)"),
        ("right_fingertip_positions", np.zeros(9, dtype=np.float32), "\\(3, 3\\)"),
    ],
)
def test_optional_kinematic_shape_validation(tmp_path, field, value, expected_shape):
    output_path = tmp_path / f"invalid_{field}.hdf5"
    recorder = TeacherDemoRecorder(
        output_path,
        control_frequency_hz=50.0,
        joint_names=JOINT_NAMES,
    )
    recorder.start_episode()
    step = _step(0, 1)
    setattr(step, field, value)
    with pytest.raises(ValueError, match=expected_shape):
        recorder.append(step)
    recorder.abort()
