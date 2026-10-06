from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "source" / "unitree_rl_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from unitree_rl_lab.utils.teacher.arena_dataset import (  # noqa: E402
    ARENA_G1_LAB_JOINT_NAMES,
    extract_dataset,
    infer_task_phases,
    load_episode_trajectory,
    pose7_wxyz_to_matrix,
    quaternion_xyzw_to_matrix,
)


REAL_ARENA_DATASET = Path(
    "/home/mjkim/IsaacLab-Arena/data/arena_g1_loco_manipulation_dataset_generated_small.hdf5"
)


def _make_phase_signals(
    num_steps: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.zeros((num_steps, 23), dtype=np.float32)
    actions[:, 5] = 1.0
    actions[:, 12] = 1.0
    actions[:, 19] = 0.75
    actions[40:80, 16] = 0.3
    box = np.zeros((num_steps, 7), dtype=np.float32)
    box[:, 3] = 1.0
    box[20:, 0] = np.linspace(0.0002, 0.10, num_steps - 20)
    box[25:80, 2] = np.linspace(0.0, 0.12, 55)
    box[80:, :3] = np.array([0.5, 0.0, 0.12])
    bin_pose = np.zeros((num_steps, 7), dtype=np.float32)
    bin_pose[:, 0] = 0.5
    bin_pose[:, 3] = 1.0
    return actions, box, bin_pose


def _write_fixture(path: Path) -> None:
    actions, box, bin_pose = _make_phase_signals()
    num_steps = len(actions)
    identity_pose = np.zeros((num_steps, 7), dtype=np.float32)
    identity_pose[:, 3] = 1.0
    wrist = np.repeat(np.eye(4, dtype=np.float32)[None], num_steps, axis=0)
    wrist[:, 0, 3] = 0.2

    with h5py.File(path, "w") as output:
        data = output.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {"env_name": "fixture", "sim_args": {"dt": 0.005, "decimation": 4}}
        )
        demo = data.create_group("demo_0")
        demo.attrs["success"] = True
        demo.create_dataset("actions", data=actions)
        demo.create_dataset(
            "processed_actions", data=np.zeros((num_steps, 43), dtype=np.float32)
        )
        action = demo.create_group("action")
        for name, value in (
            ("left_eef_pos", actions[:, 2:5]),
            ("left_eef_quat", actions[:, 5:9][:, [3, 0, 1, 2]]),
            ("right_eef_pos", actions[:, 9:12]),
            ("right_eef_quat", actions[:, 12:16][:, [3, 0, 1, 2]]),
            ("navigate_cmd", actions[:, 16:19]),
            ("base_height_cmd", actions[:, 19:20]),
            ("torso_orientation_rpy_cmd", actions[:, 20:23]),
        ):
            action.create_dataset(name, data=value)
        obs = demo.create_group("obs")
        obs.create_dataset("robot_pos", data=identity_pose[:, :3])
        obs.create_dataset("robot_quat", data=identity_pose[:, 3:7])
        obs.create_dataset("left_wrist_pose_pelvis_frame", data=wrist)
        obs.create_dataset("right_wrist_pose_pelvis_frame", data=wrist)
        camera = demo.create_group("camera_obs")
        camera.create_dataset(
            "robot_head_cam_rgb", shape=(num_steps, 1, 1, 3), dtype=np.uint8
        )

        initial = demo.create_group("initial_state")
        initial_robot = initial.create_group("articulation/robot")
        initial_robot.create_dataset("root_pose", data=identity_pose[:1])
        initial_robot.create_dataset(
            "joint_position", data=np.zeros((1, 43), dtype=np.float32)
        )
        initial_robot.create_dataset(
            "joint_velocity", data=np.zeros((1, 43), dtype=np.float32)
        )
        initial_rigid = initial.create_group("rigid_object")
        initial_rigid.create_group("brown_box").create_dataset(
            "root_pose", data=box[:1]
        )
        initial_rigid.create_group("blue_sorting_bin").create_dataset(
            "root_pose", data=bin_pose[:1]
        )

        states = demo.create_group("states")
        robot = states.create_group("articulation/robot")
        robot.create_dataset("root_pose", data=identity_pose)
        robot.create_dataset(
            "root_velocity", data=np.zeros((num_steps, 6), dtype=np.float32)
        )
        robot.create_dataset(
            "joint_position", data=np.zeros((num_steps, 43), dtype=np.float32)
        )
        robot.create_dataset(
            "joint_velocity", data=np.zeros((num_steps, 43), dtype=np.float32)
        )
        rigid = states.create_group("rigid_object")
        box_group = rigid.create_group("brown_box")
        box_group.create_dataset("root_pose", data=box)
        box_group.create_dataset(
            "root_velocity", data=np.zeros((num_steps, 6), dtype=np.float32)
        )
        bin_group = rigid.create_group("blue_sorting_bin")
        bin_group.create_dataset("root_pose", data=bin_pose)
        bin_group.create_dataset(
            "root_velocity", data=np.zeros((num_steps, 6), dtype=np.float32)
        )


def test_quaternion_conventions_are_not_interchanged() -> None:
    # WXYZ identity and XYZW identity must both produce an identity rotation.
    wxyz_pose = np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        pose7_wxyz_to_matrix(wxyz_pose)[0],
        np.eye(4)
        + np.array(
            [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 0]], dtype=float
        ),
    )
    np.testing.assert_allclose(
        quaternion_xyzw_to_matrix(np.array([[0.0, 0.0, 0.0, 1.0]]))[0], np.eye(3)
    )


def test_phase_inference_uses_signals() -> None:
    actions, box, bin_pose = _make_phase_signals()
    phase = infer_task_phases(actions, box, bin_pose, control_dt_s=0.02)
    # Synthetic motion begins at 20, lift+navigation at 40, sustained stop at 80.
    assert phase.boundaries == (0, 20, 40, 80, 100)
    assert phase.intervals["grasp"] == (20, 40)


def test_extraction_keeps_retargeting_data_but_not_rgb(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    output = tmp_path / "derived.hdf5"
    audit = tmp_path / "audit.json"
    _write_fixture(source)
    report = extract_dataset(source, output, audit)

    assert report["episodes"]["demo_0"]["timing"]["control_frequency_hz"] == 50.0
    assert (
        report["episodes"]["demo_0"]["phase_annotations"][
            "explicit_annotations_present"
        ]
        is False
    )
    with h5py.File(output, "r") as result:
        demo = result["data/demo_0"]
        assert demo["teacher/high_level_action_23d"].shape == (100, 23)
        assert demo["teacher/processed_43d_joint_target"].shape == (100, 43)
        assert demo["dex3/both_processed_joint_target"].shape == (100, 14)
        assert demo["transforms/T_box_inv_T_left_wrist_observed"].shape == (100, 4, 4)
        np.testing.assert_allclose(
            demo["transforms/T_box_inv_T_left_wrist_observed"][0, :3, 3],
            [0.2, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            demo["transforms/T_bin_inv_T_box"][0, :3, 3], [-0.5, 0.0, 0.0]
        )
        assert demo["phase/boundaries"][:].tolist() == [0, 20, 40, 80, 100]
        assert (
            len(demo["metadata/source_lab_joint_names"])
            == len(ARENA_G1_LAB_JOINT_NAMES)
            == 43
        )
        all_paths: list[str] = []
        demo.visit(all_paths.append)
        assert not any("camera" in path or "rgb" in path for path in all_paths)

    runtime = load_episode_trajectory(source)
    assert runtime.phase_boundaries == (0, 20, 40, 80, 100)
    assert runtime.left_world_wrist_command.shape == (100, 4, 4)
    assert runtime.left_dex3_joint_target.shape == (100, 7)
    assert runtime.robot_joint_position_pre_step.shape == (100, 43)


@pytest.mark.skipif(
    not REAL_ARENA_DATASET.is_file(), reason="Arena small dataset is not installed"
)
def test_installed_small_dataset_audit_and_boundaries(tmp_path: Path) -> None:
    output = tmp_path / "arena_small_derived.hdf5"
    audit = tmp_path / "arena_small_audit.json"
    report = extract_dataset(
        REAL_ARENA_DATASET, output, audit, episode_names=["demo_0"]
    )
    demo = report["episodes"]["demo_0"]
    assert demo["num_samples"] == 853
    assert demo["timing"]["control_frequency_hz"] == 50.0
    assert demo["phase_annotations"]["explicit_annotations_present"] is False
    assert demo["phase_annotations"]["boundaries"] == [0, 120, 182, 733, 853]
    with h5py.File(output, "r") as result:
        assert "camera_obs" not in result["data/demo_0"]
        assert result["data/demo_0/transforms/T_bin_inv_T_box"].shape == (853, 4, 4)
