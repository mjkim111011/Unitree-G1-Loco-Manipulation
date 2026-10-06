from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from unitree_rl_lab.utils.teacher.dagger import (
    BalancedCorrectionSelector,
    CorrectionType,
    CorrectiveState,
    DaggerCorrectionRecorder,
    DaggerSample,
    QrDropoutSchedule,
    RecoveryDisturbanceSchedule,
    classify_corrective_state,
)
from unitree_rl_lab.utils.teacher.physical_snapshots import (
    PhysicalSnapshot,
    PhysicalSnapshotStore,
    ROLLOUT_PROVENANCE,
    SnapshotKind,
    load_snapshots,
    load_snapshot_store_metadata,
)
from unitree_rl_lab.utils.teacher.physical_snapshot_runtime import (
    capture_physical_snapshot,
)
from unitree_rl_lab.utils.teacher.phase_teachers import (
    CarryTeacher,
    PlaceTeacher,
    PlaceTrajectoryCfg,
    RealizedBimanualGrasp,
)
from unitree_rl_lab.utils.teacher.skill_fsm import (
    ConditionalSuccessMetrics,
    SkillPhase,
    SkillRouter,
    SkillSignals,
    bimanual_pregrasp_ready,
    bimanual_wrists_centered,
    box_reached_ground_outside_bin,
)
from unitree_rl_lab.utils.teacher.student_observation import (
    STUDENT_FEATURE_SLICES,
    STUDENT_OBSERVATION_DIM,
    mask_qr_observation,
)


def _signals(num_envs: int, **overrides) -> SkillSignals:
    values = {
        "box_visible": np.zeros(num_envs, dtype=bool),
        "at_pregrasp": np.zeros(num_envs, dtype=bool),
        "left_contact": np.zeros(num_envs, dtype=bool),
        "right_contact": np.zeros(num_envs, dtype=bool),
        "lift_height_m": np.zeros(num_envs, dtype=np.float32),
        "hold_steps": np.zeros(num_envs, dtype=np.int64),
        "at_staging": np.zeros(num_envs, dtype=bool),
        "box_in_bin": np.zeros(num_envs, dtype=bool),
        "released": np.zeros(num_envs, dtype=bool),
        "place_settle_steps": np.zeros(num_envs, dtype=np.int64),
        "box_dropped": np.zeros(num_envs, dtype=bool),
        "bad_orientation": np.zeros(num_envs, dtype=bool),
    }
    for key, value in overrides.items():
        values[key] = np.full(num_envs, value)
    return SkillSignals(**values)


def test_explicit_fsm_cannot_skip_phases_and_routes_separate_policies():
    policies = {
        phase: (
            lambda observation, code=float(phase): np.full(
                (len(observation), 43), code, np.float32
            )
        )
        for phase in (
            SkillPhase.APPROACH,
            SkillPhase.GRASP_LIFT,
            SkillPhase.CARRY,
            SkillPhase.PLACE,
        )
    }
    router = SkillRouter(policies, num_envs=1, action_dim=43, hold_steps_required=10)
    observation = np.zeros((1, STUDENT_OBSERVATION_DIM), dtype=np.float32)

    # All predicates true can advance only one phase per call.
    all_true = _signals(
        1,
        box_visible=True,
        at_pregrasp=True,
        left_contact=True,
        right_contact=True,
        lift_height_m=0.2,
        hold_steps=50,
        at_staging=True,
        box_in_bin=True,
        released=True,
        place_settle_steps=10,
    )
    action, transition = router.step(observation, all_true)
    assert transition.approach_succeeded[0]
    assert router.phase[0] == SkillPhase.GRASP_LIFT
    assert np.all(action == float(SkillPhase.GRASP_LIFT))
    router.step(observation, all_true)
    assert router.phase[0] == SkillPhase.CARRY
    router.step(observation, all_true)
    assert router.phase[0] == SkillPhase.PLACE
    action, _ = router.step(observation, all_true)
    assert router.phase[0] == SkillPhase.COMPLETE
    assert np.all(action == 0.0)


def test_fsm_marks_carry_drop_failed_without_regrasp_or_teleport():
    policies = {
        phase: (lambda observation: np.zeros((len(observation), 43), np.float32))
        for phase in SkillPhase
        if phase != SkillPhase.COMPLETE
    }
    router = SkillRouter(policies, num_envs=1, action_dim=43)
    router.phase[0] = SkillPhase.CARRY
    transition = router.update(_signals(1, box_dropped=True))
    assert transition.failed[0]
    assert router.failed[0]
    assert router.phase[0] == SkillPhase.CARRY


def test_place_release_above_bin_is_not_a_drop_until_ground_outside_bin():
    # ``box_in_bin`` includes z, so a valid release above the aperture can be
    # outside that volume for several frames without being a physical drop.
    assert not box_reached_ground_outside_bin(
        box_height_m=0.86,
        environment_origin_z_m=0.0,
        dropped_object_height_m=0.12,
        box_in_bin=False,
    )
    assert not box_reached_ground_outside_bin(
        box_height_m=0.08,
        environment_origin_z_m=0.0,
        dropped_object_height_m=0.12,
        box_in_bin=True,
    )
    assert box_reached_ground_outside_bin(
        box_height_m=0.08,
        environment_origin_z_m=0.0,
        dropped_object_height_m=0.12,
        box_in_bin=False,
    )

    policies = {
        phase: (lambda observation: np.zeros((len(observation), 43), np.float32))
        for phase in SkillPhase
        if phase != SkillPhase.COMPLETE
    }
    router = SkillRouter(policies, num_envs=1, action_dim=43)
    router.phase[0] = SkillPhase.PLACE
    airborne_release = router.update(
        _signals(1, released=True, box_in_bin=False, box_dropped=False)
    )
    assert not airborne_release.failed[0]
    assert router.phase[0] == SkillPhase.PLACE
    ground_drop = router.update(_signals(1, box_dropped=True))
    assert ground_drop.failed[0]


def test_bimanual_pregrasp_requires_two_sided_close_cage():
    box = np.array([0.6, 0.0, 0.84], np.float32)
    left = np.array([[0.52, 0.11, 0.84], [0.4, 0.3, 0.9]], np.float32)
    right = np.array([[0.52, -0.14, 0.84], [0.4, -0.3, 0.9]], np.float32)
    assert bimanual_pregrasp_ready(box, left, right)[0]
    right += np.array([0.0, -0.2, 0.0], np.float32)
    assert not bimanual_pregrasp_ready(box, left, right)[0]


def test_bc2_handoff_requires_centered_symmetric_wrists():
    box = np.array([0.6, 0.0, 0.84], np.float32)
    assert bimanual_wrists_centered(
        box,
        np.array([0.55, 0.10, 0.84], np.float32),
        np.array([0.55, -0.10, 0.84], np.float32),
    )[0]
    assert not bimanual_wrists_centered(
        box,
        np.array([0.30, 0.10, 0.84], np.float32),
        np.array([0.30, -0.10, 0.84], np.float32),
    )[0]
    assert not bimanual_wrists_centered(
        box,
        np.array([0.55, 0.03, 0.84], np.float32),
        np.array([0.55, -0.25, 0.84], np.float32),
    )[0]


def test_place_requires_a_continuous_settle_window():
    policies = {
        phase: (lambda observation: np.zeros((len(observation), 43), np.float32))
        for phase in SkillPhase
        if phase != SkillPhase.COMPLETE
    }
    router = SkillRouter(
        policies, num_envs=1, action_dim=43, place_settle_steps_required=10
    )
    router.phase[0] = SkillPhase.PLACE
    router.update(
        _signals(1, box_in_bin=True, released=True, place_settle_steps=1)
    )
    assert router.phase[0] == SkillPhase.PLACE
    router.update(
        _signals(1, box_in_bin=True, released=True, place_settle_steps=10)
    )
    assert router.phase[0] == SkillPhase.COMPLETE


def test_conditional_metrics_denominators():
    metrics = ConditionalSuccessMetrics()
    for phase in (
        SkillPhase.APPROACH,
        SkillPhase.CARRY,
        SkillPhase.PLACE,
        SkillPhase.COMPLETE,
    ):
        metrics.add_episode(phase)
    result = metrics.as_dict()
    assert result["P(approach)"] == pytest.approx(0.75)
    assert result["P(grasp|approach)"] == pytest.approx(1.0)
    assert result["P(carry|grasp)"] == pytest.approx(2 / 3)
    assert result["P(place|carry)"] == pytest.approx(0.5)
    assert result["full_task_success"] == pytest.approx(0.25)


def _snapshot(kind: SnapshotKind = SnapshotKind.BILATERAL_HOLD) -> PhysicalSnapshot:
    return PhysicalSnapshot(
        kind=kind,
        robot_root_link_state_w=np.r_[np.zeros(3), 1.0, np.zeros(9)].astype(np.float32),
        robot_joint_position=np.zeros(43, np.float32),
        robot_joint_velocity=np.zeros(43, np.float32),
        box_root_link_state_w=np.r_[np.zeros(3), 1.0, np.zeros(9)].astype(np.float32),
        last_action=np.zeros(43, np.float32),
        student_observation=np.zeros(STUDENT_OBSERVATION_DIM, np.float32),
        task_phase=1 if kind == SnapshotKind.CARRY_AT_STAGING else 0,
        source_episode="rollout_17",
        source_step=431,
        source_env_id=0,
        provenance=ROLLOUT_PROVENANCE,
        left_contact_count=2,
        right_contact_count=2,
        lift_height_m=0.13,
        stable_hold_steps=100,
        at_staging=kind == SnapshotKind.CARRY_AT_STAGING,
        bad_orientation=False,
        box_dropped=False,
        box_size=np.full(3, 0.2, np.float32),
        box_mass_kg=0.2,
        unprocessed_wbc_raw_action=(
            np.linspace(-0.2, 0.2, 43, dtype=np.float32)
            if kind == SnapshotKind.CARRY_AT_STAGING
            else None
        ),
    )


def test_physical_snapshot_store_and_arbitrary_state_rejection(tmp_path):
    path = tmp_path / "snapshots.hdf5"
    with PhysicalSnapshotStore(
        path,
        joint_names=[f"j{i}" for i in range(43)],
        metadata={"producer": "unit_test", "checkpoint_sha256": "a" * 64},
    ) as store:
        store.append(_snapshot())
        store.append(_snapshot(SnapshotKind.CARRY_AT_STAGING))
    loaded = load_snapshots(path)
    assert [snapshot.kind for snapshot in loaded] == [
        SnapshotKind.BILATERAL_HOLD,
        SnapshotKind.CARRY_AT_STAGING,
    ]
    assert loaded[0].unprocessed_wbc_raw_action is None
    assert loaded[1].unprocessed_wbc_raw_action is not None
    assert len(
        load_snapshots(
            path,
            kind=SnapshotKind.CARRY_AT_STAGING,
            require_unprocessed_wbc_raw_action=True,
        )
    ) == 1
    assert load_snapshot_store_metadata(path) == {
        "producer": "unit_test",
        "checkpoint_sha256": "a" * 64,
    }
    legacy_path = tmp_path / "legacy_snapshots.hdf5"
    legacy_staging = _snapshot(SnapshotKind.CARRY_AT_STAGING)
    object.__setattr__(legacy_staging, "unprocessed_wbc_raw_action", None)
    with PhysicalSnapshotStore(
        legacy_path, joint_names=[f"j{i}" for i in range(43)]
    ) as store:
        store.append(legacy_staging)
    with h5py.File(legacy_path, "r+") as legacy:
        legacy.attrs["schema_version"] = "1.0.0"
    assert len(load_snapshots(legacy_path)) == 1
    with pytest.raises(ValueError, match="unprocessed Pink/WBC"):
        load_snapshots(
            legacy_path,
            kind=SnapshotKind.CARRY_AT_STAGING,
            require_unprocessed_wbc_raw_action=True,
        )
    with pytest.raises(ValueError, match="strict lift and continuous hold"):
        load_snapshots(path, hold_steps_required=101)
    invalid = _snapshot()
    object.__setattr__(invalid, "provenance", "box_teleported_into_hands")
    with pytest.raises(ValueError, match="arbitrary assembled"):
        invalid.validate()


def test_physical_snapshot_store_round_trips_randomized_work_area(tmp_path):
    snapshot = _snapshot()
    object.__setattr__(snapshot, "table_center_e", np.array([1.2, -1.0, 0.74], np.float32))
    object.__setattr__(snapshot, "bin_center_e", np.array([-1.4, 0.8, 0.28], np.float32))
    object.__setattr__(snapshot, "bin_qr_position_e", np.array([-1.786, 0.8, 0.30], np.float32))
    table_poses = np.zeros((2, 7), np.float32)
    table_poses[:, 3] = 1.0
    bin_poses = np.zeros((5, 7), np.float32)
    bin_poses[:, 3] = 1.0
    object.__setattr__(snapshot, "table_asset_root_poses_w", table_poses)
    object.__setattr__(snapshot, "bin_asset_root_poses_w", bin_poses)
    path = tmp_path / "workspace_snapshots.hdf5"
    with PhysicalSnapshotStore(path, joint_names=[f"j{i}" for i in range(43)]) as store:
        store.append(snapshot)
    loaded = load_snapshots(path)[0]
    np.testing.assert_allclose(loaded.table_center_e, snapshot.table_center_e)
    np.testing.assert_allclose(loaded.bin_center_e, snapshot.bin_center_e)
    np.testing.assert_allclose(
        loaded.bin_qr_position_e, snapshot.bin_qr_position_e
    )
    np.testing.assert_allclose(
        loaded.table_asset_root_poses_w, snapshot.table_asset_root_poses_w
    )
    np.testing.assert_allclose(
        loaded.bin_asset_root_poses_w, snapshot.bin_asset_root_poses_w
    )


def test_capture_snapshot_moves_optional_wbc_tensor_to_cpu_numpy() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    robot = SimpleNamespace(
        num_joints=43,
        data=SimpleNamespace(
            root_link_state_w=torch.zeros((1, 13), device=device),
            joint_pos=torch.zeros((1, 43), device=device),
            joint_vel=torch.zeros((1, 43), device=device),
        ),
    )
    box_state = torch.zeros((1, 13), device=device)
    box_state[0, 2] = 0.25
    box = SimpleNamespace(data=SimpleNamespace(root_link_state_w=box_state))
    env = SimpleNamespace(
        num_envs=1,
        scene={"robot": robot, "box": box},
        action_manager=SimpleNamespace(
            action=torch.zeros((1, 43), device=device), total_action_dim=43
        ),
        task_phase=torch.ones(1, dtype=torch.int64, device=device),
    )
    baseline = torch.linspace(-0.3, 0.3, 43, device=device)
    snapshot = capture_physical_snapshot(
        env,
        kind=SnapshotKind.CARRY_AT_STAGING,
        env_id=0,
        source_episode="carry_0",
        source_step=10,
        student_observation=np.zeros(STUDENT_OBSERVATION_DIM, np.float32),
        box_initial_height_m=0.1,
        left_contact_count=1,
        right_contact_count=1,
        stable_hold_steps=100,
        at_staging=True,
        bad_orientation=False,
        box_dropped=False,
        box_size=np.full(3, 0.2, np.float32),
        box_mass_kg=0.2,
        unprocessed_wbc_raw_action=baseline,
    )
    assert isinstance(snapshot.unprocessed_wbc_raw_action, np.ndarray)
    assert np.allclose(
        snapshot.unprocessed_wbc_raw_action,
        baseline.detach().cpu().numpy(),
    )


def test_correction_classifier_and_qr_mask():
    state = CorrectiveState(
        left_contact=True,
        right_contact=False,
        left_wrist_position_w=np.array([0.3, 0.2, 0.8]),
        right_wrist_position_w=np.array([0.3, 0.4, 0.8]),
        box_position_w=np.array([0.5, 0.0, 0.8]),
        box_quaternion_wxyz=np.array([0.9659, 0.0, 0.0, 0.2588]),
        reference_box_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        box_wrist_relative_speed_mps=0.2,
        lift_height_m=0.03,
        qr_visible=False,
        qr_was_visible=True,
    )
    flags = classify_corrective_state(state)
    assert flags & CorrectionType.ONE_HAND_FIRST
    assert flags & CorrectionType.WRIST_OFF_CENTER
    assert flags & CorrectionType.LIFT_SLIP
    assert flags & CorrectionType.QR_TEMPORARILY_MISSING

    rotated_bimanual = CorrectiveState(
        left_contact=True,
        right_contact=True,
        left_wrist_position_w=np.array([0.5, -0.1, 0.8]),
        right_wrist_position_w=np.array([0.5, 0.1, 0.8]),
        box_position_w=np.array([0.5, 0.0, 0.8]),
        box_quaternion_wxyz=np.array([0.9659, 0.0, 0.0, 0.2588]),
        reference_box_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        box_wrist_relative_speed_mps=0.0,
        lift_height_m=0.12,
        qr_visible=True,
        qr_was_visible=True,
    )
    assert classify_corrective_state(rotated_bimanual) & CorrectionType.BOX_ROTATED_IN_HAND

    observation = np.ones(STUDENT_OBSERVATION_DIM, np.float32)
    masked = mask_qr_observation(observation, box=True)
    assert masked[STUDENT_FEATURE_SLICES["box_visible"].as_slice()].sum() == 0.0
    assert masked[STUDENT_FEATURE_SLICES["box_position_camera"].as_slice()].sum() == 0.0
    assert masked[STUDENT_FEATURE_SLICES["bin_visible"].as_slice()].sum() == 1.0


def test_dagger_recorder_rejects_nominal_and_records_corrective_sample(tmp_path):
    path = tmp_path / "dagger.hdf5"
    recorder = DaggerCorrectionRecorder(path, joint_names=[f"j{i}" for i in range(43)])
    sample = DaggerSample(
        student_observation=np.zeros(STUDENT_OBSERVATION_DIM, np.float32),
        student_action=np.zeros(43, np.float32),
        teacher_high_level_action=np.zeros(23, np.float32),
        processed_43d_joint_target=np.zeros(43, np.float32),
        skill_phase=SkillPhase.GRASP_LIFT,
        correction_type=CorrectionType.NONE,
        qr_visible=True,
        source_episode=0,
        source_step=10,
    )
    with pytest.raises(ValueError, match="corrective states only"):
        recorder.append(sample)
    sample.correction_type = (
        CorrectionType.ONE_HAND_FIRST | CorrectionType.WRIST_OFF_CENTER
    )
    recorder.append(sample)
    recorder.finalize()
    with h5py.File(path, "r") as h5_file:
        assert h5_file.attrs["num_samples"] == 1
        assert h5_file["samples/student_observation"].shape == (
            1,
            STUDENT_OBSERVATION_DIM,
        )
        assert int(h5_file["samples/correction_type"][0]) == int(sample.correction_type)


def test_qr_dropout_schedule_changes_observation_only():
    schedule = QrDropoutSchedule(
        probability_per_step=1.0, duration_steps=(2, 2), seed=3
    )
    original = np.ones(STUDENT_OBSERVATION_DIM, np.float32)
    masked, active = schedule.apply(original, phase=SkillPhase.CARRY)
    assert active
    assert original.sum() == STUDENT_OBSERVATION_DIM
    assert masked[STUDENT_FEATURE_SLICES["bin_visible"].as_slice()].sum() == 0.0
    assert masked[STUDENT_FEATURE_SLICES["bin_position_camera"].as_slice()].sum() == 0.0


def test_seeded_dagger_disturbance_campaign_targets_all_physical_states():
    schedule = RecoveryDisturbanceSchedule(seed=0, pulse_steps=3)
    assert [schedule.reset(index) for index in range(4)] == list(schedule.TARGETS)
    schedule.reset(2)
    force, torque = schedule.wrench(
        step=20,
        close_index=10,
        bilateral_contact=True,
        lift_height_m=0.0,
    )
    assert np.allclose(force, 0.0)
    assert torque[2] > 0.0
    schedule.reset(3)
    force, torque = schedule.wrench(
        step=20,
        close_index=10,
        bilateral_contact=True,
        lift_height_m=0.03,
    )
    assert force[2] < 0.0
    assert np.allclose(torque, 0.0)


def test_balanced_dagger_selector_filters_campaign_and_bounds_invisible_samples():
    selector = BalancedCorrectionSelector(
        max_samples_per_episode=10,
        max_invisible_to_visible_ratio=0.5,
        invisible_warmup_samples=1,
    )
    selector.reset(CorrectionType.LIFT_SLIP)
    assert not selector.accept(CorrectionType.WRIST_OFF_CENTER, qr_visible=True)
    assert selector.accept(CorrectionType.VISIBLE_ANCHOR, qr_visible=True)
    assert selector.accept(CorrectionType.LIFT_SLIP, qr_visible=True)
    assert selector.accept(CorrectionType.LIFT_SLIP, qr_visible=False)
    assert not selector.accept(CorrectionType.LIFT_SLIP, qr_visible=False)
    assert selector.total == 3
    assert selector.visible == 2
    assert selector.invisible == 1


def test_carry_and_place_teacher_preserve_realized_bimanual_geometry():
    box = np.array([0.6, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0])
    left = np.array([0.6, 0.15, 0.9, 1.0, 0.0, 0.0, 0.0])
    right = np.array([0.6, -0.15, 0.9, 1.0, 0.0, 0.0, 0.0])
    pelvis = np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])
    grasp = RealizedBimanualGrasp.capture(
        box_pose_wxyz=box,
        left_wrist_pose_wxyz=left,
        right_wrist_pose_wxyz=right,
    )
    carry = CarryTeacher(grasp)
    action = carry.action(
        current_box_pose_wxyz=box,
        current_pelvis_pose_wxyz=pelvis,
        camera_navigation_velocity=np.array([0.2, -0.1, 0.05]),
    )
    np.testing.assert_allclose(action[:2], 1.0)
    np.testing.assert_allclose(action[16:19], [0.2, -0.1, 0.05])
    # In pelvis coordinates, the two target wrists retain their 30 cm spread.
    assert action[3] - action[10] == pytest.approx(0.30)

    cfg = PlaceTrajectoryCfg(
        raise_steps=2, translate_steps=2, lower_steps=2, release_steps=2
    )
    place = PlaceTeacher(
        grasp,
        start_box_position_w=box[:3],
        bin_center_w=np.array([1.5, -1.0, 0.3]),
        cfg=cfg,
    )
    assert place.action(step=0, current_pelvis_pose_wxyz=pelvis)[0] == 1.0
    release_action = place.action(
        step=cfg.total_steps - 1, current_pelvis_pose_wxyz=pelvis
    )
    np.testing.assert_allclose(release_action[:2], 0.0)
    recovery_action = place.action_for_box_position(
        box_position=box[:3],
        release=False,
        current_pelvis_pose_wxyz=pelvis,
        left_recovery_compression_m=0.012,
    )
    assert recovery_action[3] == pytest.approx(0.138)
    assert recovery_action[10] == pytest.approx(-0.15)
    np.testing.assert_allclose(recovery_action[:2], 1.0)
