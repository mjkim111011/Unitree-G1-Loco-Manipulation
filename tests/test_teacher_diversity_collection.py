from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import pytest
import torch

from scripts.teacher.collect_diverse_box_teacher import (
    audit_diverse_shard, bounds, collector_command, parser, sha256, split_bounds,
)
from unitree_rl_lab.utils.teacher.phase_bc import DIVERSE_TEACHER_SHARD_MANIFEST_SCHEMA, expand_teacher_dataset_paths, validate_complete_teacher_corpus
from tests.test_randomized_teacher_batch import _make_shard

ROOT = Path(__file__).resolve().parents[1]


def test_grid_equal_coverage_and_invalid_bounds():
    pieces = [split_bounds((-0.08, 0.08), i) for i in range(4)]
    assert pieces[0][0] == -0.08 and pieces[-1][1] == 0.08
    assert all(a[1] == b[0] for a, b in zip(pieces, pieces[1:]))
    for invalid in ((float("nan"), 0), (1, -1)):
        with pytest.raises(ValueError):
            bounds(invalid)


def test_constant_heading_uses_imu_and_rotates_standoff_correctly():
    source = ROOT / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/commands/camera_navigation_command.py"
    tree = ast.parse(source.read_text())
    command = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CameraBinNavigationCommand")
    update = next(node for node in command.body if isinstance(node, ast.FunctionDef) and node.name == "_update_command")
    branch = next(node for node in update.body if isinstance(node, ast.If) and "pregrasp_world_heading_rad" in ast.unparse(node.test))
    fake = SimpleNamespace(cfg=SimpleNamespace(pregrasp_world_heading_rad=0, pregrasp_standoff_distance=0.42),
                           robot=SimpleNamespace(data=SimpleNamespace(heading_w=torch.tensor([torch.pi/2]))),
                           pregrasp_position_error_b=torch.ones(1, 2), pregrasp_yaw_error=torch.zeros(1))
    namespace = {"self": fake, "torch": torch, "box_xy_b": torch.tensor([[0, -0.42]]),
                 "math_utils": SimpleNamespace(wrap_to_pi=lambda x: (x+torch.pi) % (2*torch.pi)-torch.pi)}
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(source), "exec"), namespace)
    assert torch.allclose(fake.pregrasp_position_error_b, torch.zeros(1, 2), atol=1e-6)
    assert fake.pregrasp_yaw_error.item() == pytest.approx(-torch.pi/2)


def test_navigation_hysteresis_holds_arrival_without_loosening_entry():
    source = ROOT / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/commands/camera_navigation_command.py"
    tree = ast.parse(source.read_text())
    command = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CameraBinNavigationCommand")
    update = next(node for node in command.body if isinstance(node, ast.FunctionDef) and node.name == "_update_command")
    start = next(i for i, node in enumerate(update.body) if ast.unparse(node).startswith("tolerance_scale ="))
    body = update.body[start:start+4]
    fake = SimpleNamespace(cfg=SimpleNamespace(pregrasp_alignment_exit_multiplier=1.5,
                           pregrasp_yaw_tolerance=0.1, pregrasp_position_tolerance=0.05, pregrasp_lateral_tolerance=0.05),
                           at_pregrasp=torch.tensor([True, False]), pregrasp_yaw_error=torch.zeros(2),
                           pregrasp_position_error_b=torch.tensor([[0.06, 0], [0.06, 0]]))
    namespace = {"self": fake, "torch": torch, "visible_pregrasp": torch.ones(2, dtype=torch.bool)}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    assert fake.at_pregrasp.tolist() == [True, False]


def test_commands_use_distinct_coordinate_frames(tmp_path):
    args = parser().parse_args(["--case", "table_box", "--output_dir", str(tmp_path)])
    cmd, cell = collector_command(args, tmp_path / "a.hdf5", 0, 0, 25)
    assert float(cmd[cmd.index("--box_x_range") + 1]) == pytest.approx(0.54)
    assert "--randomize_work_area" not in cmd
    args.case = "table_approach"
    cmd, cell = collector_command(args, tmp_path / "b.hdf5", 0, 0, 25)
    assert float(cmd[cmd.index("--box_x_range") + 1]) == -0.08
    assert "--randomize_work_area" in cmd
    assert cell[0] == pytest.approx((0.8, 0.975))


def test_navigation_is_required_before_publishing_walking_episode(tmp_path):
    shard = tmp_path / "s.hdf5"
    _make_shard(shard, 100)
    config = {"case": "table_approach", "box_local_range": [[-0.08, 0.08]] * 2}
    with h5py.File(shard, "a") as file:
        file.attrs["metadata_json"] = json.dumps({"randomization_mode": "table_and_box_xy"})
        final = {"initial_box_xy_m": [0.62, 0], "initial_table_xy_m": [0.62, 0],
                 "initial_box_table_relative_xy_m": [0, 0], "navigation_completed": True, "navigation_steps": 101,
                 "initial_root_xy_m": [0, 0], "navigation_end_root_pose_wxyz": [0.2, 0, 0.75, 1, 0, 0, 0]}
        file["data/demo_0"].attrs["final_metadata_json"] = json.dumps(final)
    assert audit_diverse_shard(shard, config, ((0.6, 0.7), (-0.1, 0.1)))[0].valid
    with h5py.File(shard, "a") as file:
        final["navigation_completed"] = False
        file["data/demo_0"].attrs["final_metadata_json"] = json.dumps(final)
    with pytest.raises(ValueError, match="physical approach"):
        audit_diverse_shard(shard, config, ((0.6, 0.7), (-0.1, 0.1)))


def test_diverse_manifest_loader_requires_complete_coverage_and_sha(tmp_path):
    shards = []
    for cell in range(16):
        path = tmp_path / f"shard_{cell:05d}.hdf5"
        path.write_bytes(b"fixture for manifest binding")
        shards.append({"path": str(path), "accepted": 32, "cell": cell, "sha256": sha256(path)})
    manifest = {"schema": DIVERSE_TEACHER_SHARD_MANIFEST_SCHEMA, "complete": True,
                "accepted_demos": 512, "target_demos": 512, "attempts": 512,
                "aggregate_teacher_success_rate": 1, "target_successes_per_cell": [32]*16, "shards": shards,
                "gates": {"minimum_success_rate": 0.8, "demo_count_reached": True,
                          "success_rate_at_least_minimum": True, "all_demos_strict_bilateral_lift_hold": True,
                          "balanced_spatial_coverage_reached": True}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert validate_complete_teacher_corpus([path]) == (path,)
    assert len(expand_teacher_dataset_paths([path])) == 16
    manifest["target_successes_per_cell"][0] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="coverage"):
        validate_complete_teacher_corpus([path])
    manifest["target_successes_per_cell"][0] -= 1
    manifest["shards"][0]["sha256"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="SHA"):
        validate_complete_teacher_corpus([path])


def test_parallel_dry_run_does_not_create_output(tmp_path):
    target = tmp_path / "uncreated"
    env = dict(os.environ, LM_RUN_DIR=str(target))
    env.pop("LD_LIBRARY_PATH", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/teacher/collect_diverse_teachers_parallel.sh"), "--dry-run"],
                            env=env, capture_output=True, text=True, check=True)
    assert "--case table_box" in result.stdout and "--case table_approach" in result.stdout
    assert not target.exists()


def test_production_plan_balances_500_without_writing(tmp_path):
    output = tmp_path / "never_created"
    result = subprocess.run([sys.executable, "-m", "scripts.teacher.collect_diverse_box_teacher",
                             "--case", "table_approach", "--output_dir", str(output), "--dry-run"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    plans = [json.loads(line) for line in result.stdout.splitlines()]
    quotas = [sum(plan["target"] for plan in plans if plan["cell"] == cell) for cell in range(16)]
    assert sum(quotas) == 500 and min(quotas) == 31 and max(quotas) == 32
    assert not output.exists()


def test_restricted_workspace_sampling_and_unchanged_default_signature():
    source = ROOT / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/events.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "reset_pick_place_work_area")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    namespace = {"torch": torch, "math_utils": SimpleNamespace(sample_uniform=lambda lo, hi, shape, device: lo + torch.rand(shape, device=device) * (hi-lo))}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    class Asset:
        def write_root_pose_to_sim(self, pose, env_ids):
            self.pose = pose
        def write_root_velocity_to_sim(self, velocity, env_ids):
            pass
    class Scene(dict):
        env_origins = torch.zeros(32, 3)
    scene = Scene(table=Asset(), bin=Asset())
    env = SimpleNamespace(num_envs=32, device="cpu", scene=scene)
    kwargs = dict(workspace_xy_range={"x": (-2.05, 2.05), "y": (-2.05, 2.05)},
                  table_origin_distance_range=(0.75, 2.2), bin_origin_distance_range=(0.75, 2.3),
                  table_bin_distance_range=(1, 3.2), table_center_e=(0.62, 0, 0.74),
                  bin_center_e=(1.55, -0.95, 0.28), bin_qr_center_e=(1.55, -0.95, 0.7),
                  table_asset_positions_e={"table": (0.62, 0, 0.7)}, bin_asset_positions_e={"bin": (1.55, -0.95, 0.28)})
    func = namespace["reset_pick_place_work_area"]
    func(env, None, **kwargs, table_xy_range={"x": (0.8, 1.5), "y": (-0.4, 0.4)}, bin_xy_range={"x": (-0.5, 0.5), "y": (-1.8, -1.3)})
    assert bool(((env.table_center_e[:, 0] >= 0.8) & (env.table_center_e[:, 0] <= 1.5)).all())
    assert bool(((env.table_center_e[:, 1] >= -0.4) & (env.table_center_e[:, 1] <= 0.4)).all())
    assert torch.allclose(scene["table"].pose[:, :2], env.table_center_e[:, :2])
    func(env, None, **kwargs)  # old callers require no new arguments
    with pytest.raises(ValueError, match="workspace"):
        func(env, None, **kwargs, table_xy_range={"x": (-5, 5), "y": (-1, 1)})
