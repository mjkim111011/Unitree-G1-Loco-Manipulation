from types import SimpleNamespace
import copy
import csv
import io
import json
import sys
from pathlib import Path

import pytest

from scripts.teacher.evaluate_transition_mass_sweep import command_for, summarize, validate_result
from scripts.teacher import evaluate_transition_mass_sweep as sweep


def test_summary_separates_pre_transition_failures():
    data = {"episode_results": [
        {"success": False, "reason": "horizon", "carry": {"success": False, "reason": "grasp_not_reached"}},
        {"success": True, "carry": {"success": False, "reason": "box_drop", "trace": [{"carry_step": -1}]}},
        {"success": True, "carry": {"success": False, "reason": "carry_horizon", "trace": [{"carry_step": 0}]}},
        {"success": True, "carry": {"success": True, "reason": "carry_success", "trace": [{"carry_step": 0}]}},
    ]}
    row = summarize(data, 2.0, "stage2")
    assert row["grasp_successes"] == 3
    assert row["transition_reached"] == 2
    assert row["pre_transition_failures"] == 2
    assert row["full_chain_success_rate"] == 0.25
    assert row["carry_success_given_transition"] == 0.5


def test_no_transition_does_not_report_zero_conditional_success():
    data = {"episode_results": [{"success": False, "carry": {"success": False}}]}
    assert summarize(data, 2.0, "instant")["carry_success_given_transition"] is None


@pytest.mark.parametrize("method", ["instant", "fixed_1s", "stage1", "stage2"])
def test_commands_keep_mass_and_frozen_policies_consistent(tmp_path, method):
    args = SimpleNamespace(
        checkpoint="bc2.pt", carry_checkpoint="bc3.pt", stage1_checkpoint="s1.pt",
        stage2_checkpoint="s2.pt", episodes=10, seed=314159,
        episode_seconds=20.0, carry_seconds=16.0, device="cuda:0", headless=True,
    )
    command = command_for(args, 1.5, method, tmp_path / "result.json")
    assert command[command.index("--box_mass_kg") + 1] == "1.5"
    assert command[command.index("--carry_checkpoint") + 1] == "bc3.pt"
    if method.startswith("stage"):
        assert "--carry_blend_steps" not in command
        assert command[command.index("--transition_checkpoint") + 1] == f"s{method[-1]}.pt"
    else:
        assert "--transition_checkpoint" not in command
        assert command[command.index("--carry_blend_steps") + 1] == ("50" if method == "fixed_1s" else "0")


@pytest.fixture
def matching_result():
    return {
        "evaluator_config": {
            "box_mass_kg": 1.5, "episodes": 1, "seed": 42,
            "episode_seconds": 20.0, "carry_seconds": 16.0, "box_size_m": 0.2,
            "box_x_range_m": [0.618, 0.622], "box_y_range_m": [-0.002, 0.002],
            "rule_based_place": False, "carry_blend_steps": 0,
            "transition_checkpoint_sha256": "s1",
        },
        "checkpoint_sha256": "bc2", "carry_checkpoint_sha256": "bc3",
        "episode_results": [{"episode": 0, "actual_box_mass_kg": 1.5}],
    }


@pytest.mark.parametrize("mismatch", [None, "mass", "checkpoint", "incomplete", "blend"])
def test_resume_accepts_only_matching_complete_results(matching_result, mismatch):
    data = matching_result
    if mismatch == "mass":
        data["episode_results"][0]["actual_box_mass_kg"] = 0.2
    elif mismatch == "checkpoint":
        data["carry_checkpoint_sha256"] = "other"
    elif mismatch == "incomplete":
        data["episode_results"] = []
    elif mismatch == "blend":
        data["evaluator_config"]["carry_blend_steps"] = 50
    args = SimpleNamespace(episodes=1, seed=42, episode_seconds=20.0, carry_seconds=16.0)
    kwargs = dict(mass=1.5, method="stage1", args=args, hashes={"bc2": "bc2", "bc3": "bc3", "stage1": "s1"})
    if mismatch:
        with pytest.raises(ValueError):
            validate_result(data, **kwargs)
    else:
        validate_result(data, **kwargs)


def test_sweep_writes_results_logs_summary_and_resumes(tmp_path, monkeypatch, matching_result):
    checkpoints = {}
    for name in ("bc2", "bc3", "stage1", "stage2"):
        checkpoints[name] = tmp_path / f"{name}.pt"
        checkpoints[name].write_bytes(name.encode())
    root = tmp_path / "results"
    argv = [
        "sweep", "--checkpoint", str(checkpoints["bc2"]),
        "--carry_checkpoint", str(checkpoints["bc3"]),
        "--stage1_checkpoint", str(checkpoints["stage1"]),
        "--stage2_checkpoint", str(checkpoints["stage2"]),
        "--masses", "1.5", "--episodes", "1", "--seed", "42",
        "--output_root", str(root), "--headless",
    ]
    launches = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            launches.append(command)
            data = copy.deepcopy(matching_result)
            data["checkpoint_sha256"] = sweep.sha256(checkpoints["bc2"])
            data["carry_checkpoint_sha256"] = sweep.sha256(checkpoints["bc3"])
            config = data["evaluator_config"]
            config["transition_checkpoint_sha256"] = (
                sweep.sha256(Path(command[command.index("--transition_checkpoint") + 1]))
                if "--transition_checkpoint" in command else None
            )
            config["carry_blend_steps"] = int(command[command.index("--carry_blend_steps") + 1]) if "--carry_blend_steps" in command else 0
            data["episode_results"][0].update({
                "success": True, "carry": {"success": True, "trace": [{"carry_step": 0}]},
            })
            Path(command[command.index("--output") + 1]).write_text(json.dumps(data))
            self.stdout = io.StringIO("[chain] test progress\n")

        def wait(self):
            return 0

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sweep.subprocess, "Popen", FakeProcess)
    sweep.main()
    assert len(launches) == 4
    with (root / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["method"] for r in rows] == list(sweep.METHODS)
    assert all(r["full_chain_success_rate"] == "1.0" for r in rows)
    assert len(list(root.glob("mass_*/*.log"))) == 4
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    sweep.main()
    assert len(launches) == 4
