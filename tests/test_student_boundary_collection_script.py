"""Check the collection wrapper without starting Isaac Sim."""

import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/teacher/collect_student_boundary_data.sh"


def invoke(*arguments, **overrides):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("LM_")}
    environment.update(overrides)
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd="/tmp",  # Must work even when invoked outside the repository.
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def commands(result):
    assert result.returncode == 0, result.stderr
    return [
        shlex.split(line.removeprefix("Command: "))
        for line in result.stdout.splitlines()
        if line.startswith("Command: ")
    ]


def value(command, flag):
    return command[command.index(flag) + 1]


def test_syntax_and_help():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    result = invoke("--help")
    assert result.returncode == 0
    assert "not guarantees" in result.stdout
    assert "Unknown argument" in invoke("--unknown").stderr
    assert invoke("--unknown").returncode != 0


def test_dry_run_preserves_student_boundary_lineage_and_creates_no_files(tmp_path):
    output = tmp_path / "must_not_be_created"
    result = invoke("--collect", "--dry-run", LM_RUN_DIR=str(output))
    screen, fsm, place, carry, dagger = commands(result)
    assert not output.exists()
    assert value(screen, "--episodes") == "10"
    assert value(fsm, "--episodes") == "2000"
    assert value(screen, "--seed") != value(fsm, "--seed")
    assert value(screen, "--snapshot_output") != value(fsm, "--snapshot_output")
    assert value(fsm, "--work_area_size_m") == "5.0"
    assert "--stop_after_carry_snapshot" in fsm
    assert "--allow_uncertified_grasp_probe" not in fsm
    assert value(place, "--target_successes") == "500"
    assert value(carry, "--target_successes") == "500"
    assert value(place, "--snapshot_input") == value(fsm, "--snapshot_output")
    assert value(carry, "--snapshot_input") == value(fsm, "--snapshot_output")
    assert value(place, "--snapshot_input") != value(carry, "--staging_snapshot_output")
    assert "--student_checkpoint" not in place
    assert "--student_teacher_relabel" not in place
    assert value(dagger, "--episodes") == "500"
    assert "NOT successful Teacher Grasp" in result.stdout
    assert all("--overwrite" not in command for command in (fsm, place, carry, dagger))


def test_smoke_and_environment_overrides():
    fsm, place, carry, dagger = commands(invoke("--smoke", "--dry-run"))
    assert value(fsm, "--episodes") == "10"
    assert value(place, "--target_successes") == "5"
    assert value(carry, "--num_trials") == "20"
    assert value(dagger, "--episodes") == "5"
    result = invoke("--smoke", "--dry-run", LM_SCREEN_EPISODES="17", LM_SEED="123")
    fsm, _, _, dagger = commands(result)
    assert value(fsm, "--episodes") == "17"
    assert value(fsm, "--seed") == "123"
    assert value(dagger, "--seed") == "124"


@pytest.mark.parametrize("command_index", range(5))
def test_every_generated_flag_exists_in_target_parser(command_index):
    command = commands(invoke("--collect", "--dry-run"))[command_index]
    tree = ast.parse((ROOT / command[1]).read_text())
    known = {"--headless", "--device"}  # Added by Isaac Lab AppLauncher.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "add_argument":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    known.add(arg.value)
                    if arg.value.startswith("--"):
                        known.add("--no-" + arg.value[2:])
    for token in command[2:]:
        if token.startswith("--"):
            assert token in known, (command[1], token)


def audit_code():
    return SCRIPT.read_text().split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


def test_no_teacher_success_is_incomplete_not_success(tmp_path, monkeypatch):
    metrics = tmp_path / "teacher.metrics.json"
    metrics.write_text(json.dumps({
        "status": "completed", "dataset_published": False, "successes": 0,
    }))
    monkeypatch.setattr(sys, "argv", ["-", "teacher", str(metrics), str(tmp_path / "absent.hdf5")])
    with pytest.raises(SystemExit) as error:
        exec(compile(audit_code(), str(SCRIPT), "exec"), {})
    assert error.value.code == 2


def test_partial_teacher_target_is_incomplete_even_with_valid_dataset(tmp_path, monkeypatch):
    from unitree_rl_lab.utils.teacher import phase_bc

    metrics = tmp_path / "teacher.metrics.json"
    metrics.write_text(json.dumps({
        "status": "completed", "dataset_published": True, "successes": 1,
        "evaluation_mode": "teacher_collection", "unique_source_snapshot_count": 1,
        "target_successes_reached": False,
    }))
    monkeypatch.setattr(phase_bc, "discover_episodes", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(sys, "argv", ["-", "teacher", str(metrics), str(tmp_path / "teacher.hdf5")])
    with pytest.raises(SystemExit) as error:
        exec(compile(audit_code(), str(SCRIPT), "exec"), {})
    assert error.value.code == 2


def test_missing_snapshots_only_allowed_when_metrics_report_zero(tmp_path, monkeypatch, capsys):
    metrics = tmp_path / "fsm.metrics.json"
    payload = {
        "episodes": 1, "episode_results": [{}],
        "grasp_checkpoint_certified_by_gate": True, "snapshot_count": 0,
    }
    metrics.write_text(json.dumps(payload))
    monkeypatch.setenv("LM_FSM_EPISODES", "1")
    monkeypatch.setattr(sys, "argv", ["-", "fsm", str(metrics), str(tmp_path / "absent.hdf5")])
    code = compile(audit_code(), str(SCRIPT), "exec")
    exec(code, {})
    assert capsys.readouterr().out == "0 0\n"
    payload["snapshot_count"] = 1
    metrics.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="snapshot file is missing"):
        exec(code, {})


def test_unfinished_metrics_rejected_even_if_process_exits_normally(tmp_path, monkeypatch):
    metrics = tmp_path / "dagger.metrics.json"
    metrics.write_text(json.dumps({"status": "aborted", "dataset_published": False}))
    monkeypatch.setattr(sys, "argv", ["-", "dagger", str(metrics), "absent.hdf5"])
    with pytest.raises(ValueError, match="Incomplete dagger"):
        exec(compile(audit_code(), str(SCRIPT), "exec"), {})


def test_default_only_screens_and_modes_are_exclusive():
    result = invoke("--dry-run")
    [screen] = commands(result)
    assert value(screen, "--episodes") == "10"
    assert "Screening only" in result.stdout
    assert invoke("--smoke", "--collect").returncode != 0


def test_alignment_probe_changes_correction_not_success_gates():
    [baseline] = commands(invoke("--dry-run"))
    [probe] = commands(invoke("--alignment-probe", "--dry-run"))
    assert value(baseline, "--approach_wrist_servo_gain") == "0.7"
    assert value(probe, "--approach_wrist_servo_gain") == "1.0"
    assert value(probe, "--approach_wrist_servo_max_correction_m") == "0.12"
    assert value(probe, "--approach_endpoint_centering_m") == "0.05"
    assert value(probe, "--approach_endpoint_offset_steps") == "10"
    assert not any("handoff_" in arg and "limit" in arg for arg in probe)
    assert value(baseline, "--bc2_gate") == value(probe, "--bc2_gate")
    [override] = commands(invoke("--alignment-probe", "--dry-run", LM_WRIST_SERVO_GAIN="0.9"))
    assert value(override, "--approach_wrist_servo_gain") == "0.9"


@pytest.mark.parametrize("grasp_count,carry_count,ready", [
    (0, 0, False), (3, 0, False), (0, 2, False), (2, 2, False), (3, 2, True),
])
def test_readiness_requires_both_grasp_and_carry_states(
    tmp_path, monkeypatch, grasp_count, carry_count, ready,
):
    metrics = tmp_path / "metrics.json"
    report = tmp_path / "readiness.json"
    metrics.write_text(json.dumps({
        "episodes": 10,
        "episode_results": [{"reason": "timeout", "transitions": []}] * 10,
    }))
    monkeypatch.setenv("LM_MIN_GRASP_STATES", "3")
    monkeypatch.setenv("LM_MIN_CARRY_STATES", "2")
    monkeypatch.setattr(sys, "argv", [
        "-", "readiness", str(metrics), str(grasp_count), str(carry_count), str(report),
    ])
    code = compile(audit_code(), str(SCRIPT), "exec")
    if ready:
        exec(code, {})
    else:
        with pytest.raises(SystemExit) as error:
            exec(code, {})
        assert error.value.code == 2
    payload = json.loads(report.read_text())
    assert payload["ready_for_collection"] is ready
    assert payload["reason_counts"] == {"timeout": 10}
    assert payload["success_conditions_unchanged"] is True


def test_screen_count_is_independent_of_production_count(tmp_path, monkeypatch, capsys):
    metrics = tmp_path / "screen.metrics.json"
    metrics.write_text(json.dumps({
        "episodes": 10, "episode_results": [{}] * 10,
        "grasp_checkpoint_certified_by_gate": True, "snapshot_count": 0,
    }))
    monkeypatch.setenv("LM_FSM_EPISODES", "2000")
    monkeypatch.setattr(sys, "argv", [
        "-", "fsm", str(metrics), str(tmp_path / "absent.hdf5"), "10",
    ])
    exec(compile(audit_code(), str(SCRIPT), "exec"), {})
    assert capsys.readouterr().out == "0 0\n"


def test_failed_screen_stops_shell_before_any_large_collection(tmp_path):
    # Simulate a cleanly completed but entirely unsuccessful screen, NOT physics.
    # Let the real inline validation run so the shell's gate is exercised end to end.
    runner = tmp_path / "fake_simulator_python"
    runner.write_text(f"#!{sys.executable}\n" + '''
import json
from pathlib import Path
import sys
if sys.argv[1] == "-":
    code = sys.stdin.read()
    if sys.argv[2] == "preflight":
        sys.exit(0)
    sys.argv = sys.argv[1:]
    exec(compile(code, "inline-audit", "exec"), {})
else:
    assert sys.argv[1].endswith("play_bc_skill_fsm.py"), sys.argv
    episodes = int(sys.argv[sys.argv.index("--episodes") + 1])
    assert episodes == 10, "Production collection must not start"
    output = Path(sys.argv[sys.argv.index("--metrics_output") + 1])
    output.write_text(json.dumps({
        "episodes": episodes, "snapshot_count": 0,
        "grasp_checkpoint_certified_by_gate": True,
        "episode_results": [{"reason": "timeout", "transitions": []}] * episodes,
    }))
''')
    runner.chmod(0o755)
    run_dir = tmp_path / "screen_run"
    result = invoke("--collect", LM_PY=str(runner), LM_RUN_DIR=str(run_dir))
    assert result.returncode == 2, result.stderr
    assert "[STOP]" in result.stdout
    assert "[01_student_end_states]" not in result.stdout
    assert "[02_place_teacher]" not in result.stdout
    assert "[04_grasp_dagger]" not in result.stdout
    assert not (run_dir / "student_end_states.metrics.json").exists()
    assert json.loads((run_dir / "readiness_report.json").read_text())["ready_for_collection"] is False
