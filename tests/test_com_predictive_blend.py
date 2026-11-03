import copy

import pytest
import torch

from unitree_rl_lab.utils.teacher.com_predictive_blend import (
    ComBlendCfg, alpha_candidates, choose_candidate, convex_hull, signed_support_margin,
)
from scripts.teacher.evaluate_com_transition_sweep import compare_starts, START_TOLERANCES, command_for, reports, METHODS


def square():
    return torch.tensor([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])


def test_support_margin_distinguishes_inside_and_outside():
    result = signed_support_margin(torch.tensor([[0., 0.], [0.12, 0.], [0.09, 0.]]), square())
    torch.testing.assert_close(result, torch.tensor([0.1, -0.02, 0.01]))


def test_hull_handles_two_feet_and_duplicate_corners():
    hull = convex_hull([[-1,-1], [1,-1], [1,1], [-1,1], [0,0], [-1,-1]])
    assert hull == [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]


def test_candidate_selection_responds_to_predicted_state():
    cfg = ComBlendCfg().validate()
    alphas = torch.tensor([0., 0.04, 0.08])
    stable = torch.tensor([[0., 0.], [0.01, 0.], [0.02, 0.]])
    unstable = torch.tensor([[0., 0.], [0.01, 0.], [0.12, 0.]])
    jumps = torch.zeros(3)
    assert choose_candidate(alphas, stable, square(), jumps, True, cfg)[0] == 2
    assert choose_candidate(alphas, unstable, square(), jumps, True, cfg)[0] == 1


def test_command_jump_can_reject_a_balanced_candidate():
    index, _, feasible = choose_candidate(torch.tensor([0., .08]), torch.zeros(2,2), square(),
                                         torch.tensor([0.01, 0.2]), True, ComBlendCfg())
    assert index == 0
    assert feasible.tolist() == [True, False]


def test_contact_loss_holds_lowest_alpha_and_reports_no_feasible_candidate():
    index, _, feasible = choose_candidate(torch.tensor([.2, .28]), torch.zeros(2,2), square(),
                                         torch.zeros(2), False, ComBlendCfg())
    assert index == 0
    assert not feasible.any()


def test_no_support_is_explicit_best_effort_fallback():
    _, margins, feasible = choose_candidate(torch.tensor([0., .08]), torch.zeros(2,2),
                                           torch.empty(0,2), torch.zeros(2), True, ComBlendCfg())
    assert not feasible.any()
    assert (margins < 0).all()
    assert torch.isfinite(margins).all()


def test_alpha_candidates_are_monotone_and_deadline_is_logged_separately():
    cfg = ComBlendCfg()
    previous = 0.0
    for step in range(100):
        values, floor = alpha_candidates(previous, step, cfg)
        assert all(previous <= v <= 1.0 for v in values)
        previous = values[0]
    assert previous == pytest.approx(1.0)
    assert floor == pytest.approx(1.0)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ComBlendCfg(tracking_fraction=2).validate()


def start():
    return {"snapshot_id": "source:1", "start_features": {k: [0., 0.] for k in START_TOLERANCES}}


def test_isolated_start_validation_rejects_previous_observed_drift():
    first, other = start(), start()
    assert compare_starts([first, other])[0]
    other["start_features"]["joint_position"][0] = .05
    matched, differences = compare_starts([first, other])
    assert not matched
    assert differences["joint_position"] == .05


def test_input_history_mismatch_is_also_rejected():
    other = start()
    other["start_features"]["student_observation"][1] = .01
    assert not compare_starts([start(), other])[0]


def test_nonfinite_start_is_rejected():
    other = start()
    other["start_features"]["box_state"][1] = float('nan')
    with pytest.raises(ValueError, match="nonfinite"):
        compare_starts([start(), other])


def test_each_command_selects_one_state_and_no_posture_residual(tmp_path):
    from types import SimpleNamespace
    args = SimpleNamespace(checkpoint="bc2", carry_checkpoint="bc3", snapshots=["snapshots"],
                           seed=314159, carry_seconds=16, device="cuda:0", headless=True,
                           box_mass_kg=0.2, snapshot_box_mass_kg=0.2)
    command = command_for(args, 2, "com_predictive", tmp_path / "result.json")
    assert command[command.index("--episodes") + 1] == "1"
    assert command[command.index("--transition_snapshot_offset") + 1] == "2"
    assert "--com_predictive_blend" in command
    assert "--transition_checkpoint" not in command
    assert "--carry_blend_steps" not in command


def test_reports_exclude_mismatched_starts_from_primary_comparison():
    group = {}
    for method in METHODS:
        e = start()
        e.update({"restore_success": True, "carry": {"success": True, "reason": "carry_success", "trace": []}})
        group[method] = e
    summary, matrix = reports({0: group})
    assert all(r["carry_success_rate"] == 1 for r in summary)
    group["com_predictive"]["start_features"]["root_state"][0] = .1
    summary, matrix = reports({0: group})
    assert not matrix[0]["paired_start_matched"]
    assert all(r["paired_valid_states"] == 0 and r["carry_success_rate"] is None for r in summary)


def test_v2_methods_build_mode_and_gate_flags(tmp_path):
    from types import SimpleNamespace
    from scripts.teacher.evaluate_com_transition_sweep import METHOD_SPECS
    args = SimpleNamespace(checkpoint="bc2", carry_checkpoint="bc3", snapshots=["snapshots"],
                           seed=314159, carry_seconds=16, device="cuda:0", headless=True,
                           box_mass_kg=0.2, snapshot_box_mass_kg=0.2)
    v2 = command_for(args, 0, "com_v2", tmp_path / "a.json")
    assert v2[v2.index("--com_predictive_mode") + 1] == "v2" and "--com_command_gate" not in v2
    assert "--com_command_gate" in command_for(args, 0, "com_v2_cmd", tmp_path / "b.json")
    walk = command_for(args, 0, "fixed_1s_walk", tmp_path / "d.json")
    assert "--walk_after_takeover" in walk and "--com_predictive_blend" not in walk
    assert walk[walk.index("--carry_blend_steps") + 1] == "50"
    fixed = command_for(args, 0, "fixed_0p5s", tmp_path / "c.json")
    assert fixed[fixed.index("--carry_blend_steps") + 1] == str(METHOD_SPECS["fixed_0p5s"][0])


def test_early_drop_counts_as_transition_failure_and_pairs_are_reported():
    def episode(success, reason, last_step):
        e = start()
        e.update({"restore_success": True, "carry": {
            "success": success, "reason": reason,
            "trace": [{"carry_step": last_step, "target_jump_rad": 0.0, "root_velocity_body": [0, 0.1, 0]}]}})
        return e
    groups = {0: {"fixed_1s": episode(False, "box_drop", 95), "com_v2": episode(True, "carry_success", 799)},
              1: {"fixed_1s": episode(False, "carry_horizon", 799), "com_v2": episode(False, "box_drop", 400)}}
    summary, _ = reports(groups, ("fixed_1s", "com_v2"), "fixed_1s")
    rows = {r["method"]: r for r in summary}
    assert rows["fixed_1s"]["transition_failures"] == 1 and rows["fixed_1s"]["other_failures"] == 1
    assert rows["com_v2"]["transition_failures"] == 0 and rows["com_v2"]["other_failures"] == 1
    assert rows["com_v2"]["only_this_succeeded_vs_fixed_1s"] == 1
    assert rows["com_v2"]["only_fixed_1s_succeeded"] == 0


def test_payload_override_passes_snapshot_mass(tmp_path):
    from types import SimpleNamespace
    args = SimpleNamespace(checkpoint="bc2", carry_checkpoint="bc3", snapshots=["snapshots"],
                           seed=314159, carry_seconds=24, device="cuda:0", headless=True,
                           box_mass_kg=0.5, snapshot_box_mass_kg=0.2)
    command = command_for(args, 0, "fixed_1s", tmp_path / "a.json")
    assert command[command.index("--box_mass_kg") + 1] == "0.5"
    assert command[command.index("--snapshot_box_mass_kg") + 1] == "0.2"
