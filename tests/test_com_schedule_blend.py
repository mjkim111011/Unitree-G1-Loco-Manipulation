import pytest
import torch

from unitree_rl_lab.utils.teacher.com_schedule_blend import (
    HOLD,
    ComScheduleCfg,
    Schedule,
    candidate_schedules,
    choose_schedule,
    forced_floor,
    predict_capture_points,
    schedule_alpha_trajectories,
    walking_gate_open,
)


def square():
    return torch.tensor([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])


def test_schedules_are_monotone_and_respect_the_deadline():
    cfg = ComScheduleCfg().validate()
    for step in (0, 40, cfg.max_steps - 20, cfg.max_steps - 1):
        for alpha in (0.0, 0.3):
            schedules = candidate_schedules(alpha, step, None, cfg)
            alphas = schedule_alpha_trajectories(schedules, step, cfg)
            assert torch.all(alphas[:, 1:] >= alphas[:, :-1])
            assert torch.all((alphas >= alpha) & (alphas <= 1.0))
            for k in range(alphas.shape[1]):
                assert torch.all(alphas[:, k] >= forced_floor(step + k, cfg) - 1e-6)
    # Even a hold reaches 1 by the deadline.
    late = schedule_alpha_trajectories([Schedule(HOLD, 0.0, cfg.max_steps - 5, cfg.max_steps - 5)],
                                       cfg.max_steps - 5, cfg)
    assert float(late[0, 4]) == pytest.approx(1.0)


def test_committed_schedule_continues_without_restarting():
    cfg = ComScheduleCfg().validate()
    committed = Schedule(0.5, 0.0, 0, 25)
    alphas = [committed.alpha(step) for step in range(25)]
    assert alphas[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(alphas, alphas[1:]))
    schedules = candidate_schedules(alphas[9], 10, committed, cfg)
    assert schedules[1] is committed


def test_predictor_distinguishes_schedules_when_blending_shifts_the_com():
    cfg = ComScheduleCfg().validate()
    schedules = candidate_schedules(0.0, 0, None, cfg)
    alphas = schedule_alpha_trajectories(schedules, 0, cfg)
    anchor = torch.zeros(3)
    bc3 = torch.tensor([0.3, 0.0, 0.0])
    jacobian = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])  # joint 0 moves the CoM laterally
    capture, _ = predict_capture_points(alphas, anchor, bc3, anchor, jacobian,
                                        torch.zeros(2), torch.zeros(2), 3.7, cfg)
    final_lateral = capture[:, -1, 1]
    assert final_lateral[0] == pytest.approx(0.0)  # hold: no shift
    assert torch.all(final_lateral[1:] > 0.0)
    assert len(set(round(float(v), 4) for v in final_lateral)) == len(schedules)


def test_keeps_the_1s_baseline_until_its_predicted_margin_is_too_small():
    cfg = ComScheduleCfg().validate()
    schedules = candidate_schedules(0.0, 0, None, cfg)  # hold, 0.26, 0.5, 1.0
    default = 3
    assert schedules[default].kind == cfg.default_schedule_seconds
    safe = torch.zeros(len(schedules), cfg.horizon_steps, 2)
    index, _, feasible, reasons = choose_schedule(schedules, safe, square(), torch.zeros(4), True, default, cfg)
    assert index == default and feasible.all() and reasons == []
    risky = safe.clone()
    risky[default, :, 1] = 0.09  # 1 cm margin: still feasible, below keep_margin
    risky[1, :, 1] = 0.05        # 0.26 s keeps 5 cm
    index, margins, feasible, _ = choose_schedule(schedules, risky, square(), torch.zeros(4), True, default, cfg)
    assert bool(feasible[default]) and float(margins[default]) < cfg.keep_margin_m
    assert schedules[index].kind in (HOLD, 0.5)  # the largest-margin schedules (10 cm), faster first
    assert schedules[index].kind == 0.5


def test_preferred_wins_ties_once_it_is_not_kept():
    cfg = ComScheduleCfg().validate()
    committed = Schedule(1.0, 0.0, 0, 50)
    schedules = candidate_schedules(0.1, 10, committed, cfg)
    capture = torch.full((len(schedules), cfg.horizon_steps, 2), 0.09)  # all 1 cm margin
    index, *_ = choose_schedule(schedules, capture, square(), torch.zeros(len(schedules)), True, 1, cfg)
    assert index == 1


def test_all_infeasible_is_a_logged_least_risk_fallback():
    cfg = ComScheduleCfg().validate()
    schedules = candidate_schedules(0.0, 0, None, cfg)
    capture = torch.full((len(schedules), cfg.horizon_steps, 2), 0.3)
    capture[2] = 0.15  # least bad
    index, margins, feasible, reasons = choose_schedule(
        schedules, capture, square(), torch.zeros(4), True, 3, cfg)
    assert index == 2 and not feasible.any()
    assert reasons == ["capture_margin"]
    assert torch.all(margins < 0)


def test_contact_loss_holds_and_reports_reason():
    cfg = ComScheduleCfg().validate()
    schedules = candidate_schedules(0.2, 5, None, cfg)
    capture = torch.zeros(len(schedules), cfg.horizon_steps, 2)
    index, _, feasible, reasons = choose_schedule(
        schedules, capture, square(), torch.zeros(4), False, 3, cfg)
    assert schedules[index].kind == HOLD and not feasible.any()
    assert "hand_contact_lost" in reasons


def test_walking_gate():
    cfg = ComScheduleCfg().validate()
    assert not walking_gate_open(0.9, 0.05, 0.0, 0, cfg)
    assert walking_gate_open(1.0, 0.05, 0.05, 0, cfg)
    assert not walking_gate_open(1.0, 0.0, 0.05, 0, cfg)
    assert not walking_gate_open(1.0, 0.05, 0.3, 10, cfg)
    assert walking_gate_open(1.0, -0.1, 0.3, int(cfg.gate_max_delay_seconds / cfg.step_dt), cfg)


def test_hold_predicts_no_shift_despite_pd_steady_state_error():
    # Joints sit 0.13 rad below their targets under load; a hold must not
    # predict them "catching up" and moving the CoM.
    cfg = ComScheduleCfg().validate()
    schedules = candidate_schedules(0.0, 0, None, cfg)
    alphas = schedule_alpha_trajectories(schedules, 0, cfg)
    anchor = torch.tensor([0.5, 0.0, 0.0])
    capture, _ = predict_capture_points(alphas, anchor, anchor + 0.1, anchor, torch.ones(2, 3) * 0.3,
                                        torch.zeros(2), torch.zeros(2), 3.7, cfg)
    torch.testing.assert_close(capture[0], torch.zeros_like(capture[0]))
