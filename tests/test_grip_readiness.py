import pytest
import torch

from unitree_rl_lab.utils.teacher.grip_readiness import (
    GripReadiness,
    GripReadinessCfg,
    advance_offsets,
    closing_directions,
    hand_ready,
)


def test_closing_direction_follows_the_grasp_away_from_open():
    cfg = GripReadinessCfg()
    targets = torch.tensor([0.5, -0.4, 0.0005, 0.0])
    open_positions = torch.zeros(4)
    torch.testing.assert_close(closing_directions(targets, open_positions, cfg), torch.tensor([1.0, -1.0, 0.0, 0.0]))


def test_hand_ready_threshold():
    cfg = GripReadinessCfg(min_contacts_per_hand=2)
    assert hand_ready(1, 2, cfg) == (False, True)
    assert hand_ready(3, 3, cfg) == (True, True)


def test_offsets_close_only_masked_joints_and_are_capped():
    cfg = GripReadinessCfg(finger_close_rate_rad=0.1, finger_close_max_rad=0.25)
    directions = torch.tensor([1.0, -1.0, 1.0])
    mask = torch.tensor([1.0, 1.0, 0.0])
    offsets = torch.zeros(3)
    for _ in range(5):
        offsets = advance_offsets(offsets, directions, mask, cfg)
    torch.testing.assert_close(offsets, torch.tensor([0.25, -0.25, 0.0]))


def test_invalid_cfg():
    with pytest.raises(ValueError):
        GripReadinessCfg(max_steps=2, hold_steps=5).validate()
    with pytest.raises(ValueError):
        GripReadinessCfg(finger_close_rate_rad=0.0).validate()


def fake_readiness(cfg):
    """GripReadiness with a 4-joint action: [left arm, left finger, right finger, right arm]."""
    regrip = object.__new__(GripReadiness)
    regrip.cfg = cfg.validate()
    regrip.scale = torch.ones(4)
    regrip.offset = torch.zeros(4)
    regrip.sides = {"left": [1], "right": [2]}
    regrip.open_positions = torch.zeros(4)
    return regrip


def test_regrip_closes_the_weak_hand_until_contacts_hold_and_keeps_the_rest():
    cfg = GripReadinessCfg(min_contacts_per_hand=2, hold_steps=3, max_steps=50,
                           finger_close_rate_rad=0.01, finger_close_max_rad=0.25)
    regrip = fake_readiness(cfg)
    reference = torch.tensor([[0.3, 0.5, -0.5, 0.7]])
    applied = []
    regrip_state = {"raw": reference.clone()}

    def step(raw):
        applied.append(raw.clone())
        regrip_state["raw"] = raw.clone()

    def contacts():
        # Left hand gains contacts once its finger closed 0.05 rad further; right is already fine.
        left_closed = float(regrip_state["raw"][0, 1] - 0.5)
        return (2 if left_closed >= 0.05 - 1e-9 else 1), 2

    final, log = regrip.run(reference, step, contacts)
    assert log["ready"] and not log["timed_out"]
    assert log["final_contacts"] == [2, 2]
    assert final[0, 1] == pytest.approx(0.55)          # left finger closed further, then held
    assert final[0, 2] == pytest.approx(-0.5)          # right finger untouched (already ready)
    assert final[0, 0] == pytest.approx(0.3) and final[0, 3] == pytest.approx(0.7)  # arms unchanged
    assert all(abs(b[0, 1] - a[0, 1]) <= cfg.finger_close_rate_rad + 1e-6 for a, b in zip(applied, applied[1:]))
    torch.testing.assert_close(final, applied[-1])


def test_regrip_times_out_without_exceeding_the_finger_limit():
    cfg = GripReadinessCfg(min_contacts_per_hand=2, hold_steps=3, max_steps=40,
                           finger_close_rate_rad=0.01, finger_close_max_rad=0.25)
    regrip = fake_readiness(cfg)
    final, log = regrip.run(torch.tensor([[0.0, 0.5, -0.5, 0.0]]), lambda raw: None, lambda: (1, 1))
    assert log["timed_out"] and log["steps"] == 40
    assert log["max_finger_offset_rad"] <= 0.25 + 1e-9
    assert final[0, 1] == pytest.approx(0.5 + 0.25) and final[0, 2] == pytest.approx(-0.5 - 0.25)


def test_already_ready_grasp_is_kept_unchanged():
    regrip = fake_readiness(GripReadinessCfg(hold_steps=5))
    reference = torch.tensor([[0.0, 0.5, -0.5, 0.0]])
    final, log = regrip.run(reference, lambda raw: None, lambda: (3, 3))
    assert log["ready"] and log["steps"] == 5
    torch.testing.assert_close(final, reference)
