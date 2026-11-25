from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unitree_rl_lab.utils.teacher.grip_residual import (
    BOX_ESTIMATE_COMPONENTS,
    PRIVILEGED_COMPONENTS,
    GripResidualCfg,
    ResidualScheduleCfg,
    actor_components,
    assemble,
    blend_weight,
    compose_residual_action,
    critic_components,
    excess,
    layout_dim,
    one_sided_excess,
    rate_limited_residual,
    residual_envelope,
    residual_joint_indices,
    residual_raw_offset,
    tilt_from_projected_gravity,
)
from unitree_rl_lab.utils.teacher.transition_gate import fixed_blend_weight
from unitree_rl_lab.utils.usd_cache import assign_unique_usd_dirs

ROOT = Path(__file__).resolve().parents[1]
BC3 = ROOT / "outputs/bc3_direct_r01/bc3_direct_mix.pt"


def _carry_contract():
    if not BC3.is_file():
        pytest.skip("mixed BC3 checkpoint not available")
    from unitree_rl_lab.utils.teacher.carry_runtime_contract import validate_carry_runtime_contract
    from unitree_rl_lab.utils.teacher.phase_bc import load_bc_policy

    return validate_carry_runtime_contract(load_bc_policy(BC3, "cpu").carry_runtime_contract)


# ---------------------------------------------------------------- timing
def test_blend_weight_matches_the_live_fixed_blend():
    cfg = ResidualScheduleCfg()
    steps = torch.arange(0, 80)
    expected = torch.tensor([fixed_blend_weight(int(k), cfg.blend_steps) for k in steps])
    torch.testing.assert_close(blend_weight(steps, cfg), expected)


def test_envelope_ramps_holds_and_fades_to_exactly_zero():
    cfg = ResidualScheduleCfg()
    k = torch.arange(-5, 200)
    beta = residual_envelope(k, cfg)
    assert torch.all(beta[k < 0] == 0)
    assert 0 < float(beta[k == 0]) < 0.01                        # gentle ramp-in, no step at the handoff
    assert torch.all(beta[(k >= cfg.fade_in_steps - 1) & (k < cfg.full_until_steps)] == 1)
    assert torch.all(beta[k >= cfg.fade_end_steps - 1] == 0)      # zero by 3 s: BC3's own command after
    # quintic slope peaks at 1.875 per unit fraction: per-step change <= 1.875 / (shortest ramp length)
    shortest = min(cfg.fade_in_steps, cfg.fade_end_steps - cfg.full_until_steps)
    assert float(beta.diff().abs().max()) <= 1.875 / shortest + 1e-6
    # the blend ends inside the full-authority window, so no envelope change coincides with it
    assert cfg.fade_in_steps < cfg.blend_steps < cfg.full_until_steps


def test_invalid_schedule_and_cfg():
    with pytest.raises(ValueError):
        ResidualScheduleCfg(full_until_steps=150, fade_end_steps=150).validate()
    with pytest.raises(ValueError):
        GripResidualCfg(max_residual_rad=0.2).validate()
    with pytest.raises(ValueError):
        GripResidualCfg(max_rate_rad=0.1, max_residual_rad=0.05).validate()
    with pytest.raises(ValueError):
        GripResidualCfg(variant="com").validate()


def test_cfg_round_trip():
    cfg = GripResidualCfg(variant="body", max_residual_rad=0.08, box_obs_noise_m=0.0,
                          schedule=ResidualScheduleCfg(full_until_steps=90))
    assert GripResidualCfg.from_dict(cfg.to_dict()) == cfg


# ---------------------------------------------------------------- residual dynamics
def test_rate_limit_and_bound():
    cfg = GripResidualCfg(max_residual_rad=0.05, max_rate_rad=0.005)
    r = torch.zeros(1, 15)
    for _ in range(3):
        r = rate_limited_residual(r, torch.ones(1, 15), cfg)
    torch.testing.assert_close(r, torch.full((1, 15), 0.015))
    for _ in range(50):
        r = rate_limited_residual(r, 10 * torch.ones(1, 15), cfg)   # commands are clipped to [-1, 1]
    torch.testing.assert_close(r, torch.full((1, 15), 0.05))


def test_worst_case_residual_target_change_is_bounded_and_ends_at_zero():
    """Bang-bang commands: the applied correction never jumps, and is exactly zero after the window."""

    cfg = GripResidualCfg()
    sched = cfg.schedule
    r = torch.zeros(1, 15)
    applied = []
    for k in range(0, 200):
        r = rate_limited_residual(r, torch.full((1, 15), 1.0 if k % 2 == 0 else -1.0), cfg)
        applied.append(residual_envelope(torch.tensor([k]), sched)[:, None] * r)
    applied = torch.cat(applied)
    max_beta_rate = float(residual_envelope(torch.arange(-1, 200), sched).diff().abs().max())
    bound = cfg.max_rate_rad + cfg.max_residual_rad * max_beta_rate
    assert float(applied.diff(dim=0).abs().max()) <= bound + 1e-7
    assert float(applied[0].abs().max()) <= cfg.max_rate_rad
    assert torch.all(applied[sched.fade_end_steps - 1:] == 0)


# ---------------------------------------------------------------- composition and joint indexing
def test_residual_joints_are_the_unlocked_leg_and_waist_joints():
    contract = _carry_contract()
    names = contract["ordered_raw_joint_names"]
    ids = residual_joint_indices(names)
    assert len(ids) == 15
    assert not set(ids) & set(contract["locked_raw_joint_indices"])
    assert {names[i].split("_")[1] for i in ids if names[i].startswith("left_")} == {"hip", "knee", "ankle"}
    assert sum("waist" in names[i] for i in ids) == 3


def test_zero_residual_is_bitwise_the_live_baseline():
    contract = _carry_contract()
    locked = contract["locked_raw_joint_indices"]
    ids = residual_joint_indices(contract["ordered_raw_joint_names"])
    torch.manual_seed(0)
    anchor, bc3 = torch.randn(4, 43), torch.randn(4, 43)
    scale = torch.full((43,), 0.25)
    for k in (0, 10, 49, 50, 120):
        w = fixed_blend_weight(k, 50)
        # evaluate_bc2_rollout.py: blend, then grasp lock
        live = anchor + w * (bc3 - anchor)
        live[:, locked] = anchor[:, locked]
        weight = blend_weight(torch.full((4,), k), ResidualScheduleCfg())
        offset = residual_raw_offset(torch.zeros(4, 15), ids, scale)
        torch.testing.assert_close(
            compose_residual_action(anchor, bc3, weight, locked_indices=locked, raw_offset=offset), live,
            rtol=0, atol=0,
        )


def test_residual_moves_only_leg_waist_targets_by_the_applied_radians():
    contract = _carry_contract()
    locked = contract["locked_raw_joint_indices"]
    ids = residual_joint_indices(contract["ordered_raw_joint_names"])
    anchor, bc3 = torch.randn(2, 43), torch.randn(2, 43)
    scale = torch.linspace(0.2, 0.6, 43)
    applied = torch.full((2, 15), 0.03)
    weight = torch.full((2,), 0.4)
    base = compose_residual_action(anchor, bc3, weight, locked_indices=locked)
    with_residual = compose_residual_action(
        anchor, bc3, weight, locked_indices=locked, raw_offset=residual_raw_offset(applied, ids, scale)
    )
    delta_rad = (with_residual - base) * scale
    torch.testing.assert_close(delta_rad[:, ids], applied)
    others = [i for i in range(43) if i not in ids]
    assert torch.all(delta_rad[:, others] == 0)
    torch.testing.assert_close(with_residual[:, locked], anchor[:, locked])


def test_residual_on_locked_joints_is_rejected():
    anchor = torch.zeros(1, 43)
    offset = torch.zeros(1, 43)
    offset[0, 30] = 0.1
    with pytest.raises(ValueError):
        compose_residual_action(anchor, anchor, torch.ones(1), locked_indices=[30], raw_offset=offset)


# ---------------------------------------------------------------- observation layout
def test_layout_dimensions_and_no_privileged_leakage():
    body, grip = actor_components("body"), actor_components("grip")
    assert layout_dim(body) == 72 and layout_dim(grip) == 79
    actor_names = {name for name, _ in grip}
    privileged = {name for name, _ in PRIVILEGED_COMPONENTS}
    assert not actor_names & privileged
    assert not any("force" in n or "friction" in n or "true" in n or "box_ang_vel" in n or "tilt" in n
                   for n in actor_names)
    assert {name for name, _ in BOX_ESTIMATE_COMPONENTS} <= actor_names
    assert not {name for name, _ in BOX_ESTIMATE_COMPONENTS} & {name for name, _ in body}
    assert critic_components("grip")[: len(grip)] == grip


def test_assemble_checks_widths():
    layout = (("a", 2), ("b", 1))
    out = assemble(layout, {"a": torch.ones(3, 2), "b": torch.zeros(3, 1)})
    assert out.shape == (3, 3)
    with pytest.raises(ValueError):
        assemble(layout, {"a": torch.ones(3, 3), "b": torch.zeros(3, 1)})


# ---------------------------------------------------------------- reward shaping
def test_soft_threshold_penalties():
    x = torch.tensor([0.0, 0.05, 0.10, 0.15, 0.30, -0.30])
    torch.testing.assert_close(excess(x, 0.10), torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0, 2.0]))
    torch.testing.assert_close(one_sided_excess(x, 0.10), torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0, 0.0]))
    with pytest.raises(ValueError):
        excess(x, 0.0)


def test_tilt_from_projected_gravity():
    g = torch.tensor([[0.0, 0.0, -1.0], [0.0, -0.0998334, -0.9950042]])
    torch.testing.assert_close(tilt_from_projected_gravity(g), torch.tensor([0.0, 0.1]), atol=1e-5, rtol=0)


# ---------------------------------------------------------------- USD cache fix
def test_unique_usd_dirs_only_for_converter_spawns_without_a_dir():
    urdf = SimpleNamespace(spawn=SimpleNamespace(asset_path="robot.urdf", usd_dir=None))
    other = SimpleNamespace(spawn=SimpleNamespace(asset_path="b.urdf", usd_dir=None))
    fixed = SimpleNamespace(spawn=SimpleNamespace(asset_path="c.urdf", usd_dir="/keep"))
    cuboid = SimpleNamespace(spawn=SimpleNamespace(size=(1, 1, 1)))
    scene = SimpleNamespace(robot=urdf, other=other, fixed=fixed, box=cuboid, sensor=None)
    assigned = assign_unique_usd_dirs(scene)
    assert set(assigned) == {"robot", "other"}
    assert urdf.spawn.usd_dir != other.spawn.usd_dir
    assert fixed.spawn.usd_dir == "/keep"
