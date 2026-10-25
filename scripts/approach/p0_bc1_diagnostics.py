#!/usr/bin/env python3
"""CPU-only P0 diagnostics for reproducing the narrow BC-1 (Approach) Student.

Read-only with respect to existing data: Teacher HDF5 files are opened in "r"
mode, checkpoints are loaded on the CPU, and every report is written under
``--output-dir`` (default ``outputs/approach_p0``, ignored by git).  Nothing
here launches Isaac Sim, uses the GPU, or trains a network.

Subcommands
  obs-stats          Rebuild the exact BC-1 training split and normalization,
                     then measure per-dimension spread and camera-FOV fingerprints.
  handoff-gate       Locate the FSM Approach->Grasp gate on Teacher trajectories
                     and compare it with the BC-2 training start.
  checkpoint         Inspect a phase-BC checkpoint (architecture, normalization).
  provenance-report  Summarize a JSONL written by run_fsm_with_provenance.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "data/teacher/box_xy_original_physx_500"
DEFAULT_BC1_METRICS = REPO_ROOT / "data/teacher/bc/bc1_approach_23d.pt.metrics.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/approach_p0"

# PolicyHistoryCfg in box_carry_with_hand_env_cfg.py.  Isaac Lab flattens each
# term's 5-frame history (oldest first) and then concatenates the terms; this
# was verified on the corpus (obs[t, slot j] == obs[t - 1, slot j + 1]).
HISTORY_LENGTH = 5
HISTORY_TERMS = (
    ("base_ang_vel", 3, 0.2),
    ("projected_gravity", 3, 1.0),
    ("joint_pos_rel", 43, 1.0),
    ("joint_vel_rel", 43, 0.05),
)
CURRENT_TERMS = (
    ("last_action", 43),
    ("velocity_commands", 3),
    ("box_visible", 1),
    ("box_position_camera", 3),
    ("bin_visible", 1),
    ("bin_position_camera", 3),
    ("table_position_camera", 3),
)
# Unoise ranges of PolicyHistoryCfg.  Isaac Lab applies noise before scale.
ENV_NOISE_HALF_RANGE = {
    "base_ang_vel": 0.2,
    "projected_gravity": 0.05,
    "joint_pos_rel": 0.01,
    "joint_vel_rel": 1.5,
}
QPOS_NEWEST = slice(30 + 4 * 43, 30 + 5 * 43)

# Camera geometry from box_carry_with_hand_env_cfg.py.
CAMERA_PITCH_DOWN_RAD = 2.0 * math.atan2(0.403545296, 0.914959668)
IMAGE_ASPECT = 120.0 / 160.0
CLIPPING_RANGE = (0.05, 8.0)
FOV_NARROW_DEG = 55.2  # forced by play_bc_skill_fsm.py without --randomize_work_area
FOV_DEFAULT_DEG = 69.4  # CAMERA_HORIZONTAL_FOV_DEG, used by every collector
BOX_HALF_HEIGHT_M = 0.10

QUATERNION_SLICES = (slice(5, 9), slice(12, 16))
STD_BANDS = (
    ("floored_to_scale_1", 0.0, 1.0e-6),
    ("fragile_1e-6_to_1e-3", 1.0e-6, 1.0e-3),
    ("small_1e-3_to_1e-2", 1.0e-3, 1.0e-2),
    ("normal_ge_1e-2", 1.0e-2, math.inf),
)


def _phase_bc():
    # Pure numpy/h5py/torch module; importing it does not start Isaac Sim.
    from unitree_rl_lab.utils.teacher import phase_bc

    return phase_bc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(values) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def _joint_group(name: str) -> str:
    if "_hand_" in name:
        return "finger"
    if "wrist" in name:
        return "wrist"
    if "shoulder" in name or "elbow" in name:
        return "arm"
    if "waist" in name:
        return "waist"
    return "leg"


def feature_layout(joint_names: list[str]) -> tuple[list[str], list[str], np.ndarray]:
    """Names, coarse groups, and obs-per-physical-unit scale of the 517-D input.

    Joint order is the processed action order stored in the Teacher HDF5; the
    observation uses the articulation order, which matches it for this asset
    (both are breadth-first and the near-constant finger dims line up).
    """

    names: list[str] = []
    groups: list[str] = []
    unit: list[float] = []
    for term, width, scale in HISTORY_TERMS:
        for lag in range(HISTORY_LENGTH - 1, -1, -1):
            for index in range(width):
                label = joint_names[index] if width == 43 else "xyz"[index]
                names.append(f"{term}[t-{lag}].{label}")
                groups.append(f"{term}:{_joint_group(label)}" if width == 43 else term)
                unit.append(scale)
    for term, width in CURRENT_TERMS:
        for index in range(width):
            if width == 43:
                names.append(f"{term}.{joint_names[index]}")
                groups.append(f"{term}:{_joint_group(joint_names[index])}")
            elif width == 1:
                names.append(term)
                groups.append(term)
            else:
                names.append(f"{term}.{'xyz'[index]}")
                groups.append(term)
            unit.append(1.0)
    if len(names) != 517:
        raise AssertionError(f"layout has {len(names)} entries, expected 517")
    return names, groups, np.asarray(unit, dtype=np.float64)


def inside_frustum(points_c: np.ndarray, fov_deg: float) -> np.ndarray:
    """Same pinhole test as observations._camera_visible."""
    tan_h = math.tan(math.radians(fov_deg) * 0.5)
    tan_v = tan_h * IMAGE_ASPECT
    x, y, z = points_c[..., 0], points_c[..., 1], points_c[..., 2]
    return (
        (x >= CLIPPING_RANGE[0])
        & (x <= CLIPPING_RANGE[1])
        & (np.abs(y) <= x * tan_h)
        & (np.abs(z) <= x * tan_v)
    )


def approximate_box_top_camera(center_c: np.ndarray) -> np.ndarray:
    """Box-top QR in camera coordinates, assuming an upright torso and box."""
    up_c = np.array(
        [-math.sin(CAMERA_PITCH_DOWN_RAD), 0.0, math.cos(CAMERA_PITCH_DOWN_RAD)]
    )
    return center_c + BOX_HALF_HEIGHT_M * up_c


def _uid_suffix(uid: str) -> str:
    path, episode = uid.rsplit("::", 1)
    return f"{Path(path).name}::{episode}"


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _write_reports(output_dir: Path, stem: str, report: dict, markdown: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"[p0] wrote {json_path}")
    print(f"[p0] wrote {md_path}")


def _load_approach_arrays(args):
    pbc = _phase_bc()
    # Keep the symlinked spelling: training UIDs embed the path string.
    corpus = Path(args.corpus).absolute()
    manifest_status = "ok"
    try:
        pbc.validate_complete_teacher_corpus([corpus])
    except Exception as exc:  # recorded, not fatal, for a diagnostic
        manifest_status = f"FAILED: {exc}"
    paths = pbc.expand_teacher_dataset_paths([corpus])
    episodes = pbc.discover_episodes(paths, success_only=True)
    split = pbc.split_episodes(episodes, ratios=(0.8, 0.1, 0.1), seed=args.seed)
    inference = pbc.PhaseInferenceConfig(
        hand_close_threshold=0.5, grasp_lead_steps=args.grasp_lead_steps
    )
    action_key = (
        "teacher_high_level_action" if args.skill == "approach" else "processed_43d_joint_target"
    )
    arrays = {
        subset: pbc.load_phase_arrays(
            episodes,
            split.ids(subset),
            skill=args.skill,
            action_key=action_key,
            inference_config=inference,
        )
        for subset in ("train", "validation", "test")
    }
    with h5py.File(paths[0], "r") as stream:
        joint_names = _decode(stream["metadata/processed_joint_names"][:])
        if action_key == "teacher_high_level_action":
            action_names = _decode(stream["metadata/teacher_high_level_action_names"][:])
        else:
            action_names = list(joint_names)
    return pbc, corpus, manifest_status, episodes, split, arrays, joint_names, action_names


def cmd_obs_stats(args) -> None:
    pbc, corpus, manifest_status, episodes, split, arrays, joint_names, action_names = (
        _load_approach_arrays(args)
    )
    train = arrays["train"]
    names, groups, unit = feature_layout(joint_names)
    report: dict = {
        "skill": args.skill,
        "corpus": str(corpus),
        "manifest_validation": manifest_status,
        "episodes": len(episodes),
        "split_seed": args.seed,
        "grasp_lead_steps": args.grasp_lead_steps,
        "sample_counts": {k: v.num_samples for k, v in arrays.items()},
        "episode_counts": {k: v.num_episodes for k, v in arrays.items()},
    }

    # 1. Does this reproduce the training split of the lab's bc1_approach_23d.pt?
    metrics_path = Path(args.metrics_json)
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        recorded = metrics.get("episode_split", {})
        report["checkpoint_metrics"] = str(metrics_path)
        report["split_matches_checkpoint_metrics"] = {
            subset: sorted(map(_uid_suffix, recorded.get(subset, [])))
            == sorted(map(_uid_suffix, split.ids(subset)))
            for subset in ("train", "validation", "test")
        }
        report["sample_counts_match_checkpoint_metrics"] = {
            subset: metrics.get("sample_counts", {}).get(subset)
            == arrays[subset].num_samples
            for subset in ("train", "validation", "test")
        }
        report["checkpoint_initial_checkpoint"] = metrics.get("initial_checkpoint")

    # 2. Exact normalization fitted by train_phase_policy (no initial checkpoint).
    obs_stats = pbc.NormalizationStats.fit(train.observations)
    act_stats = pbc.NormalizationStats.fit(train.actions)
    raw_std = train.observations.std(axis=0, dtype=np.float64)
    bands = {}
    for label, low, high in STD_BANDS:
        mask = (raw_std >= low) & (raw_std < high)
        bands[label] = {
            "count": int(mask.sum()),
            "by_group": dict(Counter(np.asarray(groups)[mask])),
        }
    report["observation_std_bands"] = bands
    fragile = np.flatnonzero((raw_std >= 1.0e-6) & (raw_std < 1.0e-3))
    report["fragile_dims"] = [
        {
            "index": int(i),
            "name": names[i],
            "std_obs": float(raw_std[i]),
            "std_physical": float(raw_std[i] / unit[i]),
            "delta_physical_for_z10": float(10.0 * obs_stats.scale[i] / unit[i]),
        }
        for i in fragile[np.argsort(raw_std[fragile])]
    ]
    floored = np.flatnonzero(raw_std < 1.0e-6)
    report["floored_dims"] = [{"index": int(i), "name": names[i]} for i in floored]

    # 3. In-distribution baseline: held-out episodes under the train statistics.
    baseline = {}
    for subset in ("validation", "test"):
        z = np.abs(obs_stats.normalize(arrays[subset].observations))
        worst_dims = np.argsort(-z.max(axis=0))[:5]
        baseline[subset] = {
            "per_sample_max_abs_z": _percentiles(z.max(axis=1)),
            "worst_dims": [[names[i], float(z[:, i].max())] for i in worst_dims],
        }
    report["held_out_z_baseline"] = baseline

    # 4. Analytic sensitivity: physical offsets on the finger channels.
    group_array = np.asarray(groups)
    finger_pos = np.flatnonzero(group_array == "joint_pos_rel:finger")
    finger_vel = np.flatnonzero(group_array == "joint_vel_rel:finger")
    sensitivity = {"finger_joint_position_offset_rad": {}, "finger_joint_velocity_offset_rad_s": {}}
    for delta in (1.0e-3, 1.0e-2, 5.0e-2):
        z = delta / obs_stats.scale[finger_pos]
        sensitivity["finger_joint_position_offset_rad"][str(delta)] = _percentiles(z)
    for delta in (1.0e-2, 1.0e-1, 1.0):
        z = delta * 0.05 / obs_stats.scale[finger_vel]
        sensitivity["finger_joint_velocity_offset_rad_s"][str(delta)] = _percentiles(z)
    noise = {}
    for term, half_range in ENV_NOISE_HALF_RANGE.items():
        idx = np.flatnonzero(np.char.startswith(np.asarray(names), f"{term}["))
        z = half_range * unit[idx] / obs_stats.scale[idx]
        noise[term] = _percentiles(z)
    sensitivity["env_observation_noise_if_corruption_enabled"] = noise
    report["sensitivity"] = sensitivity

    # 5. Camera-FOV fingerprint of the recorded corpus.
    all_obs = np.concatenate([a.observations for a in arrays.values()])
    bin_visible = all_obs[:, 510] > 0.5
    bin_c = all_obs[:, 511:514]
    box_visible = all_obs[:, 506] > 0.5
    box_top = approximate_box_top_camera(all_obs[:, 507:510])
    fov = {}
    for label, visible, points in (("bin_qr_exact", bin_visible, bin_c), ("box_top_qr_approx", box_visible, box_top)):
        in_narrow = inside_frustum(points, FOV_NARROW_DEG)
        in_default = inside_frustum(points, FOV_DEFAULT_DEG)
        fov[label] = {
            "visible_samples": int(visible.sum()),
            "visible_inside_55.2": int((visible & in_narrow).sum()),
            "visible_outside_55.2": int((visible & ~in_narrow).sum()),
            "visible_outside_69.4": int((visible & ~in_default).sum()),
        }
    fov["samples"] = int(all_obs.shape[0])
    fov["interpretation"] = (
        "visible samples outside the 55.2-deg frustum imply the corpus was "
        "recorded with a wider lens (69.4 deg); play_bc_skill_fsm.py masks them "
        "in the narrow workspace"
    )
    report["fov_fingerprint"] = fov

    # 6. What the narrow FSM would feed instead (bin/box masked outside 55.2 deg).
    masked = all_obs.copy()
    bin_hidden = bin_visible & ~inside_frustum(bin_c, FOV_NARROW_DEG)
    box_hidden = box_visible & ~inside_frustum(box_top, FOV_NARROW_DEG)
    masked[bin_hidden, 510:514] = 0.0
    masked[box_hidden, 506:510] = 0.0
    changed = bin_hidden | box_hidden
    z_masked = np.abs(obs_stats.normalize(masked[changed])) if changed.any() else np.zeros((0, 517))
    first = np.r_[True, np.diff(train.sample_episode_index) != 0]
    step0 = train.observations[first]
    step0_bin_hidden = (step0[:, 510] > 0.5) & ~inside_frustum(step0[:, 511:514], FOV_NARROW_DEG)
    report["narrow_fov_effect"] = {
        "approach_samples_changed": int(changed.sum()),
        "fraction_changed": float(changed.mean()),
        "bin_samples_hidden": int(bin_hidden.sum()),
        "box_samples_hidden_approx": int(box_hidden.sum()),
        "masked_sample_max_abs_z": _percentiles(z_masked.max(axis=1)) if changed.any() else {},
        "step0_episodes": int(step0.shape[0]),
        "step0_bin_visible_in_corpus": int((step0[:, 510] > 0.5).sum()),
        "step0_bin_hidden_under_55.2": int(step0_bin_hidden.sum()),
    }

    # 7. Action normalization: constant channels keep scale 1.0.
    act_std = train.actions.std(axis=0, dtype=np.float64)
    report["action_channels"] = [
        {
            "name": action_names[i],
            "mean": float(act_stats.mean[i]),
            "std": float(act_std[i]),
            "scale": float(act_stats.scale[i]),
            "constant_in_training": bool(act_std[i] < 1.0e-6),
        }
        for i in range(len(action_names))
    ]
    report["normalization"] = {
        "observation_mean": obs_stats.mean.tolist(),
        "observation_scale": obs_stats.scale.tolist(),
        "action_mean": act_stats.mean.tolist(),
        "action_scale": act_stats.scale.tolist(),
        "feature_names": names,
        "action_names": action_names,
    }

    md = [
        f"# P0 {args.skill} observation and normalization audit (CPU only)",
        "",
        f"- corpus: `{corpus}` ({len(episodes)} episodes), manifest: {manifest_status}",
        f"- {args.skill} samples: {report['sample_counts']} (grasp_lead_steps={args.grasp_lead_steps})",
    ]
    if "split_matches_checkpoint_metrics" in report:
        md.append(
            f"- split identical to `{metrics_path.name}`: {report['split_matches_checkpoint_metrics']}; "
            f"sample counts identical: {report['sample_counts_match_checkpoint_metrics']}"
        )
    md += ["", "## Observation std bands (train split)", "", "| band | dims | groups |", "|---|---:|---|"]
    for label, value in bands.items():
        md.append(f"| {label} | {value['count']} | {value['by_group']} |")
    md += ["", "## Held-out max |z| per sample (in-distribution baseline)", ""]
    for subset, value in baseline.items():
        md.append(f"- {subset}: {value['per_sample_max_abs_z']}, worst dims {value['worst_dims'][:3]}")
    md += ["", "## Analytic sensitivity of finger channels (|z| percentiles over finger dims)", ""]
    for key, value in sensitivity.items():
        md.append(f"- {key}: {json.dumps(value)}")
    md += ["", "## Camera FOV fingerprint", "", f"- {json.dumps(fov)}", f"- narrow-FSM effect: {json.dumps(report['narrow_fov_effect'])}"]
    md += ["", "## Action channels constant in training (normalization scale 1.0)", ""]
    for item in report["action_channels"]:
        if item["constant_in_training"]:
            md.append(f"- {item['name']}: mean {item['mean']:.4f}, scale {item['scale']}")
    stem = "bc1_observation_stats" if args.skill == "approach" else f"{args.skill}_observation_stats"
    _write_reports(Path(args.output_dir), stem, report, md)


def cmd_handoff_gate(args) -> None:
    pbc = _phase_bc()
    from unitree_rl_lab.utils.teacher.skill_fsm import (
        SkillPhase,
        bimanual_pregrasp_ready,
        bimanual_wrists_centered,
    )

    corpus = Path(args.corpus).absolute()
    episodes = pbc.discover_episodes(pbc.expand_teacher_dataset_paths([corpus]), success_only=True)
    inference = pbc.PhaseInferenceConfig(hand_close_threshold=0.5, grasp_lead_steps=args.grasp_lead_steps)
    by_path: dict[str, list] = {}
    for episode in episodes:
        by_path.setdefault(episode.path, []).append(episode)

    rows = []
    for path, group in by_path.items():
        with h5py.File(path, "r") as stream:
            joint_names = _decode(stream["metadata/processed_joint_names"][:])
            finger = np.asarray(["_hand_" in n for n in joint_names])
            arm = np.asarray([_joint_group(n) in ("arm", "wrist") for n in joint_names])
            for episode in group:
                g = stream["data"][episode.episode]
                labels = pbc.infer_bc_skill_labels(g, config=inference)
                bc2_start = int(np.flatnonzero(labels == int(SkillPhase.GRASP_LIFT))[0])
                high_level = g["teacher_high_level_action"][:]
                close = int(np.flatnonzero((high_level[:, :2] > 0.5).any(axis=1))[0])
                box = g["box_pose"][:, :3]
                left_tips = g["left_fingertip_positions"][:]
                right_tips = g["right_fingertip_positions"][:]
                left_wrist = g["left_wrist_pose"][:, :3]
                right_wrist = g["right_wrist_pose"][:, :3]
                ready = bimanual_pregrasp_ready(
                    box, left_tips, right_tips,
                    maximum_tip_distance_m=args.tip_limit,
                    maximum_cage_midpoint_error_m=args.cage_limit,
                ) & bimanual_wrists_centered(
                    box, left_wrist, right_wrist,
                    maximum_midpoint_error_m=args.wrist_midpoint_limit,
                    maximum_distance_asymmetry_m=args.asymmetry_limit,
                )
                gate = int(np.flatnonzero(ready)[0]) if ready.any() else None
                bilateral = g["left_contact"][:].astype(bool) & g["right_contact"][:].astype(bool)
                first_contact = int(np.flatnonzero(bilateral)[0]) if bilateral.any() else None
                observation = g["student_observation"][:]
                target = g["processed_43d_joint_target"][:]
                root = g["root_pose"][:, :3]

                def geometry(t: int) -> dict:
                    lt = np.linalg.norm(left_tips[t] - box[t], axis=-1)
                    rt = np.linalg.norm(right_tips[t] - box[t], axis=-1)
                    cage = 0.5 * (left_tips[t][np.argmin(lt)] + right_tips[t][np.argmin(rt)])
                    lw = np.linalg.norm(left_wrist[t] - box[t])
                    rw = np.linalg.norm(right_wrist[t] - box[t])
                    return {
                        "left_tip_m": float(lt.min()),
                        "right_tip_m": float(rt.min()),
                        "cage_midpoint_m": float(np.linalg.norm(cage - box[t])),
                        "wrist_midpoint_m": float(np.linalg.norm(0.5 * (left_wrist[t] + right_wrist[t]) - box[t])),
                        "wrist_asymmetry_m": float(abs(lw - rw)),
                    }

                row = {
                    "episode": f"{Path(path).name}::{episode.episode}",
                    "gate_step": gate,
                    "bc2_label_start": bc2_start,
                    "close_index": close,
                    "first_bilateral_contact": first_contact,
                    "bc2_start_geometry": geometry(bc2_start),
                }
                if gate is not None:
                    # Post-step state at the gate equals the pre-step observation at gate + 1.
                    fsm_first_bc2 = min(gate + 1, observation.shape[0] - 1)
                    row["gate_geometry"] = geometry(gate)
                    row["fsm_first_bc2_step"] = fsm_first_bc2
                    row["steps_before_bc2_training_start"] = bc2_start - fsm_first_bc2
                    arm_delta = observation[fsm_first_bc2, QPOS_NEWEST][arm] - observation[bc2_start, QPOS_NEWEST][arm]
                    row["arm_joint_max_abs_diff_rad"] = float(np.abs(arm_delta).max())
                    row["root_x_at_gate_minus_bc2_start_m"] = float(root[gate, 0] - root[bc2_start, 0])
                    row["teacher_velocity_cmd_at_gate"] = high_level[gate, 16:19].tolist()
                if first_contact is not None:
                    q_open = target[close - 1, finger]
                    q_closed = target[-1, finger]
                    motion = q_closed - q_open
                    moving = np.abs(motion) > 0.05
                    if moving.any():
                        row["target_closure_at_first_contact"] = float(
                            np.mean((target[first_contact, finger][moving] - q_open[moving]) / motion[moving])
                        )
                        actual_open = observation[close, QPOS_NEWEST][finger]
                        actual_closed = observation[-1, QPOS_NEWEST][finger]
                        actual_motion = actual_closed - actual_open
                        actual_moving = np.abs(actual_motion) > 0.05
                        contact_obs = min(first_contact + 1, observation.shape[0] - 1)
                        if actual_moving.any():
                            row["actual_closure_at_first_contact"] = float(
                                np.mean(
                                    (observation[contact_obs, QPOS_NEWEST][finger][actual_moving] - actual_open[actual_moving])
                                    / actual_motion[actual_moving]
                                )
                            )
                    row["contact_steps_after_close"] = first_contact - close
                rows.append(row)

    def column(key):
        return np.asarray([r[key] for r in rows if r.get(key) is not None], dtype=np.float64)

    blend = {
        str(k): float(((k + 1) / args.grasp_handoff_steps) ** 2 * (3.0 - 2.0 * (k + 1) / args.grasp_handoff_steps))
        for k in (0, 10, 20, 30, 41, 50, 60, 70, 80, 99)
    }
    report = {
        "corpus": str(corpus),
        "gate_thresholds_m": {
            "tip": args.tip_limit,
            "cage_midpoint": args.cage_limit,
            "wrist_midpoint": args.wrist_midpoint_limit,
            "wrist_asymmetry": args.asymmetry_limit,
        },
        "episodes": len(rows),
        "gate_never_fires": sum(r["gate_step"] is None for r in rows),
        "gate_step": _percentiles(column("gate_step")),
        "bc2_label_start": _percentiles(column("bc2_label_start")),
        "close_index": _percentiles(column("close_index")),
        "first_bilateral_contact": _percentiles(column("first_bilateral_contact")),
        "steps_before_bc2_training_start": _percentiles(column("steps_before_bc2_training_start")),
        "arm_joint_max_abs_diff_rad": _percentiles(column("arm_joint_max_abs_diff_rad")),
        "root_x_at_gate_minus_bc2_start_m": _percentiles(column("root_x_at_gate_minus_bc2_start_m")),
        "contact_steps_after_close": _percentiles(column("contact_steps_after_close")),
        "target_closure_at_first_contact": _percentiles(column("target_closure_at_first_contact")),
        "actual_closure_at_first_contact": _percentiles(column("actual_closure_at_first_contact")),
        "fsm_smoothstep_blend_by_grasp_step": blend,
        "rows": rows,
    }
    md = [
        "# P0 Approach->Grasp gate on Teacher trajectories (CPU only)",
        "",
        f"- episodes: {len(rows)}, gate never fires: {report['gate_never_fires']}",
        f"- FSM gate step (post-step index): {report['gate_step']}",
        f"- BC-2 training start (label): {report['bc2_label_start']}; close command: {report['close_index']}",
        f"- BC-2 would act this many steps before its training start: {report['steps_before_bc2_training_start']}",
        f"- arm joint max |diff| gate vs BC-2 start (rad): {report['arm_joint_max_abs_diff_rad']}",
        f"- first bilateral contact: {report['first_bilateral_contact']} ({report['contact_steps_after_close']} steps after close)",
        f"- finger target closure at first contact: {report['target_closure_at_first_contact']}",
        f"- finger actual closure at first contact: {report['actual_closure_at_first_contact']}",
        f"- FSM smoothstep blend vs GRASP step k: {blend}",
    ]
    _write_reports(Path(args.output_dir), "bc1_bc2_handoff_gate", report, md)


def _write_synthetic_checkpoint(directory: Path) -> Path:
    """Format fixture for the self-test: random weights, no training."""
    import torch

    pbc = _phase_bc()
    from unitree_rl_lab.utils.teacher_demo_recorder import DEFAULT_HIGH_LEVEL_ACTION_NAMES

    names = list(DEFAULT_HIGH_LEVEL_ACTION_NAMES)
    obs_stats = pbc.NormalizationStats(np.zeros(517, np.float32), np.ones(517, np.float32))
    act_stats = pbc.NormalizationStats(np.zeros(23, np.float32), np.ones(23, np.float32))
    policy = pbc.PhaseBCPolicy(action_dim=23, hidden_dims=(512, 512, 256), observation_stats=obs_stats, action_stats=act_stats)
    metadata = {
        "format": pbc.BC_CHECKPOINT_FORMAT,
        "skill_phase": "approach",
        "action_key": "teacher_high_level_action",
        "action_dim": 23,
        "hidden_dims": [512, 512, 256],
        "ordered_action_names": names,
        "action_contract_sha256": pbc.action_contract_sha256("teacher_high_level_action", names),
        "student_observation_dim": 517,
        "student_input_datasets": ["student_observation"],
        "privileged_student_inputs": [],
        "synthetic_self_test_fixture": True,
    }
    path = directory / "synthetic_bc1_fixture.pt"
    torch.save(
        {
            "metadata": metadata,
            "model_state_dict": policy.state_dict(),
            "observation_normalization": obs_stats.as_dict(),
            "action_normalization": act_stats.as_dict(),
        },
        path,
    )
    return path


def cmd_checkpoint(args) -> None:
    pbc = _phase_bc()
    with tempfile.TemporaryDirectory() as scratch:
        path = _write_synthetic_checkpoint(Path(scratch)) if args.self_test else Path(args.path)
        # weights_only=False inside _load_checkpoint: load only trusted files.
        checkpoint = pbc._load_checkpoint(path, "cpu")
        policy = pbc.load_bc_policy(path, "cpu")
        sha = _sha256(path)
        size = path.stat().st_size
    metadata = checkpoint["metadata"]
    obs = pbc.NormalizationStats.from_dict(checkpoint["observation_normalization"])
    act = pbc.NormalizationStats.from_dict(checkpoint["action_normalization"])
    joint_names = None
    teacher_files = sorted(Path(args.corpus).absolute().glob("shard_*.hdf5"))
    teacher_files = [p for p in teacher_files if not p.name.endswith(".snapshots.hdf5")]
    if teacher_files:
        with h5py.File(teacher_files[0], "r") as stream:
            joint_names = _decode(stream["metadata/processed_joint_names"][:])
    names, groups, unit = feature_layout(joint_names or [f"joint_{i}" for i in range(43)])
    small = np.flatnonzero(obs.scale < 1.0e-3)
    report = {
        "path": str(path),
        "self_test_fixture": bool(args.self_test),
        "size_bytes": size,
        "sha256": sha,
        "architecture": [[k, list(v.shape)] for k, v in checkpoint["model_state_dict"].items()],
        "metadata": {k: metadata.get(k) for k in (
            "format", "skill_phase", "action_key", "action_dim", "hidden_dims", "best_epoch",
            "initial_checkpoint", "initial_checkpoint_sha256", "train_config", "dagger_loss_weight",
            "student_observation_schema", "source_paths", "episode_counts", "sample_counts",
        )},
        "ordered_action_names": metadata.get("ordered_action_names"),
        "observation_scale_summary": _percentiles(obs.scale.astype(np.float64)),
        "observation_dims_scale_below_1e-3": {
            "count": int(small.size),
            "by_group": dict(Counter(np.asarray(groups)[small])),
        },
        "observation_dims_scale_exactly_1": int(np.sum(obs.scale == 1.0)),
        "action_scale": dict(zip(metadata.get("ordered_action_names", []), map(float, act.scale))),
        "runtime_policy_loads": policy.action_dim == int(metadata["action_dim"]),
    }
    if args.reference_stats:
        reference = json.loads(Path(args.reference_stats).read_text(encoding="utf-8"))["normalization"]
        report["matches_reconstructed_normalization"] = {
            "observation_mean_max_abs_diff": float(np.abs(obs.mean - np.asarray(reference["observation_mean"])).max()),
            "observation_scale_max_abs_diff": float(np.abs(obs.scale - np.asarray(reference["observation_scale"])).max()),
            "action_scale_max_abs_diff": float(np.abs(act.scale - np.asarray(reference["action_scale"])).max()),
        }
    md = [
        "# P0 checkpoint inspection (CPU only)",
        "",
        f"- path: `{report['path']}` ({size} bytes, sha256 {sha})",
        f"- self-test fixture: {report['self_test_fixture']}",
        f"- architecture: {report['architecture']}",
        f"- metadata: {json.dumps(report['metadata'])[:1500]}",
        f"- observation dims with scale < 1e-3: {report['observation_dims_scale_below_1e-3']}",
        f"- action scale: {report['action_scale']}",
    ]
    if "matches_reconstructed_normalization" in report:
        md.append(f"- vs reconstructed normalization: {report['matches_reconstructed_normalization']}")
    stem = "checkpoint_self_test" if args.self_test else f"checkpoint_{path.stem}"
    _write_reports(Path(args.output_dir), stem, report, md)


def _normalize_quaternions(action: np.ndarray) -> np.ndarray:
    result = action.copy()
    for part in QUATERNION_SLICES:
        norm = np.linalg.norm(result[part])
        if norm > 1.0e-8:
            result[part] = result[part] / norm
    return result


def summarize_provenance(records: list[dict], tolerance: float = 1.0e-4) -> dict:
    """Pair each APPROACH-policy call with the next WBC bridge request.

    In the APPROACH phase play_bc_skill_fsm.py calls the policy and then the
    bridge.  With --grasp_live_approach_lower_body it also calls the APPROACH
    policy during GRASP_LIFT, followed by a bridge call and then the GRASP_LIFT
    policy; those steps are reported separately.
    """

    records = sorted(records, key=lambda r: r["seq"])
    steps = []
    previous_sent = None
    for index, record in enumerate(records):
        if record.get("kind") != "policy" or record.get("skill") != "APPROACH":
            continue
        if index + 1 >= len(records) or records[index + 1].get("kind") != "bridge":
            continue
        follower = records[index + 2] if index + 2 < len(records) else None
        phase = (
            "grasp_live_lower_body"
            if follower and follower.get("kind") == "policy" and follower.get("skill") == "GRASP_LIFT"
            else "approach"
        )
        output = _normalize_quaternions(np.asarray(record["output"], dtype=np.float64))
        sent = np.asarray(records[index + 1]["high_level_action"], dtype=np.float64)
        diff = np.abs(sent - output)
        changed = np.flatnonzero(diff > tolerance)
        # play_bc_skill_fsm.py latches the last sent command with zero base
        # velocity once the physical pre-grasp gate is met.
        keep = np.r_[0:16, 19:23]
        latched = (
            previous_sent is not None
            and np.allclose(sent[keep], previous_sent[keep], atol=1.0e-7)
            and np.allclose(sent[16:19], 0.0, atol=1.0e-7)
        )
        steps.append({
            "seq": record["seq"],
            "phase": phase,
            "overwritten_dims": changed.tolist(),
            "max_abs_diff": float(diff.max()),
            "latched_after_ready": bool(latched and changed.size),
            "max_abs_z": record.get("max_abs_z"),
            "top_z": record.get("top_z"),
        })
        if phase == "approach":
            previous_sent = sent
    approach = [s for s in steps if s["phase"] == "approach"]
    pre_latch = [s for s in approach if not s["latched_after_ready"]]
    dims = Counter(d for s in pre_latch for d in s["overwritten_dims"])
    z = np.asarray([s["max_abs_z"] for s in approach if s["max_abs_z"] is not None], dtype=np.float64)
    return {
        "approach_steps": len(approach),
        "approach_steps_student_intact": sum(not s["overwritten_dims"] for s in approach),
        "approach_steps_latched_after_ready": sum(s["latched_after_ready"] for s in approach),
        "approach_steps_overwritten_before_latch": sum(bool(s["overwritten_dims"]) for s in pre_latch),
        "overwritten_dim_counts_before_latch": {str(k): v for k, v in sorted(dims.items())},
        "grasp_phase_approach_calls": sum(s["phase"] == "grasp_live_lower_body" for s in steps),
        "approach_max_abs_z": _percentiles(z),
        "approach_first10_max_abs_z": z[:10].tolist(),
        "approach_steps_with_z_above_10": int((z > 10).sum()),
        "approach_steps_with_z_above_100": int((z > 100).sum()),
        "steps": steps,
    }


def _provenance_self_test() -> list[dict]:
    rng = np.random.default_rng(0)
    base = rng.normal(size=23)
    for part in QUATERNION_SLICES:
        base[part] /= np.linalg.norm(base[part])
    intact = base.copy()
    passthrough = base.copy()
    passthrough[16:23] = [0.2, 0.0, 0.0, 0.75, 0.0, -0.4, 0.0]
    latched = base.copy()
    latched[16:19] = 0.0
    seq = iter(range(100))

    def policy(skill, output, z=1.0):
        return {"seq": next(seq), "kind": "policy", "skill": skill, "output": list(output), "max_abs_z": z, "top_z": []}

    def bridge(action):
        return {"seq": next(seq), "kind": "bridge", "high_level_action": list(action)}

    return [
        policy("APPROACH", base, 2.0), bridge(intact),            # intact
        policy("APPROACH", base, 3.0), bridge(passthrough),       # overwritten 16:23
        policy("APPROACH", base, 150.0), bridge(intact),          # intact, large |z|
        policy("APPROACH", base, 4.0), bridge(latched),           # first latched frame
        policy("APPROACH", base, 5.0), bridge(latched),           # second latched frame
        policy("APPROACH", base, 6.0), bridge(intact),            # GRASP live legs
        policy("GRASP_LIFT", np.zeros(43)),
    ]


def cmd_provenance_report(args) -> None:
    if args.self_test:
        records = _provenance_self_test()
        summary = summarize_provenance(records)
        checks = {
            "approach_steps == 5": summary["approach_steps"] == 5,
            "two intact steps": summary["approach_steps_student_intact"] == 2,
            "grasp-phase call separated": summary["grasp_phase_approach_calls"] == 1,
            "only the passthrough step is overwritten before latch": summary["approach_steps_overwritten_before_latch"] == 1,
            "passthrough dims 16..22 detected": sorted(summary["overwritten_dim_counts_before_latch"]) == [str(d) for d in range(16, 23)],
            "both latched frames recognized": summary["approach_steps_latched_after_ready"] == 2,
            "z above 100 counted": summary["approach_steps_with_z_above_100"] == 1,
        }
        print(json.dumps({"self_test_checks": checks}, indent=2))
        if not all(checks.values()):
            raise SystemExit("provenance self-test FAILED")
        print("[p0] provenance-report self-test passed")
        return
    records = [json.loads(line) for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarize_provenance(records)
    md = [
        "# BC-1 action provenance (from run_fsm_with_provenance.py)",
        "",
        f"- approach steps: {summary['approach_steps']}, intact: {summary['approach_steps_student_intact']}, "
        f"latched after ready: {summary['approach_steps_latched_after_ready']}, "
        f"overwritten before latch: {summary['approach_steps_overwritten_before_latch']}",
        f"- overwritten dims before latch: {summary['overwritten_dim_counts_before_latch']}",
        f"- approach max |z|: {summary['approach_max_abs_z']}; first 10 steps: {summary['approach_first10_max_abs_z']}",
        f"- steps with |z| > 10: {summary['approach_steps_with_z_above_10']}, > 100: {summary['approach_steps_with_z_above_100']}",
    ]
    _write_reports(Path(args.output_dir), Path(args.jsonl).stem + ".summary", summary, md)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
        p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
        p.add_argument("--grasp-lead-steps", type=int, default=2)

    p = sub.add_parser("obs-stats")
    common(p)
    p.add_argument("--skill", choices=("approach", "grasp_lift"), default="approach")
    p.add_argument("--metrics-json", default=str(DEFAULT_BC1_METRICS))
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_obs_stats)

    p = sub.add_parser("handoff-gate")
    common(p)
    p.add_argument("--tip-limit", type=float, default=0.18)
    p.add_argument("--cage-limit", type=float, default=0.11)
    p.add_argument("--wrist-midpoint-limit", type=float, default=0.222)
    p.add_argument("--asymmetry-limit", type=float, default=0.10)
    p.add_argument("--grasp-handoff-steps", type=int, default=100)
    p.set_defaults(func=cmd_handoff_gate)

    p = sub.add_parser("checkpoint")
    common(p)
    p.add_argument("--path")
    p.add_argument("--reference-stats", help="bc1_observation_stats.json from obs-stats")
    p.add_argument("--self-test", action="store_true", help="inspect a random-weight format fixture")
    p.set_defaults(func=cmd_checkpoint)

    p = sub.add_parser("provenance-report")
    common(p)
    p.add_argument("--jsonl")
    p.add_argument("--self-test", action="store_true")
    p.set_defaults(func=cmd_provenance_report)

    args = parser.parse_args()
    if args.command == "checkpoint" and not (args.self_test or args.path):
        parser.error("checkpoint requires --path or --self-test")
    if args.command == "provenance-report" and not (args.self_test or args.jsonl):
        parser.error("provenance-report requires --jsonl or --self-test")
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
