"""Clearance-aware bimanual wrist paths for the Arena Pink WBC (phase 8, Methods A and B).

Both methods command world-frame ``*_wrist_yaw_link`` poses that the caller converts to the pelvis frame every step,
exactly like the Mimic Teacher (ArenaLiftTrajectory.high_level_action).  The hands are open while these paths run,
so the whole hand is rigid on the wrist and its collision geometry can be checked against the table for every
commanded pose (hand_clearance.OpenHandGeometry, convex hulls of the URDF meshes, about 2 mm samples).

  Method A, alignment     ``alignment_path``: from the measured post-navigation wrist poses to a goal pose (the
                          Teacher's first target) in a fixed time, min-jerk timing, then lifted where needed.
  Method B, safe Teacher  ``safe_teacher_path``: the retargeted source wrist path with the first steps blended from
                          the measured start poses, then lifted where needed, approach only (before the close onset).

"Lifted where needed": for every step the smallest vertical offset of the wrist target (0..max_lift in 5 mm steps)
that gives the whole open hand at least ``margin`` clearance to the table top and pedestal; the offsets are then
spread over neighbouring steps (look-ahead / look-back maximum, then a cosine ramp) so the hand rises before it
reaches the table edge and comes back down smoothly.  The lift is computed from the full finger geometry and the
real table boxes, not from the wrist height, and the result is re-checked (``min_clearance_after``).

Tracking-aware planning (``plan_tracked``): the Arena Pink/Homie WBC does not put the wrist where it is commanded.
In the source Arena demo the observed wrist is 4.3-4.8 cm below its command over most of the approach, and the same
sag appears in this simulator (phase-8 smoke test: 2.4-4.5 cm), so a commanded path with clearance can still hit the
table.  The clearance is therefore evaluated on the predicted realized hand path (the source's observed wrist path,
blended from the measured start), and the same lift is added to the commanded path.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from hand_clearance import OpenHandGeometry, SIDES


def minjerk(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def interpolate_poses(start: np.ndarray, goal: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """(4,4), (4,4), (T,) -> (T, 4, 4): linear position, slerp orientation."""
    rotations = Rotation.from_matrix(np.stack([start[:3, :3], goal[:3, :3]]))
    slerp = Slerp([0.0, 1.0], rotations)
    out = np.tile(np.eye(4), (len(fractions), 1, 1))
    out[:, :3, :3] = slerp(np.clip(fractions, 0.0, 1.0)).as_matrix()
    out[:, :3, 3] = (1.0 - fractions)[:, None] * start[:3, 3] + fractions[:, None] * goal[:3, 3]
    return out


def blend_into(start: np.ndarray, path: np.ndarray, blend_steps: int) -> np.ndarray:
    """Replace the first ``blend_steps`` poses by a min-jerk blend from ``start`` into the moving path."""
    out = path.copy()
    if blend_steps <= 0:
        return out
    rot_start = Rotation.from_matrix(start[:3, :3])
    for t in range(min(blend_steps, len(path))):
        s = float(minjerk(np.array([(t + 1) / blend_steps]))[0])
        out[t, :3, 3] = (1.0 - s) * start[:3, 3] + s * path[t, :3, 3]
        pair = Rotation.from_matrix(np.stack([rot_start.as_matrix(), path[t, :3, :3]]))
        out[t, :3, :3] = Slerp([0.0, 1.0], pair)(s).as_matrix()
    return out


class ClearanceLifter:
    def __init__(self, table_centers: np.ndarray, table_half: np.ndarray, geometry: OpenHandGeometry | None = None) -> None:
        self.geometry = geometry or OpenHandGeometry()
        self.centers = np.asarray(table_centers, dtype=np.float64)
        self.half = np.asarray(table_half, dtype=np.float64)

    def clearance(self, side: str, poses: np.ndarray):
        return self.geometry.clearance(side, poses, self.centers, self.half)

    def lift(self, paths: dict[str, np.ndarray], active: np.ndarray, margin: float, max_lift: float = 0.15,
             resolution: float = 0.005, lookahead: int = 8, lookback: int = 8, ramp: int = 10) -> tuple[dict, dict]:
        """Lift the wrist targets (same offset for both hands per step) so every active step keeps ``margin``.

        paths: {"left": (T,4,4), "right": (T,4,4)} world wrist poses; active: (T,) bool, steps that may be lifted.
        """
        length = len(active)
        margins = np.broadcast_to(np.asarray(margin, dtype=np.float64), (length,))
        candidates = np.arange(0.0, max_lift + 1.0e-9, resolution)
        need = np.zeros(length)
        before = {}
        index = np.flatnonzero(active)
        for side in SIDES:
            base = np.full(length, np.inf)
            if len(index):
                base[index] = self.clearance(side, paths[side][index])[0]
            before[side] = base
            todo = np.flatnonzero(active & (base < margins))
            for t in todo:
                trial = np.repeat(paths[side][t][None], len(candidates), axis=0)
                trial[:, 2, 3] += candidates
                d, _, _ = self.clearance(side, trial)
                ok = np.flatnonzero(d >= margins[t])
                need[t] = max(need[t], candidates[ok[0]] if len(ok) else max_lift)
        # spread: hold the largest nearby requirement, then smooth the steps with a cosine ramp
        held = np.array([need[max(0, t - lookback): t + lookahead + 1].max() for t in range(length)])
        if ramp > 1:
            kernel = 0.5 - 0.5 * np.cos(np.linspace(0.0, 2.0 * np.pi, 2 * ramp + 1))
            kernel /= kernel.sum()
            smooth = np.convolve(np.pad(held, ramp, mode="edge"), kernel, mode="valid")
            held = np.maximum(smooth, need)  # the smoothed profile never drops below a step's own requirement
        held[~active] = 0.0
        out = {side: paths[side].copy() for side in SIDES}
        for side in SIDES:
            out[side][:, 2, 3] += held
        after = {side: self.clearance(side, out[side][index]) if len(index) else None for side in SIDES}
        info = {
            "margin_m": [round(float(margins.max()), 4), round(float(margins.min()), 4)],
            "max_lift_m": round(float(held.max()), 4),
            "steps_lifted": int((held > 1.0e-6).sum()),
            "lift_profile_m": [round(float(v), 4) for v in held],
            "min_clearance_before_m": {s: round(float(before[s][index].min()), 4) if len(index) else None for s in SIDES},
            "min_clearance_after_m": {s: round(float(after[s][0].min()), 4) if len(index) else None for s in SIDES},
            "argmin_link_after": {s: after[s][1][int(np.argmin(after[s][0]))] if len(index) else None for s in SIDES},
        }
        return out, info


def alignment_path(start: dict, goal: dict, steps: int, lifter: ClearanceLifter, margin: float) -> tuple[dict, dict]:
    """Method A: min-jerk path from the measured wrist poses to the goal poses, lifted where needed."""
    fractions = minjerk(np.arange(1, steps + 1) / steps)
    paths = {side: interpolate_poses(start[side], goal[side], fractions) for side in SIDES}
    return lifter.lift(paths, np.ones(steps, dtype=bool), margin)


def safe_teacher_path(start: dict, retargeted: dict, close_index: int, blend_steps: int, lifter: ClearanceLifter,
                      margin: float) -> tuple[dict, dict]:
    """Method B: the retargeted Teacher wrist path, blended from the measured start and lifted before the close onset."""
    paths = {side: blend_into(start[side], retargeted[side], blend_steps) for side in SIDES}
    active = np.arange(len(paths["left"])) < close_index
    return lifter.lift(paths, active, margin)


def plan_tracked(start_cmd: dict, start_act: dict, base_cmd: dict, base_act: dict, blend_steps: int, active: np.ndarray,
                 lifter: ClearanceLifter, margin: float, hold_after: int | None = None) -> tuple[dict, dict, dict]:
    """Tracking-aware plan: lift computed on the predicted realized path, applied to the commanded path.

    start_cmd / start_act: current commanded and measured wrist poses (4x4, world).
    base_cmd / base_act:   (T, 4, 4) commanded path and the realized path it is expected to produce.
    hold_after:            from this step on keep the lift reached just before it (e.g. the close onset, so the hands
                           do not drop onto the table while closing); None = no lift after the active steps.
    Returns (commanded, predicted realized, info).
    """
    cmd = {side: blend_into(start_cmd[side], base_cmd[side], blend_steps) for side in SIDES}
    act = {side: blend_into(start_act[side], base_act[side], blend_steps) for side in SIDES}
    lifted_act, info = lifter.lift(act, active, margin)
    profile = np.asarray(info["lift_profile_m"])
    if hold_after is not None and 0 < hold_after < len(profile):
        profile[hold_after:] = profile[hold_after - 1]
        for side in SIDES:
            lifted_act[side] = act[side].copy()
            lifted_act[side][:, 2, 3] += profile
        info["lift_profile_m"] = [round(float(v), 4) for v in profile]
        info["held_lift_after_m"] = round(float(profile[hold_after - 1]), 4)
    out_cmd = {}
    for side in SIDES:
        out_cmd[side] = cmd[side].copy()
        out_cmd[side][:, 2, 3] += profile
    commanded_clearance = {side: lifter.clearance(side, out_cmd[side][active])[0] for side in SIDES} if active.any() else {}
    info["commanded_min_clearance_after_m"] = {side: round(float(v.min()), 4) for side, v in commanded_clearance.items()}
    return out_cmd, lifted_act, info


class WristServo:
    """Closed-loop correction of the commanded wrist poses so that the realized wrists follow a desired path.

    The Pink/Homie WBC realizes a commanded wrist pose with a posture-dependent error (gravity sag 2-5 cm, about 10 deg
    of wrist rotation after navigation), so an open-loop plan cannot guarantee clearance.  Every step the servo adds
    an integral correction:  command = R_off * desired (orientation),  desired + p_off (position), with
      p_off <- p_off + k_pos * (desired_prev - actual)          (norm clipped to max_pos)
      R_off <- exp(k_rot * log(desired_prev * actual^T)) R_off (angle clipped to max_rot)
    Initialized with the current command - measured offsets, so the first command equals the current one.
    """

    def __init__(self, start_cmd: dict, start_act: dict, k_pos: float, k_rot: float, max_pos: float = 0.10,
                 max_rot: float = 0.6) -> None:
        self.k_pos, self.k_rot, self.max_pos, self.max_rot = k_pos, k_rot, max_pos, max_rot
        self.p_off = {s: start_cmd[s][:3, 3] - start_act[s][:3, 3] for s in SIDES}
        self.r_off = {s: Rotation.from_matrix(start_cmd[s][:3, :3] @ start_act[s][:3, :3].T) for s in SIDES}
        self.previous = None
        self.frozen = False

    def command(self, desired: dict, actual: dict) -> dict:
        if self.previous is not None and not self.frozen:
            for s in SIDES:
                self.p_off[s] = self.p_off[s] + self.k_pos * (self.previous[s][:3, 3] - actual[s][:3, 3])
                norm = float(np.linalg.norm(self.p_off[s]))
                if norm > self.max_pos:
                    self.p_off[s] *= self.max_pos / norm
                error = Rotation.from_matrix(self.previous[s][:3, :3] @ actual[s][:3, :3].T).as_rotvec()
                updated = (Rotation.from_rotvec(self.k_rot * error) * self.r_off[s]).as_rotvec()
                angle = float(np.linalg.norm(updated))
                if angle > self.max_rot:
                    updated *= self.max_rot / angle
                self.r_off[s] = Rotation.from_rotvec(updated)
        out = {}
        for s in SIDES:
            pose = np.eye(4)
            pose[:3, :3] = self.r_off[s].as_matrix() @ desired[s][:3, :3]
            pose[:3, 3] = desired[s][:3, 3] + self.p_off[s]
            out[s] = pose
        self.previous = {s: desired[s].copy() for s in SIDES}
        return out

    def state(self) -> dict:
        return {s: {"p_off_cm": [round(100 * float(v), 2) for v in self.p_off[s]],
                    "r_off_deg": round(float(np.degrees(np.linalg.norm(self.r_off[s].as_rotvec()))), 2)} for s in SIDES}


def margin_schedule(length: int, early: float, late: float | None, end_step: int, late_steps: int, ramp: int = 20) -> np.ndarray:
    """Per-step margin: ``early``, then a linear ramp of ``ramp`` steps to ``late`` that is reached ``late_steps`` steps before
    ``end_step`` (the pre-grasp, where the hands move slowly) and kept afterwards."""
    margins = np.full(length, float(early))
    if late is None or late == early:
        return margins
    start_late = max(0, end_step - late_steps)
    start_ramp = max(0, start_late - ramp)
    for t in range(length):
        if t >= start_late:
            margins[t] = late
        elif t >= start_ramp:
            margins[t] = early + (late - early) * (t - start_ramp) / max(1, start_late - start_ramp)
    return margins
