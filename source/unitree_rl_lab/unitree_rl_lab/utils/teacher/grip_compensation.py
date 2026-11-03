"""Payload-aware grip compensation for the locked BC3 arm/hand targets (no retraining).

During CARRY the 28 arm/hand joints are locked to the grasp targets realized at
the end of BC2, which hold the 0.2 kg training Box.  With a heavier payload the
arms sag under the extra weight and the Box slips between the hands.  This
layer adds, at each wrist, a quasi-static force

    F = [0, 0, dm g / 2] + s * (dm g / 2) * u_inward

(upward support plus an inward squeeze toward the other wrist, ``dm`` = extra
payload over the nominal mass) and converts it to arm joint target offsets for
the joint PD controllers: ``dq = J_wrist^T F / k``.  ``s >= 1/mu`` supplies the
extra friction the heavier Box needs.  The payload mass comes from simulation
ground truth; a real robot would need an estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


GRAVITY = 9.81
ARM_TOKENS = ("shoulder", "elbow", "wrist")


@dataclass(frozen=True)
class GripCompensationCfg:
    nominal_payload_kg: float = 0.2
    squeeze_factor: float = 1.0
    max_joint_offset_rad: float = 0.3

    def validate(self) -> "GripCompensationCfg":
        if (not math.isfinite(self.nominal_payload_kg) or self.nominal_payload_kg < 0
                or not math.isfinite(self.squeeze_factor) or self.squeeze_factor < 0
                or not math.isfinite(self.max_joint_offset_rad) or self.max_joint_offset_rad <= 0):
            raise ValueError("invalid grip compensation parameters")
        return self


def wrist_forces(extra_mass_kg: float, left_wrist: torch.Tensor, right_wrist: torch.Tensor,
                 squeeze_factor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """World-frame forces (3,) to add at the left and right wrists."""

    share = max(0.0, extra_mass_kg) * GRAVITY / 2.0
    up = torch.tensor([0.0, 0.0, share], dtype=left_wrist.dtype, device=left_wrist.device)
    inward = right_wrist - left_wrist
    inward = inward / torch.linalg.vector_norm(inward).clamp_min(1e-6)
    squeeze = squeeze_factor * share
    return up + squeeze * inward, up - squeeze * inward


def joint_offsets(jacobian: torch.Tensor, force: torch.Tensor, stiffness: torch.Tensor,
                  max_offset: float) -> torch.Tensor:
    """``J^T F / k`` for a (3, J) linear wrist Jacobian and (J,) PD stiffness."""

    torque = jacobian.T @ force
    return (torque / stiffness.clamp_min(1e-3)).clamp(-max_offset, max_offset)


class GripCompensator:
    """Single-env runtime: offsets for the locked arm targets, recomputed every step."""

    def __init__(self, env, cfg: GripCompensationCfg | None = None):
        if env.num_envs != 1:
            raise ValueError("grip compensation evaluation currently supports one environment")
        self.cfg = (cfg or GripCompensationCfg()).validate()
        self.env, self.robot, self.box = env, env.scene["robot"], env.scene["box"]
        term = env.action_manager.get_term("JointPositionAction")
        self.scale = torch.as_tensor(term._scale, device=env.device).expand(43)
        joint_ids = term._joint_ids
        joint_ids = list(range(self.robot.num_joints)) if isinstance(joint_ids, slice) else list(joint_ids)
        names = list(term._joint_names)
        self.payload = float(self.box.root_physx_view.get_masses()[0].sum())
        self.extra_mass = max(0.0, self.payload - self.cfg.nominal_payload_kg)
        stiffness = torch.zeros(self.robot.num_joints, device=env.device)
        for actuator in self.robot.actuators.values():
            indices = actuator.joint_indices
            indices = (list(range(self.robot.num_joints)) if isinstance(indices, slice)
                       else [int(i) for i in indices])
            stiffness[indices] = torch.as_tensor(actuator.stiffness, device=env.device).reshape(-1, len(indices))[0]
        self.sides = []
        for side in ("left", "right"):
            action_ids = [i for i, n in enumerate(names) if n.startswith(side) and any(t in n for t in ARM_TOKENS)]
            robot_ids = [joint_ids[i] for i in action_ids]
            wrist = self.robot.find_bodies([f"{side}_wrist_yaw_link"], preserve_order=True)[0][0]
            self.sides.append((action_ids, robot_ids, wrist, stiffness[robot_ids]))
        if any(not ids for ids, *_ in self.sides):
            raise ValueError("could not resolve the arm joints")

    def offsets(self) -> torch.Tensor:
        """(43,) raw-action offsets for the arm joints (zero at the nominal payload)."""

        result = torch.zeros(43, device=self.env.device)
        if self.extra_mass <= 0.0:
            return result
        wrists = self.robot.data.body_link_pos_w[0]
        forces = wrist_forces(self.extra_mass, wrists[self.sides[0][2]], wrists[self.sides[1][2]],
                              self.cfg.squeeze_factor)
        jacobians = self.robot.root_physx_view.get_jacobians()[0].to(self.env.device)
        for (action_ids, robot_ids, wrist, stiffness), force in zip(self.sides, forces):
            jac = jacobians[wrist, :3][:, [6 + j for j in robot_ids]]
            dq = joint_offsets(jac, force, stiffness, self.cfg.max_joint_offset_rad)
            result[action_ids] = dq / self.scale[action_ids]
        return result
