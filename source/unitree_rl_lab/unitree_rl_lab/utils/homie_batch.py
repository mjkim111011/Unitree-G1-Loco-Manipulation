"""Batched torch port of the frozen Arena Homie V2 lower-body policy.

The Arena WBC worker runs Homie through onnxruntime for a single robot.  This
module re-implements the exact ONNX graph with the original weights so that
many robots can run Homie on the GPU inside one Isaac Lab environment:

    est    = MLP_est(obs516)                                  516 -> 256 -> 256 -> 35 (ELU)
    vel    = est[:, :3]
    latent = est[:, 3:] / max(||est[:, 3:]||_2, 1e-12)
    action = MLP_actor([obs516[:, -86:], vel, latent])        121 -> 512 -> 256 -> 256 -> 15 (ELU)

``G1HomiePolicyV2`` (Arena) runs the ``stand`` network when ||[vx, vy, wz]|| < 0.05 and the ``walk``
network otherwise; :class:`HomieV2Pair` does the same per robot.  :class:`HomieBatch` reproduces
``G1HomiePolicyV2.compute_observation``/``set_observation``/``get_action`` for N robots.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CANDIDATE_MODEL_DIRS = (
    os.environ.get("UNITREE_HOMIE_V2_DIR", ""),
    str(_REPO_ROOT / "outputs" / "approach_p1" / "assets" / "homie_v2"),
    "/tmp/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/wbc_policy/models/homie_v2",
)
STAND_THRESHOLD = 0.05

# Arena G1 robot model (instantiate_g1_robot_model(waist_location="lower_body")), verified 2026-10-19.
BODY_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
LOWER_BODY_JOINTS = BODY_JOINTS[:15]  # Homie's 15 actions: legs then waist
DEFAULT_ANGLES = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0)  # g1_homie_v2.yaml
CMD_SCALE = (2.0, 2.0, 0.5)
ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE, ACTION_SCALE, HISTORY, FRAME = 0.5, 1.0, 0.05, 0.25, 6, 86


def default_model_dir() -> Path:
    for candidate in _CANDIDATE_MODEL_DIRS:
        if candidate and (Path(candidate) / "walk.onnx").is_file() and (Path(candidate) / "stand.onnx").is_file():
            return Path(candidate)
    raise FileNotFoundError(f"Homie V2 ONNX files not found in {[c for c in _CANDIDATE_MODEL_DIRS if c]}")


def _onnx_weights(path: Path) -> dict[str, np.ndarray]:
    import onnx
    from onnx import numpy_helper

    return {i.name: numpy_helper.to_array(i) for i in onnx.load(str(path)).graph.initializer}


def _mlp(weights: dict[str, np.ndarray], prefix: str, layers: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    for k in range(layers):
        w, b = weights[f"{prefix}.{2 * k}.weight"], weights[f"{prefix}.{2 * k}.bias"]
        linear = nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data = torch.from_numpy(w.copy())
        linear.bias.data = torch.from_numpy(b.copy())
        modules.append(linear)
        if k < layers - 1:
            modules.append(nn.ELU(alpha=1.0))
    return nn.Sequential(*modules)


class HomieV2Net(nn.Module):
    """One Homie V2 network (stand or walk) with the exact ONNX computation."""

    def __init__(self, onnx_path: Path) -> None:
        super().__init__()
        weights = _onnx_weights(onnx_path)
        self.estimator = _mlp(weights, "estimator", 3)
        self.actor = _mlp(weights, "actor", 4)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        est = self.estimator(obs)
        latent = est[:, 3:]
        latent = latent / torch.clamp(torch.linalg.vector_norm(latent, dim=-1, keepdim=True), min=1.0e-12)
        return self.actor(torch.cat((obs[:, -FRAME:], est[:, :3], latent), dim=-1))


class HomieV2Pair(nn.Module):
    """Stand/walk pair with Homie's per-robot switching on the command norm."""

    def __init__(self, model_dir: Path | None = None) -> None:
        super().__init__()
        model_dir = Path(model_dir) if model_dir is not None else default_model_dir()
        self.stand = HomieV2Net(model_dir / "stand.onnx")
        self.walk = HomieV2Net(model_dir / "walk.onnx")

    def forward(self, obs: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        standing = torch.linalg.vector_norm(command, dim=-1, keepdim=True) < STAND_THRESHOLD
        return torch.where(standing, self.stand(obs), self.walk(obs))


class HomieBatch:
    """Batched G1HomiePolicyV2: 86-D frame x 6 history, stand/walk switching, 15 lower-body targets.

    Frame = [cmd * (2, 2, 0.5), height cmd, torso rpy cmd, base ang vel (body) * 0.5, projected gravity,
             (q_body - padded defaults), dq_body * 0.05, last network action]; the arm defaults are padded
             with zeros exactly as in the Arena code.  ``q_body``/``dq_body`` are in BODY_JOINTS order.
    The history starts from zeros after a reset, as ``G1HomiePolicyV2.reset``/``set_observation`` do.
    """

    def __init__(self, num_envs: int, device: str | torch.device = "cpu", model_dir: Path | None = None) -> None:
        self.device = torch.device(device)
        self.net = HomieV2Pair(model_dir).to(self.device).eval()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        self.padded_defaults = torch.zeros(len(BODY_JOINTS), device=self.device)
        self.padded_defaults[:15] = torch.tensor(DEFAULT_ANGLES, device=self.device)
        self.cmd_scale = torch.tensor(CMD_SCALE, device=self.device)
        self.history = torch.zeros(num_envs, HISTORY, FRAME, device=self.device)
        self.last_action = torch.zeros(num_envs, 15, device=self.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.history[ids] = 0.0
        self.last_action[ids] = 0.0

    def frame(self, command, height, rpy, ang_vel_b, gravity_b, q_body, dq_body) -> torch.Tensor:
        return torch.cat((
            command * self.cmd_scale, height.reshape(-1, 1), rpy, ang_vel_b * ANG_VEL_SCALE, gravity_b,
            (q_body - self.padded_defaults) * DOF_POS_SCALE, dq_body * DOF_VEL_SCALE, self.last_action,
        ), dim=-1)

    @torch.no_grad()
    def step(self, command, height, rpy, ang_vel_b, gravity_b, q_body, dq_body) -> torch.Tensor:
        """Append one frame and return the 15 lower-body joint targets (LOWER_BODY_JOINTS order)."""
        frame = self.frame(command, height, rpy, ang_vel_b, gravity_b, q_body, dq_body)
        self.history = torch.cat((self.history[:, 1:], frame.unsqueeze(1)), dim=1)
        self.last_action = self.net(self.history.reshape(frame.shape[0], -1), command)
        return self.last_action * ACTION_SCALE + self.padded_defaults[:15]
