from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ..observations import object_pos_camera, object_visible_camera, point_pos_camera, point_visible_camera
from ..task_phases import CARRY_PHASE, COMPLETE_PHASE, GRASP_PHASE, PLACE_PHASE, ensure_task_phase_buffers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CameraBinNavigationCommand(CommandTerm):
    """Generate phase-specific base commands from camera-frame QR measurements.

    GRASP first uses the Box QR to walk to a centered bimanual pre-grasp
    standoff.  Once stable hand contact is established, that command becomes
    zero so lifting happens from a quiet stance.  CARRY/PLACE retain the
    camera-only Bin QR navigation command.
    """

    cfg: CameraBinNavigationCommandCfg

    def __init__(self, cfg: CameraBinNavigationCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_cfg = SceneEntityCfg(cfg.asset_name, body_names=cfg.camera_parent_body_name)
        self.robot_cfg.resolve(env.scene)

        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.bin_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.bin_position_camera = torch.zeros(self.num_envs, 3, device=self.device)
        self.position_error_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.yaw_error = torch.zeros(self.num_envs, device=self.device)
        self.at_staging = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.box_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.box_position_camera = torch.zeros(self.num_envs, 3, device=self.device)
        self.pregrasp_position_error_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.pregrasp_yaw_error = torch.zeros(self.num_envs, device=self.device)
        self.at_pregrasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_visible_pregrasp_position_error_b = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self.last_visible_pregrasp_yaw_error = torch.zeros(
            self.num_envs, device=self.device
        )
        self.last_box_visible_episode_step = torch.full(
            (self.num_envs,), -1_000_000, dtype=torch.long, device=self.device
        )
        self.last_visible_staging_position_error_b = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self.last_visible_staging_yaw_error = torch.zeros(
            self.num_envs, device=self.device
        )
        self.last_bin_visible_episode_step = torch.full(
            (self.num_envs,), -1_000_000, dtype=torch.long, device=self.device
        )

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["bin_visible"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["staging_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["box_visible"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pregrasp_position_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self):
        max_command_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.linalg.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1)
            / max_command_steps
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_steps
        )
        self.metrics["bin_visible"] += self.bin_visible.float() / max_command_steps
        self.metrics["staging_position_error"] += (
            torch.linalg.norm(self.position_error_b, dim=-1) / max_command_steps
        )
        self.metrics["box_visible"] += self.box_visible.float() / max_command_steps
        self.metrics["pregrasp_position_error"] += (
            torch.linalg.norm(self.pregrasp_position_error_b, dim=-1) / max_command_steps
        )

    def _resample_command(self, env_ids: Sequence[int]):
        self.vel_command_b[env_ids] = 0.0
        self.at_staging[env_ids] = False
        self.at_pregrasp[env_ids] = False
        self.last_visible_pregrasp_position_error_b[env_ids] = 0.0
        self.last_visible_pregrasp_yaw_error[env_ids] = 0.0
        self.last_box_visible_episode_step[env_ids] = -1_000_000
        self.last_visible_staging_position_error_b[env_ids] = 0.0
        self.last_visible_staging_yaw_error[env_ids] = 0.0
        self.last_bin_visible_episode_step[env_ids] = -1_000_000

    def _camera_measurement_to_root_yaw_frame(self, point_position_camera: torch.Tensor) -> torch.Tensor:
        """Reconstruct a camera measurement as a displacement in root yaw frame."""
        torso_pos_w = self.robot.data.body_pos_w[:, self.robot_cfg.body_ids[0]]
        torso_quat_w = self.robot.data.body_quat_w[:, self.robot_cfg.body_ids[0]]
        camera_position_b = torch.tensor(self.cfg.camera_position_b, device=self.device).repeat(self.num_envs, 1)
        camera_orientation_b = torch.tensor(self.cfg.camera_orientation_b, device=self.device).repeat(
            self.num_envs, 1
        )
        camera_pos_w = torso_pos_w + math_utils.quat_apply(torso_quat_w, camera_position_b)
        camera_quat_w = math_utils.quat_mul(torso_quat_w, camera_orientation_b)
        point_displacement_w = (
            camera_pos_w
            - self.robot.data.root_pos_w
            + math_utils.quat_apply(camera_quat_w, point_position_camera)
        )
        root_yaw_quat_w = math_utils.yaw_quat(self.robot.data.root_quat_w)
        return math_utils.quat_apply_inverse(root_yaw_quat_w, point_displacement_w)

    def _update_command(self):
        ensure_task_phase_buffers(self._env)
        self.vel_command_b.zero_()
        self.at_staging.zero_()

        self.bin_position_camera[:] = point_pos_camera(
            self._env,
            point_pos_e=self.cfg.bin_qr_position_e,
            point_attr=self.cfg.bin_qr_position_attr,
            visibility_point_pos_e=self.cfg.bin_qr_position_e,
            visibility_point_attr=self.cfg.bin_qr_position_attr,
            horizontal_fov_deg=self.cfg.horizontal_fov_deg,
            image_width=self.cfg.image_width,
            image_height=self.cfg.image_height,
            clipping_range=self.cfg.clipping_range,
            mask_if_not_visible=True,
            camera_position_b=self.cfg.camera_position_b,
            camera_orientation_b=self.cfg.camera_orientation_b,
            robot_cfg=self.robot_cfg,
        )
        self.bin_visible[:] = point_visible_camera(
            self._env,
            point_pos_e=self.cfg.bin_qr_position_e,
            point_attr=self.cfg.bin_qr_position_attr,
            horizontal_fov_deg=self.cfg.horizontal_fov_deg,
            image_width=self.cfg.image_width,
            image_height=self.cfg.image_height,
            clipping_range=self.cfg.clipping_range,
            camera_position_b=self.cfg.camera_position_b,
            camera_orientation_b=self.cfg.camera_orientation_b,
            robot_cfg=self.robot_cfg,
        ).squeeze(-1).bool()

        # Use the same rendered-camera geometry as the actor's compact Box QR
        # observation.  The command therefore cannot read an unobserved Box
        # world pose to decide how GRASP should walk.
        box_cfg = SceneEntityCfg(self.cfg.box_asset_name)
        self.box_position_camera[:] = object_pos_camera(
            self._env,
            object_cfg=box_cfg,
            visibility_on_top_surface=self.cfg.box_visibility_on_top_surface,
            horizontal_fov_deg=self.cfg.horizontal_fov_deg,
            image_width=self.cfg.image_width,
            image_height=self.cfg.image_height,
            clipping_range=self.cfg.clipping_range,
            camera_position_b=self.cfg.camera_position_b,
            camera_orientation_b=self.cfg.camera_orientation_b,
            robot_cfg=self.robot_cfg,
        )
        self.box_visible[:] = object_visible_camera(
            self._env,
            object_cfg=box_cfg,
            visibility_on_top_surface=self.cfg.box_visibility_on_top_surface,
            horizontal_fov_deg=self.cfg.horizontal_fov_deg,
            image_width=self.cfg.image_width,
            image_height=self.cfg.image_height,
            clipping_range=self.cfg.clipping_range,
            camera_position_b=self.cfg.camera_position_b,
            camera_orientation_b=self.cfg.camera_orientation_b,
            robot_cfg=self.robot_cfg,
        ).squeeze(-1).bool()

        # Reconstruct each QR displacement in the robot yaw frame from its
        # camera measurement, known extrinsics, and robot proprioception.
        qr_displacement_yaw_b = self._camera_measurement_to_root_yaw_frame(self.bin_position_camera)
        box_displacement_yaw_b = self._camera_measurement_to_root_yaw_frame(self.box_position_camera)

        qr_xy_b = qr_displacement_yaw_b[:, :2]
        qr_distance = torch.linalg.norm(qr_xy_b, dim=-1)
        qr_direction_b = qr_xy_b / torch.clamp(qr_distance.unsqueeze(-1), min=1.0e-6)
        staging_distance_error = qr_distance - self.cfg.standoff_distance
        self.position_error_b[:] = qr_direction_b * staging_distance_error.unsqueeze(-1)
        self.yaw_error[:] = torch.atan2(qr_xy_b[:, 1], qr_xy_b[:, 0])
        self.last_visible_staging_position_error_b[self.bin_visible] = (
            self.position_error_b[self.bin_visible]
        )
        self.last_visible_staging_yaw_error[self.bin_visible] = self.yaw_error[
            self.bin_visible
        ]
        self.last_bin_visible_episode_step[self.bin_visible] = (
            self._env.episode_length_buf[self.bin_visible]
        )
        bin_recently_visible = (
            self._env.episode_length_buf - self.last_bin_visible_episode_step
            <= self.cfg.bin_visibility_grace_steps
        )
        effective_staging_position_error_b = torch.where(
            self.bin_visible.unsqueeze(-1),
            self.position_error_b,
            self.last_visible_staging_position_error_b,
        )
        effective_staging_yaw_error = torch.where(
            self.bin_visible,
            self.yaw_error,
            self.last_visible_staging_yaw_error,
        )

        box_xy_b = box_displacement_yaw_b[:, :2]
        desired_box_offset_b = torch.tensor(
            (self.cfg.pregrasp_standoff_distance, 0.0), device=self.device
        ).repeat(self.num_envs, 1)
        self.pregrasp_position_error_b[:] = box_xy_b - desired_box_offset_b
        self.pregrasp_yaw_error[:] = torch.atan2(box_xy_b[:, 1], box_xy_b[:, 0])
        if self.cfg.pregrasp_world_heading_rad is not None:
            # Constant heading plus IMU, not a privileged Box waypoint.
            relative_heading = self.cfg.pregrasp_world_heading_rad - self.robot.data.heading_w
            desired_box_offset_b = self.cfg.pregrasp_standoff_distance * torch.stack(
                (torch.cos(relative_heading), torch.sin(relative_heading)), dim=-1
            )
            self.pregrasp_position_error_b[:] = box_xy_b - desired_box_offset_b
            self.pregrasp_yaw_error[:] = math_utils.wrap_to_pi(relative_heading)
        self.last_visible_pregrasp_position_error_b[self.box_visible] = (
            self.pregrasp_position_error_b[self.box_visible]
        )
        self.last_visible_pregrasp_yaw_error[self.box_visible] = (
            self.pregrasp_yaw_error[self.box_visible]
        )
        self.last_box_visible_episode_step[self.box_visible] = (
            self._env.episode_length_buf[self.box_visible]
        )
        box_recently_visible = (
            self._env.episode_length_buf - self.last_box_visible_episode_step
            <= self.cfg.box_visibility_grace_steps
        )
        effective_pregrasp_position_error_b = torch.where(
            self.box_visible.unsqueeze(-1),
            self.pregrasp_position_error_b,
            self.last_visible_pregrasp_position_error_b,
        )
        effective_pregrasp_yaw_error = torch.where(
            self.box_visible,
            self.pregrasp_yaw_error,
            self.last_visible_pregrasp_yaw_error,
        )

        phase = self._env.task_phase
        grasp = phase == GRASP_PHASE
        carry = phase == CARRY_PHASE
        place = phase == PLACE_PHASE

        # Once the hands have qualified a stable Box contact, freeze the base
        # command and let the existing double-support lift gate take over.
        stable_contact = getattr(
            self._env,
            "phase_stable_contact",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        active_pregrasp = grasp & ~stable_contact
        searching_pregrasp = active_pregrasp & ~box_recently_visible
        search_elapsed_s = self._env.episode_length_buf.float() * self._env.step_dt
        if self.cfg.pregrasp_search_forward_speed <= 0.0:
            # Compatibility path for the narrow-workspace Students: before
            # the 5 m search curriculum, QR loss produced a stationary yaw
            # scan.  Keeping this selectable prevents a new search planner
            # from silently changing an already certified policy's history.
            self.vel_command_b[searching_pregrasp, 2] = (
                self.cfg.pregrasp_search_yaw_velocity
            )
        else:
            # A top-mounted QR can be outside the vertically downward D435
            # frustum at long range. Follow an odometry-relative Archimedean
            # spiral until it becomes visible, using no privileged Box pose.
            spiral_a = self.cfg.pregrasp_search_radial_spacing / (2.0 * math.pi)
            spiral_arc_length = (
                self.cfg.pregrasp_search_forward_speed * search_elapsed_s
            )
            spiral_theta = torch.sqrt(
                torch.clamp(2.0 * spiral_arc_length / spiral_a, min=0.0)
            )
            for _ in range(4):
                root = torch.sqrt(1.0 + spiral_theta.square())
                estimated_arc_length = 0.5 * spiral_a * (
                    spiral_theta * root + torch.asinh(spiral_theta)
                )
                spiral_theta -= (
                    estimated_arc_length - spiral_arc_length
                ) / torch.clamp(spiral_a * root, min=1.0e-6)
            spiral_theta = torch.clamp(spiral_theta, min=0.0)
            signed_theta = (
                math.copysign(1.0, self.cfg.pregrasp_search_yaw_velocity)
                * spiral_theta
            )
            spiral_radius = torch.clamp(
                spiral_a * spiral_theta, max=self.cfg.pregrasp_search_max_radius
            )
            search_target_w = self._env.scene.env_origins[:, :2] + torch.stack(
                (
                    spiral_radius * torch.cos(signed_theta),
                    spiral_radius * torch.sin(signed_theta),
                ),
                dim=-1,
            )
            search_error_w = search_target_w - self.robot.data.root_pos_w[:, :2]
            search_error_w_3d = torch.cat(
                (search_error_w, torch.zeros(self.num_envs, 1, device=self.device)),
                dim=-1,
            )
            root_yaw_quat_w = math_utils.yaw_quat(self.robot.data.root_quat_w)
            search_error_b = math_utils.quat_apply_inverse(
                root_yaw_quat_w, search_error_w_3d
            )[:, :2]
            search_heading_error = torch.atan2(
                search_error_b[:, 1], search_error_b[:, 0]
            )
            search_distance = torch.linalg.vector_norm(search_error_b, dim=-1)
            search_forward = torch.clamp(
                self.cfg.pregrasp_search_position_gain * search_distance,
                max=self.cfg.pregrasp_search_forward_speed,
            ) * torch.clamp(torch.cos(search_heading_error), min=0.0)
            search_yaw = torch.clamp(
                self.cfg.pregrasp_search_yaw_gain * search_heading_error,
                min=-abs(self.cfg.pregrasp_search_yaw_velocity),
                max=abs(self.cfg.pregrasp_search_yaw_velocity),
            )
            self.vel_command_b[searching_pregrasp, 0] = search_forward[
                searching_pregrasp
            ]
            self.vel_command_b[searching_pregrasp, 2] = search_yaw[
                searching_pregrasp
            ]

        visible_pregrasp = active_pregrasp & box_recently_visible
        pregrasp_linear_command = (
            self.cfg.pregrasp_position_gain * effective_pregrasp_position_error_b
        )
        pregrasp_linear_command[:, 0] = torch.clamp(
            pregrasp_linear_command[:, 0],
            min=-self.cfg.pregrasp_max_backward_speed,
            max=self.cfg.pregrasp_max_forward_speed,
        )
        pregrasp_linear_command[:, 1] = torch.clamp(
            pregrasp_linear_command[:, 1],
            min=-self.cfg.pregrasp_max_lateral_speed,
            max=self.cfg.pregrasp_max_lateral_speed,
        )
        pregrasp_linear_command *= torch.clamp(
            torch.cos(effective_pregrasp_yaw_error), min=0.0
        ).unsqueeze(-1)
        self.vel_command_b[visible_pregrasp, :2] = pregrasp_linear_command[visible_pregrasp]
        pregrasp_yaw_command = torch.clamp(
            self.cfg.pregrasp_yaw_gain * effective_pregrasp_yaw_error,
            min=-self.cfg.pregrasp_max_yaw_speed,
            max=self.cfg.pregrasp_max_yaw_speed,
        )
        self.vel_command_b[visible_pregrasp, 2] = pregrasp_yaw_command[visible_pregrasp]
        # Optional Schmitt trigger: enter at the normal tolerance, leave at a
        # larger one. Prevents stop/walk chatter while the final step settles.
        tolerance_scale = torch.where(
            self.at_pregrasp,
            torch.full_like(self.pregrasp_yaw_error, self.cfg.pregrasp_alignment_exit_multiplier),
            torch.ones_like(self.pregrasp_yaw_error),
        )
        pregrasp_aligned = torch.abs(self.pregrasp_yaw_error) <= self.cfg.pregrasp_yaw_tolerance * tolerance_scale
        pregrasp_near = (
            (torch.abs(self.pregrasp_position_error_b[:, 0]) <= self.cfg.pregrasp_position_tolerance * tolerance_scale)
            & (torch.abs(self.pregrasp_position_error_b[:, 1]) <= self.cfg.pregrasp_lateral_tolerance * tolerance_scale)
        )
        self.at_pregrasp[:] = visible_pregrasp & pregrasp_aligned & pregrasp_near
        self.vel_command_b[self.at_pregrasp] = 0.0

        searching = (carry | place) & ~bin_recently_visible
        navigation_search_yaw = torch.sign(
            torch.full_like(search_elapsed_s, self.cfg.search_yaw_velocity)
        ) * torch.clamp(
            abs(self.cfg.search_yaw_velocity)
            / (1.0 + self.cfg.search_yaw_decay_per_s * search_elapsed_s),
            min=self.cfg.minimum_search_yaw_speed,
        )
        self.vel_command_b[searching, 0] = self.cfg.search_forward_speed
        self.vel_command_b[searching, 2] = navigation_search_yaw[searching]

        visible_carry = carry & bin_recently_visible
        linear_command = self.cfg.position_gain * effective_staging_position_error_b
        linear_command[:, 0] = torch.clamp(
            linear_command[:, 0], min=-self.cfg.max_backward_speed, max=self.cfg.max_forward_speed
        )
        linear_command[:, 1] = torch.clamp(
            linear_command[:, 1], min=-self.cfg.max_lateral_speed, max=self.cfg.max_lateral_speed
        )
        # Slow translation while the base is not facing the observed QR.
        linear_command *= torch.clamp(
            torch.cos(effective_staging_yaw_error), min=0.0
        ).unsqueeze(-1)
        self.vel_command_b[visible_carry, :2] = linear_command[visible_carry]

        visible_navigation = (carry | place) & bin_recently_visible
        yaw_command = torch.clamp(
            self.cfg.yaw_gain * effective_staging_yaw_error,
            min=-self.cfg.max_yaw_speed,
            max=self.cfg.max_yaw_speed,
        )
        self.vel_command_b[visible_navigation, 2] = yaw_command[visible_navigation]

        aligned = torch.abs(effective_staging_yaw_error) <= self.cfg.yaw_tolerance
        near_staging = (
            torch.linalg.vector_norm(effective_staging_position_error_b, dim=-1)
            <= self.cfg.position_tolerance
        )
        self.at_staging[:] = visible_carry & aligned & near_staging
        self.vel_command_b[self.at_staging] = 0.0

        # COMPLETE always stands. PLACE translates no further and only finishes
        # yaw alignment when the QR remains visible. GRASP was handled above by
        # the camera-derived Box pre-grasp command.
        self.vel_command_b[phase == COMPLETE_PHASE] = 0.0
        place_aligned = place & bin_recently_visible & aligned
        self.vel_command_b[place_aligned] = 0.0


@configclass
class CameraBinNavigationCommandCfg(CommandTermCfg):
    class_type: type = CameraBinNavigationCommand

    asset_name: str = "robot"
    camera_parent_body_name: str = "torso_link"
    box_asset_name: str = "box"
    box_visibility_on_top_surface: bool = True
    bin_qr_position_e: tuple[float, float, float] = MISSING
    bin_qr_position_attr: str | None = "bin_qr_position_e"
    camera_position_b: tuple[float, float, float] = MISSING
    camera_orientation_b: tuple[float, float, float, float] = MISSING
    horizontal_fov_deg: float = 55.2
    image_width: int = 160
    image_height: int = 120
    clipping_range: tuple[float, float] = (0.05, 8.0)

    standoff_distance: float = 0.30
    position_tolerance: float = 0.10
    yaw_tolerance: float = 0.15
    position_gain: float = 1.0
    yaw_gain: float = 1.0
    max_forward_speed: float = 0.35
    max_backward_speed: float = 0.15
    max_lateral_speed: float = 0.20
    max_yaw_speed: float = 0.30
    search_yaw_velocity: float = -0.25
    search_forward_speed: float = 0.10
    search_yaw_decay_per_s: float = 0.10
    minimum_search_yaw_speed: float = 0.08
    bin_visibility_grace_steps: int = 250

    pregrasp_standoff_distance: float = 0.42
    pregrasp_world_heading_rad: float | None = None
    pregrasp_alignment_exit_multiplier: float = 1.0
    pregrasp_position_tolerance: float = 0.06
    pregrasp_lateral_tolerance: float = 0.06
    pregrasp_yaw_tolerance: float = 0.15
    pregrasp_position_gain: float = 1.0
    pregrasp_yaw_gain: float = 1.0
    pregrasp_max_forward_speed: float = 0.25
    pregrasp_max_backward_speed: float = 0.15
    pregrasp_max_lateral_speed: float = 0.15
    pregrasp_max_yaw_speed: float = 0.25
    pregrasp_search_yaw_velocity: float = -0.20
    pregrasp_search_forward_speed: float = 0.18
    pregrasp_search_radial_spacing: float = 0.80
    pregrasp_search_max_radius: float = 2.05
    pregrasp_search_position_gain: float = 1.0
    pregrasp_search_yaw_gain: float = 1.5
    box_visibility_grace_steps: int = 100
