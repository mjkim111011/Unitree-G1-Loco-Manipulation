"""Goal-conditioned navigation to the BC-1 v2 handoff state with a frozen Homie V2 lower body.

PPO outputs ``[vx, vy, yaw_rate]`` (normalized); :class:`HomieNavAction` runs the batched torch Homie
(closed-loop parity with the Arena ONNX worker verified) and holds the nominal upper body / open hands.
The scene, robot dynamics (``configure_arena_teacher_robot``), table and 0.2 m Box are those of the
BC-1/BC-2 evaluation environment; the bin, cameras, QR markers and fingertip sensors are removed.
"""

from __future__ import annotations

import importlib
import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.utils.teacher.arena_teacher_runtime import configure_arena_teacher_robot

from . import mdp
from .homie_action import NAV_ACTION_PROFILES, HomieNavActionCfg

box_env = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.box_carry_with_hand_env_cfg")
_BoxScene = box_env.BoxPickPlaceSceneCfg


@configclass
class NavSceneCfg(_BoxScene):
    pass


@configclass
class NavActionsCfg:
    homie = HomieNavActionCfg(asset_name="robot")


@configclass
class NavObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # [GT] privileged target geometry from the simulator Box pose (replace with perception later)
        gt_goal_pos_b = ObsTerm(func=mdp.gt_goal_pos_b)
        gt_goal_heading = ObsTerm(func=mdp.gt_goal_heading)
        gt_goal_distance = ObsTerm(func=mdp.gt_goal_distance)
        # [GT now / estimator later] base linear velocity (Homie's own estimator head is a deployable source)
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        # onboard proprioception (IMU)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


@configclass
class NavRewardsCfg:
    progress = RewTerm(func=mdp.TargetProgress, weight=2.0,
                       params={"yaw_weight": 0.5, "near_radius": 0.5, "near_gain": 4.0, "near_cap": 0.3})
    handoff_bonus = RewTerm(func=mdp.termination_event, weight=10.0, params={"term": "handoff_success"})
    fall_penalty = RewTerm(func=mdp.termination_event, weight=-10.0, params={"term": "fall"})
    collision_penalty = RewTerm(func=mdp.termination_event, weight=-10.0, params={"term": "collision"})
    time_penalty = RewTerm(func=mdp.constant, weight=-0.1)
    near_target_motion = RewTerm(func=mdp.near_target_motion, weight=-1.0, params={"radius": 0.15})
    near_target_command = RewTerm(func=mdp.command_magnitude_near_target, weight=-1.0, params={"radius": 0.15})
    table_proximity = RewTerm(func=mdp.table_proximity, weight=-2.0, params={"margin": 0.05})
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.05)
    base_ang_vel_xy = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.02)


@configclass
class NavTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    handoff_success = DoneTerm(func=mdp.HandoffSuccess, params={"hold_steps": 25})
    fall = DoneTerm(func=mdp.fall, params={"min_height": 0.55, "max_tilt": 0.8})
    # Contact: a robot body within 1 cm of the table volume, or the Box pushed 3 cm (1-5 cm = penalised near-miss).
    collision = DoneTerm(func=mdp.Collision, params={"margin": 0.01, "box_shift_max": 0.03})


@configclass
class NavEventsCfg:
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )
    reset_box_and_robot = EventTerm(func=mdp.reset_box_and_robot, mode="reset", params={"fixed_stage": 2, "mix_prob": 0.2})


@configclass
class NavCurriculumCfg:
    stage = CurrTerm(func=mdp.NavStageCurriculum, params={"start_stage": 0, "window": 400, "thresholds": (0.8, 0.8, 0.75)})


@configclass
class BoxNavHomieEnvCfg(ManagerBasedRLEnvCfg):
    scene: NavSceneCfg = NavSceneCfg(num_envs=64, env_spacing=12.0, replicate_physics=True)
    observations: NavObservationsCfg = NavObservationsCfg()
    actions: NavActionsCfg = NavActionsCfg()
    rewards: NavRewardsCfg = NavRewardsCfg()
    terminations: NavTerminationsCfg = NavTerminationsCfg()
    events: NavEventsCfg = NavEventsCfg()
    curriculum: NavCurriculumCfg = NavCurriculumCfg()
    commands = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Same articulation dynamics as the BC-1/BC-2 evaluation (Arena teacher PD gains, hand stiffness 4).
        configure_arena_teacher_robot(self.scene.robot, hand_stiffness=4.0)
        self.scene.box.spawn.size = (mdp.BOX_SIZE_M,) * 3
        self.scene.box.spawn.mass_props.mass = 0.2
        self.scene.box.spawn.physics_material.static_friction = 1.5
        self.scene.box.spawn.physics_material.dynamic_friction = 1.2
        self.scene.box.spawn.physics_material.restitution = 0.0
        # Navigation needs neither the bin nor any camera / QR / fingertip sensor.
        for name in ("front_camera", "height_scanner", "box_qr", "bin_qr", "bin_bottom", "bin_front", "bin_back",
                     "bin_left", "bin_right", *box_env.BOX_CONTACT_SENSOR_NAMES):
            if hasattr(self.scene, name):
                setattr(self.scene, name, None)
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class BoxNavHomieEvalEnvCfg(BoxNavHomieEnvCfg):
    """Evaluation: no curriculum, fixed start distribution (default S3: 3-5 m, any heading)."""

    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_box_and_robot.params = {"fixed_stage": 2, "mix_prob": 0.0}
        self.episode_length_s = 40.0


# ------------------------------------------------------------------------------------------------------ v6
# v6 = v5 + the changes motivated by the v5 Arena-WBC verification (outputs/approach_p3/reports/PPO_v5_verification_2026-10-20.md):
#   actions      slower caps, slow backward walking, elliptical planar cap, command rate limit (no bang-bang, no
#                instant braking)                                     -> NAV_ACTION_PROFILES["v6"]
#   observation  + the rate-limited command actually sent to Homie (state of the action filter)
#   success      the v5 window held 50 steps by a settled robot (base/yaw/leg-joint speeds, zero command)
#   termination  + body contact (fingers / hands / hips touching the table), penalised like a collision
#   reward       + contact penalty only.  (A raw action-rate weight of -0.5 was tried first: with the exploration noise
#                alone it cost ~1/s, more than the whole progress reward, the noise collapsed and the policy stood still;
#                smoothness is enforced by the command rate limit instead.)
_V6 = NAV_ACTION_PROFILES["v6"]


@configclass
class NavActionsV6Cfg:
    homie = HomieNavActionCfg(asset_name="robot", command_scale=_V6["scale"], command_scale_negative=_V6["scale_negative"],
                              planar_ellipse_cap=_V6["ellipse"], command_rate_limit=_V6["rate_limit"])


@configclass
class NavObservationsV6Cfg:
    @configclass
    class PolicyCfg(NavObservationsCfg.PolicyCfg):
        # the rate-limited [vx, vy, wz] Homie received at the last step (the policy needs the filter state)
        homie_command = ObsTerm(func=mdp.homie_command)

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


@configclass
class NavRewardsV6Cfg(NavRewardsCfg):
    contact_penalty = RewTerm(func=mdp.termination_event, weight=-10.0, params={"term": "contact"})


@configclass
class NavTerminationsV6Cfg(NavTerminationsCfg):
    handoff_success = DoneTerm(func=mdp.StableHandoffSuccess,
                               params={"hold_steps": 50, "speed_max": 0.05, "yaw_rate_max": 0.10, "leg_joint_speed_max": 0.3})
    contact = DoneTerm(func=mdp.ContactCollision, params={"threshold": 1.0})


@configclass
class BoxNavHomieV6EnvCfg(BoxNavHomieEnvCfg):
    observations: NavObservationsV6Cfg = NavObservationsV6Cfg()
    actions: NavActionsV6Cfg = NavActionsV6Cfg()
    rewards: NavRewardsV6Cfg = NavRewardsV6Cfg()
    terminations: NavTerminationsV6Cfg = NavTerminationsV6Cfg()


@configclass
class BoxNavHomieV6EvalEnvCfg(BoxNavHomieV6EnvCfg):
    """v6 evaluation: no curriculum, fixed start distribution (default S3: 3-5 m, any heading)."""

    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_box_and_robot.params = {"fixed_stage": 2, "mix_prob": 0.0}
        self.episode_length_s = 40.0


# v6b = v6a + one reward term: standing still next to the target but outside the window costs 1/s.
@configclass
class NavRewardsV6bCfg(NavRewardsV6Cfg):
    stopped_outside_window = RewTerm(func=mdp.stopped_outside_window, weight=-1.0, params={"radius": 0.3})


@configclass
class BoxNavHomieV6bEnvCfg(BoxNavHomieV6EnvCfg):
    rewards: NavRewardsV6bCfg = NavRewardsV6bCfg()


@configclass
class BoxNavHomieV6bEvalEnvCfg(BoxNavHomieV6bEnvCfg):
    """v6b evaluation: no curriculum, fixed start distribution (default S3: 3-5 m, any heading)."""

    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_box_and_robot.params = {"fixed_stage": 2, "mix_prob": 0.0}
        self.episode_length_s = 40.0


# ------------------------------------------------------------------------------------------------------ v7
# v7 = v6b + two action-processing changes motivated by the v6 Arena-WBC traces (outputs/approach_p4, analysis in
# outputs/approach_p5/reports): no reward term changed.
#   heading gate   planar [vx, vy] scaled to 0 while the goal point is >= 90 deg off the heading (full below 45 deg, inside
#                  0.5 m always full): v6 walked 0.6-1.4 m backward/sideways *toward the goal* while turning (81 % of that
#                  path was radial = progress reward); Homie itself turns in place within ~4 cm (homie_turn_test.py).
#   gate latch     stop-gate hysteresis (on at a[3] > 0, released only below -0.5): 632/632 v6 stop->move switches near
#                  the target happened inside the deployed window with a[3] ~ -0.02 (flicker at the training-window edge).
_V7 = NAV_ACTION_PROFILES["v7"]


@configclass
class NavActionsV7Cfg:
    homie = HomieNavActionCfg(asset_name="robot", command_scale=_V7["scale"], command_scale_negative=_V7["scale_negative"],
                              planar_ellipse_cap=_V7["ellipse"], command_rate_limit=_V7["rate_limit"],
                              heading_gate=_V7["heading_gate"], stop_gate_release=_V7["gate_release"])


@configclass
class BoxNavHomieV7EnvCfg(BoxNavHomieV6bEnvCfg):
    actions: NavActionsV7Cfg = NavActionsV7Cfg()


@configclass
class BoxNavHomieV7EvalEnvCfg(BoxNavHomieV7EnvCfg):
    """v7 evaluation: no curriculum, fixed start distribution (default S3: 3-5 m, any heading)."""

    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_box_and_robot.params = {"fixed_stage": 2, "mix_prob": 0.0}
        self.episode_length_s = 40.0


_ = math  # keep the import for configs that compute ranges in radians
