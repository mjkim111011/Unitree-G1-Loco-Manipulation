"""Grip-aware residual PPO over the direct BC2 -> BC3 handoff.

Same scene and BC3 inputs as the live evaluation chain (via
:class:`CarryTransitionEnvCfg`, i.e. ``configure_fixed_teacher_scene``); every
reset restores a sampled ``BILATERAL_HOLD`` snapshot.  Episodes cover one
contact-clearing step, one warm-up step, the 1 s blend, the residual window (to 3 s) and one second of
BC3-only walking.  The controller glue lives in
:mod:`unitree_rl_lab.utils.grip_residual_policy` and must be attached before the
first reset.

Reward weights are per second (the reward manager multiplies by ``step_dt``).
Shaping terms are soft thresholds in threshold units (0 inside the normal band
of successful baseline runs), so their scale is comparable across terms.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp import carry_transition as transition_mdp
from unitree_rl_lab.tasks.locomotion.mdp import grip_residual as residual_mdp

from .box_carry_with_hand_env_cfg import BoxPickPlaceObservationsCfg
from .carry_transition_env_cfg import CarryTransitionEnvCfg

# Contact-clearing step + warm-up step + 4 s of CARRY (blend 0-1 s, residual to 3 s, BC3 alone 3-4 s).
RESIDUAL_EPISODE_SECONDS = 4.04
DEFAULT_VARIANT = "grip"

# Normal bands from the 140-state grip diagnostics (90th percentile of successful runs).
TILT_BAND_RAD = 0.10
BOX_WRIST_HEIGHT_BAND_M = 0.02
LIFT_OVERSHOOT_BAND_M = 0.05
SLIP_VELOCITY_BAND_MPS = 0.15
FRICTION_RATIO_BAND = 0.32
ANG_VEL_BAND_RADPS = 1.0


# Every collision body of each hand (the palm is merged into wrist_yaw_link by the URDF import).
HAND_LOAD_LINKS = ("wrist_yaw_link", "hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link",
                   "hand_index_0_link", "hand_index_1_link", "hand_middle_0_link", "hand_middle_1_link")


def _hand_load_sensor(link_name: str) -> ContactSensorCfg:
    """Box-filtered force on one hand link.  PhysX filtering needs one sensor body per sensor, and the
    names end in ``_box_load`` (not ``_box_contact``) so the fingertip contact counts are unchanged."""

    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_name}",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
    )


@configclass
class GripResidualObservationsCfg(BoxPickPlaceObservationsCfg):
    # BC3's policy_history/policy_current groups are inherited unchanged and are
    # declared first, so their history buffers are updated before BC3 is queried.
    @configclass
    class ResidualPolicyCfg(ObsGroup):
        features = ObsTerm(func=residual_mdp.residual_policy_observation, params={"variant": DEFAULT_VARIANT})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    residual_policy: ResidualPolicyCfg = ResidualPolicyCfg()

    @configclass
    class ResidualCriticCfg(ObsGroup):
        features = ObsTerm(func=residual_mdp.residual_critic_observation, params={"variant": DEFAULT_VARIANT})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    residual_critic: ResidualCriticCfg = ResidualCriticCfg()

    def __post_init__(self):
        self.critic_history = None
        self.critic_current = None
        self.rnd_state = None
        self.rnd_safety_mask = None


@configclass
class GripResidualRewardsCfg:
    # D. Task continuation (alive 1/s; a drop or fall additionally costs 5).
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminated = RewTerm(func=mdp.is_terminated, weight=-250.0)
    bilateral_box_hold = RewTerm(func=transition_mdp.bilateral_box_hold, weight=0.5)
    velocity_tracking = RewTerm(func=residual_mdp.velocity_tracking, weight=1.0, params={"std": 0.25})
    # A. Body stability (soft thresholds: zero inside the normal walking band).
    tilt_excess = RewTerm(func=residual_mdp.tilt_excess, weight=-1.0, params={"threshold": TILT_BAND_RAD})
    ang_vel_excess = RewTerm(func=residual_mdp.ang_vel_excess, weight=-0.25,
                             params={"threshold": ANG_VEL_BAND_RADPS})
    # B. Grip preservation (grip variant; the body-only ablation sets these to 0).
    box_wrist_height_excess = RewTerm(func=residual_mdp.box_wrist_height_excess, weight=-1.0,
                                      params={"threshold": BOX_WRIST_HEIGHT_BAND_M})
    lift_overshoot = RewTerm(func=residual_mdp.lift_overshoot, weight=-1.0,
                             params={"threshold": LIFT_OVERSHOOT_BAND_M})
    slip_velocity_excess = RewTerm(func=residual_mdp.slip_velocity_excess, weight=-0.5,
                                   params={"threshold": SLIP_VELOCITY_BAND_MPS})
    # C. Contact/friction (enabled only after the full-hand force check; see the training script).
    friction_ratio_excess = RewTerm(func=residual_mdp.friction_ratio_excess, weight=0.0,
                                    params={"threshold": FRICTION_RATIO_BAND})
    # E. Residual regularization.
    residual_magnitude = RewTerm(func=residual_mdp.residual_magnitude, weight=-0.5)
    residual_rate = RewTerm(func=residual_mdp.residual_rate, weight=-0.1)
    command_rate = RewTerm(func=residual_mdp.command_rate, weight=-0.05)


GRIP_REWARD_TERMS = ("box_wrist_height_excess", "lift_overshoot", "slip_velocity_excess", "friction_ratio_excess")


@configclass
class GripResidualEnvCfg(CarryTransitionEnvCfg):
    observations: GripResidualObservationsCfg = GripResidualObservationsCfg()
    rewards: GripResidualRewardsCfg = GripResidualRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        for side in ("left", "right"):
            for link in HAND_LOAD_LINKS:
                setattr(self.scene, f"{side}_{link}_box_load", _hand_load_sensor(f"{side}_{link}"))
        self.episode_length_s = RESIDUAL_EPISODE_SECONDS


def set_residual_variant(cfg: GripResidualEnvCfg, variant: str) -> None:
    """Select the actor/critic layout; the body-only ablation also drops the grip rewards."""

    cfg.observations.residual_policy.features.params["variant"] = variant
    cfg.observations.residual_critic.features.params["variant"] = variant
    if variant == "body":
        for name in GRIP_REWARD_TERMS:
            getattr(cfg.rewards, name).weight = 0.0


@configclass
class GripResidualPlayEnvCfg(GripResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
