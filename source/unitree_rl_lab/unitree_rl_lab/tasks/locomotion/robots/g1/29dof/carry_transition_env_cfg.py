"""Learned WBC-to-CARRY transition: training scene restored from live-chain snapshots.

The scene matches the live evaluation chain (``configure_fixed_teacher_scene``)
so that BC3 sees exactly the inputs it saw there.  Every reset writes a sampled
``LOADED_STAND_END`` snapshot; the action/observation glue lives in
:mod:`unitree_rl_lab.utils.transition_policy` and must be attached before the
first reset.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp import carry_transition as transition_mdp

from .box_carry_with_hand_env_cfg import (
    BoxPickPlaceEnvCfg,
    BoxPickPlaceEventCfg,
    BoxPickPlaceObservationsCfg,
    TABLE_SURFACE_HEIGHT,
)

# Matches the live chain in scripts/teacher/evaluate_bc2_rollout.py.
TEACHER_BOX_SIZE_M = 0.20
TEACHER_BOX_MASS_KG = 0.20
# Gate deadline (2 s) plus 3 s of BC3-only walking to score the handoff.
TRANSITION_EPISODE_SECONDS = 5.0


@configclass
class CarryTransitionObservationsCfg(BoxPickPlaceObservationsCfg):
    @configclass
    class TransitionPolicyCfg(ObsGroup):
        # One term so the live chain can call the identical function.
        features = ObsTerm(func=transition_mdp.transition_policy_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Declared after policy_history/policy_current so their history buffers
    # are already updated when bc3_preview queries BC3.
    transition_policy: TransitionPolicyCfg = TransitionPolicyCfg()

    @configclass
    class TransitionCriticCfg(TransitionPolicyCfg):
        box_lift = ObsTerm(func=transition_mdp.box_lift_height_obs)

    transition_critic: TransitionCriticCfg = TransitionCriticCfg()

    def __post_init__(self):
        # Only BC3's inputs and the transition groups are needed.
        self.critic_history = None
        self.critic_current = None
        self.rnd_state = None
        self.rnd_safety_mask = None


@configclass
class CarryTransitionRewardsCfg:
    # Task survival and payload.
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminated = RewTerm(func=mdp.is_terminated, weight=-50.0)
    bilateral_box_hold = RewTerm(func=transition_mdp.bilateral_box_hold, weight=1.0)
    # Smoothness of the handoff itself.
    joint_target_jump = RewTerm(func=transition_mdp.joint_target_jump_l2, weight=-200.0)
    transition_action_rate = RewTerm(func=transition_mdp.transition_action_rate_l2, weight=-0.5)
    # Whole-body and payload stability.
    base_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    box_relative_velocity = RewTerm(func=transition_mdp.box_relative_velocity_l2, weight=-2.0)
    com_support_violation = RewTerm(func=transition_mdp.com_support_violation, weight=-50.0)
    # Progress: finish the handoff, but never at the cost of the terms above.
    alpha_progress = RewTerm(func=transition_mdp.alpha_progress, weight=5.0)
    pending_handoff = RewTerm(func=transition_mdp.pending_handoff, weight=-0.5)
    # Downstream: the handoff is good when BC3 then walks well.
    bc3_velocity_tracking = RewTerm(func=transition_mdp.bc3_velocity_tracking, weight=2.0)
    # Stage 2 only (enabled by the training script).
    weight_shift_at_first_swing = RewTerm(func=transition_mdp.weight_shift_at_first_swing, weight=0.0)


@configclass
class CarryTransitionTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    box_dropped = DoneTerm(
        func=transition_mdp.transition_box_dropped,
        params={"table_surface_height": TABLE_SURFACE_HEIGHT},
    )


@configclass
class CarryTransitionEventCfg(BoxPickPlaceEventCfg):
    reset_carry_transition = EventTerm(
        func=transition_mdp.reset_carry_transition_from_snapshot,
        mode="reset",
    )


@configclass
class CarryTransitionEnvCfg(BoxPickPlaceEnvCfg):
    observations: CarryTransitionObservationsCfg = CarryTransitionObservationsCfg()
    rewards: CarryTransitionRewardsCfg = CarryTransitionRewardsCfg()
    terminations: CarryTransitionTerminationsCfg = CarryTransitionTerminationsCfg()
    events: CarryTransitionEventCfg = CarryTransitionEventCfg()

    def __post_init__(self):
        self.scene.num_envs = 1024
        super().__post_init__()
        # Imported lazily: the runtime module requires a running Isaac Sim.
        from unitree_rl_lab.utils.teacher.arena_teacher_runtime import configure_fixed_teacher_scene

        configure_fixed_teacher_scene(self, box_size_m=TEACHER_BOX_SIZE_M, box_mass_kg=TEACHER_BOX_MASS_KG)
        # configure_fixed_teacher_scene strips the pick-place terminations; restore ours.
        self.terminations = CarryTransitionTerminationsCfg()
        # The snapshot restore is the only reset writer.
        for name in ("reset_base", "reset_robot_joints", "reset_pick_place_scene", "reset_pick_place_work_area"):
            setattr(self.events, name, None)
        self.episode_length_s = TRANSITION_EPISODE_SECONDS


@configclass
class CarryTransitionPlayEnvCfg(CarryTransitionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
