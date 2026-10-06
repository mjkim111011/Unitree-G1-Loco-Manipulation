import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.markers import QrMarkerCfg
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_43DOF_WITH_HAND_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import ActionsCfg, EventCfg, RobotEnvCfg, RobotSceneCfg


BOX_BASE_SIZE = (0.1, 0.1, 0.1)
BOX_SIZE_SCALE_RANGE = {"x": (1.0, 3.0), "y": (1.0, 3.0), "z": (1.0, 3.0)}
TABLE_SURFACE_HEIGHT = 0.74
TABLE_HEIGHT_DELTA_RANGE = (-0.03, 0.03)
GRASP_BOX_XY_RANGE = {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}

# All asset centres remain inside this square, giving the robot an overall
# 5 m x 5 m work area while retaining collision margins for the table/bin.
WORKSPACE_XY_RANGE = {"x": (-2.05, 2.05), "y": (-2.05, 2.05)}
TABLE_ORIGIN_DISTANCE_RANGE = (0.75, 2.20)
BIN_ORIGIN_DISTANCE_RANGE = (0.75, 2.30)
TABLE_BIN_DISTANCE_RANGE = (1.00, 3.20)

BIN_CENTER_E = (1.55, -0.95, 0.28)
BIN_INNER_HALF_EXTENTS = (0.30, 0.35, 0.21)
BIN_QR_CENTER_E = (1.164, -0.95, 0.30)
TABLE_CENTER_E = (0.62, 0.0, TABLE_SURFACE_HEIGHT)
DROPPED_OBJECT_HEIGHT = 0.12
PHASE_LIFT_HEIGHT = 0.12
MIN_HOLD_LIFT_HEIGHT = 0.04
STABLE_CONTACT_DURATION_S = 0.10
STABLE_HOLD_DURATION_S = 0.20
DROP_GRACE_DURATION_S = 0.10
BOX_CONTACT_FORCE_THRESHOLD = 1.0
MIN_BOX_CONTACTS_FOR_HOLD = 2
# A strict hold must distribute its contacts across both hands.  The total
# contact count above remains as a redundant guard against a single noisy tip.
MIN_BOX_CONTACTS_PER_HAND_FOR_HOLD = 1
# The final predicate uses fingertip/palm-proxy distance to the Box centre.
# 0.36 m covers the largest 0.30 m sampled box without allowing a distant
# contact to qualify.
MAX_HAND_OBJECT_DISTANCE = 0.36
MAX_RELATIVE_OBJECT_SPEED = 0.50
# The zero-action posture settles near 0.5 rad.  Keep manipulation shaping
# available there, but remove it well before the 0.8-rad fall termination.
MAX_MANIPULATION_BASE_TILT_ANGLE = 0.60
# Positive GRASP shaping needs a tighter envelope than CARRY/PLACE: both feet
# must support the base above the table-level safety height before any arm
# progress can be collected.
GRASP_STANCE_MAX_BASE_TILT_ANGLE = 0.55
GRASP_STANCE_MIN_BASE_HEIGHT = 0.66
GRASP_STANCE_MIN_FOOT_CONTACT_FORCE = 1.0
# RND is hard-masked from physically unsafe states using the same envelope.
RND_SAFE_MAX_BASE_TILT_ANGLE = 0.55
RND_SAFE_MIN_BASE_HEIGHT = 0.66
RND_SAFE_MIN_FOOT_CONTACT_FORCE = 1.0
STABLE_CONTACT_REWARD = 1.00

# Safe pre-grasp bridge: it pays only while the table-supported Box stays
# nearly stationary, so reaching/caging is explored without rewarding pushes,
# throws, or a falling robot.
PREGRASP_CAGE_DISTANCE = 0.42
PREGRASP_MIN_CAGE_FINGERTIPS = 2
PREGRASP_MIN_CAGE_FINGERTIPS_PER_HAND = 1
PREGRASP_CONTACT_FORCE_THRESHOLD = 0.25
PREGRASP_MAX_OBJECT_DISPLACEMENT = 0.05
PREGRASP_MAX_OBJECT_SPEED = 0.50
PREGRASP_MAX_LIFT_HEIGHT = 0.03

# Contact-conditioned pre-lift bridge.  This is intentionally below the
# strict 0.12 m grasp transition and only applies while a low-force,
# multi-fingertip grasp is physically plausible.
PRELIFT_HEIGHT = 0.04
PRELIFT_MAX_RELATIVE_OBJECT_SPEED = 0.30
PRELIFT_MAX_HORIZONTAL_OBJECT_DISPLACEMENT = 0.05
PRELIFT_ESCROW_DURATION_S = 0.25

# The URDF converter merges the fixed palm links into wrist-yaw links.  The
# fingertip links below remain dynamic and are used by Box-filtered sensors.
HAND_PROXY_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
BOX_CONTACT_SENSOR_LINKS = (
    "left_hand_thumb_2_link",
    "left_hand_index_1_link",
    "left_hand_middle_1_link",
    "right_hand_thumb_2_link",
    "right_hand_index_1_link",
    "right_hand_middle_1_link",
)
BOX_CONTACT_SENSOR_NAMES = tuple(f"{link}_box_contact" for link in BOX_CONTACT_SENSOR_LINKS)
LEFT_BOX_CONTACT_SENSOR_NAMES = tuple(name for name in BOX_CONTACT_SENSOR_NAMES if name.startswith("left_"))
RIGHT_BOX_CONTACT_SENSOR_NAMES = tuple(name for name in BOX_CONTACT_SENSOR_NAMES if name.startswith("right_"))

# Intel RealSense D435 RGB horizontal field of view.  The previous 55.2 degree
# virtual lens made a top QR geometrically unobservable across a 5 m work area.
CAMERA_HORIZONTAL_FOV_DEG = 69.4
CAMERA_IMAGE_WIDTH = 160
CAMERA_IMAGE_HEIGHT = 120
CAMERA_CLIPPING_RANGE = (0.05, 8.0)
D435_POSITION_TORSO = (0.0576235, 0.01753, 0.42987)
D435_ORIENTATION_TORSO = (0.914959668, 0.0, 0.403545296, 0.0)

TABLE_ASSET_POSITIONS_E = {
    "table_top": (0.62, 0.0, 0.70),
    "table_pedestal": (0.62, 0.0, 0.33),
}
BIN_ASSET_POSITIONS_E = {
    "bin_bottom": (1.55, -0.95, 0.03),
    "bin_front": (1.195, -0.95, 0.30),
    "bin_back": (1.905, -0.95, 0.30),
    "bin_left": (1.55, -0.545, 0.30),
    "bin_right": (1.55, -1.355, 0.30),
}

def _kinematic_cuboid(
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8),
        ),
    )


def _static_cuboid(
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.85),
        ),
    )


def _box_contact_sensor(link_name: str) -> ContactSensorCfg:
    """Create a single-body fingertip sensor reporting Box-only contact forces."""
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_name}",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
    )


@configclass
class BoxPickPlaceSceneCfg(RobotSceneCfg):
    """G1 scene for visually guided box pick-and-place."""

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.84, 0.0, 0.80)),
        spawn=sim_utils.CuboidCfg(
            size=BOX_BASE_SIZE,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.05,
                angular_damping=0.05,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="multiply",
                static_friction=1.5,
                dynamic_friction=1.2,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.72, 0.46, 0.18),
                roughness=0.85,
            ),
        ),
    )
    box_qr = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Box/PickObjectQr",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.051),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=QrMarkerCfg(
            payload="PICK",
            size=0.090,
            semantic_tags=[("class", "pick_object_qr")],
        ),
    )

    # Compact 0.40 x 0.40 m pedestal table. It leaves a 5 cm margin around
    # the largest 0.30 x 0.30 m sampled box.
    table_top = _kinematic_cuboid(
        "TableTop", (0.40, 0.40, 0.08), TABLE_ASSET_POSITIONS_E["table_top"], (0.48, 0.30, 0.16)
    )
    table_pedestal = _kinematic_cuboid(
        "TablePedestal",
        (0.12, 0.12, 0.66),
        TABLE_ASSET_POSITIONS_E["table_pedestal"],
        (0.32, 0.20, 0.12),
    )

    # Open target bin, always behind and to the right of the table from the
    # robot's initial viewpoint.
    bin_bottom = _kinematic_cuboid("BinBottom", (0.77, 0.87, 0.06), BIN_ASSET_POSITIONS_E["bin_bottom"], (0.18, 0.38, 0.62))
    bin_front = _kinematic_cuboid("BinFront", (0.06, 0.87, 0.48), BIN_ASSET_POSITIONS_E["bin_front"], (0.18, 0.38, 0.62))
    bin_back = _kinematic_cuboid("BinBack", (0.06, 0.87, 0.48), BIN_ASSET_POSITIONS_E["bin_back"], (0.18, 0.38, 0.62))
    bin_left = _kinematic_cuboid("BinLeft", (0.65, 0.06, 0.48), BIN_ASSET_POSITIONS_E["bin_left"], (0.18, 0.38, 0.62))
    bin_right = _kinematic_cuboid("BinRight", (0.65, 0.06, 0.48), BIN_ASSET_POSITIONS_E["bin_right"], (0.18, 0.38, 0.62))
    bin_qr = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BinFront/TargetBinQr",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.031, 0.0, 0.0),
            rot=(0.7071068, 0.0, -0.7071068, 0.0),
        ),
        spawn=QrMarkerCfg(
            payload="BIN",
            size=0.180,
            semantic_tags=[("class", "target_bin_qr")],
        ),
    )

    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_camera",
        offset=TiledCameraCfg.OffsetCfg(
            # Exact d435_joint transform from Unitree's G1-with-Dex3 URDF.
            pos=D435_POSITION_TORSO,
            # rpy=(0, 0.8307767239493009, 0): 42.4 degrees from vertical.
            rot=D435_ORIENTATION_TORSO,
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # horizontal FOV = 2*atan(horizontal_aperture/(2*focal_length)) = 69.4 deg
            focal_length=15.131432,
            focus_distance=2.0,
            horizontal_aperture=20.955,
            clipping_range=CAMERA_CLIPPING_RANGE,
        ),
        width=CAMERA_IMAGE_WIDTH,
        height=CAMERA_IMAGE_HEIGHT,
    )

    # A broad Robot/.* contact sensor cannot distinguish the box from the
    # table or self-contact.  Each filtered sensor therefore has exactly one
    # fingertip body, as required by Isaac Lab's filtered-contact API.
    left_hand_thumb_2_link_box_contact = _box_contact_sensor("left_hand_thumb_2_link")
    left_hand_index_1_link_box_contact = _box_contact_sensor("left_hand_index_1_link")
    left_hand_middle_1_link_box_contact = _box_contact_sensor("left_hand_middle_1_link")
    right_hand_thumb_2_link_box_contact = _box_contact_sensor("right_hand_thumb_2_link")
    right_hand_index_1_link_box_contact = _box_contact_sensor("right_hand_index_1_link")
    right_hand_middle_1_link_box_contact = _box_contact_sensor("right_hand_middle_1_link")


@configclass
class BoxPickPlaceCommandsCfg:
    """Camera-derived Box pre-grasp and Bin-carry navigation commands."""

    base_velocity = mdp.CameraBinNavigationCommandCfg(
        asset_name="robot",
        camera_parent_body_name="torso_link",
        bin_qr_position_e=BIN_QR_CENTER_E,
        camera_position_b=D435_POSITION_TORSO,
        camera_orientation_b=D435_ORIENTATION_TORSO,
        horizontal_fov_deg=CAMERA_HORIZONTAL_FOV_DEG,
        image_width=CAMERA_IMAGE_WIDTH,
        image_height=CAMERA_IMAGE_HEIGHT,
        clipping_range=CAMERA_CLIPPING_RANGE,
        standoff_distance=0.30,
        position_tolerance=0.10,
        yaw_tolerance=0.15,
        position_gain=1.0,
        yaw_gain=1.0,
        max_forward_speed=0.35,
        max_backward_speed=0.15,
        max_lateral_speed=0.20,
        max_yaw_speed=0.30,
        # Scan a complete revolution before translating.  The previous
        # decaying turn covered less than 360 degrees within a 45-second
        # CARRY horizon and systematically missed wide-workspace Bin poses.
        search_yaw_velocity=-0.30,
        search_forward_speed=0.0,
        search_yaw_decay_per_s=0.0,
        minimum_search_yaw_speed=0.30,
        bin_visibility_grace_steps=250,
        pregrasp_standoff_distance=0.42,
        pregrasp_position_tolerance=0.06,
        pregrasp_lateral_tolerance=0.06,
        pregrasp_yaw_tolerance=0.15,
        pregrasp_position_gain=1.0,
        pregrasp_yaw_gain=1.0,
        pregrasp_max_forward_speed=0.25,
        pregrasp_max_backward_speed=0.15,
        pregrasp_max_lateral_speed=0.15,
        pregrasp_max_yaw_speed=0.25,
        pregrasp_search_yaw_velocity=-0.35,
        pregrasp_search_forward_speed=0.20,
        pregrasp_search_radial_spacing=0.80,
        pregrasp_search_max_radius=2.05,
        pregrasp_search_position_gain=1.0,
        pregrasp_search_yaw_gain=1.5,
        box_visibility_grace_steps=250,
        resampling_time_range=(30.0, 30.0),
        debug_vis=False,
    )


@configclass
class BoxPickPlaceObservationsCfg:
    @configclass
    class PolicyHistoryCfg(ObsGroup):
        """Five-frame proprioceptive history used by the actor."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy_history: PolicyHistoryCfg = PolicyHistoryCfg()

    @configclass
    class PolicyCurrentCfg(ObsGroup):
        """Current command, action, and compact task observations used by the actor."""

        last_action = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        box_visible = ObsTerm(
            func=mdp.object_visible_camera,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "object_cfg": SceneEntityCfg("box"),
                "camera_position_b": D435_POSITION_TORSO,
                "camera_orientation_b": D435_ORIENTATION_TORSO,
                "visibility_on_top_surface": True,
                "horizontal_fov_deg": CAMERA_HORIZONTAL_FOV_DEG,
                "image_width": CAMERA_IMAGE_WIDTH,
                "image_height": CAMERA_IMAGE_HEIGHT,
                "clipping_range": CAMERA_CLIPPING_RANGE,
            },
        )
        box_position_camera = ObsTerm(
            func=mdp.object_pos_camera,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "object_cfg": SceneEntityCfg("box"),
                "camera_position_b": D435_POSITION_TORSO,
                "camera_orientation_b": D435_ORIENTATION_TORSO,
                "visibility_on_top_surface": True,
                "horizontal_fov_deg": CAMERA_HORIZONTAL_FOV_DEG,
                "image_width": CAMERA_IMAGE_WIDTH,
                "image_height": CAMERA_IMAGE_HEIGHT,
                "clipping_range": CAMERA_CLIPPING_RANGE,
            },
        )
        bin_visible = ObsTerm(
            func=mdp.point_visible_camera,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "point_pos_e": BIN_QR_CENTER_E,
                "point_attr": "bin_qr_position_e",
                "camera_position_b": D435_POSITION_TORSO,
                "camera_orientation_b": D435_ORIENTATION_TORSO,
                "horizontal_fov_deg": CAMERA_HORIZONTAL_FOV_DEG,
                "image_width": CAMERA_IMAGE_WIDTH,
                "image_height": CAMERA_IMAGE_HEIGHT,
                "clipping_range": CAMERA_CLIPPING_RANGE,
            },
        )
        bin_position_camera = ObsTerm(
            func=mdp.point_pos_camera,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "point_pos_e": BIN_QR_CENTER_E,
                "point_attr": "bin_qr_position_e",
                "visibility_point_pos_e": BIN_QR_CENTER_E,
                "visibility_point_attr": "bin_qr_position_e",
                "camera_position_b": D435_POSITION_TORSO,
                "camera_orientation_b": D435_ORIENTATION_TORSO,
                "horizontal_fov_deg": CAMERA_HORIZONTAL_FOV_DEG,
                "image_width": CAMERA_IMAGE_WIDTH,
                "image_height": CAMERA_IMAGE_HEIGHT,
                "clipping_range": CAMERA_CLIPPING_RANGE,
                "mask_if_not_visible": True,
            },
        )
        table_position_camera = ObsTerm(
            func=mdp.point_pos_camera,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "point_pos_e": TABLE_CENTER_E,
                "point_attr": "table_center_e",
                "camera_position_b": D435_POSITION_TORSO,
                "camera_orientation_b": D435_ORIENTATION_TORSO,
                "use_randomized_table_height": True,
            },
        )

        def __post_init__(self):
            # These are true current-frame features.  A one-slot history has
            # the same width but makes ObservationManager return its previous
            # circular-buffer value when teacher/FSM tools intentionally
            # recompute with ``update_history=False`` after changing phase.
            self.history_length = 0
            self.enable_corruption = True
            self.concatenate_terms = True

    policy_current: PolicyCurrentCfg = PolicyCurrentCfg()

    @configclass
    class CriticHistoryCfg(ObsGroup):
        """Five-frame privileged proprioceptive history used by the critic."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)

        def __post_init__(self):
            self.history_length = 5
            self.concatenate_terms = True

    critic_history: CriticHistoryCfg = CriticHistoryCfg()

    @configclass
    class CriticCurrentCfg(ObsGroup):
        """Current command, action, and privileged task state used by the critic."""

        last_action = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        object_pos_b = ObsTerm(func=mdp.object_pos_b, params={"object_cfg": SceneEntityCfg("box")})
        target_pos_b = ObsTerm(
            func=mdp.point_pos_b,
            params={"point_pos_e": BIN_CENTER_E, "point_attr": "bin_center_e"},
        )
        task_phase = ObsTerm(func=mdp.task_phase)

        def __post_init__(self):
            self.history_length = 0
            self.concatenate_terms = True

    critic_current: CriticCurrentCfg = CriticCurrentCfg()

    @configclass
    class RndStateCfg(ObsGroup):
        object_bin_distance = ObsTerm(
            func=mdp.object_target_distance,
            params={
                "target_pos_e": BIN_CENTER_E,
                "maximum_distance": 2.0,
                "object_cfg": SceneEntityCfg("box"),
            },
        )
        camera_object_distance = ObsTerm(
            func=mdp.camera_object_distance,
            params={
                "maximum_distance": 2.0,
                "camera_position_b": D435_POSITION_TORSO,
                "camera_orientation_b": D435_ORIENTATION_TORSO,
                "robot_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "object_cfg": SceneEntityCfg("box"),
            },
        )
        hand_proxy_distance = ObsTerm(
            func=mdp.min_body_object_distance,
            params={
                "maximum_distance": 1.0,
                "body_cfg": SceneEntityCfg("robot", body_names=HAND_PROXY_BODY_NAMES),
                "object_cfg": SceneEntityCfg("box"),
            },
        )
        fingertip_distance = ObsTerm(
            func=mdp.min_body_object_distance,
            params={
                "maximum_distance": 1.0,
                "body_cfg": SceneEntityCfg("robot", body_names=list(BOX_CONTACT_SENSOR_LINKS)),
                "object_cfg": SceneEntityCfg("box"),
            },
        )
        box_contact_count = ObsTerm(
            func=mdp.filtered_contact_count,
            params={
                "contact_sensor_names": BOX_CONTACT_SENSOR_NAMES,
                "contact_force_threshold": BOX_CONTACT_FORCE_THRESHOLD,
                "maximum_count": len(BOX_CONTACT_SENSOR_NAMES),
            },
        )
        box_lift_height = ObsTerm(
            func=mdp.object_lift_height,
            params={"maximum_height": 0.30, "object_cfg": SceneEntityCfg("box")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    rnd_state: RndStateCfg = RndStateCfg()

    @configclass
    class RndSafetyMaskCfg(ObsGroup):
        """Internal-only mask consumed by the task-local RND PPO hook."""

        value = ObsTerm(
            func=mdp.rnd_safety_mask,
            params={
                "max_base_tilt_angle": RND_SAFE_MAX_BASE_TILT_ANGLE,
                "min_base_height": RND_SAFE_MIN_BASE_HEIGHT,
                "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                "min_foot_contact_force": RND_SAFE_MIN_FOOT_CONTACT_FORCE,
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # This group is intentionally absent from policy/critic/RND state inputs.
    # The PPO training hook reads it only to zero intrinsic reward on unsafe
    # or terminal transitions.
    rnd_safety_mask: RndSafetyMaskCfg = RndSafetyMaskCfg()


@configclass
class BoxPickPlaceRewardsCfg:
    # r_task: one-time phase bonuses.  The weight cancels Isaac Lab's dt=0.02,
    # so these values are actual per-episode bonuses: +10/+25/+80.  This keeps
    # a complete task on the same order as a normally moving 30 s episode.
    # The task state machine latches grasp/carry bonuses across retries.
    sparse_task = RewTerm(
        func=mdp.sequential_phase_task_reward,
        weight=50.0,
        params={
            "lift_height": PHASE_LIFT_HEIGHT,
            "grasp_success_reward": 10.0,
            "carry_success_reward": 25.0,
            "place_success_reward": 80.0,
            "carry_command_name": "base_velocity",
            "bin_target_pos_e": BIN_CENTER_E,
            "bin_inner_half_extents": BIN_INNER_HALF_EXTENTS,
            "dropped_object_height": DROPPED_OBJECT_HEIGHT,
            "stable_contact_duration_s": STABLE_CONTACT_DURATION_S,
            "stable_hold_duration_s": STABLE_HOLD_DURATION_S,
            "drop_grace_duration_s": DROP_GRACE_DURATION_S,
            "box_contact_sensor_names": BOX_CONTACT_SENSOR_NAMES,
            "left_box_contact_sensor_names": LEFT_BOX_CONTACT_SENSOR_NAMES,
            "right_box_contact_sensor_names": RIGHT_BOX_CONTACT_SENSOR_NAMES,
            "contact_force_threshold": BOX_CONTACT_FORCE_THRESHOLD,
            "min_box_contacts": MIN_BOX_CONTACTS_FOR_HOLD,
            "min_box_contacts_per_hand": MIN_BOX_CONTACTS_PER_HAND_FOR_HOLD,
            "max_hand_object_distance": MAX_HAND_OBJECT_DISTANCE,
            "min_hold_lift_height": MIN_HOLD_LIFT_HEIGHT,
            "max_relative_object_speed": MAX_RELATIVE_OBJECT_SPEED,
            "max_base_tilt_angle": MAX_MANIPULATION_BASE_TILT_ANGLE,
            "grasp_stance_max_base_tilt_angle": GRASP_STANCE_MAX_BASE_TILT_ANGLE,
            "grasp_stance_min_base_height": GRASP_STANCE_MIN_BASE_HEIGHT,
            "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "min_foot_contact_force": GRASP_STANCE_MIN_FOOT_CONTACT_FORCE,
            "hand_body_cfg": SceneEntityCfg("robot", body_names=HAND_PROXY_BODY_NAMES),
            "object_cfg": SceneEntityCfg("box"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    # A retry teleports the robot-box pair back to the grasp setup.  The
    # penalty makes lift/drop loops strictly worse than a stable carry.
    drop_penalty = RewTerm(func=mdp.phase_drop_penalty, weight=50.0, params={"penalty": 15.0})

    # Phase-1 bimanual bridge shaping. Stable contact/hold is updated by
    # sparse_task above, while the approach/cage/close terms only pay before
    # Box motion and require progress from both hands.
    grasp_approach = RewTerm(
        func=mdp.grasp_approach_progress,
        weight=6.0,
        params={
            "max_hand_object_distance": 0.75,
            "max_base_tilt_angle": GRASP_STANCE_MAX_BASE_TILT_ANGLE,
            "hand_body_cfg": SceneEntityCfg("robot", body_names=HAND_PROXY_BODY_NAMES),
            "box_contact_sensor_names": BOX_CONTACT_SENSOR_NAMES,
            "left_box_contact_sensor_names": LEFT_BOX_CONTACT_SENSOR_NAMES,
            "right_box_contact_sensor_names": RIGHT_BOX_CONTACT_SENSOR_NAMES,
            "max_pregrasp_object_displacement": PREGRASP_MAX_OBJECT_DISPLACEMENT,
            "max_pregrasp_object_speed": PREGRASP_MAX_OBJECT_SPEED,
            "max_pregrasp_lift_height": PREGRASP_MAX_LIFT_HEIGHT,
            "min_base_height": GRASP_STANCE_MIN_BASE_HEIGHT,
            "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "min_foot_contact_force": GRASP_STANCE_MIN_FOOT_CONTACT_FORCE,
            "object_cfg": SceneEntityCfg("box"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    grasp_cage = RewTerm(
        func=mdp.grasp_cage_progress,
        weight=8.0,
        params={
            "cage_distance": PREGRASP_CAGE_DISTANCE,
            "min_cage_fingertips": PREGRASP_MIN_CAGE_FINGERTIPS,
            "min_cage_fingertips_per_hand": PREGRASP_MIN_CAGE_FINGERTIPS_PER_HAND,
            "box_contact_sensor_names": BOX_CONTACT_SENSOR_NAMES,
            "left_box_contact_sensor_names": LEFT_BOX_CONTACT_SENSOR_NAMES,
            "right_box_contact_sensor_names": RIGHT_BOX_CONTACT_SENSOR_NAMES,
            "max_base_tilt_angle": GRASP_STANCE_MAX_BASE_TILT_ANGLE,
            "max_pregrasp_object_displacement": PREGRASP_MAX_OBJECT_DISPLACEMENT,
            "max_pregrasp_object_speed": PREGRASP_MAX_OBJECT_SPEED,
            "max_pregrasp_lift_height": PREGRASP_MAX_LIFT_HEIGHT,
            "min_base_height": GRASP_STANCE_MIN_BASE_HEIGHT,
            "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "min_foot_contact_force": GRASP_STANCE_MIN_FOOT_CONTACT_FORCE,
            "object_cfg": SceneEntityCfg("box"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    grasp_close = RewTerm(
        func=mdp.grasp_close_progress,
        weight=4.0,
        params={
            "cage_distance": PREGRASP_CAGE_DISTANCE,
            "contact_force_threshold": PREGRASP_CONTACT_FORCE_THRESHOLD,
            "min_box_contacts": MIN_BOX_CONTACTS_FOR_HOLD,
            "min_box_contacts_per_hand": MIN_BOX_CONTACTS_PER_HAND_FOR_HOLD,
            "min_cage_fingertips": PREGRASP_MIN_CAGE_FINGERTIPS,
            "min_cage_fingertips_per_hand": PREGRASP_MIN_CAGE_FINGERTIPS_PER_HAND,
            "box_contact_sensor_names": BOX_CONTACT_SENSOR_NAMES,
            "left_box_contact_sensor_names": LEFT_BOX_CONTACT_SENSOR_NAMES,
            "right_box_contact_sensor_names": RIGHT_BOX_CONTACT_SENSOR_NAMES,
            "max_base_tilt_angle": GRASP_STANCE_MAX_BASE_TILT_ANGLE,
            "max_pregrasp_object_displacement": PREGRASP_MAX_OBJECT_DISPLACEMENT,
            "max_pregrasp_object_speed": PREGRASP_MAX_OBJECT_SPEED,
            "max_pregrasp_lift_height": PREGRASP_MAX_LIFT_HEIGHT,
            "min_base_height": GRASP_STANCE_MIN_BASE_HEIGHT,
            "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "min_foot_contact_force": GRASP_STANCE_MIN_FOOT_CONTACT_FORCE,
            "object_cfg": SceneEntityCfg("box"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    # Bridge a safe, low-force bimanual fingertip contact to the strict 1 N
    # contact/hold predicate.  The potential is bounded to the first 4 cm of
    # vertical lift and is one-way, so it cannot be farmed by lowering the Box.
    grasp_pre_lift = RewTerm(
        func=mdp.grasp_pre_lift_progress,
        weight=4.0,
        params={
            "pre_lift_height": PRELIFT_HEIGHT,
            "cage_distance": PREGRASP_CAGE_DISTANCE,
            "contact_force_threshold": PREGRASP_CONTACT_FORCE_THRESHOLD,
            "min_box_contacts": MIN_BOX_CONTACTS_FOR_HOLD,
            "min_box_contacts_per_hand": MIN_BOX_CONTACTS_PER_HAND_FOR_HOLD,
            "min_cage_fingertips": PREGRASP_MIN_CAGE_FINGERTIPS,
            "min_cage_fingertips_per_hand": PREGRASP_MIN_CAGE_FINGERTIPS_PER_HAND,
            "box_contact_sensor_names": BOX_CONTACT_SENSOR_NAMES,
            "left_box_contact_sensor_names": LEFT_BOX_CONTACT_SENSOR_NAMES,
            "right_box_contact_sensor_names": RIGHT_BOX_CONTACT_SENSOR_NAMES,
            "max_hand_object_distance": MAX_HAND_OBJECT_DISTANCE,
            "max_relative_object_speed": PRELIFT_MAX_RELATIVE_OBJECT_SPEED,
            "max_object_speed": PREGRASP_MAX_OBJECT_SPEED,
            "max_horizontal_object_displacement": PRELIFT_MAX_HORIZONTAL_OBJECT_DISPLACEMENT,
            "max_base_tilt_angle": GRASP_STANCE_MAX_BASE_TILT_ANGLE,
            "min_base_height": GRASP_STANCE_MIN_BASE_HEIGHT,
            "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "min_foot_contact_force": GRASP_STANCE_MIN_FOOT_CONTACT_FORCE,
            "pre_lift_escrow_duration_s": PRELIFT_ESCROW_DURATION_S,
            "hand_body_cfg": SceneEntityCfg("robot", body_names=HAND_PROXY_BODY_NAMES),
            "object_cfg": SceneEntityCfg("box"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    grasp_box_contact = RewTerm(
        func=mdp.grasp_box_contact_reward,
        weight=2.0,
        params={
            "contact_reward": STABLE_CONTACT_REWARD,
            "max_base_tilt_angle": GRASP_STANCE_MAX_BASE_TILT_ANGLE,
            "min_base_height": GRASP_STANCE_MIN_BASE_HEIGHT,
            "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "min_foot_contact_force": GRASP_STANCE_MIN_FOOT_CONTACT_FORCE,
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    grasp_lift_progress = RewTerm(
        func=mdp.grasp_lift_progress,
        weight=6.0,
        params={
            "lift_height": PHASE_LIFT_HEIGHT,
            "max_base_tilt_angle": GRASP_STANCE_MAX_BASE_TILT_ANGLE,
            "min_base_height": GRASP_STANCE_MIN_BASE_HEIGHT,
            "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "min_foot_contact_force": GRASP_STANCE_MIN_FOOT_CONTACT_FORCE,
            "object_cfg": SceneEntityCfg("box"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    # A surviving bimanual carry is worth more than a fleeting lift, while the
    # one-shot -15 drop penalty still dominates a short hold-and-drop loop.
    carry_stable_hold = RewTerm(func=mdp.carry_hold_reward, weight=0.50)
    carry_staging_progress = RewTerm(
        func=mdp.carry_staging_progress,
        weight=10.0,
        params={"carry_command_name": "base_velocity"},
    )
    place_bin_progress = RewTerm(
        func=mdp.place_bin_progress,
        weight=10.0,
        params={
            "bin_target_pos_e": BIN_CENTER_E,
            "max_base_tilt_angle": MAX_MANIPULATION_BASE_TILT_ANGLE,
            "object_cfg": SceneEntityCfg("box"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

    # GRASP first receives a camera-derived Box pre-grasp command, then zeroes
    # it after stable contact for a quiet two-handed lift.  These tracking
    # weights are intentionally smaller than CARRY navigation tracking.
    grasp_track_lin_vel_xy = RewTerm(
        func=mdp.grasp_phase_track_lin_vel_xy_yaw_frame_exp,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "max_base_tilt_angle": GRASP_STANCE_MAX_BASE_TILT_ANGLE,
            "min_base_height": GRASP_STANCE_MIN_BASE_HEIGHT,
            "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "min_foot_contact_force": GRASP_STANCE_MIN_FOOT_CONTACT_FORCE,
        },
    )
    grasp_track_ang_vel_z = RewTerm(
        func=mdp.grasp_phase_track_ang_vel_z_exp,
        weight=0.10,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "max_base_tilt_angle": GRASP_STANCE_MAX_BASE_TILT_ANGLE,
            "min_base_height": GRASP_STANCE_MIN_BASE_HEIGHT,
            "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "min_foot_contact_force": GRASP_STANCE_MIN_FOOT_CONTACT_FORCE,
        },
    )
    # A deliberately small standing signal: it rewards quiet, double-supported
    # posture in GRASP.  Gait is enabled separately only for a safe, commanded
    # camera-guided pre-grasp approach.
    grasp_stance_stability = RewTerm(
        func=mdp.grasp_phase_stance_stability_reward,
        weight=0.05,
        params={
            "lin_vel_std": 0.20,
            "ang_vel_std": 0.75,
            "max_base_tilt_angle": GRASP_STANCE_MAX_BASE_TILT_ANGLE,
            "min_base_height": GRASP_STANCE_MIN_BASE_HEIGHT,
            "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "min_foot_contact_force": GRASP_STANCE_MIN_FOOT_CONTACT_FORCE,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Navigation tracking applies only while a stable bimanual hold is active
    # in CARRY. PLACE has a zero command after staging and deliberately
    # receives no positive locomotion reward.
    track_lin_vel_xy = RewTerm(
        func=mdp.phase_gated_track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.phase_gated_track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # Directly damp roll/pitch angular motion during the arm-reaching phase;
    # the global term above remains unchanged in every phase.
    grasp_base_angular_velocity = RewTerm(func=mdp.grasp_phase_ang_vel_xy_l2, weight=-0.20)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # Reaching needs a meaningful arm command change.  These two pose/action
    # regularizers are relaxed only in GRASP; CARRY and PLACE retain the
    # original locomotion values.
    action_rate = RewTerm(func=mdp.grasp_scaled_action_rate_l2, weight=-0.05, params={"grasp_scale": 0.50})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2.0e-5)
    joint_deviation_arms = RewTerm(
        func=mdp.grasp_scaled_joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            ),
            "grasp_scale": 0.20,
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.grasp_scaled_joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"]), "grasp_scale": 0.75},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"],
            )
        },
    )
    object_velocity = RewTerm(
        func=mdp.object_lin_vel_w_l2,
        weight=-0.01,
        params={"object_cfg": SceneEntityCfg("box")},
    )
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.78})
    gait = RewTerm(
        func=mdp.phase_gated_bimanual_feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "max_base_tilt_angle": GRASP_STANCE_MAX_BASE_TILT_ANGLE,
            "min_base_height": GRASP_STANCE_MIN_BASE_HEIGHT,
            "min_foot_contact_force": GRASP_STANCE_MIN_FOOT_CONTACT_FORCE,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.phase_gated_foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )
    bad_orientation_penalty = RewTerm(
        func=mdp.is_terminated_term,
        # Reward terms are dt-integrated: -1500 * 0.02 = -30 on a fall.  All
        # positive GRASP shaping is now stance-gated and pre-lift progress is
        # escrowed, so this remains a decisive failure signal without making
        # the critic target overwhelmingly discontinuous.
        weight=-1500.0,
        params={"term_keys": "bad_orientation"},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["pelvis", "torso_link", "waist.*", ".*_hip_.*", ".*_knee.*"],
            ),
        },
    )


@configclass
class BoxPickPlaceTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    success = DoneTerm(
        func=mdp.final_pick_place_success,
        params={
            "target_pos_e": BIN_CENTER_E,
            "inner_half_extents": BIN_INNER_HALF_EXTENTS,
            "object_cfg": SceneEntityCfg("box"),
        },
    )


@configclass
class BoxPickPlaceEventCfg(EventCfg):
    randomize_box_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "scale_range": BOX_SIZE_SCALE_RANGE,
            "asset_cfg": SceneEntityCfg("box"),
        },
    )
    preserve_box_qr_scale = EventTerm(
        func=mdp.preserve_child_world_scale,
        mode="prestartup",
        params={
            "relative_child_path": "PickObjectQr",
            "child_axis_to_parent_axis": (0, 1, 2),
            "asset_cfg": SceneEntityCfg("box"),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.08, 0.08)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )
    randomize_table_height = EventTerm(
        func=mdp.randomize_table_height_once,
        mode="startup",
        params={
            "table_asset_positions_e": TABLE_ASSET_POSITIONS_E,
            "base_surface_height": TABLE_SURFACE_HEIGHT,
            "height_delta_range": TABLE_HEIGHT_DELTA_RANGE,
        },
    )
    reset_pick_place_work_area = EventTerm(
        func=mdp.reset_pick_place_work_area,
        mode="reset",
        params={
            "workspace_xy_range": WORKSPACE_XY_RANGE,
            "table_origin_distance_range": TABLE_ORIGIN_DISTANCE_RANGE,
            "bin_origin_distance_range": BIN_ORIGIN_DISTANCE_RANGE,
            "table_bin_distance_range": TABLE_BIN_DISTANCE_RANGE,
            "table_center_e": TABLE_CENTER_E,
            "bin_center_e": BIN_CENTER_E,
            "bin_qr_center_e": BIN_QR_CENTER_E,
            "table_asset_positions_e": TABLE_ASSET_POSITIONS_E,
            "bin_asset_positions_e": BIN_ASSET_POSITIONS_E,
        },
    )
    randomize_box_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "static_friction_range": (0.8, 1.8),
            "dynamic_friction_range": (0.6, 1.5),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
        },
    )
    randomize_box_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "mass_distribution_params": (0.1, 0.4),
            "operation": "abs",
        },
    )
    reset_pick_place_scene = EventTerm(
        func=mdp.reset_pick_place_scene,
        mode="reset",
        params={
            "grasp_box_xy_range": GRASP_BOX_XY_RANGE,
            "base_table_surface_height": TABLE_SURFACE_HEIGHT,
            "box_base_size": BOX_BASE_SIZE,
            "box_xy_relative_to_table": True,
            "object_cfg": SceneEntityCfg("box"),
        },
    )
    retry_pick_place_scene = EventTerm(
        func=mdp.retry_pick_place_scene,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "object_cfg": SceneEntityCfg("box"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class BoxPickPlaceEnvCfg(RobotEnvCfg):
    """Visually guided three-phase box pick-and-place on a table."""

    scene: BoxPickPlaceSceneCfg = BoxPickPlaceSceneCfg(
        num_envs=2048,
        env_spacing=5.0,
        replicate_physics=False,
    )
    observations: BoxPickPlaceObservationsCfg = BoxPickPlaceObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: BoxPickPlaceCommandsCfg = BoxPickPlaceCommandsCfg()
    rewards: BoxPickPlaceRewardsCfg = BoxPickPlaceRewardsCfg()
    terminations: BoxPickPlaceTerminationsCfg = BoxPickPlaceTerminationsCfg()
    events: BoxPickPlaceEventCfg = BoxPickPlaceEventCfg()
    curriculum = None

    def __post_init__(self):
        # The locomotion terrain generator has only a finite set of terrain
        # origins and can assign many environments to the same origin. A flat
        # plane uses a unique grid origin for every robot-table-box-bin set.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.env_spacing = self.scene.env_spacing
        super().__post_init__()
        self.episode_length_s = 30.0
        # Training uses the exact D435 transform and geometric FOV checks, so
        # rendering thousands of RGB cameras is unnecessary.
        if self.scene.num_envs > 64:
            self.scene.front_camera = None
        else:
            self.scene.front_camera.update_period = self.decimation * self.sim.dt
        self.events.push_robot = None
        self.sim.physx.gpu_max_rigid_patch_count = 24 * 2**15


@configclass
class BoxPickPlacePlayEnvCfg(BoxPickPlaceEnvCfg):
    def __post_init__(self):
        # Play represents exactly one isolated task set.
        self.scene.num_envs = 1
        super().__post_init__()


# Backward-compatible names for existing training commands.
BoxCarryWithHandEnvCfg = BoxPickPlaceEnvCfg
BoxCarryWithHandPlayEnvCfg = BoxPickPlacePlayEnvCfg
