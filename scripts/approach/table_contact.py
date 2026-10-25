"""Table-filtered contact sensors on the arm/hand links (evaluation diagnostics only, phase 6).

The scene's broad ``contact_forces`` sensor reports the net force on each robot body and cannot tell the table from the
Box (BC-1/BC-2 touch the Box on purpose).  Isaac Lab's filtered contact reporting needs one body per sensor, so every
arm/hand link gets its own sensor filtered on the table top and the pedestal (kinematic rigid bodies of the Box scene).
Reading the sensors does not act on the simulation.
"""

from __future__ import annotations

import torch

TABLE_CONTACT_LINKS = tuple(
    f"{side}_{name}"
    for side in ("left", "right")
    for name in ("elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link", "hand_thumb_0_link", "hand_thumb_1_link",
                 "hand_thumb_2_link", "hand_index_0_link", "hand_index_1_link", "hand_middle_0_link", "hand_middle_1_link")
)
TABLE_PRIMS = ("{ENV_REGEX_NS}/TableTop", "{ENV_REGEX_NS}/TablePedestal")
BOX_FINGERTIP_SENSORS = tuple(f"{side}_hand_{tip}_link_box_contact" for side in ("left", "right") for tip in ("thumb_2", "index_1", "middle_1"))


def sensor_name(link: str) -> str:
    return f"{link}_table_contact"


def add_table_contact_sensors(scene_cfg, history_length: int = 4) -> None:
    """Add one table-filtered contact sensor per arm/hand link to a scene config (history = one control step)."""
    from isaaclab.sensors import ContactSensorCfg

    for link in TABLE_CONTACT_LINKS:
        setattr(scene_cfg, sensor_name(link), ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}", update_period=0.0, history_length=history_length,
            track_air_time=False, filter_prim_paths_expr=list(TABLE_PRIMS)))


def has_table_contact_sensors(env) -> bool:
    return sensor_name(TABLE_CONTACT_LINKS[0]) in env.scene.sensors


def table_forces(env) -> torch.Tensor:
    """(N, L) largest normal force between each link and the table (top + pedestal) over the last control step [N]."""
    forces = []
    for link in TABLE_CONTACT_LINKS:
        history = env.scene.sensors[sensor_name(link)].data.force_matrix_w_history  # (N, T, 1, M, 3)
        forces.append(torch.linalg.vector_norm(history, dim=-1).sum(dim=-1).amax(dim=1)[:, 0])
    return torch.stack(forces, dim=-1)


def box_fingertip_forces(env) -> torch.Tensor | None:
    """(N, 6) normal force between each fingertip and the Box, if the Box-filtered sensors exist (BC-1/BC-2 scene)."""
    if BOX_FINGERTIP_SENSORS[0] not in env.scene.sensors:
        return None
    return torch.stack([torch.linalg.vector_norm(env.scene.sensors[name].data.force_matrix_w, dim=-1).sum(dim=-1)[:, 0]
                        for name in BOX_FINGERTIP_SENSORS], dim=-1)


def table_impulse(env) -> torch.Tensor:
    """(N, L) table contact impulse of each link over the last control step [N s]: sum over the physics substeps kept in
    the sensor history (history_length = decimation, one update per physics step) of the normal force times dt."""
    impulses = []
    for link in TABLE_CONTACT_LINKS:
        history = env.scene.sensors[sensor_name(link)].data.force_matrix_w_history  # (N, T, 1, M, 3)
        impulses.append(torch.linalg.vector_norm(history, dim=-1).sum(dim=-1).sum(dim=1)[:, 0] * float(env.physics_dt))
    return torch.stack(impulses, dim=-1)
