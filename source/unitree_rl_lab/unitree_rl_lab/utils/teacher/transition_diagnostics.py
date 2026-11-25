"""Read-only transition telemetry; support rectangles are an approximation.

Capture point outside the current stance polygon is not alone a walking
failure: planned stepping can take it outside. No action is modified here.
"""
import math
import torch

from .com_predictive_blend import convex_hull, signed_support_margin
from .com_stabilizer import ComStabilizer


def support_polygon(feet_position, feet_quaternion, loaded, origin, rotation,
                    half_length=0.10, half_width=0.04):
    """Project yaw-oriented foot rectangles into the capture-point yaw frame."""
    corners = []
    for i in range(2):
        if not bool(loaded[i]):
            continue
        w, x, y, z = feet_quaternion[i].tolist()
        yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        c, s = math.cos(yaw), math.sin(yaw)
        for dx, dy in ((-half_length, -half_width), (half_length, -half_width),
                       (half_length, half_width), (-half_length, half_width)):
            offset = feet_position.new_tensor([c*dx-s*dy, s*dx+c*dy])
            corners.append((rotation @ (feet_position[i, :2]-origin[:2]+offset)).tolist())
    return torch.tensor(convex_hull(corners), device=feet_position.device,
                        dtype=feet_position.dtype).reshape(-1, 2)


class TransitionDiagnostics:
    def __init__(self, env, locked):
        self.model = ComStabilizer(env, locked, active=False)

    def sample(self, navigation, staging_latched, stable_staging):
        model = self.model
        robot = model.robot
        state = model.measure()
        feet = robot.data.body_link_pos_w[0, model.feet]
        loads = model.sensor.data.net_forces_w[0, model.sensor_feet, 2].clamp_min(0)
        loaded = loads >= model.cfg.minimum_foot_load_fraction*model.total_mass*9.81
        origin = (state['weights'][:, None]*feet).sum(0)
        polygon = support_polygon(feet, robot.data.body_link_quat_w[0, model.feet],
                                  loaded, origin, state['rotation'])
        margin = (float(signed_support_margin(state['capture_xy'][None], polygon)[0])
                  if len(polygon) >= 3 else None)
        return {
            'capture_xy_m': state['capture_xy'].tolist(), 'mode': state['mode'],
            'support_polygon_xy_m': polygon.tolist(), 'support_margin_m': margin,
            'support_model': 'yaw_foot_rectangles_0.20x0.08m_proxy',
            'foot_vertical_load_n': loads.tolist(),
            'root_position_w': robot.data.root_link_pos_w[0].tolist(),
            'root_velocity_w': robot.data.root_link_lin_vel_w[0].tolist(),
            'navigation_command': navigation.command[0].tolist(),
            'bin_visible': bool(navigation.bin_visible[0]),
            'staging_error_b_m': navigation.position_error_b[0].tolist(),
            'staging_yaw_error_rad': float(navigation.yaw_error[0]),
            'at_staging': bool(navigation.at_staging[0]),
            'staging_latched': bool(staging_latched),
            'stable_staging_steps_before_update': int(stable_staging),
            'grip': self.grip_sample(),
        }

    def grip_sample(self):
        """Hand-Box contact forces and Box orientation/rotation (read-only).

        Per fingertip sensor: the summed Box-filtered force vector (world frame).
        Per hand: the vector sum, from which the vertical (load-carrying friction)
        and horizontal (squeeze) components can be separated offline.
        """
        model = self.model
        env, box = model.env, model.box
        per_tip, per_hand = {}, {'left': [0.0, 0.0, 0.0], 'right': [0.0, 0.0, 0.0]}
        for name, sensor in env.scene.sensors.items():
            if not name.endswith('_box_contact') or sensor.data.force_matrix_w is None:
                continue
            vector = sensor.data.force_matrix_w[0].reshape(-1, 3).sum(0)
            per_tip[name] = vector.tolist()
            side = 'left' if name.startswith('left_') else 'right' if name.startswith('right_') else None
            if side:
                per_hand[side] = [a + b for a, b in zip(per_hand[side], vector.tolist())]
        quat = box.data.root_link_quat_w[0]
        w, x, y, z = quat.tolist()
        # Angle between the Box's own z axis and world z.
        box_z_world_z = 1.0 - 2.0 * (x * x + y * y)
        return {
            'fingertip_box_force_w_n': per_tip,
            'hand_box_force_w_n': per_hand,
            'box_quat_wxyz': [w, x, y, z],
            'box_tilt_rad': math.acos(max(-1.0, min(1.0, box_z_world_z))),
            'box_ang_vel_w': box.data.root_link_ang_vel_w[0].tolist(),
            'box_lin_vel_w': box.data.root_link_lin_vel_w[0].tolist(),
        }
