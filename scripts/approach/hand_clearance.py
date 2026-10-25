"""Arm/hand–table clearance from the robot's own collision geometry (phase 8, diagnostics only).

The G1 43-DoF asset is imported from ``g1_29dof_with_hand_rev_1_0.urdf`` with Isaac Lab's URDF converter defaults:
every collision mesh becomes a convex-hull collider and fixed joints are merged (the palm hull therefore belongs to
``*_wrist_yaw_link``).  This module rebuilds those hulls in each link frame, samples their surfaces (about 2 mm
spacing) and measures the signed distance of the samples to the table top and pedestal boxes:

  clearance > 0   gap between the link's collision shape and the table [m]
  clearance <= 0  penetration (PhysX contact)

Two uses:
  measured   link poses from the simulator (``body_link_pos_w`` / ``body_link_quat_w``)
  commanded  hand poses implied by a commanded wrist pose (Pink IK target frame ``*_wrist_yaw_link``), with the hand
             links rigidly attached at their current (or the open, all-zero) finger configuration.  This tells
             whether the wrist target itself puts the fingers into the table, independent of WBC tracking.
Nothing here acts on the simulation.
"""

from __future__ import annotations

import os
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
ARM_LINKS = ("elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")
HAND_LINKS = ("wrist_yaw_link", "hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link", "hand_index_0_link",
              "hand_index_1_link", "hand_middle_0_link", "hand_middle_1_link")
CLEARANCE_LINKS = tuple(f"{s}_{n}" for s in SIDES for n in ARM_LINKS + HAND_LINKS[1:])
FINGER_LINKS = tuple(f"{s}_{n}" for s in SIDES for n in HAND_LINKS[1:])
GROUPS = {
    "finger": tuple(f"{s}_{n}" for s in SIDES for n in HAND_LINKS[1:]),
    "wrist_palm": tuple(f"{s}_wrist_yaw_link" for s in SIDES),
    "forearm": tuple(f"{s}_{n}" for s in SIDES for n in ("elbow_link", "wrist_roll_link", "wrist_pitch_link")),
}
TABLE_TOP_SIZE = (0.40, 0.40, 0.08)
TABLE_PEDESTAL_SIZE = (0.12, 0.12, 0.66)
SAMPLE_SPACING_M = 0.002


def default_urdf() -> Path:
    root = Path(os.environ.get("UNITREE_ROS_DIR", Path(__file__).resolve().parents[2] / "external" / "unitree_ros"))
    return root / "robots" / "g1_description" / "g1_29dof_with_hand_rev_1_0.urdf"


def _load_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        values = [list(map(float, line.split()[1:4])) for line in data.decode(errors="ignore").splitlines()
                  if line.strip().startswith("vertex")]
        return np.asarray(values, dtype=np.float64)
    count = struct.unpack("<I", data[80:84])[0]
    records = np.frombuffer(data[84:84 + 50 * count], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return records["v"].reshape(-1, 3).astype(np.float64)


def _rpy_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _hull_surface_points(vertices: np.ndarray, spacing: float, rng: np.random.Generator) -> np.ndarray:
    """Hull vertices plus area-uniform samples on the hull faces (about one per spacing^2)."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(vertices)
    tri = vertices[hull.simplices]  # (F, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    count = int(np.ceil(area.sum() / spacing**2))
    face = rng.choice(len(tri), size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    samples = tri[face, 0] + u[:, None] * (tri[face, 1] - tri[face, 0]) + v[:, None] * (tri[face, 2] - tri[face, 0])
    return np.concatenate((vertices[hull.vertices], samples), axis=0)


def _box_points(size, origin, spacing: float) -> np.ndarray:
    axes = [np.linspace(-0.5 * s, 0.5 * s, max(2, int(np.ceil(s / spacing)) + 1)) for s in size]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    on_surface = np.isclose(np.abs(grid), 0.5 * np.asarray(size)).any(axis=1)
    return grid[on_surface] + np.asarray(origin)


def link_collision_points(urdf: Path | None = None, links=CLEARANCE_LINKS, spacing: float = SAMPLE_SPACING_M,
                          seed: int = 0) -> dict[str, np.ndarray]:
    """Collision-surface samples of each simulator body in its own link frame (fixed-joint children merged)."""
    urdf = Path(urdf or default_urdf())
    root = ET.parse(urdf).getroot()
    rng = np.random.default_rng(seed)
    joints = [j for j in root.findall("joint")]
    fixed_children = {}
    for joint in joints:
        if joint.get("type") == "fixed":
            origin = joint.find("origin")
            xyz = np.zeros(3) if origin is None else np.array(list(map(float, origin.get("xyz", "0 0 0").split())))
            rpy = np.zeros(3) if origin is None else np.array(list(map(float, origin.get("rpy", "0 0 0").split())))
            fixed_children.setdefault(joint.find("parent").get("link"), []).append((joint.find("child").get("link"), xyz, rpy))

    def own_points(link_name: str) -> list[np.ndarray]:
        link = root.find(f"link[@name='{link_name}']")
        result = []
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            xyz = np.zeros(3) if origin is None else np.array(list(map(float, origin.get("xyz", "0 0 0").split())))
            rpy = np.zeros(3) if origin is None else np.array(list(map(float, origin.get("rpy", "0 0 0").split())))
            geometry = collision.find("geometry")[0]
            if geometry.tag == "mesh":
                vertices = _load_stl(urdf.parent / geometry.get("filename"))
                if geometry.get("scale"):
                    vertices = vertices * np.array(list(map(float, geometry.get("scale").split())))
                points = _hull_surface_points(np.unique(vertices.round(7), axis=0), spacing, rng)
            elif geometry.tag == "box":
                points = _box_points(list(map(float, geometry.get("size").split())), (0.0, 0.0, 0.0), spacing)
            else:
                raise ValueError(f"unsupported collision geometry {geometry.tag} on {link_name}")
            result.append(points @ _rpy_matrix(rpy).T + xyz)
        return result

    output = {}
    for name in links:
        parts = own_points(name)
        for child, xyz, rpy in fixed_children.get(name, []):
            parts.extend(p @ _rpy_matrix(rpy).T + xyz for p in own_points(child))
        output[name] = np.concatenate(parts, axis=0).astype(np.float32)
    return output


def open_hand_offsets(urdf: Path | None = None) -> dict[str, np.ndarray]:
    """4x4 transform of every hand link relative to its ``*_wrist_yaw_link`` with all finger joints at 0 (open)."""
    urdf = Path(urdf or default_urdf())
    root = ET.parse(urdf).getroot()
    parent_of = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        xyz = np.array(list(map(float, origin.get("xyz", "0 0 0").split()))) if origin is not None else np.zeros(3)
        rpy = np.array(list(map(float, origin.get("rpy", "0 0 0").split()))) if origin is not None else np.zeros(3)
        transform = np.eye(4)
        transform[:3, :3] = _rpy_matrix(rpy)
        transform[:3, 3] = xyz
        parent_of[joint.find("child").get("link")] = (joint.find("parent").get("link"), transform)
    output = {}
    for side in SIDES:
        base = f"{side}_wrist_yaw_link"
        for name in HAND_LINKS:
            link = f"{side}_{name}"
            transform = np.eye(4)
            while link != base:
                parent, local = parent_of[link]
                transform = local @ transform
                link = parent
            output[f"{side}_{name}"] = transform
    return output


def signed_distance_to_boxes(points: np.ndarray, centers: np.ndarray, half_sizes: np.ndarray) -> np.ndarray:
    """(…, P, 3) points, (B, 3) axis-aligned boxes -> (…, P) signed distance to the union of the boxes."""
    q = np.abs(points[..., None, :] - centers) - half_sizes  # (…, P, B, 3)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return (outside + inside).min(axis=-1)


def table_boxes(table_top_center, table_pedestal_center) -> tuple[np.ndarray, np.ndarray]:
    centers = np.array([table_top_center, table_pedestal_center], dtype=np.float64)
    return centers, 0.5 * np.array([TABLE_TOP_SIZE, TABLE_PEDESTAL_SIZE], dtype=np.float64)


class OpenHandGeometry:
    """Numpy model of one open hand (wrist/palm + 7 finger links) rigidly attached to its ``*_wrist_yaw_link``."""

    def __init__(self, spacing: float = SAMPLE_SPACING_M) -> None:
        offsets = open_hand_offsets()
        points = link_collision_points(links=tuple(f"{s}_{n}" for s in SIDES for n in HAND_LINKS), spacing=spacing)
        self.sides = {}
        for side in SIDES:
            pts, owner, names = [], [], []
            for k, name in enumerate(HAND_LINKS):
                link = f"{side}_{name}"
                transform = offsets[link]
                pts.append(points[link].astype(np.float64) @ transform[:3, :3].T + transform[:3, 3])
                owner.append(np.full(len(points[link]), k))
                names.append(link)
            self.sides[side] = (np.concatenate(pts), np.concatenate(owner), names)

    def clearance(self, side: str, wrist: np.ndarray, centers: np.ndarray, half: np.ndarray):
        """wrist (T, 4, 4) world poses -> min signed distance (T,), owning link name per step, deepest point (T, 3)."""
        pts, owner, names = self.sides[side]
        world = np.einsum("tij,pj->tpi", wrist[:, :3, :3], pts) + wrist[:, None, :3, 3]
        q = np.abs(world[:, :, None, :] - centers) - half
        sdf = (np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)).min(axis=-1)
        arg = sdf.argmin(axis=1)
        return sdf.min(axis=1), [names[i] for i in owner[arg]], world[np.arange(len(world)), arg]


class TableClearance:
    """Torch implementation for the simulator: per-link minimum signed distance to the table top / pedestal."""

    def __init__(self, robot, table_centers, device, links=CLEARANCE_LINKS, spacing: float = SAMPLE_SPACING_M) -> None:
        import torch

        names = list(robot.data.body_names)
        self.links = tuple(links)
        self.body_ids = [names.index(link) for link in self.links]
        points = link_collision_points(links=self.links, spacing=spacing)
        self.counts = [len(points[link]) for link in self.links]
        self.points = torch.as_tensor(np.concatenate([points[link] for link in self.links]), dtype=torch.float32, device=device)
        self.owner = torch.repeat_interleave(torch.arange(len(self.links), device=device),
                                             torch.as_tensor(self.counts, device=device))
        self.centers = torch.as_tensor(np.asarray(table_centers, dtype=np.float32), device=device)
        self.half = torch.as_tensor(0.5 * np.array([TABLE_TOP_SIZE, TABLE_PEDESTAL_SIZE], dtype=np.float32), device=device)
        offsets = open_hand_offsets()
        self.hand_links = tuple(f"{s}_{n}" for s in SIDES for n in HAND_LINKS)
        self.hand_index = [self.links.index(link) if link in self.links else None for link in self.hand_links]
        self.open_offsets = torch.as_tensor(np.stack([offsets[link] for link in self.hand_links]), dtype=torch.float32, device=device)
        self.link_points = {link: torch.as_tensor(points[link], dtype=torch.float32, device=device) for link in self.hand_links}
        self.device = device

    def _sdf(self, world_points):
        import torch

        q = torch.abs(world_points[:, None, :] - self.centers) - self.half
        outside = torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.clamp(q.amax(dim=-1), max=0.0)
        return (outside + inside).amin(dim=-1)

    def measured(self, robot, env_id: int = 0):
        """(L,) minimum signed distance of each link's collision surface to the table [m] (current poses)."""
        import torch
        from isaaclab.utils import math as math_utils

        pos = robot.data.body_link_pos_w[env_id, self.body_ids]
        quat = robot.data.body_link_quat_w[env_id, self.body_ids]
        world = math_utils.quat_apply(quat[self.owner], self.points) + pos[self.owner]
        distance = self._sdf(world)
        result = torch.full((len(self.links),), float("inf"), device=self.device)
        return result.scatter_reduce(0, self.owner, distance, reduce="amin")

    def relative_hand_offsets(self, robot, env_id: int = 0):
        """Current transforms of the hand links relative to their wrist_yaw_link (realized finger configuration)."""
        import torch
        from isaaclab.utils import math as math_utils

        names = list(robot.data.body_names)
        output = []
        for side in SIDES:
            base = names.index(f"{side}_wrist_yaw_link")
            base_pos, base_quat = robot.data.body_link_pos_w[env_id, base], robot.data.body_link_quat_w[env_id, base]
            inv = math_utils.quat_inv(base_quat.unsqueeze(0))[0]
            for name in HAND_LINKS:
                index = names.index(f"{side}_{name}")
                rel_pos = math_utils.quat_apply(inv.unsqueeze(0), (robot.data.body_link_pos_w[env_id, index] - base_pos).unsqueeze(0))[0]
                rel_rot = math_utils.matrix_from_quat(math_utils.quat_mul(inv.unsqueeze(0), robot.data.body_link_quat_w[env_id, index].unsqueeze(0)))[0]
                transform = torch.eye(4, device=self.device)
                transform[:3, :3] = rel_rot
                transform[:3, 3] = rel_pos
                output.append(transform)
        return torch.stack(output)

    def commanded(self, wrist_world_left, wrist_world_right, offsets=None) -> dict[str, float]:
        """Minimum signed distance of each hand (wrist/palm + fingers) placed at a commanded wrist pose (4x4, world).

        ``offsets``: (16, 4, 4) hand-link transforms relative to the wrist (default: open hand).
        Returns {"left": d, "right": d, "left_link": name, "right_link": name}.
        """
        import torch

        offsets = self.open_offsets if offsets is None else offsets
        wrists = {"left": torch.as_tensor(np.asarray(wrist_world_left), dtype=torch.float32, device=self.device),
                  "right": torch.as_tensor(np.asarray(wrist_world_right), dtype=torch.float32, device=self.device)}
        result = {}
        for k, link in enumerate(self.hand_links):
            side = link.split("_")[0]
            transform = wrists[side] @ offsets[k]
            points = self.link_points[link] @ transform[:3, :3].T + transform[:3, 3]
            distance = float(self._sdf(points).min())
            if distance < result.get(side, float("inf")):
                result[side] = distance
                result[f"{side}_link"] = link
        return result
