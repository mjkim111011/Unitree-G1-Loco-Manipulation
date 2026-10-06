from __future__ import annotations

from collections.abc import Callable

import qrcode
from pxr import Gf, Usd, UsdGeom, Vt

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass


@clone
def spawn_qr_marker(
    prim_path: str,
    cfg: "QrMarkerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a camera-readable QR code without creating one prim per QR module."""
    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=1,
        border=cfg.border_modules,
    )
    qr.add_data(cfg.payload)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    module_count = len(matrix)
    module_size = cfg.size / module_count
    half_size = 0.5 * cfg.size

    backing_path = f"{prim_path}/Backing"
    create_prim(
        backing_path,
        "Cube",
        translation=(0.0, 0.0, -0.5 * cfg.thickness),
        scale=(cfg.size, cfg.size, cfg.thickness),
        attributes={"size": 1.0},
        stage=stage,
    )
    white_material_path = f"{prim_path}/WhiteMaterial"
    white_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=0.9)
    white_material.func(white_material_path, white_material)
    bind_visual_material(backing_path, white_material_path, stage=stage)

    points: list[Gf.Vec3f] = []
    face_counts: list[int] = []
    face_indices: list[int] = []
    marker_z = 0.5 * cfg.thickness + 1.0e-4
    for row, modules in enumerate(matrix):
        y_max = half_size - row * module_size
        y_min = y_max - module_size
        for col, is_black in enumerate(modules):
            if not is_black:
                continue
            x_min = -half_size + col * module_size
            x_max = x_min + module_size
            vertex_start = len(points)
            points.extend(
                [
                    Gf.Vec3f(x_min, y_min, marker_z),
                    Gf.Vec3f(x_max, y_min, marker_z),
                    Gf.Vec3f(x_max, y_max, marker_z),
                    Gf.Vec3f(x_min, y_max, marker_z),
                ]
            )
            face_counts.append(4)
            face_indices.extend(
                [vertex_start, vertex_start + 1, vertex_start + 2, vertex_start + 3]
            )

    mesh_path = f"{prim_path}/BlackModules"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    black_material_path = f"{prim_path}/BlackMaterial"
    black_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), roughness=0.9)
    black_material.func(black_material_path, black_material)
    bind_visual_material(mesh_path, black_material_path, stage=stage)

    return stage.GetPrimAtPath(prim_path)


@configclass
class QrMarkerCfg(sim_utils.SpawnerCfg):
    """Visual-only QR marker generated as one mesh."""

    func: Callable = spawn_qr_marker
    payload: str = "UNITREE-RL-LAB"
    size: float = 0.10
    thickness: float = 0.002
    border_modules: int = 2
