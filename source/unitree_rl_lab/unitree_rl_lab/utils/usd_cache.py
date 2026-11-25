"""Per-process USD conversion directories.

Isaac Lab's asset converters put the converted URDF/MJCF in
``/tmp/IsaacLab/usd_{YYYYmmdd_HHMMSS}_{random.randrange(10000)}`` when the
spawn config has no ``usd_dir``.  The random suffix comes from Python's global
``random`` module, which the env seeds with the experiment seed, so every
process gets the same suffix and two processes starting in the same second
write and read the same directory ("Unable to open ... .tmp.usd", or a robot
without rigid bodies: "No contact sensors added to the prim").

Assigning a directory that is unique per process (pid + ``uuid4``, which does
not use the seeded generator) removes the collision.  ``usd_dir`` is excluded
from the converter's asset hash, so the converted asset - and therefore the
simulation - is unchanged.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime


def unique_usd_dir(root: str = "/tmp/IsaacLab") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(root, f"usd_{stamp}_pid{os.getpid()}_{uuid.uuid4().hex[:8]}")


def assign_unique_usd_dirs(scene_cfg, root: str = "/tmp/IsaacLab") -> dict[str, str]:
    """Give every converter-backed asset spawn without a ``usd_dir`` its own directory.

    Returns ``{asset_name: usd_dir}`` for the assets that were changed.
    """

    assigned = {}
    for name, asset in vars(scene_cfg).items():
        spawn = getattr(asset, "spawn", None)
        if spawn is None or not hasattr(spawn, "usd_dir") or not hasattr(spawn, "asset_path"):
            continue
        if getattr(spawn, "usd_dir") is None:
            spawn.usd_dir = unique_usd_dir(root)
            assigned[name] = spawn.usd_dir
    return assigned
