"""Portable defaults shared by simulator entry points; imports only stdlib."""
from __future__ import annotations

import os
from pathlib import Path


def default_arena_root() -> str:
    configured = os.environ.get("UNITREE_ARENA_ROOT") or os.environ.get("LM_ARENA_ROOT")
    if configured:
        return str(Path(configured).expanduser().resolve())
    workspace = Path(os.environ.get("UNITREE_WORKSPACE", Path(__file__).resolve().parents[3]))
    candidate = workspace / "external" / "IsaacLab-Arena"
    legacy = Path.home() / "IsaacLab-Arena"
    # Preserve established research runs unless the new workspace was
    # explicitly activated; creating external/ must not change old defaults.
    if legacy.is_dir() and "UNITREE_WORKSPACE" not in os.environ:
        return str(legacy)
    return str(candidate if candidate.is_dir() else legacy)


def default_arena_dataset() -> str:
    configured = os.environ.get("UNITREE_ARENA_DATASET") or os.environ.get("LM_ARENA_DATASET")
    return configured or str(Path(default_arena_root()) / "data" / "arena_g1_loco_manipulation_dataset_generated_small.hdf5")
