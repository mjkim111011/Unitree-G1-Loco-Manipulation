"""Semantic contract shared by CARRY teacher collection, BC, and replay.

The CARRY targets are not self-contained unless two runtime details remain
identical to collection:

* the camera-navigation command is linearly introduced in the student
  observation over a fixed number of recorded CARRY steps; and
* a 23-D Arena action is postprocessed after WBC by restoring the realized
  phase-start arm/hand raw targets.  A processed 43-D target already embeds
  that same lock and must not apply it a second time.

This module deliberately has no Isaac Lab dependency.  The exact articulation
order and lock mask are persisted in each CARRY corpus and checkpoint instead
of being rediscovered from name substrings during replay.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


CARRY_RUNTIME_CONTRACT_SCHEMA = "unitree_box_carry_runtime_contract_v1"
CARRY_NAVIGATION_RAMP_PROFILE = "linear_recorded_step_plus_one_v1"
CARRY_GRASP_LOCK_POSTPROCESSOR_VERSION = "phase_start_raw_arm_hand_lock_v1"
CARRY_GRASP_LOCK_REFERENCE = "phase_start_snapshot_last_action_raw"
CARRY_LOCKED_JOINT_COUNT = 28
PROCESSED_JOINT_COUNT = 43

_LOCKED_JOINT_NAME_TOKENS = ("shoulder", "elbow", "wrist", "_hand_")
_CONTRACT_HASH_KEY = "contract_sha256"


def _ordered_joint_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if (
        len(names) != PROCESSED_JOINT_COUNT
        or len(set(names)) != PROCESSED_JOINT_COUNT
        or any(not name for name in names)
    ):
        raise ValueError(
            f"carry runtime contract requires {PROCESSED_JOINT_COUNT} unique "
            "non-empty raw joint names"
        )
    return names


def carry_grasp_lock_indices(ordered_joint_names: Sequence[str]) -> tuple[int, ...]:
    """Return the canonical v1 arm/hand lock indices for an exact joint order."""

    names = _ordered_joint_names(ordered_joint_names)
    indices = tuple(
        index
        for index, name in enumerate(names)
        if any(token in name for token in _LOCKED_JOINT_NAME_TOKENS)
    )
    if len(indices) != CARRY_LOCKED_JOINT_COUNT:
        raise ValueError(
            f"carry grasp-lock v1 requires {CARRY_LOCKED_JOINT_COUNT} "
            f"shoulder/elbow/wrist/hand joints, got {len(indices)}"
        )
    return indices


def _hash_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(contract)
    payload.pop(_CONTRACT_HASH_KEY, None)
    return payload


def carry_runtime_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Hash a carry contract independently of its stored hash field."""

    payload = json.dumps(
        _hash_payload(contract), separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_carry_runtime_contract(
    ordered_joint_names: Sequence[str], *, navigation_ramp_steps: int
) -> dict[str, Any]:
    """Build the canonical CARRY runtime contract for one articulation order."""

    if isinstance(navigation_ramp_steps, bool) or navigation_ramp_steps <= 0:
        raise ValueError("navigation_ramp_steps must be a positive integer")
    ramp_steps = int(navigation_ramp_steps)
    if ramp_steps != navigation_ramp_steps:
        raise ValueError("navigation_ramp_steps must be an integer")

    names = _ordered_joint_names(ordered_joint_names)
    locked_indices = carry_grasp_lock_indices(names)
    locked_index_set = set(locked_indices)
    contract: dict[str, Any] = {
        "schema": CARRY_RUNTIME_CONTRACT_SCHEMA,
        "navigation_ramp_steps": ramp_steps,
        "navigation_ramp_profile": CARRY_NAVIGATION_RAMP_PROFILE,
        "navigation_command_observation_field": "velocity_commands",
        "navigation_command_high_level_slice": [16, 19],
        "grasp_lock_postprocessor_version": (
            CARRY_GRASP_LOCK_POSTPROCESSOR_VERSION
        ),
        "grasp_lock_reference": CARRY_GRASP_LOCK_REFERENCE,
        "ordered_raw_joint_names": list(names),
        "locked_raw_joint_indices": list(locked_indices),
        "locked_raw_joint_names": [names[index] for index in locked_indices],
        "locked_raw_joint_mask": [
            index in locked_index_set for index in range(PROCESSED_JOINT_COUNT)
        ],
        "high_level_runtime_postprocessor_required": True,
        "processed_43d_targets_embed_grasp_lock": True,
    }
    contract[_CONTRACT_HASH_KEY] = carry_runtime_contract_sha256(contract)
    return contract


def validate_carry_runtime_contract(
    contract: Mapping[str, Any],
    *,
    expected_ordered_joint_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a normalized contract or reject any semantic/hash mismatch."""

    if not isinstance(contract, Mapping):
        raise ValueError("carry runtime contract must be a mapping")
    if contract.get("schema") != CARRY_RUNTIME_CONTRACT_SCHEMA:
        raise ValueError(
            f"unsupported carry runtime contract schema: {contract.get('schema')!r}"
        )

    ramp_steps = contract.get("navigation_ramp_steps")
    if (
        isinstance(ramp_steps, bool)
        or not isinstance(ramp_steps, int)
        or ramp_steps <= 0
    ):
        raise ValueError("carry navigation_ramp_steps must be a positive integer")
    if contract.get("navigation_ramp_profile") != CARRY_NAVIGATION_RAMP_PROFILE:
        raise ValueError("unsupported carry navigation ramp profile")
    if contract.get("navigation_command_observation_field") != "velocity_commands":
        raise ValueError("carry navigation command must target velocity_commands")
    if contract.get("navigation_command_high_level_slice") != [16, 19]:
        raise ValueError("carry high-level navigation slice must be [16, 19]")
    if (
        contract.get("grasp_lock_postprocessor_version")
        != CARRY_GRASP_LOCK_POSTPROCESSOR_VERSION
    ):
        raise ValueError("unsupported carry grasp-lock postprocessor version")
    if contract.get("grasp_lock_reference") != CARRY_GRASP_LOCK_REFERENCE:
        raise ValueError("unsupported carry grasp-lock reference")
    if contract.get("high_level_runtime_postprocessor_required") is not True:
        raise ValueError("carry high-level policy must require grasp-lock postprocessing")
    if contract.get("processed_43d_targets_embed_grasp_lock") is not True:
        raise ValueError("carry processed 43-D targets must embed the grasp lock")

    names = _ordered_joint_names(contract.get("ordered_raw_joint_names", ()))
    canonical_indices = carry_grasp_lock_indices(names)
    indices = contract.get("locked_raw_joint_indices")
    if not isinstance(indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in indices
    ):
        raise ValueError("carry locked_raw_joint_indices must be an integer list")
    if tuple(indices) != canonical_indices:
        raise ValueError("carry locked joint indices do not match grasp-lock v1")
    locked_names = contract.get("locked_raw_joint_names")
    canonical_locked_names = [names[index] for index in canonical_indices]
    if locked_names != canonical_locked_names:
        raise ValueError("carry locked joint names do not match their ordered indices")
    mask = contract.get("locked_raw_joint_mask")
    canonical_mask = [index in set(canonical_indices) for index in range(len(names))]
    if (
        not isinstance(mask, list)
        or any(not isinstance(value, bool) for value in mask)
        or mask != canonical_mask
    ):
        raise ValueError("carry locked joint mask does not match its exact joint order")

    if expected_ordered_joint_names is not None:
        expected = _ordered_joint_names(expected_ordered_joint_names)
        if names != expected:
            raise ValueError(
                "runtime raw joint order does not match the carry checkpoint contract"
            )

    expected_hash = carry_runtime_contract_sha256(contract)
    if contract.get(_CONTRACT_HASH_KEY) != expected_hash:
        raise ValueError("carry runtime contract hash is missing or invalid")
    normalized = _hash_payload(contract)
    normalized[_CONTRACT_HASH_KEY] = expected_hash
    return normalized


def carry_navigation_scale(carry_step: int, contract: Mapping[str, Any]) -> float:
    """Return the v1 pre-step CARRY navigation scale from a validated contract."""

    if isinstance(carry_step, bool) or not isinstance(carry_step, int) or carry_step < 0:
        raise ValueError("carry_step must be a non-negative integer")
    validated = validate_carry_runtime_contract(contract)
    fraction = (carry_step + 1) / float(validated["navigation_ramp_steps"])
    return min(1.0, fraction)
