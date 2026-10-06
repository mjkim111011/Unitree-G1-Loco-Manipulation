"""Semantic contract for decayed PLACE arm preload and bilateral release.

Pink/WBC maps the two high-level hand-state channels to generic Dex3 targets.
Those targets are not necessarily the finger configuration that physically
carried the Box into PLACE. Its unloaded current-pose arm solution also drops
the PD preload supporting the Box. A 23-D PLACE action therefore starts with
the loaded offset ``loaded - active-controller baseline``, smoothly decays that
offset while following Pink, and retains the loaded finger targets until both
hand channels explicitly request release. This preserves the physical handoff
without permanently biasing a large PLACE translation away from Pink's target.
The original transition proposal stored in a physical snapshot remains an
audit requirement but is never mixed with a newly initialized controller's
proposal. Processed 43-D teacher labels already embed both operations.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


PLACE_RUNTIME_CONTRACT_SCHEMA = "unitree_box_place_runtime_contract_v4"
PLACE_POSTPROCESSOR_VERSION = (
    "place_decayed_loaded_arm_offset_and_finger_lock_until_release_v1"
)
PLACE_LOADED_REFERENCE = "carry_to_place_transition_last_applied_action_raw"
PLACE_ACTIVE_PINK_BASELINE_REFERENCE = (
    "same_active_controller_carry_to_place_unprocessed_pink_wbc_raw"
)
PLACE_SOURCE_PINK_BASELINE_PROVENANCE = (
    "carry_at_staging_snapshot_original_transition_unprocessed_wbc_raw_audit_only"
)
PLACE_RELEASE_HAND_STATE_THRESHOLD = 0.5
DEFAULT_PLACE_ARM_OFFSET_DECAY_STEPS = 100
PLACE_ARM_OFFSET_DECAY_PROFILE = "smoothstep_one_to_zero_inclusive_v1"
PLACE_ARM_JOINT_COUNT = 14
PLACE_LOCKED_FINGER_COUNT = 14
PROCESSED_JOINT_COUNT = 43

_CONTRACT_HASH_KEY = "contract_sha256"


def _ordered_joint_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if (
        len(names) != PROCESSED_JOINT_COUNT
        or len(set(names)) != PROCESSED_JOINT_COUNT
        or any(not name for name in names)
    ):
        raise ValueError(
            f"place runtime contract requires {PROCESSED_JOINT_COUNT} unique "
            "non-empty raw joint names"
        )
    return names


def place_finger_lock_indices(ordered_joint_names: Sequence[str]) -> tuple[int, ...]:
    """Return the exact 14 Dex3 indices for an ordered raw action."""

    names = _ordered_joint_names(ordered_joint_names)
    indices = tuple(index for index, name in enumerate(names) if "_hand_" in name)
    if len(indices) != PLACE_LOCKED_FINGER_COUNT:
        raise ValueError(
            f"place finger-lock v1 requires {PLACE_LOCKED_FINGER_COUNT} hand "
            f"joints, got {len(indices)}"
        )
    return indices


def place_arm_delta_indices(ordered_joint_names: Sequence[str]) -> tuple[int, ...]:
    """Return the exact 14 shoulder/elbow/wrist delta-anchor indices."""

    names = _ordered_joint_names(ordered_joint_names)
    indices = tuple(
        index
        for index, name in enumerate(names)
        if any(token in name for token in ("shoulder", "elbow", "wrist"))
    )
    if len(indices) != PLACE_ARM_JOINT_COUNT:
        raise ValueError(
            f"place arm-delta v1 requires {PLACE_ARM_JOINT_COUNT} "
            f"shoulder/elbow/wrist joints, got {len(indices)}"
        )
    return indices


def _hash_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(contract)
    payload.pop(_CONTRACT_HASH_KEY, None)
    return payload


def place_runtime_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Hash a place contract independently of its stored hash field."""

    payload = json.dumps(
        _hash_payload(contract), separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_place_runtime_contract(
    ordered_joint_names: Sequence[str],
    *,
    arm_offset_decay_steps: int = DEFAULT_PLACE_ARM_OFFSET_DECAY_STEPS,
) -> dict[str, Any]:
    """Build the canonical PLACE runtime contract for one articulation order."""

    if (
        isinstance(arm_offset_decay_steps, bool)
        or not isinstance(arm_offset_decay_steps, int)
        or arm_offset_decay_steps < 2
    ):
        raise ValueError("place arm_offset_decay_steps must be an integer >= 2")
    names = _ordered_joint_names(ordered_joint_names)
    arm_indices = place_arm_delta_indices(names)
    finger_indices = place_finger_lock_indices(names)
    arm_index_set = set(arm_indices)
    finger_index_set = set(finger_indices)
    if arm_index_set.intersection(finger_index_set):
        raise ValueError("place arm and finger masks must be disjoint")
    contract: dict[str, Any] = {
        "schema": PLACE_RUNTIME_CONTRACT_SCHEMA,
        "postprocessor_version": PLACE_POSTPROCESSOR_VERSION,
        "loaded_raw_reference": PLACE_LOADED_REFERENCE,
        "active_pink_raw_baseline_reference": (
            PLACE_ACTIVE_PINK_BASELINE_REFERENCE
        ),
        "source_pink_raw_baseline_provenance": (
            PLACE_SOURCE_PINK_BASELINE_PROVENANCE
        ),
        "source_pink_raw_baseline_required_for_restored_collection": True,
        "source_pink_raw_baseline_used_in_arm_delta": False,
        "arm_offset_equation": (
            "pink_proposed + decay(place_step) * "
            "(loaded_ref - active_pink_baseline)"
        ),
        "arm_offset_decay_steps": arm_offset_decay_steps,
        "arm_offset_decay_profile": PLACE_ARM_OFFSET_DECAY_PROFILE,
        "arm_offset_decay_step_origin": "first_recorded_place_step_is_zero",
        "release_hand_state_slice": [0, 2],
        "release_hand_state_threshold": PLACE_RELEASE_HAND_STATE_THRESHOLD,
        "bilateral_release_predicate": "both_hand_states_lte_threshold",
        "ordered_raw_joint_names": list(names),
        "arm_raw_joint_indices": list(arm_indices),
        "arm_raw_joint_names": [names[index] for index in arm_indices],
        "arm_raw_joint_mask": [
            index in arm_index_set for index in range(PROCESSED_JOINT_COUNT)
        ],
        "finger_raw_joint_indices": list(finger_indices),
        "finger_raw_joint_names": [names[index] for index in finger_indices],
        "finger_raw_joint_mask": [
            index in finger_index_set for index in range(PROCESSED_JOINT_COUNT)
        ],
        "high_level_runtime_postprocessor_required": True,
        "processed_43d_targets_embed_decayed_arm_offset_and_finger_lock": True,
    }
    contract[_CONTRACT_HASH_KEY] = place_runtime_contract_sha256(contract)
    return contract


def validate_place_runtime_contract(
    contract: Mapping[str, Any],
    *,
    expected_ordered_joint_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a normalized contract or reject a semantic/hash mismatch."""

    if not isinstance(contract, Mapping):
        raise ValueError("place runtime contract must be a mapping")
    if contract.get("schema") != PLACE_RUNTIME_CONTRACT_SCHEMA:
        raise ValueError(
            f"unsupported place runtime contract schema: {contract.get('schema')!r}"
        )
    if (
        contract.get("postprocessor_version")
        != PLACE_POSTPROCESSOR_VERSION
    ):
        raise ValueError("unsupported place postprocessor version")
    if contract.get("loaded_raw_reference") != PLACE_LOADED_REFERENCE:
        raise ValueError("unsupported place loaded-action reference")
    if (
        contract.get("active_pink_raw_baseline_reference")
        != PLACE_ACTIVE_PINK_BASELINE_REFERENCE
    ):
        raise ValueError("unsupported place active Pink baseline reference")
    if (
        contract.get("source_pink_raw_baseline_provenance")
        != PLACE_SOURCE_PINK_BASELINE_PROVENANCE
        or contract.get(
            "source_pink_raw_baseline_required_for_restored_collection"
        )
        is not True
        or contract.get("source_pink_raw_baseline_used_in_arm_delta") is not False
    ):
        raise ValueError("unsupported place source Pink baseline provenance")
    if contract.get("arm_offset_equation") != (
        "pink_proposed + decay(place_step) * "
        "(loaded_ref - active_pink_baseline)"
    ):
        raise ValueError("unsupported place loaded-arm offset equation")
    decay_steps = contract.get("arm_offset_decay_steps")
    if (
        isinstance(decay_steps, bool)
        or not isinstance(decay_steps, int)
        or decay_steps < 2
    ):
        raise ValueError("place arm_offset_decay_steps must be an integer >= 2")
    if contract.get("arm_offset_decay_profile") != PLACE_ARM_OFFSET_DECAY_PROFILE:
        raise ValueError("unsupported place arm-offset decay profile")
    if contract.get("arm_offset_decay_step_origin") != (
        "first_recorded_place_step_is_zero"
    ):
        raise ValueError("unsupported place arm-offset decay step origin")
    if contract.get("release_hand_state_slice") != [0, 2]:
        raise ValueError("place release hand-state slice must be [0, 2]")
    threshold = contract.get("release_hand_state_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isclose(
            float(threshold),
            PLACE_RELEASE_HAND_STATE_THRESHOLD,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("unsupported place release hand-state threshold")
    if contract.get("bilateral_release_predicate") != (
        "both_hand_states_lte_threshold"
    ):
        raise ValueError("unsupported place bilateral release predicate")
    if contract.get("high_level_runtime_postprocessor_required") is not True:
        raise ValueError("place high-level policy must require v4 postprocessing")
    if (
        contract.get(
            "processed_43d_targets_embed_decayed_arm_offset_and_finger_lock"
        )
        is not True
    ):
        raise ValueError(
            "place processed 43-D targets must embed decayed arm offset and finger lock"
        )

    names = _ordered_joint_names(contract.get("ordered_raw_joint_names", ()))
    for label, canonical_indices in (
        ("arm", place_arm_delta_indices(names)),
        ("finger", place_finger_lock_indices(names)),
    ):
        indices = contract.get(f"{label}_raw_joint_indices")
        if not isinstance(indices, list) or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in indices
        ):
            raise ValueError(
                f"place {label}_raw_joint_indices must be an integer list"
            )
        if tuple(indices) != canonical_indices:
            raise ValueError(
                f"place {label} indices do not match loaded-place v4"
            )
        joint_names = contract.get(f"{label}_raw_joint_names")
        canonical_joint_names = [names[index] for index in canonical_indices]
        if joint_names != canonical_joint_names:
            raise ValueError(
                f"place {label} names do not match their ordered indices"
            )
        index_set = set(canonical_indices)
        canonical_mask = [index in index_set for index in range(len(names))]
        mask = contract.get(f"{label}_raw_joint_mask")
        if (
            not isinstance(mask, list)
            or any(not isinstance(value, bool) for value in mask)
            or mask != canonical_mask
        ):
            raise ValueError(
                f"place {label} mask does not match its exact joint order"
            )

    if expected_ordered_joint_names is not None:
        expected = _ordered_joint_names(expected_ordered_joint_names)
        if names != expected:
            raise ValueError(
                "runtime raw joint order does not match the place checkpoint contract"
            )

    expected_hash = place_runtime_contract_sha256(contract)
    if contract.get(_CONTRACT_HASH_KEY) != expected_hash:
        raise ValueError("place runtime contract hash is missing or invalid")
    normalized = _hash_payload(contract)
    normalized[_CONTRACT_HASH_KEY] = expected_hash
    return normalized


def place_finger_lock_active(
    hand_states: Sequence[float], contract: Mapping[str, Any]
) -> bool:
    """Keep both realized finger targets unless both hands request release."""

    validated = validate_place_runtime_contract(contract)
    values = tuple(float(value) for value in hand_states)
    if len(values) != 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("place hand_states must contain two finite values")
    threshold = float(validated["release_hand_state_threshold"])
    bilateral_release = all(value <= threshold for value in values)
    return not bilateral_release


def place_arm_offset_decay(place_step: int, contract: Mapping[str, Any]) -> float:
    """Return the v4 inclusive smoothstep decay for one recorded PLACE step."""

    if isinstance(place_step, bool) or not isinstance(place_step, int) or place_step < 0:
        raise ValueError("place_step must be a non-negative integer")
    validated = validate_place_runtime_contract(contract)
    decay_steps = int(validated["arm_offset_decay_steps"])
    progress = min(1.0, place_step / float(decay_steps - 1))
    smoothstep = progress * progress * (3.0 - 2.0 * progress)
    return 1.0 - smoothstep


def apply_place_runtime_postprocessor(
    pink_proposed_raw_action: Any,
    loaded_reference_raw_action: Any,
    active_pink_baseline_raw_action: Any,
    hand_states: Sequence[float],
    place_step: int,
    contract: Mapping[str, Any],
) -> Any:
    """Apply the exact v4 PLACE transform to NumPy arrays or torch tensors."""

    validated = validate_place_runtime_contract(contract)
    expected_shape = tuple(pink_proposed_raw_action.shape)
    if not expected_shape or expected_shape[-1] != PROCESSED_JOINT_COUNT:
        raise ValueError("PLACE raw actions must end in 43 ordered joints")
    for name, value in (
        ("loaded reference", loaded_reference_raw_action),
        ("active Pink baseline", active_pink_baseline_raw_action),
    ):
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"PLACE {name} shape {tuple(value.shape)} does not match "
                f"proposed action shape {expected_shape}"
            )

    result = (
        pink_proposed_raw_action.clone()
        if hasattr(pink_proposed_raw_action, "clone")
        else pink_proposed_raw_action.copy()
    )
    arm_indices = tuple(validated["arm_raw_joint_indices"])
    decay = place_arm_offset_decay(place_step, validated)
    result[..., arm_indices] = pink_proposed_raw_action[..., arm_indices] + decay * (
        loaded_reference_raw_action[..., arm_indices]
        - active_pink_baseline_raw_action[..., arm_indices]
    )
    if place_finger_lock_active(hand_states, validated):
        finger_indices = tuple(validated["finger_raw_joint_indices"])
        result[..., finger_indices] = loaded_reference_raw_action[..., finger_indices]
    return result
