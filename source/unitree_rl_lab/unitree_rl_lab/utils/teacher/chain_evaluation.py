"""Simulator-independent accounting for uninterrupted grasp/carry/place runs."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def summarize_live_chain(
    episodes: Sequence[Mapping[str, Any]], *, require_place: bool
) -> dict[str, Any]:
    if not episodes:
        raise ValueError("A chain evaluation needs at least one completed episode")
    grasp_count = carry_count = place_count = complete_count = orientation_count = 0
    failure_stages: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()

    for episode in episodes:
        grasp = episode.get("success")
        carry_result = episode.get("carry")
        if type(grasp) is not bool or not isinstance(carry_result, Mapping):
            raise ValueError("Expected a live-chain episode with grasp and Carry results")
        carry = carry_result.get("success")
        if type(carry) is not bool or (carry and not grasp):
            raise ValueError("Carry cannot succeed without grasp success")
        place_result = episode.get("place", {})
        if not isinstance(place_result, Mapping):
            raise ValueError("Expected a mapping for the Place result")
        place = place_result.get("success", False)
        if require_place and (not isinstance(place_result, Mapping) or type(place) is not bool):
            raise ValueError("Expected a boolean Place result")
        if require_place and carry and "success" not in place_result:
            raise ValueError("A successful Carry needs a completed Place result")
        if require_place and place and not carry:
            raise ValueError("Place cannot succeed without Carry success")
        complete = grasp and carry and (place or not require_place)
        if "chain_success" in episode and episode["chain_success"] is not complete:
            raise ValueError("chain_success disagrees with the individual stages")
        grasp_count += int(grasp)
        carry_count += int(carry)
        place_count += int(place) if require_place else 0
        complete_count += int(complete)
        if not grasp:
            stage, reason = "approach_or_grasp", episode.get("reason", "unknown")
        elif not carry:
            stage, reason = "carry", carry_result.get("reason", "unknown")
        elif require_place and not place:
            stage, reason = "place", place_result.get("reason", "unknown")
        else:
            continue
        failure_stages[stage] += 1
        failure_reasons[f"{stage}:{reason}"] += 1
        orientation_count += int(bool(episode.get("bad_orientation")) or "bad_orientation" in str(reason))

    def conditional(successes: int, eligible: int):
        return {"successes": successes, "eligible": eligible,
                "rate": successes / eligible if eligible else None}

    return {
        "episodes": len(episodes),
        "required_end_stage": "place" if require_place else "carry",
        "full_task_successes": complete_count,
        "full_task_success_rate": complete_count / len(episodes),
        "approach_and_grasp": conditional(grasp_count, len(episodes)),
        "carry_given_grasp": conditional(carry_count, grasp_count),
        "place_given_carry": conditional(place_count, carry_count) if require_place else None,
        "bad_orientation_episodes": orientation_count,
        "failure_stages": dict(failure_stages),
        "failure_reasons": dict(failure_reasons),
        "approach_controller": "teacher_trajectory_prefix",
        "approach_has_independent_physical_gate": False,
    }
