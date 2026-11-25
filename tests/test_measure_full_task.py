import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "measure_full_task", Path(__file__).resolve().parents[1] / "scripts/teacher/measure_full_task.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def document(grasp=True, carry=True, place=True):
    return {
        "evaluator_config": {"seed": 42, "box_mass_kg": 0.2, "rule_based_place": True,
                             "snapshot_selection": [], "residual_checkpoint": "model.pt"},
        "episode_results": [{"success": grasp, "reason": "horizon",
                             "carry": {"success": carry, "reason": "box_drop"},
                             "place": {"success": place, "reason": "place_horizon"},
                             "chain_success": grasp and carry and place}],
    }


def test_all_started_trials_in_denominator_and_failure_stage():
    docs = [document(), document(False, False, False), document(True, False, False),
            document(True, True, False)]
    rows = [module.live_result(d, 42, 0.2, "grip") for d in docs]
    summary = module.report(rows, 5)
    assert summary["full_task_success_rate"] == 0.25
    assert summary["failure_stages"] == {"approach_or_grasp": 1, "carry": 1, "place": 1}
    assert summary["stage_successes_over_all_starts"] == {"grasp": 3, "carry": 2, "place": 1}
    assert not summary["complete"]


def test_missing_place_is_not_a_robot_failure():
    d = document()
    del d["episode_results"][0]["place"]
    with pytest.raises(ValueError, match="Incomplete"):
        module.live_result(d, 42, 0.2, "grip")


def test_snapshot_or_carry_only_cannot_be_reported_as_full_task():
    for key, value in [("snapshot_selection", ["snapshot:0"]), ("rule_based_place", False)]:
        d = document()
        d["evaluator_config"][key] = value
        with pytest.raises(ValueError, match="uninterrupted"):
            module.live_result(d, 42, 0.2, "grip")


def test_wrong_method_seed_or_mass_is_rejected():
    for seed, mass, method in [(43, 0.2, "grip"), (42, 0.3, "grip"), (42, 0.2, "baseline")]:
        with pytest.raises(ValueError):
            module.live_result(document(), seed, mass, method)


def test_impossible_downstream_success_or_wrong_total_is_rejected():
    for d in [document(False, True, True), document(True, False, True)]:
        with pytest.raises(ValueError, match="Impossible"):
            module.live_result(d, 42, 0.2, "grip")
    d = document()
    d["episode_results"][0]["chain_success"] = False
    with pytest.raises(ValueError, match="disagrees"):
        module.live_result(d, 42, 0.2, "grip")
