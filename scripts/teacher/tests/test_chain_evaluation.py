from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "source/unitree_rl_lab"))

from unitree_rl_lab.utils.teacher.chain_evaluation import summarize_live_chain


def test_successful_grasp_is_not_successful_task():
    report = summarize_live_chain([
        {"success": True, "carry": {"success": False, "reason": "box_drop"},
         "place": {"success": False, "reason": "carry_not_reached"}},
    ], require_place=True)
    assert report["approach_and_grasp"]["rate"] == 1.0
    assert report["full_task_success_rate"] == 0.0
    assert report["failure_stages"] == {"carry": 1}
    assert report["place_given_carry"]["rate"] is None


def test_conditional_rates_and_failures_use_correct_denominators():
    report = summarize_live_chain([
        {"success": False, "reason": "horizon", "carry": {"success": False}},
        {"success": True, "carry": {"success": False, "reason": "bad_orientation"}},
        {"success": True, "carry": {"success": True}, "place": {"success": False, "reason": "alignment_timeout"}},
        {"success": True, "carry": {"success": True}, "place": {"success": True}},
    ], require_place=True)
    assert report["full_task_success_rate"] == 0.25
    assert report["carry_given_grasp"]["rate"] == pytest.approx(2 / 3)
    assert report["place_given_carry"]["rate"] == 0.5
    assert sum(report["failure_stages"].values()) == 3
    assert report["bad_orientation_episodes"] == 1


def test_carry_only_run_does_not_require_place():
    report = summarize_live_chain([
        {"success": True, "carry": {"success": True}, "chain_success": True},
    ], require_place=False)
    assert report["full_task_success_rate"] == 1.0


def test_cannot_claim_success_after_failed_upstream_stage():
    with pytest.raises(ValueError, match="Carry cannot"):
        summarize_live_chain([{"success": False, "carry": {"success": True}}], require_place=True)
    with pytest.raises(ValueError, match="chain_success"):
        summarize_live_chain([
            {"success": True, "carry": {"success": False}, "chain_success": True},
        ], require_place=False)


def test_missing_place_result_is_not_accepted_as_evaluation():
    with pytest.raises(ValueError, match="completed Place"):
        summarize_live_chain([{"success": True, "carry": {"success": True}}], require_place=True)
