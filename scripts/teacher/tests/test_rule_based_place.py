from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "source/unitree_rl_lab"))

from unitree_rl_lab.utils.teacher.rule_based_place import box_fits_opening


def test_center_inside_does_not_allow_box_to_overlap_wall():
    assert not box_fits_opening(
        [0.23, 0, 0.8, 1, 0, 0, 0], [0.2] * 3, [0, 0, 0.28], [0.3, 0.35, 0.21]
    )


def test_rotation_changes_required_clearance():
    kwargs = dict(box_size=[0.2] * 3, bin_center=[0, 0, 0.28], bin_half_extents=[0.3, 0.35, 0.21])
    assert box_fits_opening([0.15, 0, 0.8, 1, 0, 0, 0], **kwargs)
    assert not box_fits_opening(
        [0.15, 0, 0.8, np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)], **kwargs
    )


def test_centered_box_clears_opening():
    assert box_fits_opening(
        [1.55, -0.95, 0.8, 1, 0, 0, 0], [0.2] * 3, [1.55, -0.95, 0.28], [0.3, 0.35, 0.21]
    )
