from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "source" / "unitree_rl_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from unitree_rl_lab.utils.teacher.arena_dataset import (  # noqa: E402
    ARENA_G1_LAB_JOINT_NAMES,
)
from unitree_rl_lab.utils.teacher.carry_runtime_contract import (  # noqa: E402
    CARRY_GRASP_LOCK_POSTPROCESSOR_VERSION,
    build_carry_runtime_contract,
    carry_navigation_scale,
    validate_carry_runtime_contract,
)
from unitree_rl_lab.utils.teacher.phase_bc import (  # noqa: E402
    EpisodeSplit,
    PhaseArrays,
    TrainConfig,
    load_bc_policy,
    train_phase_policy,
    validate_carry_teacher_runtime_contract,
    validate_place_teacher_runtime_contract,
)
from unitree_rl_lab.utils.teacher.place_runtime_contract import (  # noqa: E402
    PLACE_POSTPROCESSOR_VERSION,
    apply_place_runtime_postprocessor,
    build_place_runtime_contract,
    place_arm_offset_decay,
    place_finger_lock_active,
    validate_place_runtime_contract,
)
from unitree_rl_lab.utils.teacher.skill_fsm import SkillPhase  # noqa: E402
from unitree_rl_lab.utils.teacher.student_observation import (  # noqa: E402
    STUDENT_OBSERVATION_DIM,
)
from unitree_rl_lab.utils.teacher_demo_recorder import (  # noqa: E402
    DEFAULT_HIGH_LEVEL_ACTION_NAMES,
)


def _write_contract_source(path: Path, *, ramp_steps: int) -> dict:
    contract = build_carry_runtime_contract(
        ARENA_G1_LAB_JOINT_NAMES, navigation_ramp_steps=ramp_steps
    )
    metadata = {
        "teacher_phase": "carry",
        "carry_runtime_contract": contract,
        "carry_runtime_contract_sha256": contract["contract_sha256"],
    }
    with h5py.File(path, "w") as output:
        output.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        group = output.create_group("metadata")
        string_dtype = h5py.string_dtype("utf-8")
        group.create_dataset(
            "processed_joint_names",
            data=np.asarray(ARENA_G1_LAB_JOINT_NAMES, dtype=object),
            dtype=string_dtype,
        )
        group.create_dataset(
            "teacher_high_level_action_names",
            data=np.asarray(DEFAULT_HIGH_LEVEL_ACTION_NAMES, dtype=object),
            dtype=string_dtype,
        )
    return contract


def _write_place_contract_source(
    path: Path,
    *,
    joint_names=ARENA_G1_LAB_JOINT_NAMES,
    decay_steps: int = 100,
) -> dict:
    contract = build_place_runtime_contract(
        joint_names, arm_offset_decay_steps=decay_steps
    )
    metadata = {
        "teacher_phase": "place",
        "place_runtime_contract": contract,
        "place_runtime_contract_sha256": contract["contract_sha256"],
    }
    with h5py.File(path, "w") as output:
        output.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        group = output.create_group("metadata")
        string_dtype = h5py.string_dtype("utf-8")
        group.create_dataset(
            "processed_joint_names",
            data=np.asarray(joint_names, dtype=object),
            dtype=string_dtype,
        )
        group.create_dataset(
            "teacher_high_level_action_names",
            data=np.asarray(DEFAULT_HIGH_LEVEL_ACTION_NAMES, dtype=object),
            dtype=string_dtype,
        )
    return contract


def _arrays(
    path: Path, action_key: str, action_dim: int, *, skill=SkillPhase.CARRY
) -> tuple[PhaseArrays, ...]:
    rng = np.random.default_rng(9)
    values = []
    for subset, count in (("train", 8), ("validation", 3), ("test", 3)):
        values.append(
            PhaseArrays(
                observations=rng.normal(
                    size=(count, STUDENT_OBSERVATION_DIM)
                ).astype(np.float32),
                actions=rng.normal(scale=0.1, size=(count, action_dim)).astype(
                    np.float32
                ),
                episode_ids=(f"{path.resolve()}::{subset}",),
                sample_episode_index=np.zeros(count, dtype=np.int32),
                skill=skill,
                action_key=action_key,
            )
        )
    return tuple(values)


def test_contract_binds_ramp_profile_and_exact_lock_mask() -> None:
    contract = build_carry_runtime_contract(
        ARENA_G1_LAB_JOINT_NAMES, navigation_ramp_steps=7
    )
    assert validate_carry_runtime_contract(
        contract, expected_ordered_joint_names=ARENA_G1_LAB_JOINT_NAMES
    ) == contract
    assert contract["grasp_lock_postprocessor_version"] == (
        CARRY_GRASP_LOCK_POSTPROCESSOR_VERSION
    )
    assert len(contract["locked_raw_joint_names"]) == 28
    assert sum(contract["locked_raw_joint_mask"]) == 28
    assert carry_navigation_scale(0, contract) == pytest.approx(1.0 / 7.0)
    assert carry_navigation_scale(6, contract) == 1.0
    assert carry_navigation_scale(100, contract) == 1.0

    tampered = dict(contract)
    tampered["locked_raw_joint_mask"] = list(contract["locked_raw_joint_mask"])
    tampered["locked_raw_joint_mask"][0] = not tampered["locked_raw_joint_mask"][0]
    with pytest.raises(ValueError, match="mask"):
        validate_carry_runtime_contract(tampered)


def test_all_carry_sources_must_have_the_identical_contract(tmp_path: Path) -> None:
    first = tmp_path / "carry_a.hdf5"
    second = tmp_path / "carry_b.hdf5"
    expected = _write_contract_source(first, ramp_steps=50)
    _write_contract_source(second, ramp_steps=50)
    assert validate_carry_teacher_runtime_contract([first, second]) == expected

    _write_contract_source(second, ramp_steps=25)
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_carry_teacher_runtime_contract([first, second])


def test_place_contract_requires_bilateral_release_and_exact_finger_mask() -> None:
    contract = build_place_runtime_contract(ARENA_G1_LAB_JOINT_NAMES)
    assert validate_place_runtime_contract(
        contract, expected_ordered_joint_names=ARENA_G1_LAB_JOINT_NAMES
    ) == contract
    assert contract["postprocessor_version"] == PLACE_POSTPROCESSOR_VERSION
    assert contract["schema"] == "unitree_box_place_runtime_contract_v4"
    assert contract["arm_offset_decay_steps"] == 100
    assert place_arm_offset_decay(0, contract) == 1.0
    assert place_arm_offset_decay(99, contract) == 0.0
    assert place_arm_offset_decay(1000, contract) == 0.0
    assert place_arm_offset_decay(49, contract) + place_arm_offset_decay(
        50, contract
    ) == pytest.approx(1.0)
    assert contract["source_pink_raw_baseline_used_in_arm_delta"] is False
    assert contract[
        "source_pink_raw_baseline_required_for_restored_collection"
    ] is True
    assert len(contract["arm_raw_joint_names"]) == 14
    assert sum(contract["arm_raw_joint_mask"]) == 14
    assert len(contract["finger_raw_joint_names"]) == 14
    assert sum(contract["finger_raw_joint_mask"]) == 14
    assert place_finger_lock_active((1.0, 1.0), contract)
    assert place_finger_lock_active((0.4, 1.0), contract)
    assert place_finger_lock_active((1.0, 0.4), contract)
    assert not place_finger_lock_active((0.5, 0.5), contract)

    proposed = np.zeros((1, 43), dtype=np.float32)
    loaded = np.full((1, 43), 2.0, dtype=np.float32)
    baseline = np.full((1, 43), -0.5, dtype=np.float32)
    transformed = apply_place_runtime_postprocessor(
        proposed, loaded, baseline, (1.0, 0.4), 0, contract
    )
    arms = contract["arm_raw_joint_indices"]
    fingers = contract["finger_raw_joint_indices"]
    assert np.allclose(transformed[:, arms], 2.5)
    assert np.allclose(transformed[:, fingers], 2.0)
    fully_decayed = apply_place_runtime_postprocessor(
        proposed, loaded, baseline, (1.0, 1.0), 99, contract
    )
    assert np.allclose(fully_decayed[:, arms], 0.0)
    assert np.allclose(fully_decayed[:, fingers], 2.0)
    released = apply_place_runtime_postprocessor(
        proposed, loaded, baseline, (0.5, 0.5), 0, contract
    )
    assert np.allclose(released[:, arms], 2.5)
    assert np.allclose(released[:, fingers], 0.0)


def test_all_place_sources_must_have_the_identical_contract(tmp_path: Path) -> None:
    first = tmp_path / "place_a.hdf5"
    second = tmp_path / "place_b.hdf5"
    expected = _write_place_contract_source(first)
    _write_place_contract_source(second)
    assert validate_place_teacher_runtime_contract([first, second]) == expected

    _write_place_contract_source(second, decay_steps=80)
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_place_teacher_runtime_contract([first, second])

    reordered = list(ARENA_G1_LAB_JOINT_NAMES)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    _write_place_contract_source(second, joint_names=reordered)
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_place_teacher_runtime_contract([first, second])


@pytest.mark.parametrize(
    ("action_key", "action_dim"),
    (("teacher_high_level_action", 23), ("processed_43d_joint_target", 43)),
)
def test_carry_checkpoint_stores_and_loads_runtime_contract(
    tmp_path: Path, action_key: str, action_dim: int
) -> None:
    source = tmp_path / f"carry_{action_dim}.hdf5"
    checkpoint = tmp_path / f"carry_{action_dim}.pt"
    contract = _write_contract_source(source, ramp_steps=9)
    train, validation, test = _arrays(source, action_key, action_dim)
    report = train_phase_policy(
        train_arrays=train,
        validation_arrays=validation,
        test_arrays=test,
        split=EpisodeSplit(
            train=train.episode_ids,
            validation=validation.episode_ids,
            test=test.episode_ids,
            seed=3,
            ratios=(0.6, 0.2, 0.2),
        ),
        output_path=checkpoint,
        config=TrainConfig(
            hidden_dims=(8,), batch_size=8, epochs=1, patience=1, seed=3
        ),
        device="cpu",
        source_paths=[source],
    )
    assert report["carry_runtime_contract"] == contract
    assert report["carry_runtime_contract_sha256"] == contract["contract_sha256"]

    loaded = load_bc_policy(checkpoint)
    assert loaded.carry_runtime_contract == contract
    assert loaded.carry_runtime_contract_sha256 == contract["contract_sha256"]

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved["metadata"]["carry_runtime_contract"]["navigation_ramp_steps"] = 10
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="contract hash"):
        load_bc_policy(checkpoint)


@pytest.mark.parametrize(
    ("action_key", "action_dim"),
    (("teacher_high_level_action", 23), ("processed_43d_joint_target", 43)),
)
def test_place_checkpoint_stores_and_loads_runtime_contract(
    tmp_path: Path, action_key: str, action_dim: int
) -> None:
    source = tmp_path / f"place_{action_dim}.hdf5"
    checkpoint = tmp_path / f"place_{action_dim}.pt"
    contract = _write_place_contract_source(source)
    train, validation, test = _arrays(
        source, action_key, action_dim, skill=SkillPhase.PLACE
    )
    report = train_phase_policy(
        train_arrays=train,
        validation_arrays=validation,
        test_arrays=test,
        split=EpisodeSplit(
            train=train.episode_ids,
            validation=validation.episode_ids,
            test=test.episode_ids,
            seed=4,
            ratios=(0.6, 0.2, 0.2),
        ),
        output_path=checkpoint,
        config=TrainConfig(
            hidden_dims=(8,), batch_size=8, epochs=1, patience=1, seed=4
        ),
        device="cpu",
        source_paths=[source],
    )
    assert report["place_runtime_contract"] == contract
    assert report["place_runtime_contract_sha256"] == contract["contract_sha256"]

    loaded = load_bc_policy(checkpoint)
    assert loaded.place_runtime_contract == contract
    assert loaded.place_runtime_contract_sha256 == contract["contract_sha256"]
