from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "overlays/ablation/verl/verl/anti_opd/teacher_distribution_defense.py"
)
SPEC = importlib.util.spec_from_file_location("sasr_ablation_defense", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
TeacherDefenseConfig = MODULE.TeacherDefenseConfig
apply_sparse_asymmetric_projection = MODULE.apply_sparse_asymmetric_projection


def _config(**overrides: object) -> TeacherDefenseConfig:
    values = {
        "enabled": True,
        "method": "sparse_asymmetric_projection",
        "active_ratio": 0.40,
        "selection_strategy": "top_abs_advantage",
        "selection_random_seed": 42,
        "positive_weight": 0.5,
        "negative_weight": 1.0,
        "k_positive": 0.45,
        "k_negative": -0.10,
        "lambda_value": 1.05,
        "allow_sign_flip": True,
    }
    values.update(overrides)
    config = TeacherDefenseConfig(**values)
    config.validate()
    return config


def _inputs() -> dict[str, torch.Tensor | None]:
    probabilities = torch.tensor(
        [[0.45, 0.20, 0.10, 0.25, 0.30, 0.20],
         [0.35, 0.15, 0.40, 0.20, 0.20, 0.20]]
    )
    clean_logprob = probabilities.log()
    advantage = torch.tensor(
        [[2.0, -1.5, 0.8, -0.4, 0.2, 7.0],
         [-2.2, 1.1, -0.3, 8.0, 9.0, 10.0]]
    )
    return {
        "clean_teacher_sampled_logprob": clean_logprob,
        "student_rollout_logprob": clean_logprob - advantage,
        "sampled_token_ids": torch.tensor(
            [[0, 2, 3, 1, 4, 0], [1, 2, 0, 0, 0, 0]]
        ),
        "response_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool
        ),
        "teacher_top1_ids": torch.zeros((2, 6), dtype=torch.long),
        "teacher_top1_probs": torch.tensor(
            [[0.45, 0.50, 0.55, 0.45, 0.40, 0.45],
             [0.50, 0.45, 0.40, 0.45, 0.45, 0.45]]
        ),
        "teacher_top2_probs": torch.tensor(
            [[0.25, 0.20, 0.18, 0.25, 0.30, 0.25],
             [0.35, 0.30, 0.25, 0.25, 0.25, 0.25]]
        ),
        "clean_teacher_entropy": torch.full((2, 6), 1.1),
    }


def _apply(config: TeacherDefenseConfig):
    return apply_sparse_asymmetric_projection(**_inputs(), config=config)


def test_random_selection_is_seeded_and_matches_full_counts_per_row() -> None:
    full = _apply(_config())
    random_config = _config(selection_strategy="random")
    random_first = _apply(random_config)
    random_second = _apply(random_config)

    assert torch.equal(random_first.active_mask, random_second.active_mask)
    assert full.active_mask.sum(dim=1).tolist() == [2, 2]
    assert random_first.active_mask.sum(dim=1).tolist() == [2, 2]
    assert random_first.metrics["teacher_defense/selection_count_match_rate"] == 1.0
    assert random_first.metrics["teacher_defense/selection_count_mismatch_count"] == 0.0


def test_positive_only_keeps_selected_count_but_never_perturbs_negative_branch() -> None:
    result = _apply(_config(negative_branch_enabled=False))
    negative = result.clean_advantage < 0.0

    assert not result.active_mask[negative].any()
    assert torch.count_nonzero(result.actual_delta[negative]) == 0
    assert result.metrics["teacher_defense/negative_branch_selected_token_ratio"] == 0.0
    assert result.metrics["teacher_defense/selection_count_match_rate"] == 1.0


def test_negative_only_keeps_selected_count_but_never_perturbs_positive_branch() -> None:
    result = _apply(_config(positive_branch_enabled=False))
    positive = result.clean_advantage >= 0.0

    assert not result.active_mask[positive].any()
    assert torch.count_nonzero(result.actual_delta[positive]) == 0
    assert result.metrics["teacher_defense/positive_branch_selected_token_ratio"] == 0.0
    assert result.metrics["teacher_defense/selection_count_match_rate"] == 1.0


def test_random_seed_changes_selector_without_changing_budget() -> None:
    first = _apply(_config(selection_strategy="random", selection_random_seed=42))
    second = _apply(_config(selection_strategy="random", selection_random_seed=43))

    assert not torch.equal(first.active_mask, second.active_mask)
    assert torch.equal(first.active_mask.sum(dim=1), second.active_mask.sum(dim=1))


def test_disabling_both_branches_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one active branch"):
        replace(
            _config(),
            positive_branch_enabled=False,
            negative_branch_enabled=False,
        ).validate()
