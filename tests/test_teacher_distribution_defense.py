from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest
import torch

from verl.anti_opd.teacher_distribution_defense import (
    TeacherDefenseConfig,
    apply_sparse_asymmetric_projection,
    calibrate_lambda,
)


def config(
    *,
    enabled: bool = True,
    active_ratio: float = 0.30,
    positive: float = 0.5,
    negative: float = 1.0,
    lambda_value: float | None = 0.4,
    reference_mean_kl: float | None = 0.01,
    top1_margin: float = 1.0e-6,
    allow_sign_flip: bool = False,
) -> TeacherDefenseConfig:
    cfg = TeacherDefenseConfig(
        enabled=enabled,
        method="sparse_asymmetric_projection",
        active_ratio=active_ratio,
        selection_strategy="top_abs_advantage",
        positive_weight=positive,
        negative_weight=negative,
        lambda_value=lambda_value,
        reference_mean_kl=reference_mean_kl,
        top1_margin=top1_margin,
        allow_sign_flip=allow_sign_flip,
    )
    cfg.validate()
    return cfg


def inputs(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    # [B=2, L=6], with valid lengths 5 and 3.
    p_sampled = torch.tensor(
        [[0.45, 0.20, 0.10, 0.25, 0.30, 0.20], [0.35, 0.15, 0.40, 0.20, 0.20, 0.20]],
        dtype=torch.float32,
        device=device,
    )
    clean_logprob = p_sampled.log().to(dtype)
    clean_advantage = torch.tensor(
        [[2.0, -1.5, 0.8, -0.4, 0.2, 7.0], [-2.2, 1.1, -0.3, 8.0, 9.0, 10.0]],
        dtype=torch.float32,
        device=device,
    )
    old_logprob = (clean_logprob.float() - clean_advantage).to(dtype)
    sampled_ids = torch.tensor(
        [[0, 2, 3, 1, 4, 0], [1, 2, 0, 0, 0, 0]], device=device
    )
    response_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]],
        dtype=torch.bool,
        device=device,
    )
    top1_ids = torch.tensor(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], device=device
    )
    top1_probs = torch.tensor(
        [[0.45, 0.50, 0.55, 0.45, 0.40, 0.45], [0.50, 0.45, 0.40, 0.45, 0.45, 0.45]],
        device=device,
    )
    top2_probs = torch.tensor(
        [[0.25, 0.20, 0.18, 0.25, 0.30, 0.25], [0.35, 0.30, 0.25, 0.25, 0.25, 0.25]],
        device=device,
    )
    entropy = torch.full((2, 6), 1.1, dtype=dtype, device=device)
    return {
        "clean_teacher_sampled_logprob": clean_logprob,
        "student_rollout_logprob": old_logprob,
        "sampled_token_ids": sampled_ids,
        "response_mask": response_mask,
        "teacher_top1_ids": top1_ids,
        "teacher_top1_probs": top1_probs,
        "teacher_top2_probs": top2_probs,
        "clean_teacher_entropy": entropy,
    }


def apply(
    cfg: TeacherDefenseConfig,
    values: dict[str, torch.Tensor] | None = None,
):
    return apply_sparse_asymmetric_projection(
        **(inputs() if values is None else values),
        config=cfg,
    )


def explicit_distribution(
    p: torch.Tensor, sampled_id: int, q_y: torch.Tensor
) -> torch.Tensor:
    q = p.clone()
    scale = (1.0 - q_y) / (1.0 - p[sampled_id])
    q *= scale
    q[sampled_id] = q_y
    return q


def compact_for_distribution(
    p: torch.Tensor,
    sampled_id: int,
    clean_advantage: float,
    *,
    cfg: TeacherDefenseConfig,
):
    top2_probs, top2_ids = torch.topk(p, k=2)
    clean_logprob = p[sampled_id].log().reshape(1, 1)
    old_logprob = clean_logprob - clean_advantage
    entropy = -(p * p.log()).sum().reshape(1, 1)
    return apply_sparse_asymmetric_projection(
        clean_teacher_sampled_logprob=clean_logprob,
        student_rollout_logprob=old_logprob,
        sampled_token_ids=torch.tensor([[sampled_id]]),
        response_mask=torch.ones((1, 1), dtype=torch.bool),
        teacher_top1_ids=top2_ids[0].reshape(1, 1),
        teacher_top1_probs=top2_probs[0].reshape(1, 1),
        teacher_top2_probs=top2_probs[1].reshape(1, 1),
        clean_teacher_entropy=entropy,
        config=cfg,
    )


# A. Basic behavior
def test_disabled_is_exact_elementwise_noop_and_preserves_dtype():
    values = inputs(dtype=torch.float16)
    result = apply(config(enabled=False, lambda_value=None), values)
    assert torch.equal(
        result.defended_sampled_logprob,
        values["clean_teacher_sampled_logprob"],
    )
    assert result.defended_sampled_logprob.dtype == torch.float16
    assert not result.active_mask.any()


def test_active_ratio_zero_modifies_nothing():
    result = apply(config(active_ratio=0.0))
    assert not result.active_mask.any()
    assert torch.count_nonzero(result.actual_delta) == 0
    assert torch.count_nonzero(result.token_kl) == 0


def test_masked_prompt_and_padding_positions_are_unchanged():
    values = inputs()
    result = apply(config(), values)
    outside = ~values["response_mask"]
    assert torch.equal(
        result.defended_sampled_logprob[outside],
        values["clean_teacher_sampled_logprob"][outside],
    )
    assert torch.count_nonzero(result.actual_delta[outside]) == 0
    assert torch.count_nonzero(result.token_kl[outside]) == 0


def test_entropy_shape_mismatch_is_rejected_with_context():
    values = inputs()
    values["clean_teacher_entropy"] = torch.zeros((2, 3))
    with pytest.raises(ValueError, match="clean_teacher_entropy must have shape"):
        apply(config(), values)


def test_top30_count_is_ceil_per_nonempty_row():
    result = apply(config())
    assert result.active_mask.sum(dim=1).tolist() == [2, 1]


def test_top_abs_positions_are_correct():
    result = apply(config())
    expected = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=torch.bool
    )
    assert torch.equal(result.active_mask.cpu(), expected)


def test_selection_is_deterministic_and_ties_prefer_lower_position():
    values = inputs()
    tied_clean = torch.tensor(math.log(0.20))
    tied_old = tied_clean - 1.0
    values["clean_teacher_sampled_logprob"][0, :5] = tied_clean
    values["student_rollout_logprob"][0, :5] = tied_old
    first = apply(config(), values).active_mask
    second = apply(config(), values).active_mask
    assert torch.equal(first, second)
    assert first[0, :5].tolist() == [True, True, False, False, False]


# B. Weights and shifts
def test_branch_weights_and_delta_directions():
    result = apply(config())
    active_positive = result.active_mask & (result.clean_advantage >= 0)
    active_negative = result.active_mask & (result.clean_advantage < 0)
    inactive = ~result.active_mask
    assert torch.all(result.branch_weight[active_positive] == 0.5)
    assert torch.all(result.branch_weight[active_negative] == 1.0)
    assert torch.count_nonzero(result.branch_weight[inactive]) == 0
    assert torch.all(result.actual_delta[active_positive] <= 0)
    assert torch.all(result.actual_delta[active_negative] >= 0)
    assert torch.count_nonzero(result.actual_delta[inactive]) == 0


def test_no_nonzero_advantage_sign_flip():
    result = apply(config(lambda_value=0.95))
    valid_nonzero = inputs()["response_mask"] & (result.clean_advantage != 0)
    assert torch.all(
        result.clean_advantage[valid_nonzero]
        * result.defended_advantage[valid_nonzero]
        >= 0
    )
    assert result.metrics["teacher_defense/advantage_sign_flip_fraction"] == 0.0


@pytest.mark.parametrize("advantage", [1.0e-7, -1.0e-7])
def test_tiny_nonzero_advantage_cannot_flip_from_float32_rounding(advantage):
    clean = torch.tensor([[-1.0]])
    result = apply_sparse_asymmetric_projection(
        clean_teacher_sampled_logprob=clean,
        student_rollout_logprob=clean - advantage,
        sampled_token_ids=torch.tensor([[1]]),
        response_mask=torch.ones((1, 1), dtype=torch.bool),
        teacher_top1_ids=torch.tensor([[0]]),
        teacher_top1_probs=torch.tensor([[0.50]]),
        teacher_top2_probs=torch.tensor([[0.25]]),
        clean_teacher_entropy=None,
        config=config(active_ratio=1.0, lambda_value=0.95),
    )
    assert (
        result.clean_advantage.item() * result.defended_advantage.item()
        >= 0.0
    )
    assert result.metrics["teacher_defense/advantage_sign_flip_fraction"] == 0.0


def test_lambda_zero_is_exact_noop():
    values = inputs()
    result = apply(config(lambda_value=0.0), values)
    assert torch.equal(
        result.defended_sampled_logprob,
        values["clean_teacher_sampled_logprob"],
    )
    assert torch.count_nonzero(result.actual_delta) == 0


def test_lambda_above_cap_is_rejected_not_dynamically_clipped():
    with pytest.raises(ValueError, match="lambda_cap"):
        config(lambda_value=0.951)


def test_allow_sign_flip_accepts_lambda_above_no_flip_cap_and_logs_flips():
    result = apply(
        config(
            active_ratio=1.0,
            lambda_value=2.0,
            allow_sign_flip=True,
        )
    )
    assert result.metrics["teacher_defense/advantage_sign_flip_fraction"] > 0.0
    assert result.metrics["teacher_defense/top1_preservation_rate"] == 1.0
    assert result.metrics["teacher_defense/finite_fraction"] == 1.0


# C. Full-distribution projection and entropy
@pytest.mark.parametrize(
    ("p", "sampled_id", "advantage"),
    [
        (torch.tensor([0.50, 0.25, 0.15, 0.10]), 0, 1.0),
        (torch.tensor([0.50, 0.25, 0.15, 0.10]), 2, -1.0),
    ],
)
def test_explicit_distribution_matches_analytic_logprob_kl_and_entropy(
    p, sampled_id, advantage
):
    result = compact_for_distribution(
        p, sampled_id, advantage, cfg=config(active_ratio=1.0)
    )
    q_y = result.defended_sampled_logprob.float().exp().squeeze()
    q = explicit_distribution(p, sampled_id, q_y)
    explicit_kl = (q * (q.log() - p.log())).sum()
    explicit_entropy = -(q * q.log()).sum()
    assert torch.all(q >= 0)
    assert torch.isclose(q.sum(), torch.tensor(1.0), atol=1.0e-6)
    assert torch.isclose(q[sampled_id].log(), result.defended_sampled_logprob.squeeze(), atol=1.0e-6)
    assert torch.isclose(explicit_kl, result.token_kl.squeeze(), atol=1.0e-6)
    assert torch.isclose(explicit_entropy, result.defended_entropy.squeeze(), atol=1.0e-6)
    other = [index for index in range(p.numel()) if index != sampled_id]
    assert torch.allclose(q[other] / p[other], torch.full((len(other),), (1 - q_y) / (1 - p[sampled_id])))


def test_inactive_kl_zero_and_entropy_unchanged():
    values = inputs()
    result = apply(config(), values)
    inactive = ~result.active_mask
    assert torch.count_nonzero(result.token_kl[inactive]) == 0
    assert torch.equal(
        result.defended_entropy[inactive],
        values["clean_teacher_entropy"][inactive],
    )


# D. Top-1 preservation
def test_sampled_top1_remains_top1_after_probability_decrease():
    p = torch.tensor([0.50, 0.30, 0.20])
    result = compact_for_distribution(
        p, 0, 3.0, cfg=config(active_ratio=1.0, lambda_value=0.95)
    )
    q = explicit_distribution(
        p, 0, result.defended_sampled_logprob.exp().squeeze()
    )
    assert int(torch.argmax(q)) == int(torch.argmax(p))


def test_non_top1_sample_remains_non_top1_after_probability_increase():
    p = torch.tensor([0.50, 0.30, 0.20])
    result = compact_for_distribution(
        p, 2, -3.0, cfg=config(active_ratio=1.0, lambda_value=0.95)
    )
    q = explicit_distribution(
        p, 2, result.defended_sampled_logprob.exp().squeeze()
    )
    assert int(torch.argmax(q)) == int(torch.argmax(p))


def test_infeasible_boundary_keeps_clean_probability_and_is_logged():
    clean_logprob = torch.tensor([[math.log(5.0e-8)]])
    result = apply_sparse_asymmetric_projection(
        clean_teacher_sampled_logprob=clean_logprob,
        student_rollout_logprob=clean_logprob - (-2.0),
        sampled_token_ids=torch.tensor([[2]]),
        response_mask=torch.ones((1, 1), dtype=torch.bool),
        teacher_top1_ids=torch.tensor([[0]]),
        teacher_top1_probs=torch.tensor([[5.0e-9]]),
        teacher_top2_probs=torch.tensor([[4.0e-9]]),
        clean_teacher_entropy=None,
        config=config(active_ratio=1.0, lambda_value=0.95),
    )
    assert result.infeasible_mask.item()
    assert torch.isclose(
        result.defended_sampled_logprob.squeeze(), clean_logprob.squeeze()
    )
    assert result.metrics["teacher_defense/infeasible_fraction"] == 1.0


def test_top1_preservation_metric_is_one():
    result = apply(config(lambda_value=0.95))
    assert result.metrics["teacher_defense/top1_preservation_rate"] == 1.0


def test_tied_top1_behavior_is_deterministic_and_finite():
    p = torch.tensor([0.40, 0.40, 0.20])
    first = compact_for_distribution(
        p, 0, 2.0, cfg=config(active_ratio=1.0, lambda_value=0.95)
    )
    second = compact_for_distribution(
        p, 0, 2.0, cfg=config(active_ratio=1.0, lambda_value=0.95)
    )
    assert torch.equal(
        first.defended_sampled_logprob, second.defended_sampled_logprob
    )
    assert torch.equal(
        first.defended_sampled_logprob,
        p[0].log().reshape(1, 1),
    )
    assert torch.isfinite(first.defended_sampled_logprob).all()
    assert first.metrics["teacher_defense/top1_preservation_rate"] == 1.0


def test_margin_larger_than_clean_gap_never_reverses_requested_direction():
    p = torch.tensor([0.4000004, 0.4000000, 0.1999996])
    result = compact_for_distribution(
        p,
        0,
        2.0,
        cfg=config(
            active_ratio=1.0,
            lambda_value=0.95,
            top1_margin=1.0e-6,
        ),
    )
    assert result.actual_delta.item() <= 0.0
    assert (
        result.clean_advantage.item() * result.defended_advantage.item()
        >= 0.0
    )
    q = explicit_distribution(
        p, 0, result.defended_sampled_logprob.exp().squeeze()
    )
    assert int(torch.argmax(q)) == int(torch.argmax(p))


# E. Offline lambda calibration
def calibration_statistics() -> dict[str, torch.Tensor]:
    values = inputs()
    return {
        "clean_teacher_sampled_logprob": values[
            "clean_teacher_sampled_logprob"
        ],
        "student_rollout_logprob": values["student_rollout_logprob"],
        "sampled_token_ids": values["sampled_token_ids"],
        "response_mask": values["response_mask"],
        "teacher_top1_ids": values["teacher_top1_ids"],
        "teacher_top1_probs": values["teacher_top1_probs"],
        "teacher_top2_probs": values["teacher_top2_probs"],
        "clean_teacher_entropy": values["clean_teacher_entropy"],
    }


def mean_kl_at(scale: float, cfg: TeacherDefenseConfig) -> float:
    local = replace(cfg, enabled=True, lambda_value=scale)
    return apply_sparse_asymmetric_projection(
        **inputs(), config=local
    ).metrics["teacher_defense/mean_kl_all_valid"]


def test_k_lambda_is_monotone_non_decreasing():
    cfg = config(enabled=False, lambda_value=None)
    values = [mean_kl_at(scale, cfg) for scale in (0.0, 0.2, 0.5, cfg.lambda_cap)]
    assert values == sorted(values)


def test_reachable_calibration_stays_below_budget_with_small_error():
    cfg = config(enabled=False, lambda_value=None)
    reference = 0.5 * mean_kl_at(cfg.lambda_cap, cfg)
    result = calibrate_lambda(
        statistics=calibration_statistics(),
        config=cfg,
        reference_mean_kl=reference,
        iterations=25,
    )
    assert result["actual_mean_kl"] <= reference + 1.0e-9
    assert result["relative_error"] < 1.0e-4


def test_unreachable_calibration_uses_cap_and_reports_shortfall():
    cfg = config(enabled=False, lambda_value=None)
    attainable = mean_kl_at(cfg.lambda_cap, cfg)
    reference = attainable + 0.25
    result = calibrate_lambda(
        statistics=calibration_statistics(),
        config=cfg,
        reference_mean_kl=reference,
    )
    assert result["lambda_best"] == pytest.approx(cfg.lambda_cap)
    assert result["budget_shortfall"] == pytest.approx(
        reference - attainable
    )


def test_branch_modes_are_calibrated_separately():
    stats = calibration_statistics()
    base = config(enabled=False, lambda_value=None)
    target = 0.002
    results = []
    for positive, negative in ((1.0, 1.0), (0.5, 1.0), (1.0, 0.5)):
        cfg = replace(
            base, positive_weight=positive, negative_weight=negative
        )
        results.append(
            calibrate_lambda(
                statistics=stats, config=cfg, reference_mean_kl=target
            )
        )
    assert [result["branch_mode"] for result in results] == [
        "symmetric",
        "negative_priority",
        "positive_priority",
    ]
    assert len({result["config_fingerprint"] for result in results}) == 3


def test_calibration_does_not_mutate_tensors_or_gradients_or_regenerate_rollout():
    stats = calibration_statistics()
    stats["clean_teacher_sampled_logprob"].requires_grad_(True)
    before = {
        key: value.detach().clone()
        for key, value in stats.items()
        if isinstance(value, torch.Tensor)
    }
    pointers = {
        key: value.data_ptr()
        for key, value in stats.items()
        if isinstance(value, torch.Tensor)
    }
    calibrate_lambda(
        statistics=stats,
        config=config(enabled=False, lambda_value=None),
        reference_mean_kl=0.001,
    )
    for key, expected in before.items():
        assert torch.equal(stats[key].detach(), expected)
        assert stats[key].data_ptr() == pointers[key]
    assert stats["clean_teacher_sampled_logprob"].grad is None


# F. Numerical behavior and devices
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
def test_fp32_fp16_bf16_inputs_are_finite_and_restore_logprob_dtype(dtype):
    result = apply(config(), inputs(dtype=dtype))
    assert result.defended_sampled_logprob.dtype == dtype
    for tensor in (
        result.defended_sampled_logprob,
        result.clean_advantage,
        result.defended_advantage,
        result.actual_delta,
        result.token_kl,
    ):
        assert torch.isfinite(tensor.float()).all()


def test_probabilities_near_zero_or_one_remain_finite():
    p = torch.tensor([1.0 - 2.0e-7, 1.0e-7, 1.0e-7])
    for sampled_id, advantage in ((0, 4.0), (1, -4.0)):
        result = compact_for_distribution(
            p,
            sampled_id,
            advantage,
            cfg=config(active_ratio=1.0, lambda_value=0.95),
        )
        assert torch.isfinite(result.defended_sampled_logprob).all()
        assert torch.isfinite(result.token_kl).all()


def test_exact_unit_sampled_probability_does_not_create_nan_kl():
    result = apply_sparse_asymmetric_projection(
        clean_teacher_sampled_logprob=torch.zeros((1, 1)),
        student_rollout_logprob=torch.tensor([[-1.0]]),
        sampled_token_ids=torch.tensor([[0]]),
        response_mask=torch.ones((1, 1), dtype=torch.bool),
        teacher_top1_ids=torch.tensor([[0]]),
        teacher_top1_probs=torch.ones((1, 1)),
        teacher_top2_probs=torch.zeros((1, 1)),
        clean_teacher_entropy=torch.zeros((1, 1)),
        config=config(active_ratio=1.0, lambda_value=0.95),
    )
    assert torch.isfinite(result.token_kl).all()
    assert math.isfinite(
        result.metrics["teacher_defense/mean_kl_all_valid"]
    )


def test_empty_response_and_mixed_lengths_do_not_crash():
    values = inputs()
    values["response_mask"][0] = False
    result = apply(config(), values)
    assert result.active_mask[0].sum() == 0
    assert result.active_mask[1].sum() == 1
    assert torch.isfinite(result.defended_sampled_logprob).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_device_is_preserved():
    values = inputs(device="cuda")
    result = apply(config(), values)
    for tensor in (
        result.defended_sampled_logprob,
        result.clean_advantage,
        result.actual_delta,
        result.active_mask,
        result.token_kl,
    ):
        assert tensor.device.type == "cuda"


def test_all_outputs_are_detached_and_no_vocab_shaped_tensor_is_created():
    values = inputs()
    values["clean_teacher_sampled_logprob"].requires_grad_(True)
    values["student_rollout_logprob"].requires_grad_(True)
    result = apply(config(), values)
    for tensor in (
        result.defended_sampled_logprob,
        result.clean_advantage,
        result.defended_advantage,
        result.actual_delta,
        result.active_mask,
        result.branch_weight,
        result.token_kl,
        result.defended_entropy,
        result.infeasible_mask,
    ):
        assert not tensor.requires_grad
        assert tensor.grad_fn is None
        assert tensor.ndim == 2


def test_config_result_fingerprint_and_fixed_lambda_resume(tmp_path):
    cfg = config(enabled=False, lambda_value=None)
    calibration = calibrate_lambda(
        statistics=calibration_statistics(),
        config=cfg,
        reference_mean_kl=0.001,
    )
    path = tmp_path / "negative_priority.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    loaded = TeacherDefenseConfig.from_mapping(
        {
            "enabled": True,
            "method": "sparse_asymmetric_projection",
            "selection": {
                "active_ratio": 0.30,
                "strategy": "top_abs_advantage",
            },
            "weights": {"positive": 0.5, "negative": 1.0},
            "calibration": {"result_path": str(path)},
        }
    )
    assert loaded.lambda_value == pytest.approx(calibration["lambda_best"])
    assert loaded.reference_mean_kl == pytest.approx(
        calibration["reference_mean_kl"]
    )


def test_metrics_have_all_required_scalar_keys():
    result = apply(config())
    required = {
        "teacher_defense/active_fraction",
        "teacher_defense/selected_positive_fraction",
        "teacher_defense/selected_negative_fraction",
        "teacher_defense/lambda",
        "teacher_defense/positive_weight",
        "teacher_defense/negative_weight",
        "teacher_defense/effective_lambda_positive",
        "teacher_defense/effective_lambda_negative",
        "teacher_defense/mean_abs_delta_all_valid",
        "teacher_defense/mean_abs_delta_active",
        "teacher_defense/mean_kl_all_valid",
        "teacher_defense/mean_kl_active",
        "teacher_defense/reference_mean_kl",
        "teacher_defense/budget_relative_error",
        "teacher_defense/budget_shortfall",
        "teacher_defense/top1_preservation_rate",
        "teacher_defense/top1_projection_fraction",
        "teacher_defense/infeasible_fraction",
        "teacher_defense/clean_positive_advantage_fraction",
        "teacher_defense/clean_negative_advantage_fraction",
        "teacher_defense/advantage_sign_flip_fraction",
        "teacher_defense/positive_advantage_mass_retention",
        "teacher_defense/negative_advantage_mass_retention",
        "teacher_defense/clean_defended_advantage_cosine",
        "teacher_defense/finite_fraction",
        "teacher_defense/defense_time_ms",
        "teacher_defense/teacher_entropy_shift",
        "score_perturbation/active_fraction",
        "score_perturbation/selected_positive_fraction",
        "score_perturbation/selected_negative_fraction",
        "score_perturbation/global_scale",
        "score_perturbation/effective_lambda_positive",
        "score_perturbation/effective_lambda_negative",
        "score_perturbation/reference_uniform_kl",
        "score_perturbation/actual_mean_kl",
        "score_perturbation/budget_relative_error",
        "score_perturbation/budget_shortfall",
        "score_perturbation/mean_abs_logprob_shift",
        "score_perturbation/positive_advantage_mass_retention",
        "score_perturbation/negative_advantage_mass_retention",
        "score_perturbation/clean_defended_advantage_cosine",
        "score_perturbation/advantage_sign_flip_fraction",
        "score_perturbation/top1_preservation_rate",
        "score_perturbation/top1_projection_fraction",
        "score_perturbation/teacher_entropy_shift",
        "score_perturbation/finite_fraction",
        "score_perturbation/defense_step_time_ms",
    }
    assert required.issubset(result.metrics)
    assert all(isinstance(result.metrics[key], float) for key in required)


def test_runtime_budget_shortfall_and_mass_retention_are_dynamic():
    result = apply(config(reference_mean_kl=1.0))
    actual_kl = result.metrics["teacher_defense/mean_kl_all_valid"]
    assert result.metrics["teacher_defense/budget_shortfall"] == pytest.approx(
        max(1.0 - actual_kl, 0.0)
    )
    for sign in ("positive", "negative"):
        value = result.metrics[
            f"teacher_defense/{sign}_advantage_mass_retention"
        ]
        assert math.isfinite(value)
        assert value >= 0.0
