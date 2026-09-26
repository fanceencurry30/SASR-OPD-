import math

import pytest
import torch

from verl.anti_opd.teacher_signal_recovery import (
    TeacherRecoveryConfig,
    bernoulli_coordinate_kl,
    derive_defended_top_statistics,
    evaluate_recovery,
    recover_teacher_signal,
)


def config(method=None, variant=None, enabled=True, top1=False, ratio=0.2):
    mapping = {
        "enabled": enabled,
        "method": method,
        "defense_parameters": {
            "active_ratio": ratio,
            "lambda_value": 1.05,
            "positive_weight": 0.5,
            "negative_weight": 1.0,
        },
        "global": {"variant": variant if method == "global_formula_recovery" else None},
        "sparse": {
            "variant": variant if method == "sparse_position_formula_recovery" else None
        },
        "constraints": {
            "use_top1_filter": top1,
            "top1_margin": 1e-6,
            "invalid_candidate_fallback": "identity",
        },
        "numerical": {"eps": 1e-8, "compute_dtype": "float32"},
        "diagnostics": {"enable_oracle_metrics": False, "cache_size": 256},
    }
    return TeacherRecoveryConfig.from_mapping(mapping)


def inputs(dtype=torch.float32, device="cpu"):
    student_p = torch.tensor([[0.10, 0.20, 0.30, 0.25, 0.40, 0.20]], device=device)
    defended_p = torch.tensor([[0.20, 0.10, 0.45, 0.25, 0.20, 0.20]], device=device)
    token_ids = torch.tensor([[2, 3, 4, 5, 6, 7]], device=device)
    top1_ids = torch.tensor([[9, 9, 4, 9, 9, 9]], device=device)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool, device=device)
    top1_p = torch.tensor([[0.60, 0.60, 0.45, 0.60, 0.60, 0.60]], device=device)
    top2_p = torch.tensor([[0.20, 0.20, 0.30, 0.25, 0.20, 0.20]], device=device)
    return dict(
        defended_teacher_sampled_logprob=defended_p.log().to(dtype),
        student_rollout_logprob=student_p.log().to(dtype),
        sampled_token_ids=token_ids,
        response_mask=mask,
        defended_top1_ids=top1_ids,
        defended_top1_probs=top1_p.to(dtype),
        defended_top2_probs=top2_p.to(dtype),
    )


def run(cfg, **overrides):
    values = inputs()
    values.update(overrides)
    return recover_teacher_signal(config=cfg, **values)


def test_disabled_is_exact_identity():
    out = run(config(enabled=False))
    assert torch.equal(out.recovered_teacher_sampled_logprob, inputs()["defended_teacher_sampled_logprob"])


def test_gfr_identity_is_exact_identity():
    out = run(config("global_formula_recovery", "identity"))
    assert torch.equal(out.recovered_teacher_sampled_logprob, inputs()["defended_teacher_sampled_logprob"])


@pytest.mark.parametrize("variant", ["single_positive", "single_negative", "sign_positive", "sign_negative"])
def test_mask_false_unchanged_gfr(variant):
    value = inputs()
    out = recover_teacher_signal(config=config("global_formula_recovery", variant), **value)
    assert out.recovered_teacher_sampled_logprob[0, 5] == value["defended_teacher_sampled_logprob"][0, 5]


@pytest.mark.parametrize("variant", ["negative_priority", "mixed", "positive_priority"])
def test_mask_false_unchanged_spfr(variant):
    value = inputs()
    out = recover_teacher_signal(config=config("sparse_position_formula_recovery", variant), **value)
    assert out.recovered_teacher_sampled_logprob[0, 5] == value["defended_teacher_sampled_logprob"][0, 5]


def test_config_parameters_are_exact():
    cfg = config("global_formula_recovery", "single_positive")
    assert cfg.lambda_value == 1.05
    assert cfg.positive_weight == 0.5
    assert cfg.negative_weight == 1.0
    assert cfg.positive_factor == pytest.approx(0.475)
    assert cfg.negative_factor == pytest.approx(-0.05)


@pytest.mark.parametrize("variant", ["single_positive", "single_negative", "sign_positive", "sign_negative"])
def test_all_outputs_detached(variant):
    value = inputs()
    value["defended_teacher_sampled_logprob"].requires_grad_(True)
    value["student_rollout_logprob"].requires_grad_(True)
    out = recover_teacher_signal(config=config("global_formula_recovery", variant), **value)
    for tensor in (
        out.recovered_teacher_sampled_logprob,
        out.observed_defended_advantage,
        out.recovered_advantage,
        out.guessed_active_mask,
    ):
        assert not tensor.requires_grad
        assert tensor.grad_fn is None


@pytest.mark.parametrize("variant", ["single_positive", "single_negative", "sign_positive", "sign_negative"])
def test_zero_advantage_stays_zero(variant):
    value = inputs()
    value["student_rollout_logprob"] = value["defended_teacher_sampled_logprob"].clone()
    out = recover_teacher_signal(config=config("global_formula_recovery", variant), **value)
    assert torch.allclose(out.recovered_advantage[value["response_mask"]], torch.zeros(5))


def test_positive_inverse_formula():
    value = inputs()
    out = recover_teacher_signal(config=config("global_formula_recovery", "single_positive"), **value)
    observed = value["defended_teacher_sampled_logprob"] - value["student_rollout_logprob"]
    assert torch.allclose(out.recovered_advantage[value["response_mask"]], observed[value["response_mask"]] / 0.475)


def test_negative_inverse_formula():
    value = inputs()
    safe_b = torch.tensor([[0.02, -0.02, 0.01, 0.0, -0.01, 0.0]])
    value["student_rollout_logprob"] = torch.full((1, 6), math.log(0.01))
    value["defended_teacher_sampled_logprob"] = value["student_rollout_logprob"] + safe_b
    out = recover_teacher_signal(config=config("global_formula_recovery", "single_negative"), **value)
    observed = value["defended_teacher_sampled_logprob"] - value["student_rollout_logprob"]
    assert torch.allclose(out.recovered_advantage[value["response_mask"]], -20 * observed[value["response_mask"]], atol=1e-5)


def test_logprob_equals_student_plus_recovered_advantage():
    value = inputs()
    out = recover_teacher_signal(config=config("global_formula_recovery", "sign_positive"), **value)
    valid = value["response_mask"]
    expected = value["student_rollout_logprob"].float() + out.recovered_advantage
    assert torch.allclose(out.recovered_teacher_sampled_logprob[valid], expected[valid])


@pytest.mark.parametrize("variant", ["single_positive", "single_negative", "sign_positive", "sign_negative"])
def test_extreme_values_are_finite(variant):
    value = inputs()
    value["defended_teacher_sampled_logprob"] = torch.tensor([[-1000.0] * 6])
    value["student_rollout_logprob"] = torch.tensor([[-1.0e4] * 6])
    out = recover_teacher_signal(config=config("global_formula_recovery", variant), **value)
    assert torch.isfinite(out.recovered_teacher_sampled_logprob).all()
    assert torch.isfinite(out.recovered_advantage).all()


def test_invalid_top1_candidate_falls_back_to_identity_for_top1_sample():
    value = inputs()
    value["sampled_token_ids"][0, 0] = 9
    value["defended_top1_ids"][0, 0] = 9
    value["defended_teacher_sampled_logprob"][0, 0] = math.log(0.6)
    value["student_rollout_logprob"][0, 0] = math.log(0.5)
    value["defended_top1_probs"][0, 0] = 0.6
    value["defended_top2_probs"][0, 0] = 0.3
    out = recover_teacher_signal(config=config("global_formula_recovery", "single_negative", top1=True), **value)
    assert out.recovered_teacher_sampled_logprob[0, 0] == value["defended_teacher_sampled_logprob"][0, 0]
    assert not out.guessed_active_mask[0, 0]


def test_invalid_top1_candidate_falls_back_for_non_top1_sample():
    value = inputs()
    value["defended_teacher_sampled_logprob"][0, 0] = math.log(0.2)
    value["student_rollout_logprob"][0, 0] = math.log(0.1)
    value["defended_top1_probs"][0, 0] = 0.6
    out = recover_teacher_signal(config=config("global_formula_recovery", "single_positive", top1=True), **value)
    assert out.recovered_teacher_sampled_logprob[0, 0] == value["defended_teacher_sampled_logprob"][0, 0]


def test_gfr_single_positive_applies_to_all_valid():
    out = run(config("global_formula_recovery", "single_positive"))
    assert out.guessed_positive_branch_mask.sum() == 5
    assert not out.guessed_negative_branch_mask.any()


def test_gfr_single_negative_applies_to_all_valid():
    out = run(config("global_formula_recovery", "single_negative"))
    assert out.guessed_negative_branch_mask.sum() == 5
    assert not out.guessed_positive_branch_mask.any()


@pytest.mark.parametrize("variant,branch", [("sign_positive", "positive"), ("sign_negative", "negative")])
def test_sign_aware_uses_observed_sign_only(variant, branch):
    value = inputs()
    observed = value["defended_teacher_sampled_logprob"] - value["student_rollout_logprob"]
    out = recover_teacher_signal(config=config("global_formula_recovery", variant), **value)
    selected = out.guessed_positive_branch_mask if branch == "positive" else out.guessed_negative_branch_mask
    assert torch.equal(selected, value["response_mask"] & observed.ge(0))


def spfr_inputs():
    student = torch.full((2, 10), math.log(0.1))
    b = torch.tensor([
        [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, -0.1],
        [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, -0.2],
    ])
    defended = student + b
    return dict(
        defended_teacher_sampled_logprob=defended,
        student_rollout_logprob=student,
        sampled_token_ids=torch.arange(20).reshape(2, 10),
        response_mask=torch.ones((2, 10), dtype=torch.bool),
        defended_top1_ids=torch.full((2, 10), 99),
        defended_top1_probs=torch.full((2, 10), 0.8),
        defended_top2_probs=torch.full((2, 10), 0.1),
    )


@pytest.mark.parametrize("ratio,expected", [(0.2, 2), (0.3, 3), (0.01, 1), (0.0, 0)])
def test_spfr_per_row_ceil_selection(ratio, expected):
    value = spfr_inputs()
    out = recover_teacher_signal(config=config("sparse_position_formula_recovery", "positive_priority", ratio=ratio), **value)
    assert torch.equal(out.guessed_active_mask.sum(dim=1), torch.tensor([expected, expected]))


def test_spfr_negative_selects_smallest_positive_b():
    value = spfr_inputs()
    out = recover_teacher_signal(config=config("sparse_position_formula_recovery", "negative_priority"), **value)
    assert torch.equal(torch.nonzero(out.guessed_negative_branch_mask[0]).squeeze(-1), torch.tensor([0, 2]))


def test_spfr_positive_selects_largest_positive_b():
    value = spfr_inputs()
    out = recover_teacher_signal(config=config("sparse_position_formula_recovery", "positive_priority"), **value)
    assert torch.equal(torch.nonzero(out.guessed_positive_branch_mask[0]).squeeze(-1), torch.tensor([1, 3]))


def test_spfr_mixed_selects_two_ends_without_overlap():
    value = spfr_inputs()
    out = recover_teacher_signal(config=config("sparse_position_formula_recovery", "mixed"), **value)
    assert out.guessed_positive_branch_mask[0, 1]
    assert out.guessed_negative_branch_mask[0, 0]
    assert not (out.guessed_positive_branch_mask & out.guessed_negative_branch_mask).any()


def test_spfr_rows_do_not_influence_each_other():
    value = spfr_inputs()
    out1 = recover_teacher_signal(config=config("sparse_position_formula_recovery", "positive_priority"), **value)
    sliced = {key: tensor[:1] for key, tensor in value.items()}
    out2 = recover_teacher_signal(config=config("sparse_position_formula_recovery", "positive_priority"), **sliced)
    assert torch.equal(out1.guessed_active_mask[0], out2.guessed_active_mask[0])


def test_spfr_is_deterministic():
    value = spfr_inputs()
    cfg = config("sparse_position_formula_recovery", "mixed")
    left = recover_teacher_signal(config=cfg, **value)
    right = recover_teacher_signal(config=cfg, **value)
    assert torch.equal(left.recovered_teacher_sampled_logprob, right.recovered_teacher_sampled_logprob)
    assert torch.equal(left.guessed_active_mask, right.guessed_active_mask)


def test_spfr_candidate_shortfall_is_reported():
    value = spfr_inputs()
    value["defended_teacher_sampled_logprob"] = value["student_rollout_logprob"] - 0.1
    out = recover_teacher_signal(config=config("sparse_position_formula_recovery", "negative_priority"), **value)
    assert out.guessed_active_mask.sum() == 0
    assert out.metrics["recovery/budget_shortfall"] == 1.0


def test_derived_defended_top_stats_for_sampled_top1():
    top1, top2 = derive_defended_top_statistics(
        clean_teacher_sampled_logprob=torch.tensor([[math.log(0.6)]]),
        defended_teacher_sampled_logprob=torch.tensor([[math.log(0.5)]]),
        sampled_token_ids=torch.tensor([[3]]),
        clean_top1_ids=torch.tensor([[3]]),
        clean_top1_probs=torch.tensor([[0.6]]),
        clean_top2_probs=torch.tensor([[0.2]]),
    )
    assert top1.item() == pytest.approx(0.5)
    assert top2.item() == pytest.approx(0.25)


def test_derived_defended_top_stats_for_non_top1():
    top1, _ = derive_defended_top_statistics(
        clean_teacher_sampled_logprob=torch.tensor([[math.log(0.2)]]),
        defended_teacher_sampled_logprob=torch.tensor([[math.log(0.1)]]),
        sampled_token_ids=torch.tensor([[3]]),
        clean_top1_ids=torch.tensor([[9]]),
        clean_top1_probs=torch.tensor([[0.6]]),
        clean_top2_probs=torch.tensor([[0.2]]),
    )
    assert top1.item() == pytest.approx(0.675)


def oracle_metrics(output, clean_log, defended_log, student_log, true_active):
    true_pos = true_active & (clean_log - student_log >= 0)
    true_neg = true_active & ~true_pos
    return evaluate_recovery(
        output=output,
        clean_teacher_sampled_logprob=clean_log,
        defended_teacher_sampled_logprob=defended_log,
        student_rollout_logprob=student_log,
        response_mask=torch.ones_like(true_active),
        true_active_mask=true_active,
        true_positive_active_mask=true_pos,
        true_negative_active_mask=true_neg,
        true_projection_mask=torch.zeros_like(true_active),
        sparse_method=True,
    )


def test_identity_distribution_recovery_ratio_is_zero():
    value = inputs()
    out = recover_teacher_signal(config=config(enabled=False), **value)
    clean = value["defended_teacher_sampled_logprob"] + 0.1
    m = oracle_metrics(out, clean, value["defended_teacher_sampled_logprob"], value["student_rollout_logprob"], value["response_mask"])
    assert m["mean_clean_to_recovered_kl"] == pytest.approx(m["mean_clean_to_defended_kl"])
    assert m["distribution_recovery_ratio"] == pytest.approx(0.0)


def test_perfect_recovery_ratio_is_one():
    value = inputs()
    clean = value["defended_teacher_sampled_logprob"] + 0.1
    out = recover_teacher_signal(config=config(enabled=False), **value)
    out.recovered_teacher_sampled_logprob = clean
    m = oracle_metrics(out, clean, value["defended_teacher_sampled_logprob"], value["student_rollout_logprob"], value["response_mask"])
    assert m["distribution_recovery_ratio"] == pytest.approx(1.0, abs=1e-6)


def test_mask_precision_recall_f1():
    value = inputs()
    out = recover_teacher_signal(config=config("sparse_position_formula_recovery", "positive_priority", ratio=0.4), **value)
    true_active = torch.tensor([[1, 0, 1, 0, 0, 0]], dtype=torch.bool)
    m = oracle_metrics(out, value["defended_teacher_sampled_logprob"], value["defended_teacher_sampled_logprob"], value["student_rollout_logprob"], true_active)
    guessed = out.guessed_active_mask
    tp = int((guessed & true_active).sum())
    fp = int((guessed & ~true_active).sum())
    fn = int((~guessed & true_active).sum())
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    assert m["mask_precision"] == pytest.approx(precision)
    assert m["mask_recall"] == pytest.approx(recall)
    assert m["mask_f1"] == pytest.approx(f1)


def test_analytic_kl_matches_explicit_small_vocab():
    p = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
    y = 1
    qy = torch.tensor(0.18, dtype=torch.float64)
    q = p.clone()
    q[y] = qy
    mask = torch.arange(4) != y
    q[mask] = p[mask] * (1 - qy) / (1 - p[y])
    explicit = (p * (p.log() - q.log())).sum()
    analytic = bernoulli_coordinate_kl(p[y], q[y]).double()
    assert abs(explicit.item() - analytic.item()) < 1e-6


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cpu_dtypes_are_finite(dtype):
    value = inputs(dtype=dtype)
    out = recover_teacher_signal(config=config("sparse_position_formula_recovery", "mixed"), **value)
    assert torch.isfinite(out.recovered_teacher_sampled_logprob).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_dtypes_no_device_mismatch(dtype):
    value = inputs(dtype=dtype, device="cuda")
    out = recover_teacher_signal(config=config("sparse_position_formula_recovery", "mixed"), **value)
    assert out.recovered_teacher_sampled_logprob.device.type == "cuda"
    assert torch.isfinite(out.recovered_teacher_sampled_logprob).all()


def test_no_full_vocab_tensor_is_returned():
    out = run(config("sparse_position_formula_recovery", "mixed"))
    for value in vars(out).values():
        if isinstance(value, torch.Tensor):
            assert value.ndim <= 2
