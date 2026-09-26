from __future__ import annotations

import inspect
import json
import math

import pytest
import torch

from verl.anti_opd.teacher_signal_recovery import (
    KLTRGFRConfig,
    apply_kl_tr_gfr,
    raw_full_gfr_advantage,
)


def config(rho=0.5, variant="sign_positive"):
    return KLTRGFRConfig.from_mapping(
        {
            "enabled": True,
            "method": "kl_tr_gfr",
            "source_gfr": {"variant": variant},
            "damping": {"mode": "fixed", "rho": rho},
            "defense_parameters": {
                "active_ratio": 0.20,
                "lambda_value": 1.05,
                "positive_weight": 0.5,
                "negative_weight": 1.0,
            },
            "constraints": {
                "preserve_top1": True,
                "top1_margin": 1.0e-6,
            },
            "numerical": {"eps": 1.0e-8, "compute_dtype": "float32"},
        }
    )


def inputs(dtype=torch.float32, device="cpu"):
    student = torch.full((2, 4), -3.0, dtype=dtype, device=device)
    observed = torch.tensor(
        [[0.06, -0.10, 0.20, 0.0], [0.02, 0.04, -0.03, 0.08]],
        dtype=dtype,
        device=device,
    )
    defended = student + observed
    sampled = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8]], device=device
    )
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]], device=device
    )
    top1_ids = torch.full_like(sampled, 99)
    top1_probs = torch.full((2, 4), 0.40, device=device)
    top2_probs = torch.full((2, 4), 0.20, device=device)
    return {
        "defended_teacher_sampled_logprob": defended,
        "student_rollout_logprob": student,
        "sampled_token_ids": sampled,
        "response_mask": mask,
        "defended_top1_ids": top1_ids,
        "defended_top1_probs": top1_probs,
        "defended_top2_probs": top2_probs,
    }


def run(rho=0.5, variant="sign_positive", value=None):
    value = inputs() if value is None else value
    return apply_kl_tr_gfr(
        **value,
        rho=rho,
        gfr_variant=variant,
        config=config(rho, variant),
    )


def top1_case(sampled_is_top1: bool, observed=0.06):
    q = 0.60 if sampled_is_top1 else 0.20
    student = torch.tensor([[math.log(q) - observed]])
    sampled = torch.tensor([[3]])
    return {
        "defended_teacher_sampled_logprob": torch.tensor([[math.log(q)]]),
        "student_rollout_logprob": student,
        "sampled_token_ids": sampled,
        "response_mask": torch.tensor([[True]]),
        "defended_top1_ids": (
            sampled.clone() if sampled_is_top1 else torch.tensor([[9]])
        ),
        "defended_top1_probs": torch.tensor(
            [[q if sampled_is_top1 else 0.50]]
        ),
        "defended_top2_probs": torch.tensor([[0.30]]),
    }


def test_rho_zero_advantage_is_observed():
    value = inputs()
    out = run(0.0, "single_positive", value)
    expected = (
        value["defended_teacher_sampled_logprob"]
        - value["student_rollout_logprob"]
    )
    assert torch.equal(out.damped_advantage_actual, expected.float())


def test_rho_zero_logprob_is_bitwise_defended():
    value = inputs()
    out = run(0.0, "single_negative", value)
    assert torch.equal(
        out.recovered_teacher_sampled_logprob,
        value["defended_teacher_sampled_logprob"].float(),
    )


def test_rho_one_is_full_candidate_without_projection():
    value = inputs()
    out = run(1.0, "single_positive", value)
    mask = value["response_mask"]
    assert torch.allclose(
        out.damped_advantage_actual[mask],
        out.raw_full_gfr_advantage[mask],
        atol=1e-6,
    )


def test_partial_rho_is_between_endpoints():
    out = run(0.25, "single_positive")
    low = torch.minimum(
        out.observed_defended_advantage, out.raw_full_gfr_advantage
    )
    high = torch.maximum(
        out.observed_defended_advantage, out.raw_full_gfr_advantage
    )
    assert torch.all(out.damped_advantage_actual >= low - 1e-6)
    assert torch.all(out.damped_advantage_actual <= high + 1e-6)


@pytest.mark.parametrize("rho", [0.0, 0.1, 0.5, 1.0])
def test_identity_gfr_is_identity_for_every_rho(rho):
    value = inputs()
    out = run(rho, "identity", value)
    assert torch.equal(
        out.recovered_teacher_sampled_logprob,
        value["defended_teacher_sampled_logprob"].float(),
    )


def test_response_mask_false_is_unchanged():
    value = inputs()
    out = run(1.0, "single_negative", value)
    invalid = ~value["response_mask"]
    assert torch.equal(
        out.recovered_teacher_sampled_logprob[invalid],
        value["defended_teacher_sampled_logprob"].float()[invalid],
    )
    assert torch.count_nonzero(out.recovery_delta_actual[invalid]) == 0


def test_prompt_and_padding_are_unchanged():
    value = inputs()
    out = run(0.5, "single_positive", value)
    assert torch.count_nonzero(out.recovery_active_mask[~value["response_mask"]]) == 0
    assert torch.count_nonzero(out.token_recovery_kl[~value["response_mask"]]) == 0


@pytest.mark.parametrize(
    ("rho", "expected"),
    [(0.0, 0.06), (0.1, -0.066), (0.5, -0.57), (1.0, -1.20)],
)
def test_negative_branch_damping_example(rho, expected):
    value = inputs()
    out = run(rho, "single_negative", value)
    assert out.damped_advantage_actual[0, 0].item() == pytest.approx(
        expected, abs=2e-6
    )


def test_log_r_raw_matches_defended_plus_delta():
    out = run(0.3, "single_positive")
    defended = inputs()["defended_teacher_sampled_logprob"].float()
    expected = defended + out.recovery_delta_raw
    mask = inputs()["response_mask"]
    actual = inputs()["student_rollout_logprob"].float() + out.damped_advantage_raw
    assert torch.allclose(expected[mask], actual[mask], atol=1e-6)


def test_log_r_raw_matches_student_plus_advantage():
    value = inputs()
    out = run(0.4, "single_positive", value)
    expected = value["student_rollout_logprob"].float() + out.damped_advantage_raw
    assert torch.allclose(
        expected[value["response_mask"]],
        (
            value["defended_teacher_sampled_logprob"].float()
            + out.recovery_delta_raw
        )[value["response_mask"]],
        atol=1e-6,
    )


def test_sampled_top1_is_preserved():
    out = run(1.0, "single_negative", top1_case(True))
    assert out.metrics["recovery/top1_preservation_rate"] == 1.0


def test_non_sampled_top1_is_preserved():
    out = run(1.0, "single_positive", top1_case(False, observed=1.0))
    assert out.metrics["recovery/top1_preservation_rate"] == 1.0


def test_small_rho_is_not_rejected_because_full_candidate_is_illegal():
    value = top1_case(True)
    small = run(0.1, "single_negative", value)
    full = run(1.0, "single_negative", value)
    defended = value["defended_teacher_sampled_logprob"].float()
    assert not torch.equal(small.recovered_teacher_sampled_logprob, defended)
    assert full.top1_projected_mask.item()


def test_probability_saturates_at_top1_boundary():
    value = top1_case(True)
    high = run(0.8, "single_negative", value)
    full = run(1.0, "single_negative", value)
    assert torch.allclose(
        high.recovered_teacher_sampled_logprob,
        full.recovered_teacher_sampled_logprob,
        atol=1e-6,
    )


def test_top1_preservation_metric_is_one():
    assert run(1.0, "single_negative").metrics[
        "recovery/top1_preservation_rate"
    ] == pytest.approx(1.0)


def test_rho_zero_token_kl_is_exact_zero():
    assert torch.count_nonzero(run(0.0).token_recovery_kl) == 0


def test_analytic_recovery_kl_matches_explicit_distribution():
    value = inputs()
    out = run(0.5, "single_positive", value)
    row, col = 0, 0
    q_y = value["defended_teacher_sampled_logprob"][row, col].float().exp()
    r_y = out.recovered_teacher_sampled_logprob[row, col].exp()
    q = torch.tensor([q_y, 0.30, 0.20, 1.0 - q_y - 0.50])
    scale = (1.0 - r_y) / (1.0 - q_y)
    r = torch.cat([r_y.reshape(1), q[1:] * scale])
    explicit = torch.sum(r * (r.log() - q.log()))
    assert out.token_recovery_kl[row, col].item() == pytest.approx(
        explicit.item(), abs=1e-6
    )


def test_mean_kl_denominator_is_all_valid_tokens():
    value = inputs()
    out = run(0.5, "sign_positive", value)
    expected = out.token_recovery_kl[value["response_mask"]].mean().item()
    assert out.metrics["recovery/mean_recovery_kl"] == pytest.approx(expected)


def test_k_rho_is_monotonic_on_constructed_data():
    values = [
        run(rho, "single_positive").metrics["recovery/mean_recovery_kl"]
        for rho in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    ]
    assert all(right + 1e-8 >= left for left, right in zip(values, values[1:]))


def test_rho_one_kl_is_k_full_endpoint():
    full = run(1.0, "single_positive")
    assert full.metrics["recovery/mean_recovery_kl"] == pytest.approx(
        full.token_recovery_kl[inputs()["response_mask"]].mean().item()
    )


def test_simple_bisection_stays_within_budget():
    target = 0.25 * run(1.0, "single_positive").metrics[
        "recovery/mean_recovery_kl"
    ]
    low, high = 0.0, 1.0
    selected = run(0.0, "single_positive")
    for _ in range(25):
        mid = (low + high) / 2
        candidate = run(mid, "single_positive")
        if candidate.metrics["recovery/mean_recovery_kl"] <= target:
            low, selected = mid, candidate
        else:
            high = mid
    assert selected.metrics["recovery/mean_recovery_kl"] <= target
    assert target - selected.metrics["recovery/mean_recovery_kl"] < 1e-6


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_supported_input_dtypes_are_finite(dtype):
    out = run(0.5, "single_negative", inputs(dtype=dtype))
    assert torch.isfinite(out.recovered_teacher_sampled_logprob).all()
    assert torch.isfinite(out.token_recovery_kl).all()


@pytest.mark.parametrize("q", [1.0e-7, 1.0 - 2.0e-7])
def test_probabilities_near_zero_or_one_are_stable(q):
    value = top1_case(q > 0.5)
    value["defended_teacher_sampled_logprob"][:] = math.log(q)
    value["student_rollout_logprob"][:] = math.log(q) - 0.01
    if q > 0.5:
        value["defended_top1_probs"][:] = q
        value["defended_top2_probs"][:] = 1.0 - q
    out = run(1.0, "single_negative", value)
    assert torch.isfinite(out.recovered_teacher_sampled_logprob).all()
    assert torch.isfinite(out.token_recovery_kl).all()


def test_empty_response_is_stable():
    value = inputs()
    value["response_mask"] = torch.zeros_like(value["response_mask"])
    out = run(1.0, "single_negative", value)
    assert torch.equal(
        out.recovered_teacher_sampled_logprob,
        value["defended_teacher_sampled_logprob"].float(),
    )
    assert out.metrics["recovery/mean_recovery_kl"] == 0.0


def test_different_response_lengths_are_masked_independently():
    value = inputs()
    out = run(0.5, "single_positive", value)
    assert out.recovery_active_mask[0].sum() == 3
    assert out.recovery_active_mask[1].sum() == 2


def test_cpu_device_is_preserved():
    out = run()
    assert out.recovered_teacher_sampled_logprob.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_device_is_preserved():
    out = run(value=inputs(device="cuda"))
    assert out.recovered_teacher_sampled_logprob.device.type == "cuda"


def test_every_tensor_output_is_detached():
    value = inputs()
    value["defended_teacher_sampled_logprob"].requires_grad_(True)
    value["student_rollout_logprob"].requires_grad_(True)
    out = run(value=value)
    for field in out.__dataclass_fields__:
        item = getattr(out, field)
        if isinstance(item, torch.Tensor):
            assert not item.requires_grad
            assert item.grad_fn is None


def test_no_output_has_vocab_dimension():
    out = run()
    for field in out.__dataclass_fields__:
        item = getattr(out, field)
        if isinstance(item, torch.Tensor):
            assert item.ndim == 2
            assert item.shape == inputs()["response_mask"].shape


def test_attacker_interface_has_no_clean_or_oracle_inputs():
    parameters = set(inspect.signature(apply_kl_tr_gfr).parameters)
    forbidden = {
        "clean_teacher_sampled_logprob",
        "clean_advantage",
        "true_active_mask",
        "true_branch_mask",
        "true_projection_mask",
    }
    assert not parameters & forbidden


def test_raw_full_candidate_is_not_top1_filtered():
    value = top1_case(True)
    observed = (
        value["defended_teacher_sampled_logprob"]
        - value["student_rollout_logprob"]
    )
    expected = -20.0 * observed
    full = raw_full_gfr_advantage(
        observed,
        gfr_variant="single_negative",
        positive_factor=0.475,
        negative_factor=-0.05,
    )
    assert torch.allclose(full, expected)


def test_calibrated_config_reads_fixed_rho(tmp_path):
    selected = tmp_path / "selected.json"
    selected.write_text(
        json.dumps({"best_GFR_variant": "sign_positive"}), encoding="utf-8"
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "best_gfr_variant": "sign_positive",
                "calibrations": [
                    {
                        "best_gfr_variant": "sign_positive",
                        "kappa": 0.25,
                        "rho_star": 0.4,
                        "k_full": 0.02,
                        "target_recovery_kl": 0.005,
                        "actual_recovery_kl": 0.00499,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = KLTRGFRConfig.from_mapping(
        {
            "enabled": True,
            "method": "kl_tr_gfr",
            "source_gfr": {"selected_variant_path": str(selected)},
            "damping": {
                "mode": "kl_calibrated",
                "calibration_result_path": str(calibration),
                "kappa": 0.25,
            },
        }
    )
    assert cfg.gfr_variant == "sign_positive"
    assert cfg.rho == pytest.approx(0.4)
    assert cfg.k_full == pytest.approx(0.02)


def test_calibration_variant_mismatch_is_rejected(tmp_path):
    selected = tmp_path / "selected.json"
    selected.write_text(
        json.dumps({"best_GFR_variant": "sign_positive"}), encoding="utf-8"
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "calibrations": [
                    {
                        "best_gfr_variant": "single_negative",
                        "kappa": 0.10,
                        "rho_star": 0.1,
                        "k_full": 0.02,
                        "target_recovery_kl": 0.002,
                        "actual_recovery_kl": 0.0019,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        KLTRGFRConfig.from_mapping(
            {
                "enabled": True,
                "method": "kl_tr_gfr",
                "source_gfr": {"selected_variant_path": str(selected)},
                "damping": {
                    "mode": "kl_calibrated",
                    "calibration_result_path": str(calibration),
                    "kappa": 0.10,
                },
            }
        )


def test_models_cannot_be_updated_by_recovery_math():
    assert not any(
        isinstance(value, torch.nn.Parameter)
        for value in run().__dict__.values()
    )


def test_apply_function_does_not_generate_rollouts():
    source = inspect.getsource(apply_kl_tr_gfr)
    assert ".generate(" not in source
    assert "rollout" not in source.replace("student_rollout_logprob", "")
