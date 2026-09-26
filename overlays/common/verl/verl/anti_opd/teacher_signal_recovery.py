"""Compact teacher sampled-token signal recovery for SA-TDP experiments.

The attacker-facing functions in this module only consume defended statistics.
Oracle clean tensors and true masks are accepted exclusively by
``evaluate_recovery`` for offline research diagnostics.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


_GFR_VARIANTS = {
    "identity",
    "single_positive",
    "single_negative",
    "sign_positive",
    "sign_negative",
}
_SPFR_VARIANTS = {"negative_priority", "mixed", "positive_priority"}


@dataclass(frozen=True)
class TeacherRecoveryConfig:
    enabled: bool = False
    method: str | None = None
    active_ratio: float = 0.20
    lambda_value: float = 1.05
    positive_weight: float = 0.5
    negative_weight: float = 1.0
    global_variant: str | None = None
    sparse_variant: str | None = None
    use_top1_filter: bool = True
    top1_margin: float = 1.0e-6
    invalid_candidate_fallback: str = "identity"
    eps: float = 1.0e-8
    compute_dtype: str = "float32"
    enable_oracle_metrics: bool = False
    cache_size: int = 256

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "TeacherRecoveryConfig":
        root = dict(mapping or {})
        defense = dict(root.get("defense_parameters", {}) or {})
        global_cfg = dict(root.get("global", {}) or {})
        sparse_cfg = dict(root.get("sparse", {}) or {})
        constraints = dict(root.get("constraints", {}) or {})
        numerical = dict(root.get("numerical", {}) or {})
        diagnostics = dict(root.get("diagnostics", {}) or {})
        config = cls(
            enabled=bool(root.get("enabled", False)),
            method=root.get("method", None),
            active_ratio=float(defense.get("active_ratio", 0.20)),
            lambda_value=float(defense.get("lambda_value", 1.05)),
            positive_weight=float(defense.get("positive_weight", 0.5)),
            negative_weight=float(defense.get("negative_weight", 1.0)),
            global_variant=global_cfg.get("variant", None),
            sparse_variant=sparse_cfg.get("variant", None),
            use_top1_filter=bool(constraints.get("use_top1_filter", True)),
            top1_margin=float(constraints.get("top1_margin", 1.0e-6)),
            invalid_candidate_fallback=str(
                constraints.get("invalid_candidate_fallback", "identity")
            ),
            eps=float(numerical.get("eps", 1.0e-8)),
            compute_dtype=str(numerical.get("compute_dtype", "float32")),
            enable_oracle_metrics=bool(
                diagnostics.get("enable_oracle_metrics", False)
            ),
            cache_size=int(diagnostics.get("cache_size", 256)),
        )
        config.validate()
        return config

    @property
    def positive_factor(self) -> float:
        return 1.0 - self.lambda_value * self.positive_weight

    @property
    def negative_factor(self) -> float:
        return 1.0 - self.lambda_value * self.negative_weight

    @property
    def variant(self) -> str | None:
        if self.method == "global_formula_recovery":
            return self.global_variant
        if self.method == "sparse_position_formula_recovery":
            return self.sparse_variant
        return None

    def validate(self) -> None:
        if not 0.0 <= self.active_ratio <= 1.0:
            raise ValueError("teacher_recovery active_ratio must be in [0, 1]")
        if self.lambda_value < 0 or self.positive_weight < 0 or self.negative_weight < 0:
            raise ValueError("recovery defense parameters must be non-negative")
        if self.eps <= 0 or self.eps >= 0.5:
            raise ValueError("teacher_recovery eps must be in (0, 0.5)")
        if self.top1_margin < 0:
            raise ValueError("top1_margin must be non-negative")
        if self.compute_dtype != "float32":
            raise ValueError("v1 supports compute_dtype=float32 only")
        if self.invalid_candidate_fallback != "identity":
            raise ValueError("v1 supports identity fallback only")
        if not self.enabled:
            return
        if self.method == "global_formula_recovery":
            if self.global_variant not in _GFR_VARIANTS:
                raise ValueError(f"unknown GFR variant: {self.global_variant}")
        elif self.method == "sparse_position_formula_recovery":
            if self.sparse_variant not in _SPFR_VARIANTS:
                raise ValueError(f"unknown SPFR variant: {self.sparse_variant}")
        else:
            raise ValueError(f"unknown teacher_recovery method: {self.method}")
        if abs(self.positive_factor) < self.eps:
            raise ValueError("positive inverse is singular")
        if abs(self.negative_factor) < self.eps:
            raise ValueError("negative inverse is singular")


@dataclass
class RecoveryOutput:
    recovered_teacher_sampled_logprob: torch.Tensor
    observed_defended_advantage: torch.Tensor
    recovered_advantage: torch.Tensor
    guessed_active_mask: torch.Tensor
    guessed_positive_branch_mask: torch.Tensor
    guessed_negative_branch_mask: torch.Tensor
    candidate_valid_mask: torch.Tensor
    metrics: dict[str, float]


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"recovery configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {resolved}")
    return payload


def _selected_gfr_variant(payload: Mapping[str, Any]) -> str | None:
    for key in ("best_GFR_variant", "best_gfr_variant", "gfr_variant"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    selected = payload.get("selected")
    if isinstance(selected, Mapping):
        return _selected_gfr_variant(selected)
    return None


def _calibration_row(
    payload: Mapping[str, Any], kappa: float
) -> Mapping[str, Any] | None:
    rows = payload.get("calibrations", payload.get("results", []))
    if isinstance(rows, Mapping):
        rows = list(rows.values())
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, Mapping) and math.isclose(
            float(row.get("kappa", -1.0)), float(kappa), rel_tol=0.0, abs_tol=1.0e-12
        ):
            return row
    return None


@dataclass(frozen=True)
class KLTRGFRConfig:
    """Fixed, globally damped GFR with an offline-calibrated KL budget."""

    enabled: bool = False
    method: str = "kl_tr_gfr"
    selected_variant_path: str | None = None
    gfr_variant: str | None = None
    damping_mode: str = "kl_calibrated"
    rho: float | None = None
    calibration_result_path: str | None = None
    kappa: float | None = None
    k_full: float | None = None
    target_recovery_kl: float | None = None
    calibration_actual_kl: float | None = None
    active_ratio: float = 0.20
    lambda_value: float = 1.05
    positive_weight: float = 0.5
    negative_weight: float = 1.0
    preserve_top1: bool = True
    top1_margin: float = 1.0e-6
    eps: float = 1.0e-8
    compute_dtype: str = "float32"
    enable_oracle_metrics: bool = False
    calibration_size: int = 256
    recovery_active_tolerance: float = 1.0e-7

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "KLTRGFRConfig":
        root = dict(mapping or {})
        enabled = bool(root.get("enabled", False)) and root.get("method") == "kl_tr_gfr"
        if not enabled:
            return cls(enabled=False)

        source = dict(root.get("source_gfr", {}) or {})
        damping = dict(root.get("damping", {}) or {})
        defense = dict(root.get("defense_parameters", {}) or {})
        constraints = dict(root.get("constraints", {}) or {})
        numerical = dict(root.get("numerical", {}) or {})
        diagnostics = dict(root.get("diagnostics", {}) or {})

        selected_path = source.get("selected_variant_path")
        selected_variant = source.get("variant")
        if selected_path:
            from_file = _selected_gfr_variant(_read_json(selected_path))
            if from_file is None:
                raise ValueError(
                    f"best_GFR_variant is missing from {selected_path}"
                )
            if selected_variant is not None and selected_variant != from_file:
                raise ValueError(
                    "source_gfr.variant disagrees with selected_variant_path: "
                    f"{selected_variant!r} != {from_file!r}"
                )
            selected_variant = from_file

        mode = str(damping.get("mode", "kl_calibrated"))
        rho_value = damping.get("rho")
        kappa_value = damping.get("kappa")
        calibration_path = damping.get("calibration_result_path")
        k_full = None
        target = None
        actual = None
        if mode == "kl_calibrated":
            if calibration_path is None or kappa_value is None:
                raise ValueError(
                    "kl_calibrated mode requires calibration_result_path and kappa"
                )
            calibration_payload = _read_json(calibration_path)
            row = _calibration_row(calibration_payload, float(kappa_value))
            if row is None:
                raise ValueError(
                    f"kappa={kappa_value} is missing from {calibration_path}"
                )
            calibrated_variant = row.get(
                "best_gfr_variant",
                calibration_payload.get("best_gfr_variant"),
            )
            if selected_variant is None:
                selected_variant = calibrated_variant
            elif calibrated_variant is not None and selected_variant != calibrated_variant:
                raise ValueError(
                    "selected GFR variant does not match calibration result"
                )
            calibrated_rho = float(row["rho_star"])
            if rho_value is not None and not math.isclose(
                float(rho_value), calibrated_rho, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError("configured rho disagrees with calibration result")
            rho_value = calibrated_rho
            k_full = float(row["k_full"])
            target = float(row["target_recovery_kl"])
            actual = float(row["actual_recovery_kl"])
        elif mode == "fixed":
            if rho_value is None:
                raise ValueError("fixed damping mode requires rho")
        else:
            raise ValueError(f"unknown KL-TR-GFR damping mode: {mode}")

        config = cls(
            enabled=True,
            selected_variant_path=(
                str(selected_path) if selected_path is not None else None
            ),
            gfr_variant=selected_variant,
            damping_mode=mode,
            rho=float(rho_value),
            calibration_result_path=(
                str(calibration_path) if calibration_path is not None else None
            ),
            kappa=float(kappa_value) if kappa_value is not None else None,
            k_full=k_full,
            target_recovery_kl=target,
            calibration_actual_kl=actual,
            active_ratio=float(defense.get("active_ratio", 0.20)),
            lambda_value=float(defense.get("lambda_value", 1.05)),
            positive_weight=float(defense.get("positive_weight", 0.5)),
            negative_weight=float(defense.get("negative_weight", 1.0)),
            preserve_top1=bool(constraints.get("preserve_top1", True)),
            top1_margin=float(constraints.get("top1_margin", 1.0e-6)),
            eps=float(numerical.get("eps", 1.0e-8)),
            compute_dtype=str(numerical.get("compute_dtype", "float32")),
            enable_oracle_metrics=bool(
                diagnostics.get("enable_oracle_metrics", False)
            ),
            calibration_size=int(diagnostics.get("calibration_size", 256)),
            recovery_active_tolerance=float(
                numerical.get("recovery_active_tolerance", 1.0e-7)
            ),
        )
        config.validate()
        return config

    @property
    def positive_factor(self) -> float:
        return 1.0 - self.lambda_value * self.positive_weight

    @property
    def negative_factor(self) -> float:
        return 1.0 - self.lambda_value * self.negative_weight

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.gfr_variant not in _GFR_VARIANTS:
            raise ValueError(f"unknown source GFR variant: {self.gfr_variant}")
        if self.rho is None or not 0.0 <= self.rho <= 1.0:
            raise ValueError("KL-TR-GFR rho must be in [0, 1]")
        if self.kappa is not None and self.kappa not in {0.10, 0.25, 0.50}:
            raise ValueError("KL-TR-GFR kappa must be one of 0.10, 0.25, 0.50")
        if not math.isclose(self.active_ratio, 0.20, abs_tol=1.0e-12):
            raise ValueError("KL-TR-GFR requires SA-TDP Low active_ratio=0.20")
        if self.lambda_value < 0 or self.positive_weight < 0 or self.negative_weight < 0:
            raise ValueError("KL-TR-GFR defense parameters must be non-negative")
        if abs(self.positive_factor) < self.eps or abs(self.negative_factor) < self.eps:
            raise ValueError("KL-TR-GFR inverse factor is singular")
        if not self.preserve_top1:
            raise ValueError("KL-TR-GFR requires preserve_top1=true")
        if self.top1_margin < 0:
            raise ValueError("top1_margin must be non-negative")
        if self.eps <= 0 or self.eps >= 0.5:
            raise ValueError("KL-TR-GFR eps must be in (0, 0.5)")
        if self.compute_dtype != "float32":
            raise ValueError("KL-TR-GFR supports compute_dtype=float32 only")
        if self.calibration_size <= 0:
            raise ValueError("calibration_size must be positive")


@dataclass
class KLTRGFRRecoveryOutput:
    recovered_teacher_sampled_logprob: torch.Tensor
    observed_defended_advantage: torch.Tensor
    raw_full_gfr_advantage: torch.Tensor
    raw_full_recovery_logprob: torch.Tensor
    damped_advantage_raw: torch.Tensor
    damped_advantage_actual: torch.Tensor
    recovery_delta_raw: torch.Tensor
    recovery_delta_actual: torch.Tensor
    token_recovery_kl: torch.Tensor
    recovery_active_mask: torch.Tensor
    top1_projected_mask: torch.Tensor
    infeasible_mask: torch.Tensor
    metrics: dict[str, float]


def _finite_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value[mask]
    return float(selected.float().mean().item()) if selected.numel() else 0.0


def _fraction(mask: torch.Tensor, denominator_mask: torch.Tensor) -> float:
    denominator = int(denominator_mask.sum().item())
    return float(mask.sum().item()) / denominator if denominator else 0.0


def _effective_eps(eps: float) -> float:
    """Return a boundary representable below one in float32."""

    return max(float(eps), float(torch.finfo(torch.float32).eps))


@torch.no_grad()
def derive_defended_top_statistics(
    *,
    clean_teacher_sampled_logprob: torch.Tensor,
    defended_teacher_sampled_logprob: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    clean_top1_ids: torch.Tensor,
    clean_top1_probs: torch.Tensor,
    clean_top2_probs: torch.Tensor,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive compact defended top-1/top-2 probabilities analytically.

    The exact defended top-2 value is only required when the sampled token is
    top-1.  For non-top-1 samples the filter only consumes defended top-1.
    """

    bound = _effective_eps(eps)
    p_y = clean_teacher_sampled_logprob.detach().float().exp().clamp(bound, 1 - bound)
    q_y = defended_teacher_sampled_logprob.detach().float().exp().clamp(bound, 1 - bound)
    scale = ((1.0 - q_y) / (1.0 - p_y).clamp_min(bound)).clamp_min(0.0)
    sampled_is_top1 = sampled_token_ids.detach().eq(clean_top1_ids.detach())
    top1 = clean_top1_probs.detach().float()
    top2 = clean_top2_probs.detach().float()
    defended_top1 = torch.where(sampled_is_top1, q_y, top1 * scale)
    defended_top2 = torch.where(
        sampled_is_top1,
        top2 * scale,
        torch.maximum(q_y, top2 * scale),
    )
    return defended_top1.detach(), defended_top2.detach()


@torch.no_grad()
def _candidate_top1_valid(
    *,
    defended_logprob: torch.Tensor,
    candidate_logprob: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    defended_top1_ids: torch.Tensor,
    defended_top1_probs: torch.Tensor,
    defended_top2_probs: torch.Tensor,
    margin: float,
    eps: float,
) -> torch.Tensor:
    bound = _effective_eps(eps)
    q_y = defended_logprob.float().exp().clamp(bound, 1 - bound)
    p_hat = candidate_logprob.float().exp().clamp(bound, 1 - bound)
    scale = ((1.0 - p_hat) / (1.0 - q_y).clamp_min(bound)).clamp_min(0.0)
    sampled_is_top1 = sampled_token_ids.eq(defended_top1_ids)
    max_other = defended_top2_probs.float() * scale
    restored_top1 = defended_top1_probs.float() * scale
    top1_case = p_hat + 1.0e-12 >= max_other + margin
    non_top1_case = restored_top1 + 1.0e-12 >= p_hat + margin
    return torch.where(sampled_is_top1, top1_case, non_top1_case)


def _candidate_advantage(
    observed: torch.Tensor, branch: str, config: TeacherRecoveryConfig
) -> torch.Tensor:
    if branch == "identity":
        return observed
    if branch == "positive":
        return observed / config.positive_factor
    if branch == "negative":
        return observed / config.negative_factor
    raise ValueError(f"unknown inverse branch: {branch}")


def raw_full_gfr_advantage(
    observed_defended_advantage: torch.Tensor,
    *,
    gfr_variant: str,
    positive_factor: float = 0.475,
    negative_factor: float = -0.05,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return the unfiltered full GFR candidate using attacker-visible inputs.

    This helper deliberately performs no top-1 filtering.  KL-TR-GFR must
    interpolate first and project each candidate rho independently.
    """

    observed = observed_defended_advantage.detach().float()
    if gfr_variant not in _GFR_VARIANTS:
        raise ValueError(f"unknown GFR variant: {gfr_variant}")
    if abs(positive_factor) < eps or abs(negative_factor) < eps:
        raise ValueError("GFR inverse factor is singular")
    if gfr_variant == "identity":
        candidate = observed
    elif gfr_variant == "single_positive":
        candidate = observed / positive_factor
    elif gfr_variant == "single_negative":
        candidate = observed / negative_factor
    elif gfr_variant == "sign_positive":
        candidate = torch.where(
            observed.ge(0), observed / positive_factor, observed
        )
    else:
        candidate = torch.where(
            observed.ge(0), observed / negative_factor, observed
        )
    candidate = torch.where(torch.isfinite(candidate), candidate, observed)
    return candidate.detach()


@torch.no_grad()
def apply_kl_tr_gfr(
    *,
    defended_teacher_sampled_logprob: torch.Tensor,
    student_rollout_logprob: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    defended_top1_ids: torch.Tensor,
    defended_top1_probs: torch.Tensor,
    defended_top2_probs: torch.Tensor,
    rho: float,
    gfr_variant: str,
    config: KLTRGFRConfig | Mapping[str, Any],
) -> KLTRGFRRecoveryOutput:
    """Apply one fixed globally damped GFR action under a top-1 trust region."""

    started = time.perf_counter()
    if not isinstance(config, KLTRGFRConfig):
        root = dict(config)
        root["enabled"] = True
        root["method"] = "kl_tr_gfr"
        damping = dict(root.get("damping", {}) or {})
        damping.update({"mode": "fixed", "rho": float(rho)})
        root["damping"] = damping
        source = dict(root.get("source_gfr", {}) or {})
        source["variant"] = gfr_variant
        source["selected_variant_path"] = None
        root["source_gfr"] = source
        config = KLTRGFRConfig.from_mapping(root)
    else:
        config.validate()
        if not math.isclose(float(rho), float(config.rho), rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("rho argument disagrees with KLTRGFRConfig")
        if gfr_variant != config.gfr_variant:
            raise ValueError("gfr_variant argument disagrees with KLTRGFRConfig")

    defended = defended_teacher_sampled_logprob.detach().float()
    student = student_rollout_logprob.detach().float()
    token_ids = sampled_token_ids.detach()
    mask = response_mask.detach().bool()
    top1_ids = defended_top1_ids.detach()
    top1_probs = defended_top1_probs.detach().float()
    top2_probs = defended_top2_probs.detach().float()
    observed = (defended - student).detach()

    full = raw_full_gfr_advantage(
        observed,
        gfr_variant=gfr_variant,
        positive_factor=config.positive_factor,
        negative_factor=config.negative_factor,
        eps=config.eps,
    )
    raw_full_logprob = student + full
    full_finite = torch.isfinite(full) & torch.isfinite(raw_full_logprob)
    full = torch.where(full_finite, full, observed)
    raw_full_logprob = torch.where(full_finite, raw_full_logprob, defended)
    full = torch.where(mask, full, observed)
    raw_full_logprob = torch.where(mask, raw_full_logprob, defended)

    recovery_delta_raw = torch.where(
        mask,
        float(rho) * (full - observed),
        torch.zeros_like(observed),
    )
    damped_raw = observed + recovery_delta_raw
    log_r_raw = student + damped_raw
    bound = _effective_eps(config.eps)
    low = math.log(bound)
    high = math.log1p(-bound)
    log_r_raw = log_r_raw.clamp(min=low, max=high)
    r_raw = log_r_raw.exp().clamp(bound, 1.0 - bound)
    q_y = defended.exp().clamp(bound, 1.0 - bound)

    sampled_is_top1 = token_ids.eq(top1_ids)
    one_minus_q = (1.0 - q_y).clamp_min(bound)
    top1_denom = one_minus_q + top2_probs
    top1_numer = top2_probs + config.top1_margin * one_minus_q
    r_min = top1_numer / top1_denom.clamp_min(bound)
    top1_boundary_valid = (
        torch.isfinite(r_min)
        & torch.isfinite(top1_denom)
        & top1_denom.gt(bound)
        & r_min.le(1.0 - bound)
    )
    r_min = r_min.clamp(bound, 1.0 - bound)

    non_top1_denom = top1_probs + one_minus_q
    non_top1_numer = top1_probs - config.top1_margin * one_minus_q
    r_max = non_top1_numer / non_top1_denom.clamp_min(bound)
    non_top1_boundary_valid = (
        torch.isfinite(r_max)
        & torch.isfinite(non_top1_denom)
        & non_top1_denom.gt(bound)
        & r_max.ge(bound)
    )
    r_max = r_max.clamp(bound, 1.0 - bound)

    boundary_valid = torch.where(
        sampled_is_top1, top1_boundary_valid, non_top1_boundary_valid
    )
    projected = torch.where(
        sampled_is_top1,
        torch.maximum(r_raw, r_min),
        torch.minimum(r_raw, r_max),
    )
    infeasible = mask & ~boundary_valid
    r_actual = torch.where(boundary_valid, projected, q_y)
    # rho=0 is the exact passive endpoint even when the observed distribution
    # has only a numerical tie rather than the requested positive margin.
    exact_identity = float(rho) == 0.0 or gfr_variant == "identity"
    if exact_identity:
        r_actual = q_y
        infeasible = torch.zeros_like(mask)
    r_actual = torch.where(mask, r_actual, q_y).clamp(bound, 1.0 - bound)
    log_r = r_actual.log()
    log_r = torch.where(mask, log_r, defended)
    if exact_identity:
        log_r = defended

    recovery_delta_actual = torch.where(mask, log_r - defended, torch.zeros_like(defended))
    damped_actual = torch.where(mask, log_r - student, observed)
    token_kl = bernoulli_coordinate_kl(r_actual, q_y, config.eps)
    token_kl = torch.where(mask, token_kl, torch.zeros_like(token_kl))
    if exact_identity:
        token_kl = torch.zeros_like(token_kl)
    recovery_active = mask & recovery_delta_actual.abs().gt(
        config.recovery_active_tolerance
    )
    top1_projected = (
        mask
        & boundary_valid
        & ~torch.isclose(r_actual, r_raw, rtol=1.0e-6, atol=1.0e-8)
    )
    if exact_identity:
        top1_projected = torch.zeros_like(mask)

    scale = ((1.0 - r_actual) / one_minus_q).clamp_min(0.0)
    restored_top1 = top1_probs * scale
    restored_second = top2_probs * scale
    top1_preserved = torch.where(
        sampled_is_top1,
        r_actual + 1.0e-7 >= restored_second,
        restored_top1 + 1.0e-7 >= r_actual,
    )
    top1_preserved = torch.where(infeasible, torch.ones_like(top1_preserved), top1_preserved)
    finite = (
        torch.isfinite(log_r)
        & torch.isfinite(damped_actual)
        & torch.isfinite(token_kl)
    )

    valid_count = max(1, int(mask.sum().item()))
    actual_mean_kl = _finite_mean(token_kl, mask)
    active_mean_kl = _finite_mean(token_kl, recovery_active)
    target = config.target_recovery_kl
    if target is not None and target > 0:
        budget_relative_error = abs(actual_mean_kl - target) / target
        budget_shortfall = max(0.0, target - actual_mean_kl)
    else:
        budget_relative_error = 0.0
        budget_shortfall = 0.0
    variant_id = {
        "identity": 0,
        "single_positive": 1,
        "single_negative": 2,
        "sign_positive": 3,
        "sign_negative": 4,
    }[gfr_variant]
    metrics = {
        "recovery/enabled": 1.0,
        "recovery/method": 3.0,
        "recovery/method_id": 3.0,
        "recovery/gfr_variant": float(variant_id),
        "recovery/variant_id": float(variant_id),
        "recovery/kappa": float(config.kappa) if config.kappa is not None else -1.0,
        "recovery/rho": float(rho),
        "recovery/k_full": float(config.k_full) if config.k_full is not None else -1.0,
        "recovery/target_recovery_kl": (
            float(target) if target is not None else -1.0
        ),
        "recovery/mean_recovery_kl": actual_mean_kl,
        "recovery/active_recovery_kl": active_mean_kl,
        "recovery/budget_relative_error": budget_relative_error,
        "recovery/budget_shortfall": budget_shortfall,
        "recovery/mean_abs_delta_raw": _finite_mean(
            recovery_delta_raw.abs(), mask
        ),
        "recovery/mean_abs_delta_actual": _finite_mean(
            recovery_delta_actual.abs(), mask
        ),
        "recovery/recovery_active_fraction": float(recovery_active.sum().item())
        / valid_count,
        "recovery/top1_projection_fraction": float(
            top1_projected.sum().item()
        )
        / valid_count,
        "recovery/top1_preservation_rate": _fraction(
            mask & top1_preserved, mask
        ),
        "recovery/infeasible_fraction": float(infeasible.sum().item())
        / valid_count,
        "recovery/observed_positive_fraction": _fraction(
            mask & observed.ge(0), mask
        ),
        "recovery/recovered_positive_fraction": _fraction(
            mask & damped_actual.ge(0), mask
        ),
        "recovery/advantage_sign_change_fraction": _fraction(
            mask & torch.sign(observed).ne(torch.sign(damped_actual)), mask
        ),
        "recovery/finite_fraction": _fraction(mask & finite, mask),
        "recovery/recovery_time_ms": (time.perf_counter() - started) * 1000.0,
    }
    return KLTRGFRRecoveryOutput(
        recovered_teacher_sampled_logprob=log_r.detach(),
        observed_defended_advantage=observed.detach(),
        raw_full_gfr_advantage=full.detach(),
        raw_full_recovery_logprob=raw_full_logprob.detach(),
        damped_advantage_raw=damped_raw.detach(),
        damped_advantage_actual=damped_actual.detach(),
        recovery_delta_raw=recovery_delta_raw.detach(),
        recovery_delta_actual=recovery_delta_actual.detach(),
        token_recovery_kl=token_kl.detach(),
        recovery_active_mask=recovery_active.detach(),
        top1_projected_mask=top1_projected.detach(),
        infeasible_mask=infeasible.detach(),
        metrics=metrics,
    )


@torch.no_grad()
def _candidate_bundle(
    *,
    observed: torch.Tensor,
    student_logprob: torch.Tensor,
    defended_logprob: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    defended_top1_ids: torch.Tensor,
    defended_top1_probs: torch.Tensor,
    defended_top2_probs: torch.Tensor,
    branch: str,
    config: TeacherRecoveryConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    advantage = _candidate_advantage(observed, branch, config)
    raw_logprob = student_logprob + advantage
    finite = torch.isfinite(advantage) & torch.isfinite(raw_logprob)
    bound = _effective_eps(config.eps)
    low = math.log(bound)
    high = math.log1p(-bound)
    candidate_logprob = raw_logprob.clamp(min=low, max=high)
    if config.use_top1_filter:
        top1_valid = _candidate_top1_valid(
            defended_logprob=defended_logprob,
            candidate_logprob=candidate_logprob,
            sampled_token_ids=sampled_token_ids,
            defended_top1_ids=defended_top1_ids,
            defended_top1_probs=defended_top1_probs,
            defended_top2_probs=defended_top2_probs,
            margin=config.top1_margin,
            eps=config.eps,
        )
    else:
        top1_valid = torch.ones_like(finite)
    valid = finite & top1_valid
    return advantage.detach(), candidate_logprob.detach(), valid.detach()


def _ordered_positions(
    values: torch.Tensor, mask: torch.Tensor, descending: bool
) -> list[int]:
    positions = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if positions.numel() == 0:
        return []
    order = torch.argsort(values[positions], descending=descending, stable=True)
    return [int(x) for x in positions[order].detach().cpu().tolist()]


@torch.no_grad()
def recover_teacher_signal(
    *,
    defended_teacher_sampled_logprob: torch.Tensor,
    student_rollout_logprob: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    defended_top1_ids: torch.Tensor,
    defended_top1_probs: torch.Tensor,
    defended_top2_probs: torch.Tensor,
    config: TeacherRecoveryConfig | Mapping[str, Any],
) -> RecoveryOutput:
    """Recover compact teacher scores without clean/oracle inputs."""

    started = time.perf_counter()
    if not isinstance(config, TeacherRecoveryConfig):
        config = TeacherRecoveryConfig.from_mapping(config)

    defended = defended_teacher_sampled_logprob.detach().float()
    student = student_rollout_logprob.detach().float()
    token_ids = sampled_token_ids.detach()
    mask = response_mask.detach().bool()
    top1_ids = defended_top1_ids.detach()
    top1_probs = defended_top1_probs.detach().float()
    top2_probs = defended_top2_probs.detach().float()
    observed = (defended - student).detach()

    recovered_advantage = observed.clone()
    guessed_positive = torch.zeros_like(mask)
    guessed_negative = torch.zeros_like(mask)
    candidate_valid = torch.zeros_like(mask)
    attempted_positive = torch.zeros_like(mask)
    attempted_negative = torch.zeros_like(mask)
    rejected_positive = torch.zeros_like(mask)
    rejected_negative = torch.zeros_like(mask)
    total_target = 0

    if config.enabled:
        pos_adv, pos_logprob, pos_valid = _candidate_bundle(
            observed=observed,
            student_logprob=student,
            defended_logprob=defended,
            sampled_token_ids=token_ids,
            defended_top1_ids=top1_ids,
            defended_top1_probs=top1_probs,
            defended_top2_probs=top2_probs,
            branch="positive",
            config=config,
        )
        neg_adv, neg_logprob, neg_valid = _candidate_bundle(
            observed=observed,
            student_logprob=student,
            defended_logprob=defended,
            sampled_token_ids=token_ids,
            defended_top1_ids=top1_ids,
            defended_top1_probs=top1_probs,
            defended_top2_probs=top2_probs,
            branch="negative",
            config=config,
        )

        if config.method == "global_formula_recovery":
            variant = config.global_variant
            if variant == "identity":
                pass
            elif variant in {"single_positive", "sign_positive"}:
                attempted_positive = mask & (
                    torch.ones_like(mask) if variant == "single_positive" else observed.ge(0)
                )
                guessed_positive = attempted_positive & pos_valid
                rejected_positive = attempted_positive & ~pos_valid
            elif variant in {"single_negative", "sign_negative"}:
                attempted_negative = mask & (
                    torch.ones_like(mask) if variant == "single_negative" else observed.ge(0)
                )
                guessed_negative = attempted_negative & neg_valid
                rejected_negative = attempted_negative & ~neg_valid
            else:
                raise ValueError(f"unknown GFR variant: {variant}")
        elif config.method == "sparse_position_formula_recovery":
            variant = config.sparse_variant
            for row in range(mask.shape[0]):
                valid_count = int(mask[row].sum().item())
                if valid_count == 0 or config.active_ratio == 0:
                    continue
                target = min(valid_count, max(1, math.ceil(config.active_ratio * valid_count)))
                total_target += target
                positive_observed = mask[row] & observed[row].gt(0)
                ascending = _ordered_positions(observed[row], positive_observed, False)
                descending = _ordered_positions(observed[row], positive_observed, True)

                if variant == "negative_priority":
                    accepted = 0
                    for col in ascending:
                        attempted_negative[row, col] = True
                        if bool(neg_valid[row, col]):
                            guessed_negative[row, col] = True
                            accepted += 1
                            if accepted == target:
                                break
                        else:
                            rejected_negative[row, col] = True
                elif variant == "positive_priority":
                    accepted = 0
                    for col in descending:
                        attempted_positive[row, col] = True
                        if bool(pos_valid[row, col]):
                            guessed_positive[row, col] = True
                            accepted += 1
                            if accepted == target:
                                break
                        else:
                            rejected_positive[row, col] = True
                elif variant == "mixed":
                    target_pos = target // 2
                    target_neg = target - target_pos
                    proposals: list[tuple[int, int, str, int]] = []
                    for rank, col in enumerate(ascending):
                        proposals.append((rank, 0, "negative", col))
                    for rank, col in enumerate(descending):
                        proposals.append((rank, 1, "positive", col))
                    proposals.sort()
                    used: set[int] = set()
                    accepted_pos = 0
                    accepted_neg = 0
                    for _, _, branch, col in proposals:
                        if col in used:
                            continue
                        if branch == "negative":
                            if accepted_neg >= target_neg:
                                continue
                            attempted_negative[row, col] = True
                            if bool(neg_valid[row, col]):
                                guessed_negative[row, col] = True
                                used.add(col)
                                accepted_neg += 1
                            else:
                                rejected_negative[row, col] = True
                        else:
                            if accepted_pos >= target_pos:
                                continue
                            attempted_positive[row, col] = True
                            if bool(pos_valid[row, col]):
                                guessed_positive[row, col] = True
                                used.add(col)
                                accepted_pos += 1
                            else:
                                rejected_positive[row, col] = True
                        if accepted_pos >= target_pos and accepted_neg >= target_neg:
                            break
                else:
                    raise ValueError(f"unknown SPFR variant: {variant}")
        else:
            raise ValueError(f"unknown teacher recovery method: {config.method}")

        recovered_advantage = torch.where(
            guessed_positive, pos_adv, recovered_advantage
        )
        recovered_advantage = torch.where(
            guessed_negative, neg_adv, recovered_advantage
        )
        candidate_valid = guessed_positive | guessed_negative

    guessed_active = guessed_positive | guessed_negative
    recovered_logprob = student + recovered_advantage
    bound = _effective_eps(config.eps)
    low = math.log(bound)
    high = math.log1p(-bound)
    recovered_logprob = recovered_logprob.clamp(min=low, max=high)
    recovered_logprob = torch.where(mask, recovered_logprob, defended)
    recovered_advantage = torch.where(mask, recovered_logprob - student, observed)

    attempted = attempted_positive | attempted_negative
    rejected = rejected_positive | rejected_negative
    valid_count = max(1, int(mask.sum().item()))
    attempt_count = int(attempted.sum().item())
    guessed_count = int(guessed_active.sum().item())
    if config.method == "sparse_position_formula_recovery" and config.enabled:
        denominator = max(1, total_target)
        budget_shortfall = max(0, total_target - guessed_count) / denominator
    else:
        budget_shortfall = 0.0

    checked_top1_valid = _candidate_top1_valid(
        defended_logprob=defended,
        candidate_logprob=recovered_logprob,
        sampled_token_ids=token_ids,
        defended_top1_ids=top1_ids,
        defended_top1_probs=top1_probs,
        defended_top2_probs=top2_probs,
        margin=0.0,
        eps=config.eps,
    )
    # Identity positions inherit the defense's guaranteed top-1 unchanged.
    final_top1_valid = torch.where(
        guessed_active, checked_top1_valid, torch.ones_like(checked_top1_valid)
    )
    finite = torch.isfinite(recovered_logprob) & torch.isfinite(recovered_advantage)
    variant = config.variant or "identity"
    metrics = {
        "recovery/enabled": float(config.enabled),
        "recovery/method_id": float(
            {None: 0, "global_formula_recovery": 1, "sparse_position_formula_recovery": 2}.get(
                config.method, -1
            )
        ),
        "recovery/variant_id": float(
            {
                "identity": 0,
                "single_positive": 1,
                "single_negative": 2,
                "sign_positive": 3,
                "sign_negative": 4,
                "negative_priority": 5,
                "mixed": 6,
                "positive_priority": 7,
            }.get(variant, -1)
        ),
        "recovery/observed_positive_fraction": _fraction(mask & observed.ge(0), mask),
        "recovery/observed_negative_fraction": _fraction(mask & observed.lt(0), mask),
        "recovery/guessed_active_fraction": guessed_count / valid_count,
        "recovery/guessed_positive_fraction": int(guessed_positive.sum().item()) / valid_count,
        "recovery/guessed_negative_fraction": int(guessed_negative.sum().item()) / valid_count,
        "recovery/top1_candidate_rejection_fraction": (
            int(rejected.sum().item()) / attempt_count if attempt_count else 0.0
        ),
        "recovery/top1_positive_candidate_rejection_fraction": (
            int(rejected_positive.sum().item()) / int(attempted_positive.sum().item())
            if attempted_positive.any()
            else 0.0
        ),
        "recovery/top1_negative_candidate_rejection_fraction": (
            int(rejected_negative.sum().item()) / int(attempted_negative.sum().item())
            if attempted_negative.any()
            else 0.0
        ),
        "recovery/top1_candidate_acceptance_fraction": (
            guessed_count / attempt_count if attempt_count else 1.0
        ),
        "recovery/recovered_top1_preservation_rate": _fraction(mask & final_top1_valid, mask),
        "recovery/mean_abs_advantage_change": _finite_mean(
            (recovered_advantage - observed).abs(), mask
        ),
        "recovery/recovered_advantage_positive_fraction": _fraction(
            mask & recovered_advantage.ge(0), mask
        ),
        "recovery/recovered_advantage_negative_fraction": _fraction(
            mask & recovered_advantage.lt(0), mask
        ),
        "recovery/finite_fraction": _fraction(mask & finite, mask),
        "recovery/recovered_fraction": guessed_count / valid_count,
        "recovery/budget_shortfall": float(budget_shortfall),
        "recovery/recovery_time_ms": (time.perf_counter() - started) * 1000.0,
    }
    return RecoveryOutput(
        recovered_teacher_sampled_logprob=recovered_logprob.detach(),
        observed_defended_advantage=observed.detach(),
        recovered_advantage=recovered_advantage.detach(),
        guessed_active_mask=guessed_active.detach(),
        guessed_positive_branch_mask=guessed_positive.detach(),
        guessed_negative_branch_mask=guessed_negative.detach(),
        candidate_valid_mask=candidate_valid.detach(),
        metrics=metrics,
    )


@torch.no_grad()
def bernoulli_coordinate_kl(
    source_probability: torch.Tensor,
    target_probability: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    bound = _effective_eps(eps)
    p = source_probability.detach().float().clamp(bound, 1 - bound)
    q = target_probability.detach().float().clamp(bound, 1 - bound)
    return p * (p.log() - q.log()) + (1 - p) * (
        torch.log1p(-p) - torch.log1p(-q)
    )


def _masked_or_nan(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value[mask]
    return float(selected.float().mean().item()) if selected.numel() else float("nan")


@torch.no_grad()
def evaluate_recovery(
    *,
    output: RecoveryOutput,
    clean_teacher_sampled_logprob: torch.Tensor,
    defended_teacher_sampled_logprob: torch.Tensor,
    student_rollout_logprob: torch.Tensor,
    response_mask: torch.Tensor,
    true_active_mask: torch.Tensor,
    true_positive_active_mask: torch.Tensor,
    true_negative_active_mask: torch.Tensor,
    true_projection_mask: torch.Tensor,
    sparse_method: bool,
    eps: float = 1.0e-8,
) -> dict[str, float]:
    """Oracle-only diagnostics; none of these tensors affect attack decisions."""

    clean_log = clean_teacher_sampled_logprob.detach().float()
    defended_log = defended_teacher_sampled_logprob.detach().float()
    student = student_rollout_logprob.detach().float()
    valid = response_mask.detach().bool()
    true_active = true_active_mask.detach().bool() & valid
    true_pos = true_positive_active_mask.detach().bool() & valid
    true_neg = true_negative_active_mask.detach().bool() & valid
    true_projected = true_projection_mask.detach().bool() & valid
    true_unmodified = valid & ~true_active
    recovered_log = output.recovered_teacher_sampled_logprob.detach().float()
    clean_adv = clean_log - student
    recovered_adv = recovered_log - student
    abs_error = (recovered_log - clean_log).abs()

    sign_accuracy = _masked_or_nan(
        torch.sign(recovered_adv).eq(torch.sign(clean_adv)).float(), valid
    )
    clean_flat = clean_adv[valid]
    recovered_flat = recovered_adv[valid]
    if clean_flat.numel() and clean_flat.norm() > 0 and recovered_flat.norm() > 0:
        cosine = float(torch.nn.functional.cosine_similarity(
            clean_flat.unsqueeze(0), recovered_flat.unsqueeze(0)
        ).item())
    else:
        cosine = float("nan")

    clean_prob = clean_log.exp()
    defended_prob = defended_log.exp()
    recovered_prob = recovered_log.exp()
    clean_to_defended = bernoulli_coordinate_kl(clean_prob, defended_prob, eps)
    clean_to_recovered = bernoulli_coordinate_kl(clean_prob, recovered_prob, eps)
    mean_defended = _masked_or_nan(clean_to_defended, valid)
    mean_recovered = _masked_or_nan(clean_to_recovered, valid)
    if math.isfinite(mean_defended) and mean_defended > eps:
        recovery_ratio = 1.0 - mean_recovered / mean_defended
    elif math.isfinite(mean_recovered) and mean_recovered <= eps:
        recovery_ratio = 1.0
    else:
        recovery_ratio = float("nan")

    metrics = {
        "logprob_mae_all_valid": _masked_or_nan(abs_error, valid),
        "logprob_mae_true_unmodified": _masked_or_nan(abs_error, true_unmodified),
        "logprob_mae_true_active_positive": _masked_or_nan(abs_error, true_pos),
        "logprob_mae_true_active_negative": _masked_or_nan(abs_error, true_neg),
        "logprob_mae_true_projected": _masked_or_nan(abs_error, true_projected),
        "advantage_mae": _masked_or_nan((recovered_adv - clean_adv).abs(), valid),
        "advantage_sign_accuracy": sign_accuracy,
        "advantage_cosine": cosine,
        "mean_clean_to_defended_kl": mean_defended,
        "mean_clean_to_recovered_kl": mean_recovered,
        "distribution_recovery_ratio": recovery_ratio,
        "top1_candidate_rejection_fraction": output.metrics[
            "recovery/top1_candidate_rejection_fraction"
        ],
        "top1_positive_candidate_rejection_fraction": output.metrics[
            "recovery/top1_positive_candidate_rejection_fraction"
        ],
        "top1_negative_candidate_rejection_fraction": output.metrics[
            "recovery/top1_negative_candidate_rejection_fraction"
        ],
        "recovered_top1_preservation_rate": output.metrics[
            "recovery/recovered_top1_preservation_rate"
        ],
    }

    if sparse_method:
        guessed = output.guessed_active_mask & valid
        guessed_pos = output.guessed_positive_branch_mask & valid
        guessed_neg = output.guessed_negative_branch_mask & valid
        tp = int((guessed & true_active).sum().item())
        fp = int((guessed & ~true_active & valid).sum().item())
        fn = int((~guessed & true_active).sum().item())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        positive_recall = (
            int((guessed_pos & true_pos).sum().item()) / int(true_pos.sum().item())
            if true_pos.any()
            else float("nan")
        )
        negative_recall = (
            int((guessed_neg & true_neg).sum().item()) / int(true_neg.sum().item())
            if true_neg.any()
            else float("nan")
        )
        true_predicted = guessed & true_active
        correct_branch = (guessed_pos & true_pos) | (guessed_neg & true_neg)
        branch_accuracy = (
            int((correct_branch & true_predicted).sum().item())
            / int(true_predicted.sum().item())
            if true_predicted.any()
            else float("nan")
        )
        metrics.update(
            {
                "mask_precision": precision,
                "mask_recall": recall,
                "mask_f1": f1,
                "positive_branch_recall": positive_recall,
                "negative_branch_recall": negative_recall,
                "branch_accuracy_on_true_positive_predictions": branch_accuracy,
            }
        )
    else:
        metrics.update(
            {
                "mask_precision": float("nan"),
                "mask_recall": float("nan"),
                "mask_f1": float("nan"),
                "positive_branch_recall": float("nan"),
                "negative_branch_recall": float("nan"),
                "branch_accuracy_on_true_positive_predictions": float("nan"),
            }
        )
    return metrics


@torch.no_grad()
def evaluate_kl_tr_gfr_recovery(
    *,
    output: KLTRGFRRecoveryOutput,
    clean_teacher_sampled_logprob: torch.Tensor,
    defended_teacher_sampled_logprob: torch.Tensor,
    student_rollout_logprob: torch.Tensor,
    response_mask: torch.Tensor,
    eps: float = 1.0e-8,
) -> dict[str, float]:
    """Oracle-only research diagnostics for KL-TR-GFR.

    Clean teacher statistics are intentionally isolated in this function and
    never enter ``apply_kl_tr_gfr`` or rho calibration.
    """

    clean_log = clean_teacher_sampled_logprob.detach().float()
    defended_log = defended_teacher_sampled_logprob.detach().float()
    student = student_rollout_logprob.detach().float()
    recovered_log = output.recovered_teacher_sampled_logprob.detach().float()
    valid = response_mask.detach().bool()
    clean_advantage = clean_log - student
    recovered_advantage = recovered_log - student

    clean_flat = clean_advantage[valid]
    recovered_flat = recovered_advantage[valid]
    if clean_flat.numel() and clean_flat.norm() > 0 and recovered_flat.norm() > 0:
        cosine = float(
            torch.nn.functional.cosine_similarity(
                clean_flat.unsqueeze(0), recovered_flat.unsqueeze(0)
            ).item()
        )
    else:
        cosine = float("nan")

    clean_probability = clean_log.exp()
    defended_probability = defended_log.exp()
    recovered_probability = recovered_log.exp()
    clean_to_defended = bernoulli_coordinate_kl(
        clean_probability, defended_probability, eps
    )
    clean_to_recovered = bernoulli_coordinate_kl(
        clean_probability, recovered_probability, eps
    )
    mean_defended = _masked_or_nan(clean_to_defended, valid)
    mean_recovered = _masked_or_nan(clean_to_recovered, valid)
    if math.isfinite(mean_defended) and mean_defended > eps:
        distribution_ratio = 1.0 - mean_recovered / mean_defended
    elif math.isfinite(mean_recovered) and mean_recovered <= eps:
        distribution_ratio = 1.0
    else:
        distribution_ratio = float("nan")
    return {
        "logprob_mae_to_clean": _masked_or_nan(
            (recovered_log - clean_log).abs(), valid
        ),
        "advantage_mae_to_clean": _masked_or_nan(
            (recovered_advantage - clean_advantage).abs(), valid
        ),
        "advantage_sign_accuracy": _masked_or_nan(
            torch.sign(recovered_advantage)
            .eq(torch.sign(clean_advantage))
            .float(),
            valid,
        ),
        "advantage_cosine_to_clean": cosine,
        "clean_to_defended_kl": mean_defended,
        "clean_to_recovered_kl": mean_recovered,
        "distribution_recovery_ratio": distribution_ratio,
    }
