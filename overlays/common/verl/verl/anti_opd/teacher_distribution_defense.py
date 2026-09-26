"""Sparse Asymmetric Teacher-Distribution Projection (SA-TDP).

This module contains only detached, framework-independent probability
operations.  It changes the teacher probability of the student-sampled token
and analytically rescales the rest of the vocabulary.  It never materializes a
defended ``[batch, sequence, vocabulary]`` tensor.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch


_METHODS = {"none", "sparse_asymmetric_projection"}
_REFERENCE_KL_MODES = {"precomputed", "compute_uniform"}
_BRANCH_MODES = {
    "symmetric": (1.0, 1.0),
    "negative_priority": (0.5, 1.0),
    "positive_priority": (1.0, 0.5),
}
_FLOAT32_MAX_PROBABILITY = (
    1.0 - torch.finfo(torch.float32).eps / 2.0
)
_IMPLEMENTATION_VERSION = "sa-tdp-v1.2"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _branch_mode(positive: float, negative: float) -> str:
    for name, expected in _BRANCH_MODES.items():
        if math.isclose(positive, expected[0]) and math.isclose(negative, expected[1]):
            return name
    raise ValueError(
        "SA-TDP v1 supports only fixed weights 1:1, 0.5:1, or 1:0.5; "
        f"got {positive}:{negative}"
    )


@dataclass(frozen=True)
class TeacherDefenseConfig:
    """Hydra-independent SA-TDP configuration."""

    enabled: bool = False
    method: str = "sparse_asymmetric_projection"
    active_ratio: float = 0.30
    selection_strategy: str = "top_abs_advantage"
    positive_weight: float = 0.5
    negative_weight: float = 1.0
    lambda_value: float | None = None
    calibration_result_path: str | None = None
    reference_kl_mode: str = "precomputed"
    reference_mean_kl: float | None = None
    calibration_iterations: int = 25
    tolerance_ratio: float = 0.05
    allow_sign_flip: bool = False
    max_effective_strength: float = 0.95
    preserve_teacher_top1: bool = True
    top1_margin: float = 1.0e-6
    eps: float = 1.0e-8
    compute_dtype: str = "float32"
    use_defended_teacher_statistics_for_fire: bool = True
    calibration_budget_shortfall: float = 0.0
    calibration_fingerprint: str | None = None
    calibration_capture_only: bool = False
    calibration_capture_path: str | None = None
    calibration_uniform_alpha: float = 0.20

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TeacherDefenseConfig":
        value = _mapping(value)
        selection = _mapping(value.get("selection"))
        weights = _mapping(value.get("weights"))
        calibration = _mapping(value.get("calibration"))
        constraints = _mapping(value.get("constraints"))
        numerical = _mapping(value.get("numerical"))
        integration = _mapping(value.get("integration"))
        capture = _mapping(value.get("capture"))

        config = cls(
            enabled=bool(value.get("enabled", False)),
            method=str(value.get("method", "sparse_asymmetric_projection")),
            active_ratio=float(selection.get("active_ratio", 0.30)),
            selection_strategy=str(selection.get("strategy", "top_abs_advantage")),
            positive_weight=float(weights.get("positive", 0.5)),
            negative_weight=float(weights.get("negative", 1.0)),
            lambda_value=(
                None
                if calibration.get("lambda_value") is None
                else float(calibration["lambda_value"])
            ),
            calibration_result_path=calibration.get("result_path"),
            reference_kl_mode=str(calibration.get("reference_kl_mode", "precomputed")),
            reference_mean_kl=(
                None
                if calibration.get("reference_mean_kl") is None
                else float(calibration["reference_mean_kl"])
            ),
            calibration_iterations=int(calibration.get("iterations", 25)),
            tolerance_ratio=float(calibration.get("tolerance_ratio", 0.05)),
            allow_sign_flip=bool(constraints.get("allow_sign_flip", False)),
            max_effective_strength=float(
                constraints.get("max_effective_strength", 0.95)
            ),
            preserve_teacher_top1=bool(
                constraints.get("preserve_teacher_top1", True)
            ),
            top1_margin=float(constraints.get("top1_margin", 1.0e-6)),
            eps=float(numerical.get("eps", 1.0e-8)),
            compute_dtype=str(numerical.get("compute_dtype", "float32")),
            use_defended_teacher_statistics_for_fire=bool(
                integration.get("use_defended_teacher_statistics_for_fire", True)
            ),
            calibration_capture_only=bool(capture.get("enabled", False)),
            calibration_capture_path=capture.get("path"),
            calibration_uniform_alpha=float(capture.get("uniform_alpha", 0.20)),
        )
        config = config.resolve_calibration()
        config.validate()
        return config

    @property
    def is_enabled(self) -> bool:
        return self.enabled and self.method == "sparse_asymmetric_projection"

    @property
    def needs_teacher_statistics(self) -> bool:
        return self.is_enabled or self.calibration_capture_only

    @property
    def branch_mode(self) -> str:
        return _branch_mode(self.positive_weight, self.negative_weight)

    @property
    def lambda_cap(self) -> float:
        return self.max_effective_strength / max(
            self.positive_weight, self.negative_weight
        )

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "implementation_version": _IMPLEMENTATION_VERSION,
            "method": self.method,
            "active_ratio": self.active_ratio,
            "selection_strategy": self.selection_strategy,
            "positive_weight": self.positive_weight,
            "negative_weight": self.negative_weight,
            "allow_sign_flip": self.allow_sign_flip,
            "max_effective_strength": self.max_effective_strength,
            "preserve_teacher_top1": self.preserve_teacher_top1,
            "top1_margin": self.top1_margin,
            "eps": self.eps,
            "compute_dtype": self.compute_dtype,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.fingerprint_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolve_calibration(self) -> "TeacherDefenseConfig":
        if not self.calibration_result_path:
            return self
        path = Path(self.calibration_result_path).expanduser()
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        expected = self.fingerprint()
        actual = result.get("config_fingerprint")
        if actual != expected:
            raise ValueError(
                f"SA-TDP calibration fingerprint mismatch: expected {expected}, "
                f"found {actual} in {path}"
            )
        result_lambda = float(result["lambda_best"])
        if self.lambda_value is not None and not math.isclose(
            self.lambda_value, result_lambda, rel_tol=1.0e-9, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"configured lambda_value={self.lambda_value} disagrees with "
                f"calibration lambda_best={result_lambda}"
            )
        return replace(
            self,
            lambda_value=result_lambda,
            reference_mean_kl=float(result["reference_mean_kl"]),
            calibration_budget_shortfall=float(
                result.get("budget_shortfall", 0.0)
            ),
            calibration_fingerprint=actual,
        )

    def validate(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(
                f"unknown teacher defense method {self.method!r}; "
                f"expected {sorted(_METHODS)}"
            )
        if not 0.0 <= self.active_ratio <= 1.0:
            raise ValueError(
                f"active_ratio must be in [0, 1], got {self.active_ratio}"
            )
        if self.selection_strategy != "top_abs_advantage":
            raise ValueError(
                "SA-TDP v1 permits only deterministic top_abs_advantage selection"
            )
        _branch_mode(self.positive_weight, self.negative_weight)
        if not 0.0 < self.max_effective_strength < 1.0:
            raise ValueError("max_effective_strength must be in (0, 1)")
        if not self.preserve_teacher_top1:
            raise ValueError("SA-TDP v1 requires preserve_teacher_top1=true")
        if self.top1_margin < 0.0:
            raise ValueError("top1_margin must be non-negative")
        if self.eps <= 0.0 or self.eps >= 0.5:
            raise ValueError("eps must be in (0, 0.5)")
        if self.compute_dtype != "float32":
            raise ValueError(
                "SA-TDP v1 supports numerical.compute_dtype=float32 only"
            )
        if self.reference_kl_mode not in _REFERENCE_KL_MODES:
            raise ValueError(
                f"reference_kl_mode must be one of {sorted(_REFERENCE_KL_MODES)}"
            )
        if self.reference_mean_kl is not None and self.reference_mean_kl < 0.0:
            raise ValueError("reference_mean_kl must be non-negative")
        if self.calibration_iterations <= 0:
            raise ValueError("calibration iterations must be positive")
        if self.tolerance_ratio <= 0.0:
            raise ValueError("calibration tolerance_ratio must be positive")
        if self.lambda_value is not None:
            if self.lambda_value < 0.0:
                raise ValueError("lambda_value must be non-negative")
            if (
                not self.allow_sign_flip
                and self.lambda_value > self.lambda_cap + 1.0e-12
            ):
                raise ValueError(
                    f"lambda_value={self.lambda_value} exceeds no-sign-flip "
                    f"lambda_cap={self.lambda_cap}"
                )
        if self.is_enabled and self.lambda_value is None:
            raise ValueError(
                "enabled SA-TDP requires calibration.lambda_value or result_path"
            )
        if self.calibration_capture_only and not self.calibration_capture_path:
            raise ValueError("calibration capture requires capture.path")
        if not 0.0 <= self.calibration_uniform_alpha < 1.0:
            raise ValueError("capture uniform_alpha must be in [0, 1)")


@dataclass
class TeacherDefenseOutput:
    """Compact SA-TDP outputs, all detached from model parameters."""

    defended_sampled_logprob: torch.Tensor
    clean_advantage: torch.Tensor
    defended_advantage: torch.Tensor
    actual_delta: torch.Tensor
    active_mask: torch.Tensor
    branch_weight: torch.Tensor
    token_kl: torch.Tensor
    defended_entropy: torch.Tensor | None
    infeasible_mask: torch.Tensor
    metrics: dict[str, float]


def _stable_top_abs_mask(
    clean_advantage: torch.Tensor,
    response_mask: torch.Tensor,
    active_ratio: float,
) -> torch.Tensor:
    """Select ceil(ratio * valid length) per row; ties prefer lower positions."""

    active = torch.zeros_like(response_mask, dtype=torch.bool)
    if active_ratio == 0.0:
        return active
    for row in range(response_mask.shape[0]):
        positions = torch.nonzero(response_mask[row], as_tuple=False).squeeze(-1)
        count = int(positions.numel())
        if count == 0:
            continue
        selected_count = min(
            count, max(1, int(math.ceil(active_ratio * count)))
        )
        values = clean_advantage[row, positions].abs()
        order = torch.argsort(values, descending=True, stable=True)
        active[row, positions[order[:selected_count]]] = True
    return active


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value[mask]
    return float(selected.float().mean().item()) if selected.numel() else 0.0


def _advantage_mass_retention(
    clean_advantage: torch.Tensor,
    defended_advantage: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> float:
    clean_mass = clean_advantage[mask].abs().float().sum()
    if clean_mass.item() <= eps:
        return 1.0
    defended_mass = defended_advantage[mask].abs().float().sum()
    return float((defended_mass / clean_mass).item())


def _add_score_perturbation_metric_aliases(
    metrics: dict[str, float],
) -> dict[str, float]:
    """Expose the requested experiment metric names for the new defense."""

    alias_sources = {
        "active_fraction": "active_fraction",
        "selected_positive_fraction": "selected_positive_fraction",
        "selected_negative_fraction": "selected_negative_fraction",
        "global_scale": "lambda",
        "effective_lambda_positive": "effective_lambda_positive",
        "effective_lambda_negative": "effective_lambda_negative",
        "reference_uniform_kl": "reference_mean_kl",
        "actual_mean_kl": "mean_kl_all_valid",
        "budget_relative_error": "budget_relative_error",
        "budget_shortfall": "budget_shortfall",
        "mean_abs_logprob_shift": "mean_abs_delta_all_valid",
        "positive_advantage_mass_retention": (
            "positive_advantage_mass_retention"
        ),
        "negative_advantage_mass_retention": (
            "negative_advantage_mass_retention"
        ),
        "clean_defended_advantage_cosine": (
            "clean_defended_advantage_cosine"
        ),
        "advantage_sign_flip_fraction": "advantage_sign_flip_fraction",
        "top1_preservation_rate": "top1_preservation_rate",
        "top1_projection_fraction": "top1_projection_fraction",
        "teacher_entropy_shift": "teacher_entropy_shift",
        "finite_fraction": "finite_fraction",
        "defense_step_time_ms": "defense_time_ms",
    }
    for alias, source in alias_sources.items():
        metrics[f"score_perturbation/{alias}"] = metrics[
            f"teacher_defense/{source}"
        ]
    return metrics


def _binary_kl(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    value = q * (q.log() - p.log()) + (1.0 - q) * (
        torch.log1p(-q) - torch.log1p(-p)
    )
    return value.clamp_min(0.0)


def _project_preserving_top1(
    *,
    p_sampled: torch.Tensor,
    q_raw: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    teacher_top1_ids: torch.Tensor,
    teacher_top1_probs: torch.Tensor,
    teacher_top2_probs: torch.Tensor,
    active_mask: torch.Tensor,
    config: TeacherDefenseConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project q_y to the exact top-1 preserving boundary from the prompt."""

    expected = p_sampled.shape
    for name, tensor in {
        "teacher_top1_ids": teacher_top1_ids,
        "teacher_top1_probs": teacher_top1_probs,
        "teacher_top2_probs": teacher_top2_probs,
    }.items():
        if tensor.shape != expected:
            raise ValueError(f"{name} must have shape {tuple(expected)}")

    top1_ids = teacher_top1_ids.detach()
    p1 = teacher_top1_probs.detach().float()
    p2 = teacher_top2_probs.detach().float()
    sampled_is_top1 = sampled_token_ids == top1_ids
    one_minus_p = 1.0 - p_sampled
    max_probability = min(
        1.0 - config.eps, _FLOAT32_MAX_PROBABILITY
    )
    clean_top1_gap = torch.where(
        sampled_is_top1,
        p_sampled - p2,
        p1 - p_sampled,
    ).clamp_min(0.0)
    # A fixed probability margin larger than the clean top-1 gap would make
    # the clean distribution itself infeasible and could reverse the requested
    # perturbation direction.  Limit the per-token margin to half that gap;
    # exact clean ties therefore remain unchanged.
    effective_margin = torch.minimum(
        torch.full_like(p_sampled, config.top1_margin),
        0.5 * clean_top1_gap,
    )

    denominator_top1 = one_minus_p + p2
    q_min = (p2 + effective_margin * one_minus_p) / denominator_top1.clamp_min(
        config.eps
    )
    numerator_non_top1 = p1 - effective_margin * one_minus_p
    denominator_non_top1 = p1 + one_minus_p
    q_max = numerator_non_top1 / denominator_non_top1.clamp_min(config.eps)

    valid_top1_boundary = (
        torch.isfinite(q_min)
        & (denominator_top1 > config.eps)
        & (q_min >= config.eps)
        & (q_min <= max_probability)
    )
    valid_non_top1_boundary = (
        torch.isfinite(q_max)
        & (numerator_non_top1 >= 0.0)
        & (denominator_non_top1 > config.eps)
        & (q_max >= config.eps)
        & (q_max <= max_probability)
    )
    boundary_valid = torch.where(
        sampled_is_top1, valid_top1_boundary, valid_non_top1_boundary
    )
    infeasible = active_mask & ~boundary_valid

    projected = q_raw.clone()
    projected = torch.where(
        sampled_is_top1, torch.maximum(projected, q_min), projected
    )
    projected = torch.where(
        ~sampled_is_top1, torch.minimum(projected, q_max), projected
    )
    numerical_tie = clean_top1_gap <= config.eps
    projected = torch.where(
        active_mask & numerical_tie, p_sampled, projected
    )
    projected = torch.where(infeasible, p_sampled, projected)
    projected = torch.where(active_mask, projected, p_sampled)
    projected = projected.clamp(config.eps, max_probability)

    largest_other = torch.where(sampled_is_top1, p2, p1)
    scaled_other = largest_other * (1.0 - projected) / one_minus_p.clamp_min(
        config.eps
    )
    checked_preserved = torch.where(
        sampled_is_top1,
        projected + config.eps >= scaled_other,
        scaled_other + config.eps >= projected,
    )
    verification_failed = active_mask & ~checked_preserved
    # If compact top-1 statistics are too close for a reliable boundary check,
    # fall back to the exact clean sampled probability.  The clean distribution
    # is authoritative and therefore preserves its own deterministic argmax.
    projected = torch.where(verification_failed, p_sampled, projected)
    infeasible = infeasible | verification_failed
    projection_mask = active_mask & (
        (projected - q_raw).abs() > config.eps
    )
    largest_other = torch.where(sampled_is_top1, p2, p1)
    scaled_other = largest_other * (1.0 - projected) / one_minus_p.clamp_min(
        config.eps
    )
    preserved = torch.where(
        sampled_is_top1,
        projected + config.eps >= scaled_other,
        scaled_other + config.eps >= projected,
    )
    preserved = torch.where(
        infeasible,
        torch.ones_like(preserved),
        preserved,
    )
    preserved = torch.where(active_mask, preserved, torch.ones_like(preserved))
    return projected, projection_mask, infeasible, preserved


def _empty_metrics(
    *,
    config: TeacherDefenseConfig,
    clean_advantage: torch.Tensor,
    valid: torch.Tensor,
    elapsed_ms: float,
) -> dict[str, float]:
    valid_count = int(valid.sum().item())
    positive = valid & (clean_advantage >= 0.0)
    negative = valid & (clean_advantage < 0.0)
    reference = float(config.reference_mean_kl or 0.0)
    lambda_value = float(config.lambda_value or 0.0)
    metrics = {
        "teacher_defense/active_fraction": 0.0,
        "teacher_defense/selected_positive_fraction": 0.0,
        "teacher_defense/selected_negative_fraction": 0.0,
        "teacher_defense/lambda": lambda_value,
        "teacher_defense/positive_weight": config.positive_weight,
        "teacher_defense/negative_weight": config.negative_weight,
        "teacher_defense/effective_lambda_positive": (
            lambda_value * config.positive_weight
        ),
        "teacher_defense/effective_lambda_negative": (
            lambda_value * config.negative_weight
        ),
        "teacher_defense/mean_abs_delta_all_valid": 0.0,
        "teacher_defense/mean_abs_delta_active": 0.0,
        "teacher_defense/mean_kl_all_valid": 0.0,
        "teacher_defense/mean_kl_active": 0.0,
        "teacher_defense/reference_mean_kl": reference,
        "teacher_defense/budget_relative_error": (
            1.0 if reference > config.eps else 0.0
        ),
        "teacher_defense/budget_shortfall": reference,
        "teacher_defense/top1_preservation_rate": 1.0,
        "teacher_defense/top1_projection_fraction": 0.0,
        "teacher_defense/infeasible_fraction": 0.0,
        "teacher_defense/clean_positive_advantage_fraction": (
            int(positive.sum().item()) / valid_count if valid_count else 0.0
        ),
        "teacher_defense/clean_negative_advantage_fraction": (
            int(negative.sum().item()) / valid_count if valid_count else 0.0
        ),
        "teacher_defense/advantage_sign_flip_fraction": 0.0,
        "teacher_defense/positive_advantage_mass_retention": 1.0,
        "teacher_defense/negative_advantage_mass_retention": 1.0,
        "teacher_defense/clean_defended_advantage_cosine": 1.0,
        "teacher_defense/finite_fraction": 1.0,
        "teacher_defense/defense_time_ms": elapsed_ms,
        "teacher_defense/teacher_entropy_shift": 0.0,
    }
    return _add_score_perturbation_metric_aliases(metrics)


@torch.no_grad()
def apply_sparse_asymmetric_projection(
    *,
    clean_teacher_sampled_logprob: torch.Tensor,
    student_rollout_logprob: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_top1_ids: torch.Tensor | None,
    teacher_top1_probs: torch.Tensor | None,
    teacher_top2_probs: torch.Tensor | None,
    clean_teacher_entropy: torch.Tensor | None,
    config: TeacherDefenseConfig,
) -> TeacherDefenseOutput:
    """Apply SA-TDP to compact ``[batch, response_length]`` statistics."""

    started = time.perf_counter()
    config.validate()
    expected = clean_teacher_sampled_logprob.shape
    if clean_teacher_sampled_logprob.ndim != 2:
        raise ValueError(
            "SA-TDP expects clean_teacher_sampled_logprob with shape [B, L]"
        )
    for name, tensor in {
        "student_rollout_logprob": student_rollout_logprob,
        "sampled_token_ids": sampled_token_ids,
        "response_mask": response_mask,
    }.items():
        if tensor.shape != expected:
            raise ValueError(f"{name} must have shape {tuple(expected)}")
    if (
        clean_teacher_entropy is not None
        and clean_teacher_entropy.shape != expected
    ):
        raise ValueError(
            "clean_teacher_entropy must have shape "
            f"{tuple(expected)}, got {tuple(clean_teacher_entropy.shape)}"
        )

    output_dtype = clean_teacher_sampled_logprob.dtype
    device = clean_teacher_sampled_logprob.device
    clean_logprob = clean_teacher_sampled_logprob.detach().float()
    old_logprob = student_rollout_logprob.detach().to(device=device).float()
    sampled_ids = sampled_token_ids.detach().to(device=device)
    valid = response_mask.detach().to(device=device).bool()
    clean_advantage_raw = clean_logprob - old_logprob
    zero = torch.zeros_like(clean_logprob)
    clean_advantage = torch.where(valid, clean_advantage_raw, zero)

    no_op = (
        not config.is_enabled
        or config.active_ratio == 0.0
        or float(config.lambda_value or 0.0) == 0.0
    )
    if no_op:
        entropy = (
            None
            if clean_teacher_entropy is None
            else clean_teacher_entropy.detach().to(device=device)
        )
        metrics = _empty_metrics(
            config=config,
            clean_advantage=clean_advantage,
            valid=valid,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return TeacherDefenseOutput(
            defended_sampled_logprob=clean_teacher_sampled_logprob.detach(),
            clean_advantage=clean_advantage.detach(),
            defended_advantage=clean_advantage.detach(),
            actual_delta=zero.detach(),
            active_mask=torch.zeros_like(valid),
            branch_weight=zero.detach(),
            token_kl=zero.detach(),
            defended_entropy=entropy,
            infeasible_mask=torch.zeros_like(valid),
            metrics=metrics,
        )

    if (
        teacher_top1_ids is None
        or teacher_top1_probs is None
        or teacher_top2_probs is None
    ):
        raise ValueError(
            "enabled SA-TDP requires clean teacher top-1 ids/probabilities "
            "and top-2 probabilities"
        )

    active = _stable_top_abs_mask(
        clean_advantage, valid, config.active_ratio
    )
    branch_weight = torch.where(
        clean_advantage >= 0.0,
        torch.full_like(clean_advantage, config.positive_weight),
        torch.full_like(clean_advantage, config.negative_weight),
    )
    branch_weight = torch.where(active, branch_weight, zero)
    strength = float(config.lambda_value) * branch_weight
    if (
        not config.allow_sign_flip
        and torch.any(
            strength[active]
            > config.max_effective_strength + 1.0e-7
        )
    ):
        raise ValueError(
            "lambda * branch weight exceeds max_effective_strength; "
            "SA-TDP v1 does not dynamically clip strength"
        )

    requested_delta = -strength * clean_advantage
    max_probability = min(
        1.0 - config.eps, _FLOAT32_MAX_PROBABILITY
    )
    p_sampled = clean_logprob.exp().clamp(
        config.eps, max_probability
    )
    min_log_probability = math.log(config.eps)
    max_log_probability = math.log(max_probability)
    requested_log_probability = (
        clean_logprob + requested_delta
    ).clamp(min_log_probability, max_log_probability)
    q_raw = requested_log_probability.exp()
    q_raw = torch.where(active, q_raw, p_sampled)

    q_sampled, projection_mask, infeasible, top1_preserved = (
        _project_preserving_top1(
            p_sampled=p_sampled,
            q_raw=q_raw,
            sampled_token_ids=sampled_ids,
            teacher_top1_ids=teacher_top1_ids.detach().to(device=device),
            teacher_top1_probs=teacher_top1_probs.detach().to(device=device),
            teacher_top2_probs=teacher_top2_probs.detach().to(device=device),
            active_mask=active,
            config=config,
        )
    )

    projected_logprob = q_sampled.log()
    if config.allow_sign_flip:
        guarded_logprob = projected_logprob
    else:
        lower_logprob = torch.minimum(clean_logprob, old_logprob)
        upper_logprob = torch.maximum(clean_logprob, old_logprob)
        guarded_logprob = torch.minimum(
            torch.maximum(projected_logprob, lower_logprob),
            upper_logprob,
        )
        sign_guard_projection = active & (
            (guarded_logprob - projected_logprob).abs() > config.eps
        )
        projection_mask = projection_mask | sign_guard_projection
    probability_unchanged = active & (
        (guarded_logprob - clean_logprob).abs() <= config.eps
    )
    defended_logprob_float = torch.where(
        active & ~probability_unchanged,
        guarded_logprob,
        clean_logprob,
    )
    actual_delta = torch.where(
        active,
        defended_logprob_float - clean_logprob,
        zero,
    )
    q_sampled = torch.where(
        active & ~probability_unchanged,
        defended_logprob_float.exp().clamp(config.eps, max_probability),
        p_sampled,
    )
    defended_logprob = defended_logprob_float.to(dtype=output_dtype)
    defended_advantage_raw = defended_logprob_float - old_logprob
    defended_advantage = torch.where(valid, defended_advantage_raw, zero)
    token_kl = torch.where(active, _binary_kl(q_sampled, p_sampled), zero)

    defended_entropy: torch.Tensor | None
    entropy_shift = 0.0
    if clean_teacher_entropy is None:
        defended_entropy = None
    else:
        clean_entropy_dtype = clean_teacher_entropy.dtype
        clean_entropy = (
            clean_teacher_entropy.detach().to(device=device).float()
        )
        scale_other = (1.0 - q_sampled) / (1.0 - p_sampled).clamp_min(
            config.eps
        )
        projected_entropy = (
            -torch.special.xlogy(q_sampled, q_sampled)
            + scale_other
            * (
                clean_entropy
                + torch.special.xlogy(p_sampled, p_sampled)
            )
            - torch.special.xlogy(1.0 - q_sampled, scale_other)
        )
        defended_entropy_float = torch.where(
            active, projected_entropy, clean_entropy
        )
        defended_entropy = defended_entropy_float.to(
            dtype=clean_entropy_dtype
        ).detach()
        entropy_shift = _masked_mean(
            defended_entropy_float - clean_entropy, valid
        )

    active_count = int(active.sum().item())
    valid_count = int(valid.sum().item())
    selected_positive = active & (clean_advantage >= 0.0)
    selected_negative = active & (clean_advantage < 0.0)
    valid_positive = valid & (clean_advantage >= 0.0)
    valid_negative = valid & (clean_advantage < 0.0)
    sign_flip = (
        valid
        & (clean_advantage != 0.0)
        & (clean_advantage * defended_advantage < 0.0)
    )
    clean_vector = clean_advantage[valid].float()
    defended_vector = defended_advantage[valid].float()
    norm_product = clean_vector.norm() * defended_vector.norm()
    cosine = (
        float(torch.dot(clean_vector, defended_vector).div(norm_product).item())
        if clean_vector.numel() and norm_product.item() > 0.0
        else 1.0
    )
    mean_kl = _masked_mean(token_kl, valid)
    reference = float(config.reference_mean_kl or 0.0)
    budget_relative_error = (
        abs(mean_kl - reference) / reference
        if reference > config.eps
        else 0.0
    )
    budget_shortfall = max(reference - mean_kl, 0.0)
    positive_mass_retention = _advantage_mass_retention(
        clean_advantage,
        defended_advantage,
        valid_positive,
        config.eps,
    )
    negative_mass_retention = _advantage_mass_retention(
        clean_advantage,
        defended_advantage,
        valid_negative,
        config.eps,
    )
    finite_tensors = [
        defended_logprob_float[valid],
        defended_advantage[valid],
        actual_delta[valid],
        token_kl[valid],
    ]
    if defended_entropy is not None:
        finite_tensors.append(defended_entropy.float()[valid])
    finite_values = torch.cat(
        [tensor.reshape(-1) for tensor in finite_tensors]
    )

    metrics = {
        "teacher_defense/active_fraction": (
            active_count / valid_count if valid_count else 0.0
        ),
        "teacher_defense/selected_positive_fraction": (
            int(selected_positive.sum().item()) / active_count
            if active_count
            else 0.0
        ),
        "teacher_defense/selected_negative_fraction": (
            int(selected_negative.sum().item()) / active_count
            if active_count
            else 0.0
        ),
        "teacher_defense/lambda": float(config.lambda_value),
        "teacher_defense/positive_weight": config.positive_weight,
        "teacher_defense/negative_weight": config.negative_weight,
        "teacher_defense/effective_lambda_positive": (
            float(config.lambda_value) * config.positive_weight
        ),
        "teacher_defense/effective_lambda_negative": (
            float(config.lambda_value) * config.negative_weight
        ),
        "teacher_defense/mean_abs_delta_all_valid": _masked_mean(
            actual_delta.abs(), valid
        ),
        "teacher_defense/mean_abs_delta_active": _masked_mean(
            actual_delta.abs(), active
        ),
        "teacher_defense/mean_kl_all_valid": mean_kl,
        "teacher_defense/mean_kl_active": _masked_mean(token_kl, active),
        "teacher_defense/reference_mean_kl": reference,
        "teacher_defense/budget_relative_error": budget_relative_error,
        "teacher_defense/budget_shortfall": budget_shortfall,
        "teacher_defense/top1_preservation_rate": (
            int(top1_preserved[valid].sum().item()) / valid_count
            if valid_count
            else 1.0
        ),
        "teacher_defense/top1_projection_fraction": (
            int(projection_mask.sum().item()) / active_count
            if active_count
            else 0.0
        ),
        "teacher_defense/infeasible_fraction": (
            int(infeasible.sum().item()) / active_count
            if active_count
            else 0.0
        ),
        "teacher_defense/clean_positive_advantage_fraction": (
            int(valid_positive.sum().item()) / valid_count
            if valid_count
            else 0.0
        ),
        "teacher_defense/clean_negative_advantage_fraction": (
            int(valid_negative.sum().item()) / valid_count
            if valid_count
            else 0.0
        ),
        "teacher_defense/advantage_sign_flip_fraction": (
            int(sign_flip.sum().item()) / valid_count
            if valid_count
            else 0.0
        ),
        "teacher_defense/positive_advantage_mass_retention": (
            positive_mass_retention
        ),
        "teacher_defense/negative_advantage_mass_retention": (
            negative_mass_retention
        ),
        "teacher_defense/clean_defended_advantage_cosine": cosine,
        "teacher_defense/finite_fraction": (
            float(torch.isfinite(finite_values).float().mean().item())
            if finite_values.numel()
            else 1.0
        ),
        "teacher_defense/defense_time_ms": (
            time.perf_counter() - started
        )
        * 1000.0,
        "teacher_defense/teacher_entropy_shift": entropy_shift,
    }
    metrics = _add_score_perturbation_metric_aliases(metrics)
    return TeacherDefenseOutput(
        defended_sampled_logprob=defended_logprob.detach(),
        clean_advantage=clean_advantage.detach(),
        defended_advantage=defended_advantage.detach(),
        actual_delta=actual_delta.detach(),
        active_mask=active.detach(),
        branch_weight=branch_weight.detach(),
        token_kl=token_kl.detach(),
        defended_entropy=defended_entropy,
        infeasible_mask=infeasible.detach(),
        metrics=metrics,
    )


def _config_with_lambda(
    config: TeacherDefenseConfig,
    lambda_value: float,
    reference_mean_kl: float,
) -> TeacherDefenseConfig:
    return replace(
        config,
        enabled=True,
        method="sparse_asymmetric_projection",
        lambda_value=float(lambda_value),
        calibration_result_path=None,
        reference_mean_kl=float(reference_mean_kl),
        calibration_budget_shortfall=0.0,
    )


def _statistics_output(
    statistics: Mapping[str, torch.Tensor],
    config: TeacherDefenseConfig,
    lambda_value: float,
    reference_mean_kl: float,
) -> TeacherDefenseOutput:
    return apply_sparse_asymmetric_projection(
        clean_teacher_sampled_logprob=statistics[
            "clean_teacher_sampled_logprob"
        ],
        student_rollout_logprob=statistics["student_rollout_logprob"],
        sampled_token_ids=statistics["sampled_token_ids"],
        response_mask=statistics["response_mask"],
        teacher_top1_ids=statistics["teacher_top1_ids"],
        teacher_top1_probs=statistics["teacher_top1_probs"],
        teacher_top2_probs=statistics["teacher_top2_probs"],
        clean_teacher_entropy=statistics.get("clean_teacher_entropy"),
        config=_config_with_lambda(
            config, lambda_value, reference_mean_kl
        ),
    )


@torch.no_grad()
def calibrate_lambda(
    *,
    statistics: Mapping[str, torch.Tensor],
    config: TeacherDefenseConfig,
    reference_mean_kl: float,
    iterations: int | None = None,
) -> dict[str, Any]:
    """Calibrate one global lambda using one immutable statistics cache."""

    required = {
        "clean_teacher_sampled_logprob",
        "student_rollout_logprob",
        "sampled_token_ids",
        "response_mask",
        "teacher_top1_ids",
        "teacher_top1_probs",
        "teacher_top2_probs",
    }
    missing = sorted(required.difference(statistics))
    if missing:
        raise ValueError(f"calibration statistics missing keys: {missing}")
    if reference_mean_kl < 0.0:
        raise ValueError("reference_mean_kl must be non-negative")
    search_iterations = (
        config.calibration_iterations if iterations is None else iterations
    )
    if search_iterations <= 0:
        raise ValueError("calibration iterations must be positive")

    high = config.lambda_cap
    high_output = _statistics_output(
        statistics, config, high, reference_mean_kl
    )
    attainable = high_output.metrics[
        "teacher_defense/mean_kl_all_valid"
    ]
    if not math.isfinite(attainable):
        raise FloatingPointError(
            "SA-TDP calibration produced non-finite K(lambda_cap); "
            "inspect cached probabilities and advantages"
        )
    if attainable < reference_mean_kl:
        lambda_best = high
        chosen = high_output
        shortfall = reference_mean_kl - attainable
    else:
        low = 0.0
        for _ in range(search_iterations):
            middle = 0.5 * (low + high)
            candidate = _statistics_output(
                statistics, config, middle, reference_mean_kl
            )
            candidate_kl = candidate.metrics[
                "teacher_defense/mean_kl_all_valid"
            ]
            if not math.isfinite(candidate_kl):
                raise FloatingPointError(
                    "SA-TDP calibration produced non-finite K(lambda) "
                    f"at lambda={middle}"
                )
            if candidate_kl <= reference_mean_kl:
                low = middle
            else:
                high = middle
        lambda_best = low
        chosen = _statistics_output(
            statistics, config, lambda_best, reference_mean_kl
        )
        shortfall = max(
            reference_mean_kl
            - chosen.metrics["teacher_defense/mean_kl_all_valid"],
            0.0,
        )

    actual = chosen.metrics["teacher_defense/mean_kl_all_valid"]
    if not math.isfinite(actual):
        raise FloatingPointError(
            "SA-TDP calibration selected a non-finite mean KL"
        )
    relative_error = (
        abs(actual - reference_mean_kl) / reference_mean_kl
        if reference_mean_kl > config.eps
        else 0.0
    )
    return {
        "schema_version": 1,
        "method": "sparse_asymmetric_projection",
        "branch_mode": config.branch_mode,
        "positive_weight": config.positive_weight,
        "negative_weight": config.negative_weight,
        "active_ratio": config.active_ratio,
        "config": config.fingerprint_payload(),
        "config_fingerprint": config.fingerprint(),
        "reference_mean_kl": float(reference_mean_kl),
        "lambda_best": float(lambda_best),
        "lambda_cap": float(config.lambda_cap),
        "actual_mean_kl": float(actual),
        "active_mean_kl": float(
            chosen.metrics["teacher_defense/mean_kl_active"]
        ),
        "attainable_kl": float(attainable),
        "relative_error": float(relative_error),
        "budget_shortfall": float(shortfall),
        "top1_projection_fraction": float(
            chosen.metrics["teacher_defense/top1_projection_fraction"]
        ),
        "infeasible_fraction": float(
            chosen.metrics["teacher_defense/infeasible_fraction"]
        ),
        "iterations": int(search_iterations),
    }


def save_calibration_result(
    path: str | Path, result: Mapping[str, Any]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            dict(result), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def config_as_dict(config: TeacherDefenseConfig) -> dict[str, Any]:
    return asdict(config)
