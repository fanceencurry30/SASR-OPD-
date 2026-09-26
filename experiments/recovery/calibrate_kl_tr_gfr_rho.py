#!/usr/bin/env python3
"""Calibrate one global KL-TR-GFR rho for each recovery-action KL budget."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from verl.anti_opd.teacher_signal_recovery import (
    KLTRGFRConfig,
    apply_kl_tr_gfr,
)


KAPPAS = (0.10, 0.25, 0.50)
RHO_GRID = tuple(index * 0.05 for index in range(21))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_config(rho: float, variant: str) -> KLTRGFRConfig:
    return KLTRGFRConfig.from_mapping(
        {
            "enabled": True,
            "method": "kl_tr_gfr",
            "source_gfr": {"variant": variant},
            "damping": {"mode": "fixed", "rho": rho},
            "defense_parameters": {
                "active_ratio": 0.20,
                "lambda_value": 0.75,
                "positive_weight": 0.5,
                "negative_weight": 1.0,
            },
            "constraints": {
                "preserve_top1": True,
                "top1_margin": 1.0e-6,
            },
            "numerical": {"eps": 1.0e-8, "compute_dtype": "float32"},
            "diagnostics": {
                "enable_oracle_metrics": False,
                "calibration_size": 256,
            },
        }
    )


def attack_inputs(cache: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    forbidden_fragments = ("clean", "true_", "branch_mask", "projection_mask")
    leaked = [
        key
        for key in cache
        if any(fragment in key.lower() for fragment in forbidden_fragments)
    ]
    if leaked:
        raise RuntimeError(f"attacker cache contains oracle keys: {leaked}")
    required = (
        "defended_teacher_sampled_logprob",
        "student_rollout_logprob",
        "sampled_token_ids",
        "response_mask",
        "defended_top1_ids",
        "defended_top1_probs",
        "defended_top2_probs",
    )
    missing = [name for name in required if name not in cache]
    if missing:
        raise KeyError(f"attacker cache is missing: {missing}")
    return {name: cache[name].detach() for name in required}


def evaluate_rho(
    cache: Mapping[str, Any], rho: float, variant: str
) -> dict[str, float]:
    output = apply_kl_tr_gfr(
        **attack_inputs(cache),
        rho=rho,
        gfr_variant=variant,
        config=fixed_config(rho, variant),
    )
    return {
        "rho": float(rho),
        "mean_recovery_kl": output.metrics["recovery/mean_recovery_kl"],
        "active_recovery_kl": output.metrics["recovery/active_recovery_kl"],
        "top1_projection_fraction": output.metrics[
            "recovery/top1_projection_fraction"
        ],
        "infeasible_fraction": output.metrics["recovery/infeasible_fraction"],
        "mean_abs_recovery_delta": output.metrics[
            "recovery/mean_abs_delta_actual"
        ],
        "recovery_active_fraction": output.metrics[
            "recovery/recovery_active_fraction"
        ],
        "top1_preservation_rate": output.metrics[
            "recovery/top1_preservation_rate"
        ],
        "finite_fraction": output.metrics["recovery/finite_fraction"],
    }


def build_rho_curve(
    cache: Mapping[str, Any],
    variant: str,
    *,
    grid: tuple[float, ...] = RHO_GRID,
    monotonic_tolerance: float = 1.0e-8,
) -> list[dict[str, float]]:
    curve = [evaluate_rho(cache, rho, variant) for rho in grid]
    for left, right in zip(curve, curve[1:]):
        if (
            right["mean_recovery_kl"] + monotonic_tolerance
            < left["mean_recovery_kl"]
        ):
            raise RuntimeError(
                "K(rho) is non-monotonic: "
                f"K({left['rho']})={left['mean_recovery_kl']} > "
                f"K({right['rho']})={right['mean_recovery_kl']}"
            )
    return curve


def bisect_rho(
    cache: Mapping[str, Any],
    variant: str,
    target_kl: float,
    *,
    iterations: int = 25,
) -> dict[str, float]:
    if target_kl < 0:
        raise ValueError("target_kl must be non-negative")
    low = 0.0
    high = 1.0
    low_row = evaluate_rho(cache, low, variant)
    for _ in range(iterations):
        mid = (low + high) / 2.0
        row = evaluate_rho(cache, mid, variant)
        if row["mean_recovery_kl"] <= target_kl:
            low = mid
            low_row = row
        else:
            high = mid
    # Guard against a final float32 rounding overshoot.
    if low_row["mean_recovery_kl"] > target_kl:
        low = math.nextafter(low, 0.0)
        low_row = evaluate_rho(cache, low, variant)
    return low_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacker-cache", type=Path, required=True)
    parser.add_argument(
        "--gfr-variant",
        choices=(
            "identity",
            "single_positive",
            "single_negative",
            "sign_positive",
            "sign_negative",
        ),
        default="sign_positive",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--monotonic-tolerance", type=float, default=1.0e-8)
    args = parser.parse_args()

    cache = torch.load(args.attacker_cache, map_location="cpu", weights_only=False)
    kwargs = attack_inputs(cache)
    if int(kwargs["response_mask"].shape[0]) != 256:
        raise ValueError("KL-TR-GFR calibration requires exactly 256 samples")
    metadata = dict(cache.get("metadata", {}) or {})
    if int(metadata.get("seed", 42)) != 42:
        raise ValueError("KL-TR-GFR calibration requires seed=42")
    variant = args.gfr_variant

    curve = build_rho_curve(
        cache,
        variant,
        monotonic_tolerance=args.monotonic_tolerance,
    )
    k_full = float(curve[-1]["mean_recovery_kl"])
    if k_full <= 1.0e-12:
        raise RuntimeError(
            "full recovery action KL is approximately zero; calibration stopped"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.output_dir / "rho_kl_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)

    calibrations = []
    for kappa in KAPPAS:
        target = kappa * k_full
        selected = bisect_rho(
            cache,
            variant,
            target,
            iterations=args.iterations,
        )
        actual = float(selected["mean_recovery_kl"])
        if actual > target + 1.0e-10:
            raise RuntimeError(
                f"calibrated KL exceeds budget for kappa={kappa}: {actual}>{target}"
            )
        calibrations.append(
            {
                "best_gfr_variant": variant,
                "kappa": kappa,
                "k_full": k_full,
                "target_recovery_kl": target,
                "rho_star": float(selected["rho"]),
                "actual_recovery_kl": actual,
                "relative_budget_error": (
                    abs(actual - target) / target if target > 0 else 0.0
                ),
                "budget_shortfall": max(0.0, target - actual),
                "top1_projection_fraction": selected[
                    "top1_projection_fraction"
                ],
                "infeasible_fraction": selected["infeasible_fraction"],
                "finite_fraction": selected["finite_fraction"],
                "calibration_size": 256,
                "seed": 42,
                "teacher_checkpoint": metadata.get("teacher_checkpoint"),
                "student_checkpoint": metadata.get("student_checkpoint"),
                "defense_lambda": 0.75,
                "defense_positive_weight": 0.5,
                "defense_negative_weight": 1.0,
            }
        )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "kl_tr_gfr",
        "best_gfr_variant": variant,
        "attacker_cache": str(args.attacker_cache),
        "attacker_cache_sha256": sha256(args.attacker_cache),
        "k_full": k_full,
        "rho_grid": list(RHO_GRID),
        "monotonic_tolerance": args.monotonic_tolerance,
        "bisection_iterations": args.iterations,
        "calibrations": calibrations,
        "oracle_inputs_used_for_calibration": False,
    }
    json_path = args.output_dir / "kl_tr_gfr_calibration.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_path = args.output_dir / "kl_tr_gfr_calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(calibrations[0]))
        writer.writeheader()
        writer.writerows(calibrations)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
