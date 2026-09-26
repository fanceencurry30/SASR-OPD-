#!/usr/bin/env python3
"""Materialize the fixed attacker-view cache used by KL-TR-GFR calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from verl.anti_opd.teacher_distribution_defense import (
    TeacherDefenseConfig,
    apply_sparse_asymmetric_projection,
)
from verl.anti_opd.teacher_signal_recovery import derive_defended_top_statistics


FORBIDDEN_ATTACK_KEYS = (
    "clean_teacher_sampled_logprob",
    "clean_advantage",
    "true_active_mask",
    "true_branch_mask",
    "true_projection_mask",
    "projection_mask",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_high_defense_config() -> TeacherDefenseConfig:
    return TeacherDefenseConfig.from_mapping(
        {
            "enabled": True,
            "method": "sparse_asymmetric_projection",
            "selection": {
                "active_ratio": 0.20,
                "strategy": "top_abs_advantage",
            },
            "weights": {"positive": 0.5, "negative": 1.0},
            "calibration": {"lambda_value": 0.75},
            "constraints": {
                "allow_sign_flip": False,
                "max_effective_strength": 0.95,
                "preserve_teacher_top1": True,
                "top1_margin": 1.0e-6,
            },
            "numerical": {"eps": 1.0e-8, "compute_dtype": "float32"},
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-stats", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, default=256)
    args = parser.parse_args()

    source = torch.load(args.source_stats, map_location="cpu", weights_only=False)
    clean = source["clean_teacher_sampled_logprob"].detach()
    student = source["student_rollout_logprob"].detach()
    sampled = source["sampled_token_ids"].detach()
    mask = source["response_mask"].detach().bool()
    top1_ids = source["teacher_top1_ids"].detach()
    top1_probs = source["teacher_top1_probs"].detach()
    top2_probs = source["teacher_top2_probs"].detach()
    if clean.shape[0] != args.expected_size:
        raise ValueError(
            f"expected {args.expected_size} calibration samples, got {clean.shape[0]}"
        )

    defense = apply_sparse_asymmetric_projection(
        clean_teacher_sampled_logprob=clean,
        student_rollout_logprob=student,
        sampled_token_ids=sampled,
        response_mask=mask,
        teacher_top1_ids=top1_ids,
        teacher_top1_probs=top1_probs,
        teacher_top2_probs=top2_probs,
        clean_teacher_entropy=source.get("clean_teacher_entropy"),
        config=stable_high_defense_config(),
    )
    defended_top1, defended_top2 = derive_defended_top_statistics(
        clean_teacher_sampled_logprob=clean,
        defended_teacher_sampled_logprob=defense.defended_sampled_logprob,
        sampled_token_ids=sampled,
        clean_top1_ids=top1_ids,
        clean_top1_probs=top1_probs,
        clean_top2_probs=top2_probs,
    )
    attacker_view = {
        "defended_teacher_sampled_logprob": (
            defense.defended_sampled_logprob.detach().cpu()
        ),
        "student_rollout_logprob": student.cpu(),
        "sampled_token_ids": sampled.cpu(),
        "response_mask": mask.cpu(),
        "defended_top1_ids": top1_ids.cpu(),
        "defended_top1_probs": defended_top1.detach().cpu(),
        "defended_top2_probs": defended_top2.detach().cpu(),
        "metadata": {
            "schema_version": 1,
            "attack_view": True,
            "teacher_checkpoint": source.get("teacher_checkpoint"),
            "student_checkpoint": source.get("student_checkpoint"),
            "seed": 42,
            "calibration_size": int(clean.shape[0]),
            "valid_tokens": int(mask.sum().item()),
            "defense_active_ratio": 0.20,
            "defense_lambda": 0.75,
            "defense_positive_weight": 0.5,
            "defense_negative_weight": 1.0,
            "allow_sign_flip": False,
            "preserve_top1": True,
        },
    }
    forbidden_present = [
        key for key in FORBIDDEN_ATTACK_KEYS if key in attacker_view
    ]
    if forbidden_present:
        raise RuntimeError(
            f"oracle keys leaked into attacker cache: {forbidden_present}"
        )

    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(attacker_view, args.output_cache)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_stats": str(args.source_stats),
        "source_stats_sha256": sha256(args.source_stats),
        "attacker_cache": str(args.output_cache),
        "attacker_cache_sha256": sha256(args.output_cache),
        "calibration_size": int(clean.shape[0]),
        "seed": 42,
        "forbidden_attack_inputs_absent": list(FORBIDDEN_ATTACK_KEYS),
        "note": (
            "Clean statistics are used only by this research-side materialization "
            "step to reproduce the observable SA-TDP Stable-High distribution. The saved "
            "attacker cache contains only defended/student/top-2 compact statistics."
        ),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
