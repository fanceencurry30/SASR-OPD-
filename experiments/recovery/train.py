#!/usr/bin/env python3
"""Launch the three adaptive recovery attacks from the paper."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = EXPERIMENTS_ROOT.parent
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from train import build_command  # noqa: E402


def replace_override(command: list[str], key: str, value: str) -> None:
    prefix = f"{key}="
    for index, item in enumerate(command):
        if item.startswith(prefix):
            command[index] = f"{prefix}{value}"
            return
    command.append(f"{prefix}{value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("full-gfr", "spfr", "kl-gfr"))
    parser.add_argument("gpu_ids")
    parser.add_argument("steps", type=int, nargs="?", default=40)
    args = parser.parse_args()

    base = argparse.Namespace(
        suite="strong-to-weak",
        method="opd",
        protection="sasr",
        gpu_ids=args.gpu_ids,
        steps=args.steps,
    )
    command, _, run_env = build_command(base)
    gopd_root = Path(
        os.environ.get("GOPD_ROOT", RELEASE_ROOT / "third_party/G-OPD")
    ).expanduser().resolve()
    working_directory = gopd_root / "verl"
    output_root = Path(
        os.environ.get("OUTPUT_ROOT", RELEASE_ROOT / "outputs")
    ).expanduser().resolve()
    output = output_root / "recovery" / args.mode
    output.mkdir(parents=True, exist_ok=True)

    replace_override(command, "trainer.project_name", "sasr-recovery")
    replace_override(command, "trainer.experiment_name", args.mode)
    replace_override(command, "trainer.default_local_dir", str(output))

    recovery = [
        "teacher_recovery.enabled=true",
        "teacher_recovery.defense_parameters.active_ratio=0.20",
        "teacher_recovery.defense_parameters.lambda_value=0.75",
        "teacher_recovery.defense_parameters.positive_weight=0.5",
        "teacher_recovery.defense_parameters.negative_weight=1.0",
        "teacher_recovery.constraints.use_top1_filter=true",
        "teacher_recovery.constraints.preserve_top1=true",
        "teacher_recovery.constraints.top1_margin=1.0e-6",
        "teacher_recovery.constraints.invalid_candidate_fallback=identity",
        "teacher_recovery.numerical.eps=1.0e-8",
        "teacher_recovery.numerical.compute_dtype=float32",
        "teacher_recovery.diagnostics.enable_oracle_metrics=false",
    ]
    if args.mode == "full-gfr":
        recovery.extend(
            [
                "teacher_recovery.method=global_formula_recovery",
                "teacher_recovery.global.variant=sign_positive",
                "teacher_recovery.sparse.variant=null",
            ]
        )
    elif args.mode == "spfr":
        recovery.extend(
            [
                "teacher_recovery.method=sparse_position_formula_recovery",
                "teacher_recovery.global.variant=null",
                "teacher_recovery.sparse.variant=positive_priority",
            ]
        )
    else:
        calibration_root = Path(
            os.environ.get("CALIBRATION_ROOT", RELEASE_ROOT / "calibration")
        ).expanduser().resolve()
        calibration = calibration_root / "kl_tr_gfr_calibration.json"
        if os.environ.get("SASR_CONFIG_ONLY") != "1" and not calibration.is_file():
            raise FileNotFoundError(
                f"KL-GFR calibration is missing: {calibration}; run the included calibration scripts"
            )
        recovery.extend(
            [
                "teacher_recovery.method=kl_tr_gfr",
                "teacher_recovery.source_gfr.selected_variant_path=null",
                "teacher_recovery.source_gfr.variant=sign_positive",
                "teacher_recovery.damping.mode=kl_calibrated",
                "teacher_recovery.damping.rho=null",
                f"teacher_recovery.damping.calibration_result_path={calibration}",
                f"teacher_recovery.damping.kappa={os.environ.get('KAPPA', '0.25')}",
                "teacher_recovery.diagnostics.calibration_size=256",
            ]
        )

    flag_index = command.index("--cfg") if "--cfg" in command else len(command)
    command[flag_index:flag_index] = recovery
    if not working_directory.is_dir():
        raise FileNotFoundError(f"G-OPD checkout is missing: {gopd_root}")
    print("working directory:", working_directory)
    print("command:", shlex.join(command))
    subprocess.run(command, cwd=working_directory, env=run_env, check=True)


if __name__ == "__main__":
    main()

