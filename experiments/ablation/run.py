#!/usr/bin/env python3
"""Run the Step-5 Strong-to-Weak SASR component ablation."""

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


VARIANTS = {
    "full": ("top_abs_advantage", True, True),
    "positive-only": ("top_abs_advantage", True, False),
    "negative-only": ("top_abs_advantage", False, True),
    "random-selection": ("random", True, True),
}


def replace_override(command: list[str], key: str, value: str) -> None:
    prefix = f"{key}="
    for index, item in enumerate(command):
        if item.startswith(prefix):
            command[index] = f"{prefix}{value}"
            return
    command.append(f"{prefix}{value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("vanilla", *VARIANTS))
    parser.add_argument("gpu_ids")
    args = parser.parse_args()

    protection = "vanilla" if args.variant == "vanilla" else "sasr"
    base = argparse.Namespace(
        suite="strong-to-weak",
        method="opd",
        protection=protection,
        gpu_ids=args.gpu_ids,
        steps=5,
    )
    command, working_directory, run_env = build_command(base)
    output_root = Path(
        os.environ.get("OUTPUT_ROOT", RELEASE_ROOT / "outputs")
    ).expanduser().resolve()
    output = output_root / "ablation" / args.variant
    output.mkdir(parents=True, exist_ok=True)

    replace_override(command, "trainer.project_name", "sasr-step5-ablation")
    replace_override(command, "trainer.experiment_name", args.variant)
    replace_override(command, "trainer.default_local_dir", str(output / "checkpoints"))
    replace_override(command, "trainer.save_freq", "-1")
    replace_override(command, "trainer.test_freq", "5")
    replace_override(command, "actor_rollout_ref.rollout.val_kwargs.n", "2")

    additions = [
        "+actor_rollout_ref.rollout.seed=42",
        f"trainer.validation_data_dir={output / 'evaluation'}",
        f"+trainer.batch_fingerprint_dir={output / 'batch_fingerprints'}",
    ]
    if args.variant != "vanilla":
        strategy, positive, negative = VARIANTS[args.variant]
        replace_override(command, "teacher_defense.selection.strategy", strategy)
        additions.extend(
            [
                "teacher_defense.selection.random_seed=42",
                f"teacher_defense.branches.positive_enabled={str(positive).lower()}",
                f"teacher_defense.branches.negative_enabled={str(negative).lower()}",
            ]
        )

    flag_index = command.index("--cfg") if "--cfg" in command else len(command)
    command[flag_index:flag_index] = additions
    if not working_directory.is_dir():
        raise FileNotFoundError(
            f"FiRe-OPD checkout is missing: {working_directory.parent}"
        )
    print("working directory:", working_directory)
    print("command:", shlex.join(command))
    subprocess.run(command, cwd=working_directory, env=run_env, check=True)


if __name__ == "__main__":
    main()

