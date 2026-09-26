#!/usr/bin/env python3
"""Launch a paper training configuration without embedding machine paths."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Protocol:
    name: str
    student: str
    teacher: str
    batch_size: int
    response_length: int
    max_batched_tokens: int
    tensor_parallel_size: int
    lambda_value: float
    allow_sign_flip: bool
    default_steps: int = 40


PROTOCOLS = {
    "strong-to-weak": Protocol(
        name="strong_to_weak",
        student="Qwen2.5-0.5B-Instruct",
        teacher="Qwen2.5-3B-Instruct",
        batch_size=256,
        response_length=4096,
        max_batched_tokens=8192,
        tensor_parallel_size=1,
        lambda_value=0.75,
        allow_sign_flip=False,
    ),
    "single-teacher": Protocol(
        name="single_teacher",
        student="Qwen3-4B",
        teacher="Qwen3-4B-Non-Thinking-RL-Math-Step500",
        batch_size=1024,
        response_length=16384,
        max_batched_tokens=32768,
        tensor_parallel_size=4,
        lambda_value=1.05,
        allow_sign_flip=True,
    ),
}


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--suite", choices=sorted(PROTOCOLS), required=True)
    result.add_argument("method", choices=("opd", "exopd", "fire"))
    result.add_argument("protection", choices=("vanilla", "sasr"))
    result.add_argument("gpu_ids")
    result.add_argument("steps", type=int, nargs="?")
    return result


def boolean(value: bool) -> str:
    return str(value).lower()


def build_command(args: argparse.Namespace) -> tuple[list[str], Path, dict[str, str]]:
    protocol = PROTOCOLS[args.suite]
    steps = protocol.default_steps if args.steps is None else args.steps
    if steps <= 0:
        raise ValueError("steps must be positive")
    gpu_ids = [item for item in args.gpu_ids.split(",") if item]
    required = 4 if args.method == "exopd" or args.suite == "single-teacher" else 2
    if len(gpu_ids) != required:
        raise ValueError(f"{args.suite}/{args.method} requires {required} GPUs")

    gopd_root = env_path("GOPD_ROOT", RELEASE_ROOT / "third_party/G-OPD")
    fire_root = env_path("FIRE_OPD_ROOT", RELEASE_ROOT / "third_party/FiRe-OPD")
    model_root = env_path("MODEL_ROOT", RELEASE_ROOT / "models")
    data_root = env_path("DATA_ROOT", RELEASE_ROOT / "data")
    output_root = env_path("OUTPUT_ROOT", RELEASE_ROOT / "outputs")
    upstream = gopd_root if args.method == "exopd" else fire_root
    python_bin = os.environ.get("PYTHON", "python")

    student = model_root / protocol.student
    teacher = model_root / protocol.teacher
    train_file = data_root / "DeepMath-103K/train_filtered_level6.parquet"
    val_files = [
        data_root / "AIME2024/test.parquet",
        data_root / "AIME2025/test.parquet",
    ]
    experiment = f"{protocol.name}-{args.method}-{args.protection}-seed42"
    output = output_root / protocol.name / args.method / args.protection / experiment
    output.mkdir(parents=True, exist_ok=True)

    val_literal = "[" + ",".join(f"'{path}'" for path in val_files) + "]"
    command = [
        python_bin,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=5.0",
        "algorithm.rollout_correction.rollout_rs=null",
        "algorithm.rollout_correction.bypass_mode=false",
        "actor_rollout_ref.rollout.calculate_log_probs=true",
        f"data.train_files={train_file}",
        f"data.val_files={val_literal}",
        f"data.train_batch_size={protocol.batch_size}",
        "data.max_prompt_length=2048",
        f"data.max_response_length={protocol.response_length}",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",
        "data.shuffle=true",
        "data.seed=42",
        "data.return_raw_chat=true",
        "+data.apply_chat_template_kwargs.enable_thinking=false",
        f"actor_rollout_ref.model.path={student}",
        f"+actor_rollout_ref.ref.model.path={teacher}",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=true",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={protocol.batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=true",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={protocol.max_batched_tokens}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.model.enable_activation_offload=true",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={protocol.tensor_parallel_size}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={os.environ.get('VLLM_GPU_MEMORY_UTILIZATION', '0.60')}",
        "actor_rollout_ref.rollout.n=1",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={protocol.max_batched_tokens}",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        "actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=true",
        "score_perturbation.enabled=false",
        "score_perturbation.mode=identity",
    ]

    if args.method == "exopd":
        command.extend(
            [
                f"+actor_rollout_ref.model.base_model_path={student}",
                f"+actor_rollout_ref.ref.model.base_model_path={student}",
                "actor_rollout_ref.actor.policy_loss.lambda_vals=1.25",
            ]
        )
    elif args.method == "fire":
        command.extend(
            [
                "actor_rollout_ref.actor.policy_loss.entropy_aware_distill=true",
                "actor_rollout_ref.actor.policy_loss.traj_skip_percentile=20.0",
                "actor_rollout_ref.actor.policy_loss.entropy_alpha=1.0",
                "actor_rollout_ref.actor.policy_loss.entropy_beta=1.0",
            ]
        )

    if args.protection == "vanilla":
        command.extend(
            [
                "teacher_defense.enabled=false",
                "teacher_defense.method=none",
            ]
        )
    else:
        command.extend(
            [
                "teacher_defense.enabled=true",
                "teacher_defense.method=sparse_asymmetric_projection",
                "teacher_defense.selection.active_ratio=0.20",
                "teacher_defense.selection.strategy=top_abs_advantage",
                "teacher_defense.weights.positive=0.5",
                "teacher_defense.weights.negative=1.0",
                f"teacher_defense.calibration.lambda_value={protocol.lambda_value}",
                "teacher_defense.calibration.result_path=null",
                "teacher_defense.calibration.reference_kl_mode=precomputed",
                f"teacher_defense.calibration.reference_mean_kl={os.environ.get('REFERENCE_MEAN_KL', 'null')}",
                f"teacher_defense.constraints.allow_sign_flip={boolean(protocol.allow_sign_flip)}",
                "teacher_defense.constraints.max_effective_strength=0.95",
                "teacher_defense.constraints.preserve_teacher_top1=true",
                "teacher_defense.constraints.top1_margin=1.0e-6",
                "teacher_defense.numerical.eps=1.0e-8",
                "teacher_defense.numerical.compute_dtype=float32",
                "teacher_defense.integration.use_defended_teacher_statistics_for_fire=true",
            ]
        )

    capture_path = os.environ.get("CALIBRATION_CAPTURE_PATH")
    if capture_path:
        command.extend(
            [
                "teacher_defense.capture.enabled=true",
                f"teacher_defense.capture.path={Path(capture_path).expanduser().resolve()}",
                "teacher_defense.capture.uniform_alpha=0.20",
            ]
        )

    command.extend(
        [
            "algorithm.use_kl_in_reward=false",
            "reward_model.reward_manager=naive",
            "trainer.critic_warmup=0",
            "trainer.val_before_train=false",
            'trainer.logger=["console"]',
            "trainer.log_val_generations=0",
            "trainer.project_name=sasr-reproduction",
            f"trainer.experiment_name={experiment}",
            f"trainer.n_gpus_per_node={len(gpu_ids)}",
            "trainer.nnodes=1",
            "trainer.save_freq=10",
            "trainer.resume_mode=auto",
            "trainer.max_actor_ckpt_to_keep=2",
            "trainer.max_critic_ckpt_to_keep=2",
            f"trainer.default_local_dir={output}",
            "trainer.test_freq=-1",
            "trainer.total_epochs=3",
            f"trainer.total_training_steps={steps}",
            "ray_kwargs.ray_init.num_cpus=24",
        ]
    )
    if os.environ.get("SASR_CONFIG_ONLY") == "1":
        command.extend(["--cfg", "job", "--resolve"])

    run_env = os.environ.copy()
    run_env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu_ids,
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": "offline",
            "USED_MODEL": "no_api",
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
            "TOKENIZERS_PARALLELISM": "false",
            "HYDRA_FULL_ERROR": "1",
            "RAY_TMPDIR": f"/tmp/sasr-{protocol.name}-{args.method}-{os.getpid()}",
        }
    )
    run_env.pop("RAY_ADDRESS", None)
    return command, upstream / "verl", run_env


def main() -> None:
    args = parser().parse_args()
    command, working_directory, run_env = build_command(args)
    if not working_directory.is_dir():
        raise FileNotFoundError(
            f"upstream repository is missing: {working_directory.parent}"
        )
    print("working directory:", working_directory)
    print("command:", shlex.join(command))
    subprocess.run(command, cwd=working_directory, env=run_env, check=True)


if __name__ == "__main__":
    main()
