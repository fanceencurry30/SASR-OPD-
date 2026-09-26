# SASR reproducibility code

This repository contains the code needed to reproduce the experiments in
**Same Answer, Different Lesson: Defending On-Policy Distillation via Signal
Decoupling**. It is a source-only release: datasets, model weights,
checkpoints, logs, generated predictions, and reported result files are not
included.

## What is included

- The SASR sampled-token teacher-distribution transformation.
- Adaptive recovery implementations for Full GFR, SPFR, and KL-GFR.
- Minimal overlays for the public G-OPD and FiRe-OPD codebases.
- Reproduction launchers for Strong-to-Weak, Single-Teacher, recovery, and
  component-ablation experiments.
- Avg@8 evaluation code for the seven benchmarks used in the paper.
- Unit tests and a release-content validator.

The detailed paper-to-code map is in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Repository layout

```text
experiments/        Reproduction entry points
overlays/common/    SASR and recovery algorithm code
overlays/g_opd/     G-OPD integration files
overlays/fire_opd/  FiRe-OPD integration files
overlays/ablation/  Positive/negative/random component gates
scripts/            Overlay installer and release validation
tests/              Algorithm-level tests
data/               Empty dataset placeholder
models/             Empty model placeholder
checkpoints/        Empty checkpoint placeholder
outputs/            Empty generated-output placeholder
logs/               Empty log placeholder
calibration/        Empty locally generated calibration placeholder
third_party/        Empty upstream-repository placeholder
```

## Environment

The recorded training environment used Python 3.10, PyTorch 2.6.0,
Transformers 4.51.1, Ray 2.43.0, vLLM 0.8.5.post1, Hydra 1.3.4, and
FlashAttention 2.7.4.post1. CUDA-capable Linux is required for the training
launchers. The upstream OPD repositories provide their own installation
instructions and dependencies.

## Setup

1. Clone the public upstream repositories into `third_party/`:

   ```bash
   git clone https://github.com/RUCBM/G-OPD.git third_party/G-OPD
   git clone https://github.com/YuYingLi0/FiRe-OPD.git third_party/FiRe-OPD
   ```

2. Apply the corresponding source overlays:

   ```bash
   bash scripts/apply_overlay.sh g_opd third_party/G-OPD
   bash scripts/apply_overlay.sh fire_opd third_party/FiRe-OPD
   ```

3. Put locally obtained assets under `models/` and `data/`, or export custom
   absolute roots. Nothing in those directories should be committed.

   ```bash
   export PYTHON=/path/to/python
   export GOPD_ROOT=$PWD/third_party/G-OPD
   export FIRE_OPD_ROOT=$PWD/third_party/FiRe-OPD
   export MODEL_ROOT=$PWD/models
   export DATA_ROOT=$PWD/data
   export OUTPUT_ROOT=$PWD/outputs
   export CHECKPOINT_ROOT=$PWD/checkpoints
   export CALIBRATION_ROOT=$PWD/calibration
   ```

4. Resolve a Hydra configuration without starting GPU work:

   ```bash
   SASR_CONFIG_ONLY=1 bash experiments/strong_to_weak/train.sh opd sasr 0,1 40
   SASR_CONFIG_ONLY=1 bash experiments/single_teacher/train.sh opd sasr 0,1,2,3 40
   ```

## Main commands

Strong-to-Weak training:

```bash
bash experiments/strong_to_weak/train.sh opd sasr 0,1 40
bash experiments/strong_to_weak/train.sh exopd sasr 0,1,2,3 40
bash experiments/strong_to_weak/train.sh fire sasr 0,1 40
```

Single-Teacher training:

```bash
bash experiments/single_teacher/train.sh opd sasr 0,1,2,3 40
bash experiments/single_teacher/train.sh exopd sasr 0,1,2,3 40
bash experiments/single_teacher/train.sh fire sasr 0,1,2,3 40
```

Adaptive recovery attacks:

```bash
bash experiments/recovery/train.sh full-gfr 0,1 40
bash experiments/recovery/train.sh spfr 0,1 40
bash experiments/recovery/train.sh kl-gfr 0,1 40
```

Five-step component ablation:

```bash
bash scripts/apply_overlay.sh ablation third_party/FiRe-OPD
bash experiments/ablation/run.sh full 0,1
bash experiments/ablation/run.sh positive-only 0,1
bash experiments/ablation/run.sh negative-only 0,1
bash experiments/ablation/run.sh random-selection 0,1
```

Avg@8 evaluation:

```bash
bash experiments/evaluate_avg8.sh /path/to/merged/model 0,1,2,3 4096
```

The launchers write only to ignored runtime directories. Set the second
training argument to `vanilla` to reproduce the unprotected counterpart with
the same protocol.

## Source-only release policy

Before committing, run:

```bash
python scripts/validate_release.py
pytest -q
```

The validator rejects large files, known weight/checkpoint formats,
machine-specific paths, credentials, and result artifacts. Runtime directories
remain visible through `.gitkeep` files only.

## License

The release is provided under Apache-2.0. Files adapted from upstream VERL,
G-OPD, or FiRe-OPD retain their original copyright notices.

