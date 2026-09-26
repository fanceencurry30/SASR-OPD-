# Experiment-to-code map

This document maps each experiment family in the paper to its public launcher
and fixed protocol. It intentionally contains no experimental outcomes.

## Common method configuration

SASR changes only the sampled-token teacher supervision consumed by the OPD
method. Student rollout generation, the downstream OPD objective, and the
teacher's generated answer are otherwise unchanged.

The common selection ratio is `rho=0.20`, branch weights are
`w_pos=0.5` and `w_neg=1.0`, the selection rule is
`top_abs_advantage`, and top-1 preservation is enabled. The legacy Hydra key
used by the upstream integration is `teacher_defense`.

## Strong-to-Weak

Entry point: `experiments/strong_to_weak/train.sh`

| Item | Configuration |
|---|---|
| Student | `Qwen2.5-0.5B-Instruct` |
| Teacher | `Qwen2.5-3B-Instruct` |
| Optimizer updates | 40 |
| Global batch size | 256 |
| Learning rate | `1e-6` |
| Maximum response length | 4096 |
| SASR lambda | `0.75` |
| Branch weights | `(0.5, 1.0)` |
| Seed | 42 |

The first argument selects `opd`, `exopd`, or `fire`. The second argument is
`vanilla` or `sasr`, allowing each protected run to share the same launcher as
its control.

## Single-Teacher

Entry point: `experiments/single_teacher/train.sh`

| Item | Configuration |
|---|---|
| Student | `Qwen3-4B` |
| Teacher | `Qwen3-4B-Non-Thinking-RL-Math-Step500` |
| Optimizer updates | 40 |
| Global batch size | 1024 |
| Learning rate | `1e-6` |
| Maximum response length | 16384 |
| SASR lambda | `1.05` |
| Branch weights | `(0.5, 1.0)` |
| Seed | 42 |

The method and protection arguments have the same meaning as in the
Strong-to-Weak launcher.

## Adaptive recovery attacks

Entry point: `experiments/recovery/train.sh`

Recovery is evaluated against the Strong-to-Weak SASR setting and therefore
uses `lambda=0.75`, `w_pos=0.5`, `w_neg=1.0`, batch size 256, response length
4096, and 40 optimizer updates. Supported modes are:

| CLI mode | Recovery implementation |
|---|---|
| `full-gfr` | Global formula recovery with `sign_positive` candidate |
| `spfr` | Sparse position formula recovery with `positive_priority` ordering |
| `kl-gfr` | KL-budgeted global recovery using an offline calibration file |

The KL-GFR calibration product is generated locally under `calibration/` and
is ignored by Git. The calibration programs are included under
`experiments/recovery/`; they do not bundle a fitted calibration value.

One reproducible calibration sequence is:

```bash
CALIBRATION_CAPTURE_PATH="$CALIBRATION_ROOT/source_stats.pt" \
  bash experiments/strong_to_weak/train.sh opd sasr 0,1 1

PYTHONPATH="$GOPD_ROOT/verl" "$PYTHON" \
  experiments/recovery/prepare_kl_tr_gfr_cache.py \
  --source-stats "$CALIBRATION_ROOT/source_stats.pt" \
  --output-cache "$CALIBRATION_ROOT/attacker_cache.pt" \
  --output-manifest "$CALIBRATION_ROOT/attacker_cache_manifest.json"

PYTHONPATH="$GOPD_ROOT/verl" "$PYTHON" \
  experiments/recovery/calibrate_kl_tr_gfr_rho.py \
  --attacker-cache "$CALIBRATION_ROOT/attacker_cache.pt" \
  --gfr-variant sign_positive \
  --output-dir "$CALIBRATION_ROOT"
```

## Component ablation

Entry point: `experiments/ablation/run.sh`

The paper's Step-5 Strong-to-Weak component study uses the same models and
SASR parameters as the Strong-to-Weak setting. The four protected variants
are `full`, `positive-only`, `negative-only`, and `random-selection`.
`vanilla` is available as the matched unprotected control. Apply the
`ablation` overlay to FiRe-OPD before using this launcher.

## Evaluation

Entry point: `experiments/evaluate_avg8.sh`

The script evaluates a merged model with eight sampled responses per problem,
temperature 1.0, top-p 1.0, and seed 42 on:

- AIME 2024
- AIME 2025
- MATH-500
- AMC 2023
- OlympiadBench
- Minerva Math
- HMMT February 2025

Pass `4096` as the maximum generation length for Strong-to-Weak and recovery
runs, and `16384` for Single-Teacher runs. Generated JSONL files and logs are
written under ignored runtime directories.

## External assets

The repository contains empty placeholders only. Expected relative locations
are:

```text
models/Qwen2.5-0.5B-Instruct/
models/Qwen2.5-3B-Instruct/
models/Qwen3-4B/
models/Qwen3-4B-Non-Thinking-RL-Math-Step500/
data/DeepMath-103K/train_filtered_level6.parquet
data/AIME2024/test.parquet
data/AIME2025/test.parquet
data/eval/amc2023/test.jsonl
data/eval/hmmt25_feb/test.jsonl
data/eval/math500/test.jsonl
data/eval/minervamath/test.jsonl
data/eval/olympiadbench/test.jsonl
```

Equivalent absolute paths can be supplied with the environment variables
shown in `.env.example`.
