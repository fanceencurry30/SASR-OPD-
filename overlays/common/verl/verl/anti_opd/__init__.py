"""Output-preserving score-channel defenses for on-policy distillation."""

from .score_perturbation import (
    PerturbationResult,
    ScorePerturbationConfig,
    ScorePerturber,
    infer_model_vocab_size,
)

__all__ = ["PerturbationResult", "ScorePerturbationConfig", "ScorePerturber", "infer_model_vocab_size"]
