"""Teacher score perturbations for the official G-OPD training path.

The teacher still generates/scores with its original model.  This module only
transforms the token-level probabilities exposed to the student's OPD loss.
Every implemented transform is the sampled-token marginal of a normalized
full-vocabulary distribution; no isolated log-probability is edited ad hoc.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

import torch


_TINY = 1e-30
_CRITICAL_RE = re.compile(r"(?:\d|[+\-*/=<>%^]|\\(?:frac|boxed|sqrt)|####)")
_VALID_MODES = {"identity", "global_smoothing", "critical_smoothing", "random_noise"}


def infer_model_vocab_size(model_path: str | None, tokenizer: Any) -> int:
    """Prefer the model output vocabulary recorded in a local config.json."""

    if model_path:
        config_path = os.path.join(os.path.expanduser(str(model_path)), "config.json")
        try:
            with open(config_path, encoding="utf-8") as handle:
                value = int(json.load(handle)["vocab_size"])
            if value > 0:
                return value
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    value = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    if int(value) <= 0:
        raise ValueError("a positive tokenizer vocabulary size is required")
    return int(value)


@dataclass(frozen=True)
class ScorePerturbationConfig:
    """Configuration independent of Hydra/verl internals."""

    enabled: bool = False
    mode: str = "identity"
    strength: float = 0.2
    random_seed: int = 42
    critical_answer_window: int = 16
    keep_clean_scores: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ScorePerturbationConfig":
        if value is None:
            return cls()
        config = cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"unknown score perturbation mode {self.mode!r}; expected {sorted(_VALID_MODES)}")
        if not 0.0 <= self.strength < 1.0:
            raise ValueError(f"score perturbation strength must be in [0, 1), got {self.strength}")
        if self.critical_answer_window < 0:
            raise ValueError("critical_answer_window must be non-negative")


@dataclass
class PerturbationResult:
    teacher_logprobs: torch.Tensor
    active_mask: torch.Tensor
    metrics: dict[str, float]


class ScorePerturber:
    """Transform teacher scores while leaving teacher generation untouched."""

    def __init__(self, config: ScorePerturbationConfig, tokenizer: Any, vocab_size: int | None = None):
        config.validate()
        self.config = config
        self.tokenizer = tokenizer
        inferred_vocab = vocab_size or getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        if inferred_vocab is None or int(inferred_vocab) <= 0:
            raise ValueError("a positive tokenizer vocabulary size is required")
        self.vocab_size = int(inferred_vocab)
        self._critical_token_cache: dict[int, bool] = {}
        self._random_reference_cpu: torch.Tensor | None = None

    @property
    def is_identity(self) -> bool:
        return not self.config.enabled or self.config.mode == "identity" or self.config.strength == 0.0

    def _is_critical_token(self, token_id: int) -> bool:
        cached = self._critical_token_cache.get(token_id)
        if cached is not None:
            return cached
        text = self.tokenizer.decode([token_id], skip_special_tokens=False)
        value = bool(_CRITICAL_RE.search(text))
        self._critical_token_cache[token_id] = value
        return value

    def build_critical_mask(self, sampled_token_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """Mark numbers/operators plus the final valid answer window."""

        if sampled_token_ids.shape != response_mask.shape:
            raise ValueError("sampled_token_ids and response_mask must have identical shapes")
        result = torch.zeros_like(response_mask, dtype=torch.bool)
        valid_ids = sampled_token_ids[response_mask.bool()].detach().cpu().unique().tolist()
        critical_ids = [token_id for token_id in valid_ids if self._is_critical_token(int(token_id))]
        if critical_ids:
            critical_tensor = torch.tensor(critical_ids, device=sampled_token_ids.device, dtype=sampled_token_ids.dtype)
            result |= (sampled_token_ids.unsqueeze(-1) == critical_tensor).any(dim=-1)
        result &= response_mask.bool()

        window = self.config.critical_answer_window
        if window:
            for row in range(response_mask.shape[0]):
                positions = torch.nonzero(response_mask[row].bool(), as_tuple=False).flatten()
                if positions.numel():
                    result[row, positions[-window:]] = True
        return result

    def _random_reference(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return a repeat-query-stable normalized random vocabulary distribution."""

        if self._random_reference_cpu is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.config.random_seed)
            # Exponential samples normalized to the simplex (Dirichlet(1)).
            uniform = torch.rand(self.vocab_size, generator=generator, dtype=torch.float64).clamp_min(_TINY)
            weights = -uniform.log()
            self._random_reference_cpu = (weights / weights.sum()).to(torch.float32)
        return self._random_reference_cpu.to(device=device, dtype=dtype)

    def transform(
        self,
        *,
        sampled_token_ids: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> PerturbationResult:
        """Return scores consumed by OPD/G-OPD and diagnostics.

        For smoothing, the full distribution is
        ``q(v|c)=(1-alpha)p(v|c)+alpha/|V|``.  This is normalized and strictly
        preserves every teacher rank (hence top-1) whenever ``alpha < 1``.
        The random baseline replaces the uniform component with one fixed,
        normalized random vocabulary distribution.
        """

        if (
            sampled_token_ids.shape != teacher_logprobs.shape
            or response_mask.shape != teacher_logprobs.shape
        ):
            raise ValueError(
                "sampled ids, teacher logprobs, and response mask must have identical [batch, time] shapes"
            )

        valid = response_mask.bool()
        empty_metrics = {
            "score_perturbation/active_fraction": 0.0,
            "score_perturbation/mean_abs_logprob_shift": 0.0,
            "score_perturbation/mean_signed_logprob_shift": 0.0,
            "score_perturbation/top1_guaranteed": 1.0,
        }
        if self.is_identity:
            return PerturbationResult(
                teacher_logprobs=teacher_logprobs,
                active_mask=torch.zeros_like(valid),
                metrics=empty_metrics,
            )

        if self.config.mode == "critical_smoothing":
            active = self.build_critical_mask(sampled_token_ids, valid)
        else:
            active = valid

        alpha = torch.zeros_like(teacher_logprobs)
        alpha[active] = self.config.strength
        p_sampled = teacher_logprobs.exp()

        if self.config.mode == "random_noise":
            random_reference = self._random_reference(teacher_logprobs.device, teacher_logprobs.dtype)
            r_sampled = random_reference[sampled_token_ids.long()]
            top1_guaranteed = 0.0
        else:
            r_sampled = torch.full_like(p_sampled, 1.0 / float(self.vocab_size))
            top1_guaranteed = 1.0

        q_sampled = ((1.0 - alpha) * p_sampled + alpha * r_sampled).clamp_min(_TINY)
        perturbed = torch.where(active, q_sampled.log(), teacher_logprobs)
        shift = perturbed - teacher_logprobs
        valid_count = int(valid.sum().item())
        active_count = int(active.sum().item())
        metrics = {
            "score_perturbation/active_fraction": active_count / valid_count if valid_count else 0.0,
            "score_perturbation/mean_abs_logprob_shift": (
                float(shift[active].abs().mean().item()) if active_count else 0.0
            ),
            "score_perturbation/mean_signed_logprob_shift": float(shift[active].mean().item()) if active_count else 0.0,
            "score_perturbation/top1_guaranteed": top1_guaranteed,
            "score_perturbation/repeat_query_consistent": 1.0,
        }
        return PerturbationResult(teacher_logprobs=perturbed, active_mask=active, metrics=metrics)
