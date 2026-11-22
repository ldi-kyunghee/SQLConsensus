"""Confidence of a prediction from execution agreement, clause consistency and clause-level verification."""
from __future__ import annotations

import collections
import math
from dataclasses import asdict, dataclass

from src.scoring.normalize import clause_forms, form

# Default weight of the execution agreement.
AGREE_WEIGHT = 0.60
NEUTRAL = 0.5


@dataclass(frozen=True)
class Components:
    """The three signals of one prediction and the confidence derived from them."""
    agree: float
    consist: float
    verify: float

    @property
    def struct(self) -> float:
        """Structural score, the mean of Consist and Verify."""
        return (self.consist + self.verify) / 2

    def confidence(self, agree_weight: float = AGREE_WEIGHT) -> float:
        """Weighted combination of the execution agreement and the structural score."""
        return agree_weight * self.agree + (1 - agree_weight) * self.struct

    def as_dict(self, agree_weight: float = AGREE_WEIGHT) -> dict:
        """Signals, structural score and confidence as a JSON-ready dict."""
        return {**asdict(self), "struct": self.struct, "confidence": self.confidence(agree_weight)}


def agreement_mask(prediction_result, sample_results) -> list[bool]:
    """Which generated queries return the result of the prediction; none when the prediction returns no rows."""
    return [bool(prediction_result) and result == prediction_result for result in sample_results]


def clause_consistency(sample_forms, mask) -> float:
    """How consistently the agreeing queries write each clause, judged by the least consistent clause."""
    clauses = set().union(*sample_forms) if sample_forms else set()
    if not clauses:
        return 1.0
    n = len(sample_forms)
    if n < 2:
        raise ValueError("Clause consistency needs at least two generated queries")
    worst = 0.0
    for clause in clauses:
        counts = collections.Counter(form(s, clause) if agrees else f"outside-{i}"
                                     for i, (s, agrees) in enumerate(zip(sample_forms, mask)))
        worst = max(worst, -sum(c / n * math.log(c / n) for c in counts.values()) / math.log(n))
    return 1.0 - worst


def score_prediction(prediction: str, samples: list[str], verify: float | None,
                     prediction_result, sample_results) -> Components:
    """Components of one prediction from its generated queries, their execution results and Verify."""
    if len(samples) != len(sample_results):
        raise ValueError("One execution result is needed for each generated query")
    mask = agreement_mask(prediction_result, sample_results)
    forms = clause_forms([prediction] + list(samples))
    return Components(agree=sum(mask) / len(samples), consist=clause_consistency(forms[1:], mask),
                      verify=NEUTRAL if verify is None else float(verify))
