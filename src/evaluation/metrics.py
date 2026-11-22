"""AUROC and risk at fixed coverage, with tied confidences handled evenly."""
from __future__ import annotations

import math
from itertools import groupby


def auroc(confidences: list[float], correct: list[bool]) -> float:
    """Probability that a correct prediction receives a higher confidence than an incorrect one."""
    pairs = sorted(zip(confidences, correct), key=lambda p: p[0])
    positives = sum(correct)
    negatives = len(correct) - positives
    if not positives or not negatives:
        return float("nan")
    rank_sum, position = 0.0, 0
    for _, group in groupby(pairs, key=lambda p: p[0]):
        group = list(group)
        mean_rank = position + (len(group) + 1) / 2
        rank_sum += mean_rank * sum(y for _, y in group)
        position += len(group)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def risk_at_coverage(confidences: list[float], correct: list[bool], coverage: float) -> float:
    """Error rate among the most confident predictions at the given coverage, averaging over tied confidences."""
    n = len(confidences)
    k = max(1, math.ceil(coverage * n))
    ordered = sorted(zip(confidences, correct), key=lambda p: -p[0])
    errors, taken = 0.0, 0
    for _, group in groupby(ordered, key=lambda p: p[0]):
        group = list(group)
        wrong = sum(not y for _, y in group)
        take = min(len(group), k - taken)
        errors += wrong * take / len(group)
        taken += take
        if taken >= k:
            break
    return errors / k


def summary(confidences: list[float], correct: list[bool]) -> dict:
    """Accuracy, AUROC, R@10% and R@20% in percent."""
    return {"n": len(correct), "accuracy": 100 * sum(correct) / len(correct),
            "auroc": 100 * auroc(confidences, correct),
            "r@10": 100 * risk_at_coverage(confidences, correct, 0.10),
            "r@20": 100 * risk_at_coverage(confidences, correct, 0.20)}
