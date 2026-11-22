"""Evaluation: execution correctness against the reference SQL, AUROC, and risk at fixed coverage."""

from src.evaluation.correctness import is_correct
from src.evaluation.metrics import auroc, risk_at_coverage, summary
from src.evaluation.report import evaluate

__all__ = ["auroc", "evaluate", "is_correct", "risk_at_coverage", "summary"]
