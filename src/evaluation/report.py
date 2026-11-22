"""Evaluation of scored predictions: execution correctness and confidence metrics."""
from __future__ import annotations

from pathlib import Path

from src.evaluation.correctness import is_correct
from src.evaluation.metrics import summary
from src.utils.ehrsql import rewriter
from src.utils.io import load_json, read_jsonl, save_json


def evaluate(data_path: Path, scored: Path, out: Path | None = None, *, ehrsql_repo: Path | None = None) -> dict:
    """Judge each prediction against the reference SQL and return accuracy, AUROC, R@10% and R@20%."""
    rewrite = rewriter(str(ehrsql_repo)) if ehrsql_repo else None
    records = {r["id"]: r for r in load_json(data_path)}
    rows = read_jsonl(scored)
    correct = [is_correct(records[r["id"]]["db_path"], records[r["id"]]["gold_sql"], r["pred_sql"], rewrite=rewrite)
               for r in rows]
    metrics = summary([r["confidence"] for r in rows], correct)
    if out:
        save_json({"metrics": metrics, "correct": dict(zip([r["id"] for r in rows], correct))}, out)
    return metrics
