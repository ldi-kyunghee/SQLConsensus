"""Experiment pipeline: prepare benchmark records and run generation, scoring and evaluation for one setting."""
from __future__ import annotations

from pathlib import Path

from src.data.benchmarks import build
from src.evaluation.report import evaluate
from src.generation.sampler import generate
from src.scoring.scorer import score
from src.utils.io import save_json


def prepare(benchmark: str, out: Path, **sources) -> int:
    """Build the records of one benchmark, save them as JSON, and return their number."""
    records = build(benchmark, **sources)
    save_json(records, out)
    return len(records)


def run(data_path: Path, backend: str, model: str, out_dir: Path, *, ehrsql_repo: Path | None = None,
        **generation_options) -> dict:
    """Generate, score, and evaluate one model on one benchmark; outputs go to out_dir."""
    out_dir = Path(out_dir)
    generate(data_path, backend, model, out_dir / "generations.jsonl", **generation_options)
    score(data_path, out_dir / "generations.jsonl", out_dir / "scored.jsonl", ehrsql_repo=ehrsql_repo)
    return evaluate(data_path, out_dir / "scored.jsonl", out_dir / "evaluation.json", ehrsql_repo=ehrsql_repo)
