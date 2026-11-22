"""Scoring step: execute each prediction and its generated queries and compute the confidence."""
from __future__ import annotations

from pathlib import Path

from src.scoring.confidence import AGREE_WEIGHT, NEUTRAL, score_prediction
from src.scoring.execution import EXECUTION_TIMEOUT, execute
from src.utils.ehrsql import rewriter
from src.utils.io import jsonl_line, load_json, read_jsonl


def score(data_path: Path, generations: Path, out: Path, *, agree_weight: float = AGREE_WEIGHT,
          timeout: float = EXECUTION_TIMEOUT, ehrsql_repo: Path | None = None) -> int:
    """Write the components and confidence of every generated prediction and return their number."""
    rewrite = rewriter(str(ehrsql_repo)) if ehrsql_repo else None
    records = {r["id"]: r for r in load_json(data_path)}
    rows = read_jsonl(generations)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("w", encoding="utf-8") as stream:
        for g in rows:
            database = records[g["id"]]["db_path"]
            row = {"id": g["id"], "pred_sql": g["pred_sql"]}
            if not g["pred_sql"].strip():
                row.update(confidence=NEUTRAL, imputed=True)
            else:
                cache = {}

                def execute_once(sql: str):
                    """Execute a query once per question, reusing the result of an identical query."""
                    if sql not in cache:
                        cache[sql] = execute(database, sql, timeout=timeout, rewrite=rewrite)
                    return cache[sql]

                parts = score_prediction(g["pred_sql"], g["samples"], g["verify"], execute_once(g["pred_sql"]),
                                         [execute_once(s) for s in g["samples"]])
                row.update(parts.as_dict(agree_weight), imputed=False)
            stream.write(jsonl_line(row))
    return len(rows)
