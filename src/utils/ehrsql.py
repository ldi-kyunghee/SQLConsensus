"""EHRSQL: the official SQL rewrite applied to queries before execution."""
from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path

CURRENT_TIME = "2105-12-31 23:59:00"


@lru_cache(maxsize=None)
def _post_process(evaluator: str):
    """Load post_process_sql from the official evaluator without importing its dependencies."""
    tree = ast.parse(Path(evaluator).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "post_process_sql"]
    if len(nodes) != 1:
        raise ValueError(f"Expected one post_process_sql in {evaluator}")
    namespace = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), evaluator, "exec"), namespace)
    return namespace["post_process_sql"]


def rewriter(ehrsql_repo: str):
    """SQL rewrite of the official evaluator in a clone of the EHRSQL repository."""
    function = _post_process(str(Path(ehrsql_repo) / "evaluate.py"))
    return lambda sql: function(sql, current_time=CURRENT_TIME)
