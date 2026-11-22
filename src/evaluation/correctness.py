"""Execution correctness: whether a prediction returns the same rows as the reference SQL."""
from __future__ import annotations

import sqlite3
from typing import Callable

from func_timeout import FunctionTimedOut, func_timeout

TIMEOUT = 10


def _compare(database: str, gold_sql: str, pred_sql: str) -> bool:
    """Run both queries in a rolled-back transaction and compare their row sets."""
    connection = sqlite3.connect(database)
    try:
        connection.execute("BEGIN TRANSACTION;")
        predicted = connection.execute(pred_sql).fetchall()
        gold = connection.execute(gold_sql).fetchall()
        return set(predicted) == set(gold)
    except Exception:
        return False
    finally:
        try:
            connection.rollback()
        finally:
            connection.close()


def is_correct(database: str, gold_sql: str | None, pred_sql: str, *, timeout: float = TIMEOUT,
               rewrite: Callable[[str], str] | None = None) -> bool:
    """Execution correctness of a prediction; False without a reference SQL, on error, or on timeout."""
    if not gold_sql or gold_sql.strip().lower() in {"null", "nan"} or not pred_sql.strip():
        return False
    if rewrite is not None:
        gold_sql, pred_sql = rewrite(gold_sql), rewrite(pred_sql)
    try:
        return bool(func_timeout(timeout, _compare, args=(database, gold_sql, pred_sql)))
    except (FunctionTimedOut, Exception):
        return False
