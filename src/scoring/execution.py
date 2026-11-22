"""Read-only execution of a query on its SQLite database, with a time limit."""
from __future__ import annotations

import sqlite3
import time
from typing import Callable

EXECUTION_TIMEOUT = 5.0


def execute(database: str, sql: str, *, timeout: float = EXECUTION_TIMEOUT,
            rewrite: Callable[[str], str] | None = None) -> list[str] | None:
    """Result as a sorted set of rows, or None when the query is empty, fails, or exceeds the time limit.

    Results are compared as sets of rows, without regard to order or duplicates.
    """
    if not sql or not sql.strip():
        return None
    if rewrite is not None:
        try:
            sql = rewrite(sql)
        except Exception:
            return None
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    deadline = time.monotonic() + timeout
    connection.set_progress_handler(lambda: time.monotonic() > deadline, 10000)
    try:
        return sorted({repr(row) for row in connection.execute(sql).fetchall()})
    except Exception:
        return None
    finally:
        connection.close()
