"""Reading and writing JSON and JSON Lines files."""
from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path):
    """Content of a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(value, path: Path, *, indent: int = 1) -> None:
    """Write a value to a JSON file, creating its directory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSON Lines file."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def jsonl_line(row: dict) -> str:
    """One JSON Lines row with its line break."""
    return json.dumps(row, ensure_ascii=False) + "\n"
