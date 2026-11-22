"""Final SQL extraction from model responses."""
from __future__ import annotations

import re

SQL_FENCE = re.compile(r"```sql\b\s*(.*?)\s*```", re.I | re.S)
OPEN_SQL_FENCE = re.compile(r"```sql\b\s*", re.I)


def _last_fence(response: str, left: int, right: int, *, allow_open: bool):
    """Offsets of the last ```sql block in a region; an unclosed block only when allowed."""
    region = response[left:right]
    blocks = list(SQL_FENCE.finditer(region))
    opened = list(OPEN_SQL_FENCE.finditer(region))
    if allow_open and opened and (not blocks or opened[-1].start() >= blocks[-1].end()):
        start, end = opened[-1].end(), len(region.rstrip())
        return (left + start, left + end) if region[start:end].strip() else None
    if blocks:
        start, end = blocks[-1].span(1)
        return left + start, left + end
    return None


def _reasoning_span(response: str):
    """The last SQL block of the last <answer>, else the last block after </think>; never inside thinking."""
    depth, opened, last_think_end, tagged = 0, None, None, False
    answers = []
    for tag in re.finditer(r"<(\/?)(think|answer)>", response, re.I):
        tagged = True
        closing, kind = bool(tag.group(1)), tag.group(2).lower()
        if kind == "think":
            if not closing and not depth and opened is not None:
                answers.append((opened, tag.start(), False))
                opened = None
            if closing and depth:
                depth -= 1
                if not depth:
                    last_think_end = tag.end()
            elif not closing:
                depth += 1
            continue
        if depth:
            continue
        if not closing:
            if opened is not None:
                answers.append((opened, tag.start(), False))
            opened = tag.end()
        elif opened is not None:
            answers.append((opened, tag.start(), True))
            opened = None
    if opened is not None and not depth:
        answers.append((opened, len(response), False))
    for left, right, closed in reversed(answers):
        span = _last_fence(response, left, right, allow_open=closed)
        if span:
            return span
    if depth:
        return None
    if not tagged:
        return _last_fence(response, 0, len(response), allow_open=False)
    if last_think_end is None:
        return None
    return _last_fence(response, last_think_end, len(response), allow_open=False)


def extract_reasoning_sql(response: str) -> str:
    """Open-weight models: the SQL inside <answer> after the <think> reasoning."""
    span = _reasoning_span(response)
    return response[slice(*span)].strip() if span else ""


def extract_final_sql(response: str) -> str:
    """API models: the last complete ```sql block outside any thinking."""
    depth, start, regions = 0, 0, []
    for tag in re.finditer(r"<(/?)think>", response, re.I):
        if not tag.group(1):
            if not depth:
                regions.append((start, tag.start()))
            depth += 1
        elif depth:
            depth -= 1
            if not depth:
                start = tag.end()
    if depth:
        return ""
    regions.append((start, len(response)))
    found = [(left + m.start(1), left + m.end(1)) for left, right in regions
             for m in SQL_FENCE.finditer(response[left:right])]
    if not found:
        return ""
    sql = response[slice(*found[-1])].strip()
    return "" if sql.lower().rstrip(";").strip() == "null" else sql
