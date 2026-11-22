"""Prompts for SQL generation and for the clause check of a prediction."""
from __future__ import annotations

import sqlglot
from sqlglot import exp

OPEN_WEIGHT_INSTRUCTION = """You are a helpful SQL expert assistant.
The assistant first thinks about how to write the SQL query by analyzing the question,
database schema and external knowledge, then provides the final SQL query.

Output format (strictly required):
1. Reasoning inside <think> </think> tags.
2. SQL query inside <answer> </answer> tags, with the SQL itself in a ```sql ``` code block.

Required structure:
<think>
...reasoning...
</think>
<answer> ```sql SELECT ... ``` </answer>"""

API_INSTRUCTION = """You are a SQL expert. Write a SQL query that answers the question using the database schema and external knowledge provided.
Return only the final SQL query in one complete ```sql ``` code block. Do not include explanations or intermediate reasoning."""

CLAUSE_CHECK_PROMPT = """Below is a Text-to-SQL question, its database schema, and a candidate SQL answer with its clauses listed separately.
Check each clause of the candidate against the question and the schema:
- FROM/JOIN: the tables are the ones the question needs, and every join uses the correct key columns.
- WHERE/HAVING: every condition stated in the question is present with the correct column, operator, and value, and no condition is added that the question does not state.
- SELECT: the returned columns, aggregations, and DISTINCT match what the question asks for.
- GROUP BY/ORDER BY/LIMIT: grouping, ordering, and row limits match the question.
Output "Yes" only if every clause is correct; otherwise output "No". Only output "Yes" or "No".

Question: {question}

Database schema:
{schema}

Candidate SQL:
{candidate_sql}

Clauses of the candidate SQL:
{clauses}"""


def _context(record: dict) -> str:
    """Database engine, schema, evidence and question of a record."""
    return ("Database Engine:\n" + str(record.get("dialect") or "SQLite")
            + "\n\nDatabase Schema:\n" + str(record["schema"])
            + "\n\nExternal Knowledge:\n" + str(record.get("evidence") or "")
            + "\n\nQuestion:\n" + record["question"])


def generation_messages(record: dict, *, api: bool = False) -> list[dict[str, str]]:
    """One user message: the instruction followed by the database engine, schema, evidence and question."""
    instruction = API_INSTRUCTION if api else OPEN_WEIGHT_INSTRUCTION
    return [{"role": "user", "content": instruction + "\n\n" + _context(record)}]


def clause_outline(sql: str) -> str:
    """Top-level clauses of a candidate, one per line; the whole query when it cannot be split."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        tree = None
    if not isinstance(tree, exp.Select):
        return f"- QUERY: {' '.join(str(sql).split())}"
    lines = [f"- SELECT: {', '.join(e.sql(dialect='sqlite') for e in tree.expressions)}"]
    if tree.args.get("distinct") is not None:
        lines[0] += " (DISTINCT)"
    if tree.args.get("from_") is not None:
        lines.append(f"- FROM: {tree.args['from_'].this.sql(dialect='sqlite')}")
    for join in tree.args.get("joins") or []:
        lines.append(f"- JOIN: {join.sql(dialect='sqlite')}")
    for name, key in (("WHERE", "where"), ("GROUP BY", "group"), ("HAVING", "having"),
                      ("ORDER BY", "order"), ("LIMIT", "limit")):
        node = tree.args.get(key)
        if node is not None:
            text = node.sql(dialect="sqlite")
            lines.append(f"- {name}: {text[len(name):].strip() if text.upper().startswith(name) else text}")
    return "\n".join(lines)


def clause_check_messages(record: dict, candidate_sql: str) -> list[dict[str, str]]:
    """One user message asking whether every clause of the candidate is correct."""
    content = CLAUSE_CHECK_PROMPT.format(question=record["question"], schema=record["schema"],
                                         candidate_sql=candidate_sql, clauses=clause_outline(candidate_sql))
    return [{"role": "user", "content": content}]
