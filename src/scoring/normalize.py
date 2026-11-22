"""Normalized clause forms of SQL queries, so that queries differing only in aliases, case or term order match."""
from __future__ import annotations

import collections
import json
import re

import sqlglot
from sqlglot import exp

CLAUSES = ("set_op", "distinct", "select", "from", "join", "on", "where", "group_by", "having", "order_by", "limit")


def _canonical_sql(node: exp.Expression) -> str:
    """Normalized SQL text of an expression."""
    return node.sql(normalize=True, pretty=False, comments=False)


def _string_key(sql: str) -> str:
    """Lowercased, whitespace-collapsed SQL without a trailing semicolon."""
    return re.sub(r"\s+", " ", (sql or "").strip().rstrip(";").strip().lower())


def _scope_tables(select: exp.Select) -> dict[str, str]:
    """Alias or name -> table name for the base tables of one SELECT."""
    out = {}
    relations = []
    if select.args.get("from_") is not None and select.args["from_"].this is not None:
        relations.append(select.args["from_"].this)
    relations.extend(join.this for join in select.args.get("joins") or [])
    for relation in relations:
        if isinstance(relation, exp.Table):
            out[relation.alias_or_name.lower()] = relation.name.lower()
            out[relation.name.lower()] = relation.name.lower()
    return out


def _canonical_clause(node: exp.Expression, aliases: dict[str, str]) -> str:
    """Normalized text of one clause with aliases resolved and commutative terms ordered."""
    expression = node.copy()

    def rewrite(current: exp.Expression) -> exp.Expression:
        """Resolve table aliases and order the operands of = and <>."""
        if isinstance(current, exp.Column) and current.table:
            table = aliases.get(current.table.lower())
            if table:
                current.set("table", exp.to_identifier(table))
        elif isinstance(current, exp.Table):
            current.set("alias", None)
        elif isinstance(current, (exp.EQ, exp.NEQ)):
            left, right = current.this, current.expression
            if _canonical_sql(left) > _canonical_sql(right):
                current.set("this", right)
                current.set("expression", left)
        return current

    expression = expression.transform(rewrite)

    def sort_boolean(current: exp.Expression) -> exp.Expression:
        """Sort the terms of AND and OR chains."""
        current = current.copy()
        for key, value in list(current.args.items()):
            if isinstance(value, exp.Expression):
                current.set(key, sort_boolean(value))
            elif isinstance(value, list):
                current.set(key, [sort_boolean(v) if isinstance(v, exp.Expression) else v for v in value])
        if isinstance(current, (exp.And, exp.Or)):
            operator = type(current)
            terms = []

            def collect(item: exp.Expression) -> None:
                """Gather the operands of a chain of the same Boolean operator."""
                if isinstance(item, operator):
                    collect(item.this)
                    collect(item.expression)
                else:
                    terms.append(item)

            collect(current)
            terms.sort(key=_canonical_sql)
            rebuilt = terms[0]
            for term in terms[1:]:
                rebuilt = operator(this=rebuilt, expression=term)
            return rebuilt
        return current

    return _canonical_sql(sort_boolean(expression))


def clause_signature(sql: str) -> dict[str, tuple[str, ...]]:
    """Clause type -> sorted tuple of normalized expressions, collected from the outer and nested queries.

    An absent clause has no entry, so a clause present in only one of two queries counts as a difference.
    """
    try:
        tree = sqlglot.parse_one(sql)
    except (sqlglot.errors.SqlglotError, ValueError):
        normalized = " ".join(str(sql).strip().lower().split())
        return {"raw": (normalized,)} if normalized else {}
    values: dict[str, list[str]] = {name: [] for name in CLAUSES}
    if isinstance(tree, (exp.Union, exp.Intersect, exp.Except)):
        values["set_op"].append(tree.key.upper())
    for query in tree.find_all(exp.Select):
        aliases = _scope_tables(query)
        if query.args.get("distinct") is not None:
            values["distinct"].append("DISTINCT")
        values["select"].extend(_canonical_clause(item.this if isinstance(item, exp.Alias) else item, aliases)
                                for item in query.expressions)
        from_clause = query.args.get("from_")
        relations = []
        if from_clause is not None and from_clause.this is not None:
            relations.append(_canonical_clause(from_clause.this, aliases))
        for join in query.args.get("joins") or []:
            relation = _canonical_clause(join.this, aliases)
            kind = " ".join(v.upper() for v in (join.args.get("side"), join.args.get("kind")) if v) or "INNER"
            if kind in {"INNER", "CROSS"}:
                relations.append(relation)
                values["join"].append(kind)
            else:
                values["join"].append(f"{kind}:{relation}")
            if join.args.get("on") is not None:
                values["on"].append(_canonical_clause(join.args["on"], aliases))
        values["from"].extend(relations)
        for name, key in (("where", "where"), ("group_by", "group"), ("having", "having"),
                          ("order_by", "order"), ("limit", "limit")):
            if query.args.get(key) is not None:
                values[name].append(_canonical_clause(query.args[key], aliases))
    signature = {name: tuple(sorted(items)) for name, items in values.items() if items}
    return signature or {"raw": (_canonical_sql(tree),)}


def _safe_signature(sql: str) -> dict[str, tuple[str, ...]]:
    """Clause signature that falls back to the normalized string on any error."""
    try:
        return clause_signature(sql)
    except Exception:
        return {"raw": (_string_key(sql),)}


def _parse(sql: str):
    """SQLite AST of a query, or None when it cannot be parsed."""
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None


def clause_forms(sqls: list[str]) -> list[dict[str, tuple[str, ...]]]:
    """Clause forms of the queries of one question (the prediction first, then the generated queries)."""
    trees = [_parse(sql) for sql in sqls]
    owner = collections.defaultdict(set)
    for tree in trees:
        if tree is None:
            continue
        for column in tree.find_all(exp.Column):
            select = column.find_ancestor(exp.Select)
            if column.table and select is not None:
                owner[column.name.lower()].add(_scope_tables(select).get(column.table.lower(), column.table.lower()))
    out = []
    for tree, sql in zip(trees, sqls):
        if tree is None:
            out.append(_safe_signature(sql))
            continue
        tree = tree.copy()
        for column in list(tree.find_all(exp.Column)):
            select = column.find_ancestor(exp.Select)
            if select is None:
                continue
            tables = _scope_tables(select)
            if column.table:
                resolved = tables.get(column.table.lower())
                if resolved:
                    column.set("table", exp.to_identifier(resolved))
                continue
            if column.name.lower() in {e.alias.lower() for e in select.expressions if isinstance(e, exp.Alias)}:
                continue
            in_scope = set(tables.values())
            candidates = owner.get(column.name.lower(), set()) & in_scope
            if len(in_scope) == 1:
                column.set("table", exp.to_identifier(next(iter(in_scope))))
            elif len(candidates) == 1:
                column.set("table", exp.to_identifier(next(iter(candidates))))
        try:
            out.append(clause_signature(tree.sql()))
        except Exception:
            out.append(_safe_signature(sql))
    return out


def form(signature: dict[str, tuple[str, ...]], clause: str) -> str:
    """Comparable text of one clause form; an absent clause compares as null."""
    return json.dumps(signature.get(clause), default=str)
