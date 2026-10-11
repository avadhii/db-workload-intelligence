"""
candidates.py — Index candidate generation (Phase 5).

Parses queries from pg_stat_statements or ad-hoc queries to extract
WHERE / JOIN / ORDER BY columns, validates them against the PostgreSQL schema,
checks for existing indexes, and generates candidate B-tree indexes.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Set

logger = logging.getLogger(__name__)


try:
    import sqlglot
    import sqlglot.expressions as exp
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False


@dataclass
class IndexCandidate:
    table_name: str
    columns: list[str]
    index_type: str = "btree"
    is_partial: bool = False
    where_clause: str | None = None
    reason: str = "Filter/Join predicate"
    index_name: str = ""

    def __post_init__(self):
        if not self.index_name:
            col_part = "_".join(self.columns)
            self.index_name = f"idx_{self.table_name}_{col_part}"[:63]

    @property
    def create_sql(self) -> str:
        col_str = ", ".join(self.columns)
        partial = f" WHERE {self.where_clause}" if self.where_clause else ""
        return (
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {self.index_name} "
            f"ON {self.table_name} ({col_str}){partial}"
        )

    @property
    def drop_sql(self) -> str:
        return f"DROP INDEX CONCURRENTLY IF EXISTS {self.index_name}"


def _normalize_query_for_parser(sql: str) -> str:
    """Normalize query so SQLGlot can parse generic PostgreSQL queries cleanly."""
    clean = sql.strip()
    return clean


def extract_candidates(
    query: str,
    top_n_composite: int = 2,
    existing_indexes: list[dict] | None = None,
    valid_schema: dict[str, list[str]] | None = None,
) -> list[IndexCandidate]:
    """
    Parse a SQL query and return validated candidate index suggestions.

    Extracts columns from:
    - WHERE clause equality and range predicates
    - JOIN ON conditions
    - ORDER BY clause

    Generates:
    - Single-column candidates
    - Composite candidates (WHERE columns + ORDER BY, up to top_n_composite)

    Validates:
    - Tables and columns exist (if valid_schema provided)
    - Deduplicates candidates
    - Skips indexes that already exist (if existing_indexes provided)
    """
    candidates: list[IndexCandidate] = []
    sql = _normalize_query_for_parser(query)

    # 1. Parse query
    parsed = None
    if HAS_SQLGLOT:
        try:
            parsed = sqlglot.parse_one(sql, dialect="postgres")
        except Exception:
            # Try replacing placeholders with arbitrary literals if parse fails
            try:
                subbed = re.sub(r"\$\d+", "'sample'", sql)
                parsed = sqlglot.parse_one(subbed, dialect="postgres")
            except Exception as exc:
                logger.debug("SQLGlot failed to parse query: %s (%s)", sql[:80], exc)

    where_cols: dict[str, list[str]] = {}  # table -> [columns]
    order_cols: dict[str, list[str]] = {}  # table -> [columns]
    join_cols: dict[str, list[str]] = {}   # table -> [columns]

    if parsed is not None:
        # Resolve aliases
        alias_map: dict[str, str] = {}
        from_tables: list[str] = []

        for tbl in parsed.find_all(exp.Table):
            real_name = tbl.name.lower() if tbl.name else ""
            alias = tbl.alias.lower() if tbl.alias else ""
            if real_name:
                from_tables.append(real_name)
            if alias and real_name:
                alias_map[alias] = real_name

        def resolve(t: str) -> str:
            t_low = t.lower()
            if t_low in alias_map:
                return alias_map[t_low]
            if t_low:
                return t_low
            return from_tables[0] if from_tables else "unknown"

        # WHERE predicates
        for condition in parsed.find_all(exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In):
            for col_node in condition.find_all(exp.Column):
                tbl = resolve(col_node.table or "")
                col = col_node.name.lower()
                if col:
                    where_cols.setdefault(tbl, [])
                    if col not in where_cols[tbl]:
                        where_cols[tbl].append(col)

        # JOIN ON
        for join in parsed.find_all(exp.Join):
            on = join.args.get("on")
            if on:
                for col_node in on.find_all(exp.Column):
                    tbl = resolve(col_node.table or "")
                    col = col_node.name.lower()
                    if col:
                        join_cols.setdefault(tbl, [])
                        if col not in join_cols[tbl]:
                            join_cols[tbl].append(col)

        # ORDER BY
        for order in parsed.find_all(exp.Order):
            for expr in order.find_all(exp.Column):
                tbl = resolve(expr.table or "")
                col = expr.name.lower()
                if col:
                    order_cols.setdefault(tbl, [])
                    if col not in order_cols[tbl]:
                        order_cols[tbl].append(col)

    else:
        # Regex fallback
        from_match = re.search(r"FROM\s+([a-zA-Z0-9_]+)", sql, re.IGNORECASE)
        tbl = from_match.group(1).lower() if from_match else "unknown"
        where_matches = re.findall(r"([a-zA-Z0-9_]+)\s*(?:=|>=|<=|>|<|IN|LIKE)", sql, re.IGNORECASE)
        for col in where_matches:
            col_low = col.lower()
            if col_low not in ("where", "and", "or", "select", "from"):
                where_cols.setdefault(tbl, [])
                if col_low not in where_cols[tbl]:
                    where_cols[tbl].append(col_low)

    # Combine where and join columns
    combined_predicates: dict[str, list[str]] = {}
    for tbl, cols in where_cols.items():
        combined_predicates.setdefault(tbl, []).extend(cols)
    for tbl, cols in join_cols.items():
        combined_predicates.setdefault(tbl, [])
        for c in cols:
            if c not in combined_predicates[tbl]:
                combined_predicates[tbl].append(c)

    # Build known existing index lookup: (table_name, tuple(columns_prefix))
    existing_set: Set[tuple[str, tuple[str, ...]]] = set()
    if existing_indexes:
        for idx in existing_indexes:
            t = idx.get("table_name", "").lower()
            cols = tuple(c.lower() for c in idx.get("columns", []))
            if t and cols:
                # Add all prefix sub-tuples since a leading column index satisfies single-column index requirements
                for i in range(1, len(cols) + 1):
                    existing_set.add((t, cols[:i]))

    # Helper for schema validation
    def is_valid_column(t: str, c: str) -> bool:
        if valid_schema is None:
            return True
        t_low = t.lower()
        if t_low not in valid_schema:
            return False
        return c.lower() in [col.lower() for col in valid_schema[t_low]]

    seen: Set[tuple[str, tuple[str, ...]]] = set()

    # 1. Single-column candidates from predicates
    for tbl, cols in combined_predicates.items():
        if tbl == "unknown":
            continue
        for col in cols:
            if not is_valid_column(tbl, col):
                continue
            key = (tbl, (col,))
            if key not in seen:
                seen.add(key)
                if key not in existing_set:
                    reason = "JOIN ON clause" if col in join_cols.get(tbl, []) else "WHERE filter predicate"
                    candidates.append(IndexCandidate(table_name=tbl, columns=[col], reason=reason))

    # 2. Composite candidates: (WHERE columns + ORDER BY columns)
    for tbl, o_cols in order_cols.items():
        if tbl == "unknown":
            continue
        p_cols = [c for c in combined_predicates.get(tbl, []) if is_valid_column(tbl, c)]
        valid_o_cols = [c for c in o_cols if is_valid_column(tbl, c)]
        if valid_o_cols:
            composite = p_cols[:top_n_composite] + valid_o_cols[:1]
            composite = list(dict.fromkeys(composite))
            if len(composite) > 1:
                key = (tbl, tuple(composite))
                if key not in seen:
                    seen.add(key)
                    if key not in existing_set:
                        candidates.append(
                            IndexCandidate(
                                table_name=tbl,
                                columns=composite,
                                reason=f"Composite for filter ({', '.join(p_cols[:top_n_composite])}) + sorting ({valid_o_cols[0]})",
                            )
                        )

    return candidates
