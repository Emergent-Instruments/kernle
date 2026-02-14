"""Search implementation extracted from SQLiteStorage.

Contains the text-based search logic and supporting static helpers
(tokenization, token filtering, scoring). The top-level search()
coordinator and vector search remain on SQLiteStorage because they
depend on cloud credentials, embedding providers, and many self.*
methods that would require excessive parameter passing.

All functions receive dependencies explicitly to enable independent
testing and avoid circular imports.
"""

import logging
import sqlite3
from typing import Callable, Dict, List, Optional, Tuple

from .base import SearchResult
from .raw_entries import escape_like_pattern

logger = logging.getLogger(__name__)


def tokenize_query(query: str) -> List[str]:
    """Split a search query into meaningful tokens (words with 3+ chars)."""
    return [w for w in query.split() if len(w) >= 3]


def build_token_filter(tokens: List[str], columns: List[str]) -> Tuple[str, list]:
    """Build a tokenized OR filter for multiple columns.

    Returns (sql_fragment, params) where sql_fragment is a parenthesized
    OR expression matching any token in any column, and params is the
    list of LIKE pattern values. Tokens are escaped to prevent LIKE
    metacharacter injection.
    """
    clauses = []
    params: list = []
    for token in tokens:
        escaped = escape_like_pattern(token)
        pattern = f"%{escaped}%"
        for col in columns:
            clauses.append(f"{col} LIKE ? ESCAPE '\\'")
            params.append(pattern)
    sql = f"({' OR '.join(clauses)})"
    return sql, params


def token_match_score(text: str, tokens: List[str]) -> float:
    """Score a text by fraction of query tokens it contains (case-insensitive)."""
    if not tokens:
        return 1.0
    lower = text.lower()
    hits = sum(1 for t in tokens if t.lower() in lower)
    return hits / len(tokens)


def text_search(
    conn: sqlite3.Connection,
    stack_id: str,
    query: str,
    limit: int,
    types: List[str],
    row_converters: Dict[str, Callable],
    build_access_filter: Callable,
    requesting_entity: Optional[str] = None,
) -> List[SearchResult]:
    """Text-based search using tokenized LIKE matching.

    This is the fallback search strategy when vector search (sqlite-vec)
    is unavailable. Extracted from SQLiteStorage._text_search.

    Args:
        conn: Open sqlite3 connection.
        stack_id: The stack identifier for record isolation.
        query: Search query string.
        limit: Maximum results to return.
        types: Memory types to search (episode, note, belief, value, goal).
        row_converters: Dict mapping type name to row-to-record converter callable.
            Expected keys: "episode", "note", "belief", "value", "goal".
        build_access_filter: Callable(requesting_entity) -> (sql_fragment, params)
            for privacy filtering.
        requesting_entity: If provided, filter by access_grants.

    Returns:
        List of SearchResult sorted by token match score descending.
    """
    results = []
    tokens = tokenize_query(query)
    access_filter, access_params = build_access_filter(requesting_entity)

    # If no meaningful tokens, fall back to full-phrase match
    if not tokens:
        escaped_query = escape_like_pattern(query)
        search_term = f"%{escaped_query}%"
    else:
        search_term = None

    if "episode" in types:
        columns = ["objective", "outcome", "lessons"]
        if tokens:
            filt, filt_params = build_token_filter(tokens, columns)
        else:
            filt = "(objective LIKE ? ESCAPE '\\' OR outcome LIKE ? ESCAPE '\\' OR lessons LIKE ? ESCAPE '\\')"
            filt_params = [search_term, search_term, search_term]
        rows = conn.execute(
            f"""SELECT * FROM episodes
               WHERE stack_id = ? AND deleted = 0 AND COALESCE(strength, 1.0) > 0.0
               AND {filt}{access_filter}
               LIMIT ?""",
            [stack_id] + filt_params + access_params + [limit],
        ).fetchall()
        converter = row_converters["episode"]
        for row in rows:
            ep = converter(row)
            combined = f"{ep.objective or ''} {ep.outcome or ''} {ep.lessons or ''}"
            score = token_match_score(combined, tokens) if tokens else 1.0
            results.append(SearchResult(record=ep, record_type="episode", score=score))

    if "note" in types:
        columns = ["content"]
        if tokens:
            filt, filt_params = build_token_filter(tokens, columns)
        else:
            filt = "content LIKE ? ESCAPE '\\'"
            filt_params = [search_term]
        rows = conn.execute(
            f"""SELECT * FROM notes
               WHERE stack_id = ? AND deleted = 0 AND COALESCE(strength, 1.0) > 0.0
               AND {filt}{access_filter}
               LIMIT ?""",
            [stack_id] + filt_params + access_params + [limit],
        ).fetchall()
        converter = row_converters["note"]
        for row in rows:
            note = converter(row)
            score = token_match_score(note.content or "", tokens) if tokens else 1.0
            results.append(SearchResult(record=note, record_type="note", score=score))

    if "belief" in types:
        columns = ["statement"]
        if tokens:
            filt, filt_params = build_token_filter(tokens, columns)
        else:
            filt = "statement LIKE ? ESCAPE '\\'"
            filt_params = [search_term]
        rows = conn.execute(
            f"""SELECT * FROM beliefs
               WHERE stack_id = ? AND deleted = 0 AND COALESCE(strength, 1.0) > 0.0
               AND {filt}{access_filter}
               LIMIT ?""",
            [stack_id] + filt_params + access_params + [limit],
        ).fetchall()
        converter = row_converters["belief"]
        for row in rows:
            belief = converter(row)
            score = token_match_score(belief.statement or "", tokens) if tokens else 1.0
            results.append(SearchResult(record=belief, record_type="belief", score=score))

    if "value" in types:
        columns = ["name", "statement"]
        if tokens:
            filt, filt_params = build_token_filter(tokens, columns)
        else:
            filt = "(name LIKE ? ESCAPE '\\' OR statement LIKE ? ESCAPE '\\')"
            filt_params = [search_term, search_term]
        rows = conn.execute(
            f"""SELECT * FROM agent_values
               WHERE stack_id = ? AND deleted = 0 AND COALESCE(strength, 1.0) > 0.0
               AND {filt}{access_filter}
               LIMIT ?""",
            [stack_id] + filt_params + access_params + [limit],
        ).fetchall()
        converter = row_converters["value"]
        for row in rows:
            val = converter(row)
            combined = f"{val.name or ''} {val.statement or ''}"
            score = token_match_score(combined, tokens) if tokens else 1.0
            results.append(SearchResult(record=val, record_type="value", score=score))

    if "goal" in types:
        columns = ["title", "description"]
        if tokens:
            filt, filt_params = build_token_filter(tokens, columns)
        else:
            filt = "(title LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\')"
            filt_params = [search_term, search_term]
        rows = conn.execute(
            f"""SELECT * FROM goals
               WHERE stack_id = ? AND deleted = 0 AND COALESCE(strength, 1.0) > 0.0
               AND {filt}{access_filter}
               LIMIT ?""",
            [stack_id] + filt_params + access_params + [limit],
        ).fetchall()
        converter = row_converters["goal"]
        for row in rows:
            goal = converter(row)
            combined = f"{goal.title or ''} {goal.description or ''}"
            score = token_match_score(combined, tokens) if tokens else 1.0
            results.append(SearchResult(record=goal, record_type="goal", score=score))

    # Sort by token match score descending
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]
