"""FTS5 / BM25 wrapper.

`chunks_fts` was created in Phase 2 Step 1 as an external-content FTS5
table over `chunks.text`, with triggers keeping it in sync. SQLite's
built-in `bm25()` aux function gives us a per-row score (more negative
= more relevant; we sort ASC and treat row position as the rank).

The user-supplied query is sanitized into a quoted-OR FTS5 expression:
each whitespace-delimited token is wrapped in double quotes (escaping
internal quotes by doubling) and joined with `OR`. This:
  - neutralizes FTS5 operators (NEAR, AND, OR, +, -, *, ^, :)
  - prevents an injection from breaking the MATCH parse
  - keeps recall reasonable (a missing word doesn't disqualify a chunk)

BM25's IDF weighting naturally downweights common words like "how" /
"should" / "the" so the OR-joined query still ranks correctly.
"""
from __future__ import annotations

import re
import sqlite3

_TOKEN_SPLIT_RE = re.compile(r"\s+")


def _sanitize_fts5_query(query: str) -> str:
    if not query:
        return ""
    tokens = [t for t in _TOKEN_SPLIT_RE.split(query.strip()) if t]
    if not tokens:
        return ""
    quoted = ['"' + t.replace('"', '""') + '"' for t in tokens]
    return " OR ".join(quoted)


def search_bm25(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 50,
) -> list[tuple[int, float]]:
    """Return [(chunk_id, bm25_score)] ordered best-first. BM25 scores
    are negative; closer to 0 = more relevant. The fusion layer only
    cares about rank order, not raw value."""
    fts_query = _sanitize_fts5_query(query)
    if not fts_query:
        return []
    rows = conn.execute(
        "SELECT rowid, bm25(chunks_fts) AS score "
        "FROM chunks_fts WHERE chunks_fts MATCH ? "
        "ORDER BY score LIMIT ?",
        (fts_query, int(limit)),
    ).fetchall()
    return [(row["rowid"], float(row["score"])) for row in rows]
