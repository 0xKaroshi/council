"""Hybrid retrieval: BM25 (FTS5) + dense (sqlite-vec) → RRF fusion.

Public API:
    Snippet            — dataclass returned per result
    retrieve(...)      — async, returns list[Snippet]

The `retrieve` coroutine kicks off two parallel branches via
`asyncio.gather`:
  - BM25 (sync sqlite call, ~5 ms)
  - embed query → vector search (OpenAI HTTP call ~300 ms then sync
    sqlite call ~5 ms)

The OpenAI embed dominates wall time; running BM25 alongside it hides
its cost. Both branches return ranked lists which RRF fuses into a
single ordering. Optional `recency_bias` then nudges newer chunks up
by up to 10% before truncation to top-k.

Nothing here registers as an MCP tool — that is Phase 3 Step 3.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from app.ingest.db import open_mentor_db

__all__ = ["Snippet", "retrieve"]


@dataclass
class Snippet:
    chunk_id: int
    mentor_slug: str
    text: str
    source_url: str
    source_type: str
    date: str                 # ISO 8601
    score: float              # RRF score (post recency / priority boost if applied)
    bm25_rank: int | None     # rank in BM25 top-50, None if absent
    vec_rank: int | None      # rank in vector top-50, None if absent
    source_priority: int = 1  # 1=tweet, 2=podcast, 3=canonical essay


_BM25_LIMIT = 50
_VEC_LIMIT = 50
_RRF_K = 60
_RECENCY_MAX_BOOST = 0.10
_RECENCY_WINDOW_DAYS = 365
_SOURCE_PRIORITY_BOOST_MULTIPLIER = 1.2
_CANONICAL_PRIORITY = 3


async def retrieve(
    mentor_slug: str,
    query: str,
    k: int = 8,
    recency_bias: bool = False,
    source_priority_boost: bool = False,
    query_embedding: list[float] | None = None,
    stats: dict[str, float] | None = None,
) -> list[Snippet]:
    """Run hybrid retrieval against `<mentor_slug>.db`.

    `recency_bias=True` nudges chunks within the last 365 days up by
    up to +10%. `source_priority_boost=True` multiplies the RRF score
    of canonical chunks (`source_priority == 3` — essays, newsletter
    posts) by 1.2 so they outrank tweet-shaped chunks at near-equal
    relevance. Both boosts are additive to the score before the
    final top-k truncation.

    `query_embedding`, if provided, skips the OpenAI embed call and
    uses the precomputed 1536-dim vector for the dense branch.
    BM25 still consumes the raw query text. Used by `council_retrieve`
    so the same embedding feeds three parallel mentor retrievals.

    `stats`, if provided, is populated with per-phase wall times:
      embed_ms, bm25_ms, vec_ms, total_ms
    The CLI uses this to surface timing without changing the return
    type. Tests can ignore it.
    """
    from app.retrieval.bm25 import search_bm25
    from app.retrieval.fusion import rrf_fuse
    from app.retrieval.query import embed_query, normalize_query
    from app.retrieval.vector import search_vector

    t_total = time.monotonic()
    normalized = normalize_query(query)
    if not normalized:
        if stats is not None:
            stats.update(embed_ms=0.0, bm25_ms=0.0, vec_ms=0.0, total_ms=0.0)
        return []

    # Read path: fall back to the bundled example DB if the user
    # hasn't ingested this mentor yet. Write paths (ingest, embed)
    # always materialize against user data — the bundled archive is
    # read-only by design.
    db = open_mentor_db(mentor_slug, fallback_to_bundled=True)
    try:
        async def _bm25_branch() -> tuple[list[tuple[int, float]], float]:
            t = time.monotonic()
            r = search_bm25(db, normalized, limit=_BM25_LIMIT)
            return r, (time.monotonic() - t) * 1000

        async def _embed_then_vec_branch() -> tuple[list[tuple[int, float]], float, float]:
            if query_embedding is not None:
                emb = query_embedding
                embed_ms = 0.0
            else:
                t_e = time.monotonic()
                emb = await embed_query(normalized)
                embed_ms = (time.monotonic() - t_e) * 1000
            t_v = time.monotonic()
            r = search_vector(db, emb, limit=_VEC_LIMIT)
            vec_ms = (time.monotonic() - t_v) * 1000
            return r, embed_ms, vec_ms

        (bm25_results, bm25_ms), (vec_results, embed_ms, vec_ms) = await asyncio.gather(
            _bm25_branch(),
            _embed_then_vec_branch(),
        )

        fused = rrf_fuse(bm25_results, vec_results, k_rrf=_RRF_K)

        bm25_ranks = {cid: i + 1 for i, (cid, _) in enumerate(bm25_results)}
        vec_ranks = {cid: i + 1 for i, (cid, _) in enumerate(vec_results)}

        if recency_bias and fused:
            fused = _apply_recency_bias(db, fused)
            fused.sort(key=lambda pair: pair[1], reverse=True)

        if source_priority_boost and fused:
            fused = _apply_source_priority_boost(db, fused)
            fused.sort(key=lambda pair: pair[1], reverse=True)

        top_k = fused[:k]
        snippets = _hydrate_snippets(db, mentor_slug, top_k, bm25_ranks, vec_ranks)

        total_ms = (time.monotonic() - t_total) * 1000
        if stats is not None:
            stats.update(
                embed_ms=embed_ms,
                bm25_ms=bm25_ms,
                vec_ms=vec_ms,
                total_ms=total_ms,
            )
        return snippets
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _hydrate_snippets(
    conn: sqlite3.Connection,
    mentor_slug: str,
    top_k: list[tuple[int, float]],
    bm25_ranks: dict[int, int],
    vec_ranks: dict[int, int],
) -> list[Snippet]:
    if not top_k:
        return []
    ids = [cid for cid, _ in top_k]
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id, text, source_url, source_type, date, source_priority "
        f"FROM chunks WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    by_id = {row["id"]: row for row in rows}

    out: list[Snippet] = []
    for cid, score in top_k:
        row = by_id.get(cid)
        if row is None:
            continue
        out.append(
            Snippet(
                chunk_id=cid,
                mentor_slug=mentor_slug,
                text=row["text"],
                source_url=row["source_url"],
                source_type=row["source_type"],
                date=row["date"],
                score=score,
                bm25_rank=bm25_ranks.get(cid),
                vec_rank=vec_ranks.get(cid),
                source_priority=row["source_priority"] if "source_priority" in row.keys() else 1,
            )
        )
    return out


def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _apply_recency_bias(
    conn: sqlite3.Connection,
    fused: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Boost RRF scores by up to +10% for chunks closer to the corpus's
    newest date. Linear falloff over a 365-day window; chunks older
    than the window get no boost."""
    newest_row = conn.execute("SELECT MAX(date) AS m FROM chunks").fetchone()
    if newest_row is None or not newest_row["m"]:
        return fused
    try:
        newest_dt = _parse_iso(newest_row["m"])
    except (TypeError, ValueError):
        return fused

    ids = [cid for cid, _ in fused]
    placeholders = ",".join("?" for _ in ids)
    date_rows = conn.execute(
        f"SELECT id, date FROM chunks WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    chunk_dates = {row["id"]: row["date"] for row in date_rows}

    boosted: list[tuple[int, float]] = []
    for cid, score in fused:
        date_str = chunk_dates.get(cid)
        if not date_str:
            boosted.append((cid, score))
            continue
        try:
            chunk_dt = _parse_iso(date_str)
        except (TypeError, ValueError):
            boosted.append((cid, score))
            continue
        days_old = (newest_dt - chunk_dt).days
        factor = max(0.0, 1.0 - days_old / _RECENCY_WINDOW_DAYS)
        boosted.append((cid, score * (1.0 + _RECENCY_MAX_BOOST * factor)))
    return boosted


def _apply_source_priority_boost(
    conn: sqlite3.Connection,
    fused: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Multiply the score of canonical-priority chunks
    (`source_priority == 3`) by `_SOURCE_PRIORITY_BOOST_MULTIPLIER`.
    Chunks with priority < 3 pass through unchanged. Used by the
    `search` and `convene` tools so canonical essays outrank
    tweet-shaped chunks at near-equal relevance."""
    if not fused:
        return fused
    ids = [cid for cid, _ in fused]
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id, source_priority FROM chunks WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    priorities = {row["id"]: row["source_priority"] for row in rows}
    out: list[tuple[int, float]] = []
    for cid, score in fused:
        if priorities.get(cid) == _CANONICAL_PRIORITY:
            out.append((cid, score * _SOURCE_PRIORITY_BOOST_MULTIPLIER))
        else:
            out.append((cid, score))
    return out
