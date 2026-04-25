"""Unit tests for app.embeddings.

No network. `OpenAIEmbedder` accepts an injected `client` so tests can
hand in a `MagicMock` with an `AsyncMock`'d `embeddings.create`. The
store tests use an in-memory SQLite with sqlite-vec loaded — fast
(~milliseconds per test) and exercises the real vec0 virtual-table
path without hitting disk or the network.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.embeddings.providers import (
    _COST_PER_1M_TOKENS,
    EmbedderStats,
    OpenAIEmbedder,
)
from app.embeddings.store import get_missing_chunks, upsert_embeddings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embeddings_response(n_vectors: int, tokens: int) -> MagicMock:
    """Shape-match openai.types.CreateEmbeddingResponse: .data is a list
    of items with .embedding; .usage has .total_tokens."""
    resp = MagicMock()
    resp.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(n_vectors)]
    resp.usage = MagicMock(total_tokens=tokens)
    return resp


def _in_memory_db() -> sqlite3.Connection:
    """Fresh in-memory DB matching the Phase 3 schema: `chunks` table
    plus a `chunks_vec` vec0 virtual table. Mirrors what
    `app.ingest.db.open_mentor_db` builds."""
    import sqlite_vec  # lazy — matches production db.py

    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.executescript(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            source_url TEXT,
            source_type TEXT,
            date TEXT,
            speaker TEXT,
            text TEXT NOT NULL,
            content_hash TEXT UNIQUE,
            source_priority INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute(
        "CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[1536])"
    )
    return conn


def _seed_chunks(conn: sqlite3.Connection, rows: list[tuple[int, str, str]]) -> None:
    conn.executemany(
        "INSERT INTO chunks(id, source_url, source_type, date, text, content_hash) "
        "VALUES (?, 'url', 'twitter', '2024-01-01', ?, ?)",
        rows,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embedder_returns_1536_dim_vectors() -> None:
    """OpenAIEmbedder.embed() wraps the SDK call and surfaces
    1536-dim vectors + token usage back through EmbedderStats."""
    fake_client = MagicMock()
    fake_client.embeddings = MagicMock()
    fake_client.embeddings.create = AsyncMock(
        return_value=_fake_embeddings_response(n_vectors=3, tokens=42)
    )

    embedder = OpenAIEmbedder(api_key="sk-test", client=fake_client, batch_size=100)
    result = await embedder.embed(["a", "b", "c"])

    assert len(result.vectors) == 3
    assert all(len(v) == 1536 for v in result.vectors)
    assert result.tokens == 42

    # Stats accumulate.
    assert embedder.stats.batches == 1
    assert embedder.stats.vectors == 3
    assert embedder.stats.tokens_embedded == 42

    # One API call, not the batch_size. The client was called with the
    # exact text list we passed in.
    fake_client.embeddings.create.assert_awaited_once()
    call_kwargs = fake_client.embeddings.create.call_args.kwargs
    assert call_kwargs["model"] == "text-embedding-3-small"
    assert call_kwargs["input"] == ["a", "b", "c"]


def test_upsert_embeddings_and_get_missing_chunks_round_trip() -> None:
    """Seed two chunks, confirm both report missing; upsert vectors;
    confirm they're now in chunks_vec and not in the missing list."""
    conn = _in_memory_db()
    _seed_chunks(conn, [(1, "one", "h1"), (2, "two", "h2")])

    assert {cid for cid, _ in get_missing_chunks(conn)} == {1, 2}

    inserted = upsert_embeddings(
        conn, [(1, [0.1] * 1536), (2, [0.2] * 1536)]
    )
    assert inserted == 2

    count = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
    assert count == 2
    assert get_missing_chunks(conn) == []

    # Re-upserting the same ids must be idempotent — still 2 rows, no error.
    upsert_embeddings(conn, [(1, [0.9] * 1536), (2, [0.8] * 1536)])
    assert conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 2


def test_get_missing_chunks_skips_already_embedded() -> None:
    """Partial-embedding scenario: rows 1 and 3 get vectors; row 2 stays
    unembedded and must be the sole entry returned by get_missing_chunks."""
    conn = _in_memory_db()
    _seed_chunks(conn, [(1, "one", "h1"), (2, "two", "h2"), (3, "three", "h3")])

    upsert_embeddings(conn, [(1, [0.1] * 1536), (3, [0.3] * 1536)])

    missing = get_missing_chunks(conn)
    assert [cid for cid, _ in missing] == [2]
    assert missing[0][1] == "two"

    # --limit honored.
    _seed_chunks(conn, [(4, "four", "h4"), (5, "five", "h5")])
    limited = get_missing_chunks(conn, limit=2)
    assert [cid for cid, _ in limited] == [2, 4]


def test_cost_estimation_matches_openai_pricing() -> None:
    """text-embedding-3-small is billed at $0.02 per 1M input tokens
    (April 2026). Verify the stats property implements that formula
    and the module constant matches."""
    assert _COST_PER_1M_TOKENS == 0.02

    # 1M tokens → $0.02
    stats = EmbedderStats(tokens_embedded=1_000_000)
    assert abs(stats.estimated_cost_usd - 0.02) < 1e-9

    # 500k → $0.01
    stats = EmbedderStats(tokens_embedded=500_000)
    assert abs(stats.estimated_cost_usd - 0.01) < 1e-9

    # 314,845 (from a typical 5K-chunk corpus) → $0.0063
    stats = EmbedderStats(tokens_embedded=314_845)
    assert abs(stats.estimated_cost_usd - 0.0062969) < 1e-7
