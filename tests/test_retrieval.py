"""Unit + integration tests for app.retrieval.

Strategy:
- BM25 sanitization: pure-function checks, no DB.
- RRF fusion: pure-function checks (math + dedup) parameterized.
- Recency bias: builds a tiny in-memory DB so we can verify the
  multiplier matches the spec (`1 + 0.1 * factor`, factor = max(0,
  1 - days/365)).
- End-to-end: real per-mentor SQLite on a tmp_path with sqlite-vec
  loaded, 5 chunks + 5 distinct embeddings. Mock `embed_query` so
  no OpenAI call is made; assert top-1 is the chunk whose embedding
  matches the mocked query vector.

No real network. The mentor registry already has "testmentor", which is
the slug used by both the production code and the integration test.
"""
from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.retrieval.bm25 import _sanitize_fts5_query
from app.retrieval.fusion import rrf_fuse

# ---------------------------------------------------------------------------
# Test 1 — BM25 sanitization
# ---------------------------------------------------------------------------

def test_bm25_sanitize_strips_dangerous_chars_and_handles_empty() -> None:
    """Operators, quotes, and structural FTS5 chars must be neutralized
    by quoting each token; whitespace-only input yields an empty string
    so search_bm25 short-circuits without hitting MATCH parse errors."""
    # Whitespace-only → empty.
    assert _sanitize_fts5_query("") == ""
    assert _sanitize_fts5_query("   \t  \n ") == ""

    # Plain query → quoted-OR.
    assert _sanitize_fts5_query("price newsletter") == '"price" OR "newsletter"'

    # Embedded quotes are escaped via doubling so they stay inside the
    # quoted token rather than terminating it.
    out = _sanitize_fts5_query('foo"bar baz')
    assert out == '"foo""bar" OR "baz"'

    # FTS5 operators / specials end up as literal tokens, not parser
    # directives. The "AND" token below is now just the string "AND",
    # not the AND operator.
    out = _sanitize_fts5_query("price AND community")
    assert out == '"price" OR "AND" OR "community"'

    # Operator chars (* + - ^ : NEAR) are quoted, so MATCH won't see them.
    out = _sanitize_fts5_query("a* +b -c NEAR")
    assert '"a*"' in out
    assert '"+b"' in out
    assert '"-c"' in out
    assert '"NEAR"' in out


# ---------------------------------------------------------------------------
# Test 2 — RRF fusion (math + dedup, parameterized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case_name, bm25, vec, expected_first, expected_present",
    [
        # Distinct lists, no overlap. Highest score wins; with k=60
        # the top of each list scores 1/61. After fusion, doc 1 and
        # doc 100 tie at 1/61 — sorted is stable, dict order preserved.
        (
            "distinct_top_ranks_tie",
            [(1, -10.0), (2, -8.0), (3, -7.0)],
            [(100, 0.1), (200, 0.2), (300, 0.3)],
            1,                 # 1 inserted into scores dict first
            {1, 2, 3, 100, 200, 300},
        ),
        # Overlapping doc shows up in both lists → should beat anything
        # appearing in only one. Doc 7 scores 1/(60+1) + 1/(60+2) ≈
        # 0.03268; lone-list docs only get one half ≈ 0.01639.
        (
            "overlap_wins_over_singletons",
            [(7, -10.0), (5, -8.0)],
            [(7, 0.1), (9, 0.2)],
            7,
            {7, 5, 9},
        ),
    ],
)
def test_rrf_fuse_math_and_dedup(case_name, bm25, vec, expected_first, expected_present):
    fused = rrf_fuse(bm25, vec, k_rrf=60)
    fused_ids = [cid for cid, _ in fused]

    assert set(fused_ids) == expected_present, f"{case_name}: id-set mismatch"
    assert fused[0][0] == expected_first, f"{case_name}: top-1 mismatch"

    # Specifically verify the dedup case: doc 7 beats both singletons.
    if case_name == "overlap_wins_over_singletons":
        score_7 = dict(fused)[7]
        score_5 = dict(fused)[5]
        score_9 = dict(fused)[9]
        assert score_7 > score_5
        assert score_7 > score_9
        # Exact RRF math. Doc 7 sits at rank 1 in BOTH lists → 2 * 1/61.
        assert abs(score_7 - (2 / 61)) < 1e-12
        assert abs(score_5 - 1 / 62) < 1e-12
        assert abs(score_9 - 1 / 62) < 1e-12


# ---------------------------------------------------------------------------
# Test 3 — recency bias
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recency_bias_applies_correct_multiplier(tmp_path: Path, monkeypatch) -> None:
    """Newest chunk → +10% boost. A chunk one full window (365 days)
    older → +0% boost. A chunk in between → linear interpolation.

    Expected multipliers are computed from the real date arithmetic
    (not hand-counted) so a calendar mistake can't land as a test
    assertion mistake."""
    from datetime import datetime

    from app.config import settings
    from app.embeddings.store import upsert_embeddings
    from app.ingest.db import open_mentor_db

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    # Dates chosen so the span crosses typical boundaries; exact day
    # counts computed at test time rather than hand-written.
    newest_iso = "2025-01-01T00:00:00+00:00"
    mid_iso = "2024-07-01T00:00:00+00:00"
    old_iso = "2024-01-01T00:00:00+00:00"  # >365 days before newest → clamped to 0

    newest_dt = datetime.fromisoformat(newest_iso.replace("Z", "+00:00"))
    mid_days = (newest_dt - datetime.fromisoformat(mid_iso.replace("Z", "+00:00"))).days
    expected_mid = 1.0 + 0.1 * max(0.0, 1.0 - mid_days / 365)

    db = open_mentor_db("testmentor")
    db.executemany(
        "INSERT INTO chunks(id, source_url, source_type, date, text, content_hash) "
        "VALUES (?, 'u', 'twitter', ?, ?, ?)",
        [
            (1, old_iso, "old chunk", "h1"),
            (2, mid_iso, "mid chunk", "h2"),
            (3, newest_iso, "newest chunk", "h3"),
        ],
    )
    upsert_embeddings(
        db,
        [(1, [0.1] * 1536), (2, [0.2] * 1536), (3, [0.3] * 1536)],
    )
    db.close()

    from app.retrieval import _apply_recency_bias

    db = open_mentor_db("testmentor")
    fused = [(1, 1.0), (2, 1.0), (3, 1.0)]
    boosted = dict(_apply_recency_bias(db, fused))
    db.close()

    # Newest chunk (3): factor = 1.0 → multiplier 1.10.
    assert abs(boosted[3] - 1.10) < 1e-9
    # Mid chunk (2): linear interpolation based on actual day count.
    assert abs(boosted[2] - expected_mid) < 1e-9
    # Old chunk (1): >365 days old → factor clamped to 0 → multiplier 1.0.
    assert abs(boosted[1] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 4 — end-to-end retrieve()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_end_to_end_with_mocked_embedding(tmp_path: Path, monkeypatch) -> None:
    """Build a tiny testmentor.db on a tmp_path, mock embed_query so no
    network is touched, run retrieve() and assert the right chunk
    bubbles to the top with the correct Snippet shape."""
    from app.config import settings
    from app.embeddings.store import upsert_embeddings
    from app.ingest.db import open_mentor_db

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    # Seed testmentor.db with five chunks + five distinct embeddings.
    rng = random.Random(42)
    vectors = [[rng.uniform(-1.0, 1.0) for _ in range(1536)] for _ in range(5)]

    db = open_mentor_db("testmentor")
    db.executemany(
        "INSERT INTO chunks(id, source_url, source_type, date, text, content_hash) "
        "VALUES (?, ?, 'twitter', ?, ?, ?)",
        [
            (1, "https://x.com/a/1", "2024-01-01", "how to price a paid newsletter", "h1"),
            (2, "https://x.com/a/2", "2024-02-01", "writing a hook that stops the scroll", "h2"),
            (3, "https://x.com/a/3", "2024-03-01", "burnout and recovery for solopreneurs", "h3"),
            (4, "https://x.com/a/4", "2024-04-01", "LinkedIn posting frequency", "h4"),
            (5, "https://x.com/a/5", "2024-05-01", "audience building fundamentals", "h5"),
        ],
    )
    upsert_embeddings(db, [(i + 1, vectors[i]) for i in range(5)])
    db.close()

    # Mock embed_query: hand back chunk 1's vector, so vector search
    # ranks chunk 1 first by L2 distance = 0. BM25 on "price newsletter"
    # also matches chunk 1 strongly. RRF should put chunk 1 at the top.
    monkeypatch.setattr(
        "app.retrieval.query.embed_query",
        AsyncMock(return_value=vectors[0]),
    )

    from app.retrieval import Snippet, retrieve

    stats: dict[str, float] = {}
    snippets = await retrieve(
        mentor_slug="testmentor",
        query="price newsletter",
        k=3,
        stats=stats,
    )

    assert isinstance(snippets, list)
    assert 1 <= len(snippets) <= 3
    assert all(isinstance(s, Snippet) for s in snippets)

    # Top-1: chunk 1 (matches both the lexical query and the mocked vector).
    top = snippets[0]
    assert top.chunk_id == 1
    assert top.mentor_slug == "testmentor"
    assert top.text == "how to price a paid newsletter"
    assert top.source_url == "https://x.com/a/1"
    assert top.source_type == "twitter"
    assert top.date == "2024-01-01"
    # Both retrievers found chunk 1 → both ranks populated.
    assert top.bm25_rank == 1
    assert top.vec_rank == 1
    # RRF fusion math: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.03279
    assert abs(top.score - (2 / 61)) < 1e-6

    # Stats dict was populated by retrieve().
    assert {"embed_ms", "bm25_ms", "vec_ms", "total_ms"} <= stats.keys()


# ---------------------------------------------------------------------------
# Test 5 — source_priority boost lifts canonical chunks (Phase 5 Step 1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_source_priority_boost_lifts_canonical_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    """Build a tiny mentor DB with two chunks at differing
    source_priority. With boost=False, the higher fused score wins.
    With boost=True, a canonical (priority 3) chunk gets multiplied
    by 1.2 and the rank order can flip when scores were close."""
    from app.config import settings
    from app.embeddings.store import upsert_embeddings
    from app.ingest.db import open_mentor_db

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    # Seed: chunk 1 = tweet (priority 1) — perfect query match.
    # chunk 2 = canonical essay (priority 3) — slightly weaker match.
    db = open_mentor_db("testmentor")
    db.executemany(
        "INSERT INTO chunks(id, source_url, source_type, date, text, "
        "content_hash, source_priority) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "https://x.com/a/1", "twitter", "2024-04-01",
             "pricing is positioning", "ph1", 1),
            (2, "https://blog.example/p", "blog_post", "2024-04-02",
             "pricing positioning lever", "ph2", 3),
        ],
    )
    rng = random.Random(42)
    vecs = [[rng.uniform(-1.0, 1.0) for _ in range(1536)] for _ in range(2)]
    upsert_embeddings(db, [(1, vecs[0]), (2, vecs[1])])
    db.close()

    # Mock embed_query to return chunk 1's vector — chunk 1 wins
    # vector search (distance 0). BM25 also favors chunk 1 (the
    # phrase appears verbatim in chunk 1's text). So without boost,
    # chunk 1 ranks first.
    monkeypatch.setattr(
        "app.retrieval.query.embed_query",
        AsyncMock(return_value=vecs[0]),
    )

    from app.retrieval import retrieve

    no_boost = await retrieve(
        mentor_slug="testmentor", query="pricing positioning", k=2,
    )
    assert no_boost[0].chunk_id == 1
    score_no_boost_2 = next(s for s in no_boost if s.chunk_id == 2).score

    # With boost on, chunk 2's score is multiplied by 1.2. Chunk 2
    # may or may not flip to #1 depending on the raw score gap, but
    # its absolute score must rise by exactly the multiplier.
    boosted = await retrieve(
        mentor_slug="testmentor", query="pricing positioning", k=2,
        source_priority_boost=True,
    )
    boosted_2 = next(s for s in boosted if s.chunk_id == 2)
    assert abs(boosted_2.score - score_no_boost_2 * 1.2) < 1e-9

    # Chunk 1 (priority 1) is unaffected by the boost.
    score_no_boost_1 = next(s for s in no_boost if s.chunk_id == 1).score
    score_boost_1 = next(s for s in boosted if s.chunk_id == 1).score
    assert abs(score_boost_1 - score_no_boost_1) < 1e-9

    # Snippets carry source_priority through.
    assert {s.source_priority for s in boosted} == {1, 3}


# ---------------------------------------------------------------------------
# Test 6 — query_embedding override skips the embed call (Phase 6)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_uses_provided_embedding_skips_embed_call(
    tmp_path: Path, monkeypatch
) -> None:
    """When `query_embedding` is supplied, retrieve() must NOT call
    embed_query() — the council fan-out shares one embedding across
    three mentor calls precisely to skip the redundant OpenAI round-
    trips. Vector search must consume the supplied embedding directly."""
    from app.config import settings
    from app.embeddings.store import upsert_embeddings
    from app.ingest.db import open_mentor_db

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    rng = random.Random(7)
    vectors = [[rng.uniform(-1.0, 1.0) for _ in range(1536)] for _ in range(3)]

    db = open_mentor_db("testmentor")
    db.executemany(
        "INSERT INTO chunks(id, source_url, source_type, date, text, content_hash) "
        "VALUES (?, ?, 'twitter', ?, ?, ?)",
        [
            (1, "https://x.com/a/1", "2024-01-01", "alpha topic", "h1"),
            (2, "https://x.com/a/2", "2024-02-01", "beta topic", "h2"),
            (3, "https://x.com/a/3", "2024-03-01", "gamma topic", "h3"),
        ],
    )
    upsert_embeddings(db, [(i + 1, vectors[i]) for i in range(3)])
    db.close()

    # Sentinel mock: if retrieve() ever calls embed_query, the test
    # fails immediately — we want strict proof that the override
    # branch skips the network call.
    embed_mock = AsyncMock(side_effect=AssertionError(
        "embed_query was called even though query_embedding was provided"
    ))
    monkeypatch.setattr("app.retrieval.query.embed_query", embed_mock)

    from app.retrieval import retrieve

    snippets = await retrieve(
        mentor_slug="testmentor",
        query="beta topic",
        k=3,
        query_embedding=vectors[1],   # → vector search will favor chunk 2
    )

    embed_mock.assert_not_awaited()
    assert snippets, "expected at least one snippet from the supplied embedding"

    # Vector search ran on vectors[1] → chunk 2 is the L2 = 0 match,
    # so it must appear with vec_rank == 1.
    chunk_2 = next((s for s in snippets if s.chunk_id == 2), None)
    assert chunk_2 is not None, "chunk 2 should be retrieved when its embedding is supplied"
    assert chunk_2.vec_rank == 1
