"""Unit tests for app.tools.council_retrieve.

The generic version reads the mentor lineup from YAML at call time
rather than baking in a fixed three-mentor list. Tests
plant a 3-mentor YAML in tmp_path, then exercise the orchestration:

  1. Schema clamp + plumb-through (k_per_mentor in [3, 15],
     recency_bias propagates, shared embedding goes to every call).
  2. Parallel fan-out (timestamps prove asyncio.gather is concurrent).
  3. Output formatting (one labeled section per mentor, mentor-
     prefixed citations like [alpha_1] / [beta_2]).
  4. Single-mentor failure isolation (one raises, the others render,
     unavailable notice names survivors, no exception escapes).
  5. All-fail returns the canonical "Council unavailable" message.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def three_mentor_config(tmp_path: Path, monkeypatch):
    yaml = """
    mentors:
      - slug: alpha
        display_name: Alpha
        domain_focus: "first thing"
        sources: []
      - slug: beta
        display_name: Beta
        domain_focus: "second thing"
        sources: []
      - slug: gamma
        display_name: Gamma
        domain_focus: "third thing"
        sources: []
    """
    cfg = tmp_path / "mentors.yaml"
    cfg.write_text(yaml)
    monkeypatch.setenv("COUNCIL_CONFIG", str(cfg))
    return cfg


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch):
    """Every test runs against a fake embedding so we never call OpenAI."""
    monkeypatch.setattr(
        "app.tools.council_retrieve.embed_query",
        AsyncMock(return_value=[0.0] * 1536),
    )


@pytest.mark.asyncio
async def test_council_schema_clamp_and_plumb_through(monkeypatch, three_mentor_config):
    from app.tools.council_retrieve import (
        TOOL_NAME,
        build_tool_description,
        council_retrieve,
    )

    assert TOOL_NAME == "council_retrieve"
    desc = build_tool_description()
    # Description names every configured mentor so the LLM router
    # has explicit triggers.
    for name in ("Alpha", "Beta", "Gamma"):
        assert name in desc

    captured: list[dict[str, object]] = []

    async def capturing_retrieve(**kwargs):
        captured.append(dict(kwargs))
        return []

    monkeypatch.setattr("app.tools.council_retrieve.retrieve", capturing_retrieve)

    captured.clear()
    await council_retrieve(question="x")
    assert len(captured) == 3
    slugs = {c["mentor_slug"] for c in captured}
    assert slugs == {"alpha", "beta", "gamma"}
    for c in captured:
        assert c["k"] == 8
        assert c["recency_bias"] is False
        assert c["source_priority_boost"] is True
        assert c["query_embedding"] == [0.0] * 1536

    # k_per_mentor clamps to [3, 15].
    captured.clear()
    await council_retrieve(question="x", k_per_mentor=99)
    assert all(c["k"] == 15 for c in captured)
    captured.clear()
    await council_retrieve(question="x", k_per_mentor=1)
    assert all(c["k"] == 3 for c in captured)

    # Empty question short-circuits before any retrieve runs.
    captured.clear()
    out = await council_retrieve(question="")
    assert out == "No question provided."
    assert captured == []


@pytest.mark.asyncio
async def test_council_runs_retrievals_in_parallel(monkeypatch, three_mentor_config):
    """Each retrieve() sleeps PER_CALL_SLEEP. If gather were serial,
    total would be ~3x. We assert (a) all three start within ~50ms of
    each other, and (b) total wall time is much closer to one sleep
    than to three."""
    PER_CALL_SLEEP = 0.20
    start_times: list[float] = []

    async def slow_retrieve(**_):
        start_times.append(time.perf_counter())
        await asyncio.sleep(PER_CALL_SLEEP)
        return []

    monkeypatch.setattr("app.tools.council_retrieve.retrieve", slow_retrieve)

    from app.tools.council_retrieve import council_retrieve

    t0 = time.perf_counter()
    await council_retrieve(question="multi-lens question")
    elapsed = time.perf_counter() - t0

    assert len(start_times) == 3
    spread_ms = (max(start_times) - min(start_times)) * 1000
    assert spread_ms < 50, f"retrieve() calls did not start in parallel: spread={spread_ms:.1f}ms"
    assert elapsed < PER_CALL_SLEEP * 1.6, (
        f"council total {elapsed:.3f}s suggests serialization (expected near {PER_CALL_SLEEP:.3f}s)"
    )


@pytest.mark.asyncio
async def test_council_formats_three_sections_with_labeled_snippets(
    monkeypatch, three_mentor_config
):
    from app.retrieval import Snippet
    from app.tools.council_retrieve import council_retrieve

    fakes = {
        "alpha": [
            Snippet(
                chunk_id=10,
                mentor_slug="alpha",
                text="Alpha take one.",
                source_url="https://alpha.example.test/1",
                source_type="twitter",
                date="2025-01-01",
                score=0.04,
                bm25_rank=1,
                vec_rank=1,
                source_priority=1,
            ),
        ],
        "beta": [
            Snippet(
                chunk_id=20,
                mentor_slug="beta",
                text="Beta take one.",
                source_url="https://beta.example.test/1",
                source_type="blog_post",
                date="2025-02-01",
                score=0.05,
                bm25_rank=1,
                vec_rank=1,
                source_priority=3,
            ),
            Snippet(
                chunk_id=21,
                mentor_slug="beta",
                text="Beta take two.",
                source_url="https://beta.example.test/2",
                source_type="blog_post",
                date="2025-02-02",
                score=0.04,
                bm25_rank=2,
                vec_rank=None,
                source_priority=3,
            ),
        ],
        "gamma": [
            Snippet(
                chunk_id=30,
                mentor_slug="gamma",
                text="Gamma take one.",
                source_url="https://gamma.example.test/1",
                source_type="podcast",
                date="2025-03-01",
                score=0.03,
                bm25_rank=None,
                vec_rank=1,
                source_priority=2,
            ),
        ],
    }

    async def fake_retrieve(**kwargs):
        return fakes[kwargs["mentor_slug"]]

    monkeypatch.setattr("app.tools.council_retrieve.retrieve", fake_retrieve)

    out = await council_retrieve(question="What should I do?", k_per_mentor=3)

    assert out.startswith('Council retrieved snippets for: "What should I do?"')
    # One section per mentor, in YAML order.
    assert "=== ALPHA" in out
    assert "=== BETA" in out
    assert "=== GAMMA" in out
    assert out.index("ALPHA") < out.index("BETA") < out.index("GAMMA")

    # Mentor-prefixed citations.
    assert "[alpha_1]" in out
    assert "[beta_1]" in out
    assert "[beta_2]" in out
    assert "[gamma_1]" in out

    for snips in fakes.values():
        for s in snips:
            assert s.text in out
            assert s.source_url in out

    assert "currently unavailable" not in out


@pytest.mark.asyncio
async def test_council_isolates_single_mentor_failure(monkeypatch, three_mentor_config, caplog):
    """beta raises; alpha + gamma still render; unavailable notice
    names survivors; tool returns a string, never raises."""
    from app.retrieval import Snippet
    from app.tools.council_retrieve import council_retrieve

    alpha_snip = Snippet(
        chunk_id=1,
        mentor_slug="alpha",
        text="Alpha take",
        source_url="https://alpha.example.test/1",
        source_type="twitter",
        date="2025-01-01",
        score=0.05,
        bm25_rank=1,
        vec_rank=1,
        source_priority=1,
    )
    gamma_snip = Snippet(
        chunk_id=2,
        mentor_slug="gamma",
        text="Gamma take",
        source_url="https://gamma.example.test/1",
        source_type="twitter",
        date="2025-02-01",
        score=0.04,
        bm25_rank=1,
        vec_rank=1,
        source_priority=1,
    )

    async def maybe_failing(**kwargs):
        slug = kwargs["mentor_slug"]
        if slug == "beta":
            raise RuntimeError("beta.db missing")
        if slug == "alpha":
            return [alpha_snip]
        if slug == "gamma":
            return [gamma_snip]
        return []

    monkeypatch.setattr("app.tools.council_retrieve.retrieve", maybe_failing)

    with caplog.at_level(logging.ERROR, logger="app.tools.council_retrieve"):
        out = await council_retrieve(question="anything")

    assert "Alpha take" in out
    assert "Gamma take" in out
    assert "[alpha_1]" in out
    assert "[gamma_1]" in out

    assert "Beta's archive is currently unavailable" in out
    assert "Alpha + Gamma" in out
    assert "[beta_1]" not in out

    matching = [
        r
        for r in caplog.records
        if r.exc_info is not None and "beta.db missing" in str(r.exc_info[1])
    ]
    assert matching


@pytest.mark.asyncio
async def test_council_all_fail_returns_canonical_message(monkeypatch, three_mentor_config):
    from app.tools.council_retrieve import council_retrieve

    async def boom(**_):
        raise RuntimeError("data dir unmounted")

    monkeypatch.setattr("app.tools.council_retrieve.retrieve", boom)

    out = await council_retrieve(question="anything")
    assert out == ("Council unavailable: every configured mentor archive failed to respond.")
