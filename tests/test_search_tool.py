"""Unit tests for app.tools.search — the generic single-mentor tool.

The tool replaces mentor-brain's per-mentor search_<slug>.py
modules; mentor selection is now a runtime parameter driven by the
YAML config rather than baked into the tool name."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest


@pytest.fixture
def tiny_config(tmp_path: Path, monkeypatch):
    """A 2-mentor YAML lineup so search() has something to dispatch
    against without needing real DBs."""
    yaml = """
    mentors:
      - slug: alpha
        display_name: Alpha Author
        domain_focus: "things alpha is good at"
        sources:
          - type: blog
            domain: alpha.example.test
      - slug: beta
        display_name: Beta Author
        sources:
          - type: twitter
            handle: beta_h
    """
    cfg = tmp_path / "mentors.yaml"
    cfg.write_text(yaml)
    monkeypatch.setenv("COUNCIL_CONFIG", str(cfg))
    return cfg


@pytest.mark.asyncio
async def test_search_schema_clamp_and_plumb_through(monkeypatch, tiny_config):
    from app.tools.search import TOOL_NAME, build_tool_description, search

    assert TOOL_NAME == "search"
    desc = build_tool_description()
    # Description enumerates configured mentors so the LLM router
    # has explicit slug options.
    assert "alpha" in desc
    assert "beta" in desc

    captured: dict[str, object] = {}

    async def capturing_retrieve(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("app.tools.search.retrieve", capturing_retrieve)

    # Defaults: k=8, recency_bias=False, source_priority_boost=True.
    await search(mentor_slug="alpha", question="anything")
    assert captured["mentor_slug"] == "alpha"
    assert captured["k"] == 8
    assert captured["recency_bias"] is False
    assert captured["source_priority_boost"] is True

    # k clamps to [1, 20].
    await search(mentor_slug="alpha", question="x", k=99)
    assert captured["k"] == 20
    await search(mentor_slug="alpha", question="x", k=0)
    assert captured["k"] == 1
    await search(mentor_slug="alpha", question="x", k=-3)
    assert captured["k"] == 1

    # recency_bias plumbs through.
    await search(mentor_slug="alpha", question="x", recency_bias=True)
    assert captured["recency_bias"] is True


@pytest.mark.asyncio
async def test_search_unknown_mentor_returns_canonical_message(tiny_config):
    from app.tools.search import search
    out = await search(mentor_slug="gamma", question="anything")
    assert "Unknown mentor" in out
    assert "alpha" in out and "beta" in out


@pytest.mark.asyncio
async def test_search_empty_question_short_circuits(tiny_config):
    from app.tools.search import search
    assert await search(mentor_slug="alpha", question="") == "No question provided."
    assert await search(mentor_slug="alpha", question="   \t\n ") == "No question provided."


@pytest.mark.asyncio
async def test_search_formats_snippets_with_kind_tags(monkeypatch, tiny_config):
    """Mocked snippets at three priority levels should render with
    correct kind tags ('essay' / 'long-form' / 'tweet')."""
    from app.retrieval import Snippet
    from app.tools.search import search

    snippets = [
        Snippet(
            chunk_id=1, mentor_slug="alpha",
            text="Long-form essay body here.",
            source_url="https://alpha.example.test/p/one",
            source_type="blog_post",
            date="2025-08-01T00:00:00+00:00",
            score=0.05, bm25_rank=1, vec_rank=1, source_priority=3,
        ),
        Snippet(
            chunk_id=2, mentor_slug="alpha",
            text="Podcast snippet.",
            source_url="https://alpha.example.test/podcast/ep1",
            source_type="podcast",
            date="2025-09-01T00:00:00+00:00",
            score=0.04, bm25_rank=2, vec_rank=2, source_priority=2,
        ),
        Snippet(
            chunk_id=3, mentor_slug="alpha",
            text="A pithy tweet.",
            source_url="https://x.com/alpha/status/1",
            source_type="twitter",
            date="2025-10-01T00:00:00+00:00",
            score=0.02, bm25_rank=3, vec_rank=None, source_priority=1,
        ),
    ]

    async def fake_retrieve(**_):
        return snippets

    monkeypatch.setattr("app.tools.search.retrieve", fake_retrieve)

    out = await search(mentor_slug="alpha", question="any")
    assert "Found 3 relevant snippets from Alpha Author's archive:" in out
    assert "[1] (essay" in out
    assert "[2] (long-form" in out
    assert "[3] (tweet" in out
    for s in snippets:
        assert s.text in out
        assert s.source_url in out


@pytest.mark.asyncio
async def test_search_handles_retrieve_exception_gracefully(
    monkeypatch, tiny_config, caplog
):
    from app.tools.search import search

    async def boom(**_):
        raise RuntimeError("synthetic db missing")

    monkeypatch.setattr("app.tools.search.retrieve", boom)

    with caplog.at_level(logging.ERROR, logger="app.tools.search"):
        out = await search(mentor_slug="alpha", question="anything")

    assert "Alpha Author" in out
    assert "currently unavailable" in out
    matching = [
        r for r in caplog.records
        if r.exc_info is not None
        and "synthetic db missing" in str(r.exc_info[1])
    ]
    assert matching, "expected ERROR log carrying the underlying RuntimeError"
