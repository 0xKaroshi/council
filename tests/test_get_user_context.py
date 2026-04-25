"""Unit tests for app.tools.get_user_context — the user-context layer.

The tool returns markdown content from a configurable directory.
Tests cover: dynamic file discovery (anything *.md, skipping
.example.md templates), unknown-file message, empty / missing /
unreadable file handling, and missing-directory error message."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest


@pytest.fixture
def context_dir(tmp_path: Path, monkeypatch):
    """A fresh user_context directory with two real files + one
    template (which should be ignored by the tool)."""
    d = tmp_path / "user_context"
    d.mkdir()
    (d / "brand.md").write_text("# Brand\n\nWe ship simple things.\n")
    (d / "audience.md").write_text("")  # intentionally empty
    (d / "constraints.example.md").write_text("# Template — should be ignored\n")
    monkeypatch.setattr("app.tools.get_user_context.settings", _settings_with(d))
    return d


def _settings_with(path: Path):
    class _Stub:
        user_context_dir = path
    return _Stub()


@pytest.mark.asyncio
async def test_all_concatenates_files_skipping_example_templates(context_dir):
    from app.tools.get_user_context import get_user_context

    out = await get_user_context(file=None)

    # Real files appear with their full filename as a section header.
    assert "# brand.md" in out
    assert "We ship simple things." in out
    assert "# audience.md" in out
    # Empty slot is surfaced explicitly.
    assert "(empty: audience.md" in out
    # Template was filtered out.
    assert "constraints.example.md" not in out
    assert "Template — should be ignored" not in out


@pytest.mark.asyncio
async def test_single_file_by_stem(context_dir):
    from app.tools.get_user_context import get_user_context
    out = await get_user_context(file="brand")
    assert out == "# Brand\n\nWe ship simple things."


@pytest.mark.asyncio
async def test_unknown_file_returns_helpful_message(context_dir):
    from app.tools.get_user_context import get_user_context
    out = await get_user_context(file="metrics")
    assert "Invalid file name" in out
    # Lists what IS available so the LLM can self-correct.
    assert "audience" in out
    assert "brand" in out


@pytest.mark.asyncio
async def test_all_with_no_files_surfaces_empty_directory_hint(tmp_path: Path, monkeypatch):
    """A directory exists but contains no .md files — the LLM should
    get a clear "no context yet" hint rather than an empty string."""
    d = tmp_path / "user_context_empty"
    d.mkdir()
    monkeypatch.setattr("app.tools.get_user_context.settings", _settings_with(d))

    from app.tools.get_user_context import get_user_context
    out = await get_user_context()
    assert "no user context files yet" in out
    assert ".example.md" in out


@pytest.mark.asyncio
async def test_missing_directory_returns_unavailable_with_path(tmp_path: Path, monkeypatch, caplog):
    nonexistent = tmp_path / "definitely-not-here"
    monkeypatch.setattr("app.tools.get_user_context.settings", _settings_with(nonexistent))

    from app.tools.get_user_context import get_user_context

    with caplog.at_level(logging.ERROR, logger="app.tools.get_user_context"):
        out = await get_user_context()

    assert "not present" in out
    # Path appears so the user can fix the misconfiguration.
    assert str(nonexistent) in out
    assert any("directory missing" in r.message for r in caplog.records)
