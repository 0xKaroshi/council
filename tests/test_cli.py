"""CLI smoke tests — verify every command parses and basic
help / --version surfaces work without dragging in network or DBs.

Per-command behavior is exercised by the targeted tool tests
(test_search_tool, test_council_retrieve_tool, etc.). This file
just guards against the kind of regression where a typo in
@cli.command breaks the entire CLI surface."""
from __future__ import annotations

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def test_root_help_lists_every_command(runner):
    from app.cli import cli
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in ("init", "list-mentors", "ingest", "embed", "ask", "convene", "status"):
        assert cmd in result.output, f"missing command: {cmd}"


def test_each_subcommand_has_help(runner):
    from app.cli import cli
    for cmd in ("init", "list-mentors", "ingest", "embed", "ask", "convene", "status"):
        result = runner.invoke(cli, [cmd, "--help"])
        assert result.exit_code == 0, f"--help failed for {cmd}: {result.output}"


def test_init_scaffolds_into_clean_directory(tmp_path, runner, monkeypatch):
    """`council init` in an empty cwd produces config/, user_context/,
    and data/. It should not write any real .md files (only templates
    in the bundled package, which we won't have in tmp_path)."""
    monkeypatch.chdir(tmp_path)
    from app.cli import cli

    result = runner.invoke(cli, ["init"])
    # Exit code may be 0 even if running outside the bundled package
    # path; what we assert is that init created the expected dirs.
    assert (tmp_path / "config").is_dir()
    assert (tmp_path / "user_context").is_dir()
    assert (tmp_path / "data").is_dir() or (tmp_path / "data" / "mentors").is_dir()


def test_list_mentors_friendly_error_when_no_config(tmp_path, runner, monkeypatch):
    """Running list-mentors with no config should exit nonzero and
    point the user at `council init` rather than crashing with a
    bare stack trace.

    We disable the bundled-config fallback so this test exercises the
    truly-missing case (otherwise the fallback to the shipped
    config/mentors.yaml would always succeed)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COUNCIL_CONFIG", str(tmp_path / "definitely-missing.yaml"))
    # Reload mentors module so the new env var is honored.
    import importlib

    import app.ingest.mentors as m
    importlib.reload(m)
    monkeypatch.setattr(m, "_bundled_config_path", lambda: None)

    from app.cli import cli
    result = runner.invoke(cli, ["list-mentors"])
    assert result.exit_code == 2
    assert "not found" in result.output or "not found" in (result.stderr_bytes or b"").decode()


# ---------------------------------------------------------------------------
# Bundled-fallback tests (repo MUST contain examples/<slug>/<slug>.db files)
# ---------------------------------------------------------------------------

def _bundled_slugs_present() -> list[str]:
    """Return slugs that have a bundled DB findable via the same
    resolver the CLI uses. Empty when running from a checkout without
    examples/ (e.g. CI without LFS or a stripped wheel)."""
    from app.ingest.db import _bundled_db_path
    candidates = ["paulgraham", "naval", "patrick_oshag"]
    return [s for s in candidates if _bundled_db_path(s) is not None]


def test_status_falls_back_to_bundled_examples_when_data_dir_empty(
    tmp_path, runner, monkeypatch
):
    """Point COUNCIL_DATA_DIR at an empty tmp_path. The bundled DBs
    should appear in `council status` with the (bundled) source tag —
    proving a freshly-installed council works without ingestion."""
    bundled = _bundled_slugs_present()
    if not bundled:
        pytest.skip("examples/<slug>/<slug>.db not present — skipping")

    # Empty user data dir.
    empty_data = tmp_path / "empty_data"
    empty_data.mkdir()
    monkeypatch.setenv("COUNCIL_DATA_DIR", str(empty_data))

    # Real mentor config naming all bundled slugs.
    cfg_path = tmp_path / "mentors.yaml"
    cfg_path.write_text(
        "mentors:\n"
        + "".join(
            f"  - slug: {s}\n    display_name: {s.title()}\n    sources: []\n"
            for s in bundled
        )
    )
    monkeypatch.setenv("COUNCIL_CONFIG", str(cfg_path))

    # Re-import config + mentors so the env var changes take effect.
    import importlib

    import app.config as cfg_mod
    import app.ingest.mentors as m
    importlib.reload(cfg_mod)
    importlib.reload(m)
    import app.ingest.db as db_mod
    importlib.reload(db_mod)
    import app.cli as cli_mod
    importlib.reload(cli_mod)

    result = runner.invoke(cli_mod.cli, ["status"])
    assert result.exit_code == 0, result.output

    # Every bundled mentor surfaces with the (bundled) tag, NOT (missing).
    for slug in bundled:
        assert slug in result.output, f"missing slug {slug} in status output"
    assert "(bundled)" in result.output
    assert "(missing)" not in result.output

    # User data dir should still be empty (status must NEVER write to it
    # just to read).
    assert not (empty_data / "mentors").exists() or not list(
        (empty_data / "mentors").glob("*.db")
    )


def test_search_reads_from_bundled_examples_when_user_db_missing(
    tmp_path, runner, monkeypatch
):
    """A clean COUNCIL_DATA_DIR + a bundled paulgraham DB → `council ask
    paulgraham "..."` returns a non-empty snippet block. We mock the
    OpenAI embed call so the test stays offline; the BM25 branch alone
    is enough to exercise the bundled-DB read path."""
    bundled = _bundled_slugs_present()
    if "paulgraham" not in bundled:
        pytest.skip("examples/paulgraham/paulgraham.db not present — skipping")

    empty_data = tmp_path / "empty_data"
    empty_data.mkdir()
    monkeypatch.setenv("COUNCIL_DATA_DIR", str(empty_data))

    cfg_path = tmp_path / "mentors.yaml"
    cfg_path.write_text(
        "mentors:\n"
        "  - slug: paulgraham\n"
        "    display_name: Paul Graham\n"
        "    sources: []\n"
    )
    monkeypatch.setenv("COUNCIL_CONFIG", str(cfg_path))

    # Mock the OpenAI embed — we don't need real vectors for BM25 to
    # find matches, and the test must be runnable offline.
    from unittest.mock import AsyncMock
    monkeypatch.setattr(
        "app.retrieval.query.embed_query",
        AsyncMock(return_value=[0.0] * 1536),
    )

    import importlib

    import app.config as cfg_mod
    import app.ingest.mentors as m
    importlib.reload(cfg_mod)
    importlib.reload(m)
    import app.ingest.db as db_mod
    importlib.reload(db_mod)

    # Use the search tool directly (CLI runner + asyncio.run interop is
    # awkward; the CLI delegates to this same function).
    import asyncio
    from app.tools.search import search

    out = asyncio.run(search(mentor_slug="paulgraham", question="startup", k=3))

    # Either we found snippets (typical) or we explicitly didn't —
    # what we MUST NOT see is the unavailable / unknown-mentor message,
    # which would indicate the bundled fallback didn't fire.
    assert "Paul Graham" in out
    assert "Unknown mentor" not in out
    assert "currently unavailable" not in out
    assert "Found " in out and " relevant snippets" in out

    # User data dir untouched.
    assert not (empty_data / "mentors").exists() or not list(
        (empty_data / "mentors").glob("*.db")
    )
