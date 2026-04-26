"""Unit tests for app.ingest.mentors — the YAML config loader.

The loader is the seam every other module reaches through to find
the mentor lineup. Bad YAML, missing slug fields, or a missing
config file all need predictable behavior."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_load_mentors_parses_minimal_valid_config(tmp_path: Path):
    yaml = """
    mentors:
      - slug: alpha
        display_name: Alpha Mentor
        domain_focus: "first thing"
        sources:
          - type: blog
            domain: alpha.example.test
            rss_url: https://alpha.example.test/atom.xml
          - type: twitter
            handle: alpha_h
        source_priority:
          blog: 3
          twitter: 1
      - slug: beta
        display_name: Beta Mentor
        sources:
          - type: blog
            domain: beta.example.test
    """
    config = tmp_path / "mentors.yaml"
    config.write_text(yaml)

    from app.ingest.mentors import load_mentors

    result = load_mentors(path=config)

    assert set(result.keys()) == {"alpha", "beta"}

    alpha = result["alpha"]
    assert alpha.display_name == "Alpha Mentor"
    assert alpha.domain_focus == "first thing"
    assert len(alpha.sources) == 2
    blog = alpha.sources[0]
    assert blog.type == "blog"
    assert blog.domain == "alpha.example.test"
    assert blog.rss_url == "https://alpha.example.test/atom.xml"
    twitter = alpha.sources[1]
    assert twitter.type == "twitter"
    assert twitter.handle == "alpha_h"
    # Priority overrides land; defaults fill in for unmentioned types.
    assert alpha.priority_for("blog") == 3
    assert alpha.priority_for("twitter") == 1
    assert alpha.priority_for("podcast") == 2  # default

    beta = result["beta"]
    # Convenience accessors.
    assert beta.twitter_handle is None
    assert beta.blog_domains == ("beta.example.test",)


def test_load_mentors_raises_on_missing_file(tmp_path: Path):
    from app.ingest.mentors import load_mentors

    with pytest.raises(FileNotFoundError) as exc:
        load_mentors(path=tmp_path / "nope.yaml")
    assert "council init" in str(exc.value)


def test_load_mentors_skips_invalid_entries_keeps_valid(tmp_path: Path, caplog):
    """An entry missing 'slug' should be logged + skipped, not crash
    the whole load. Other entries in the same file still come through."""
    import logging

    yaml = """
    mentors:
      - display_name: Missing Slug
        sources: []
      - slug: ok
        display_name: Has Both
        sources: []
      - slug: ok            # duplicate slug, second occurrence dropped
        display_name: Duplicate
    """
    config = tmp_path / "mentors.yaml"
    config.write_text(yaml)

    from app.ingest.mentors import load_mentors

    with caplog.at_level(logging.WARNING):
        result = load_mentors(path=config)

    assert set(result.keys()) == {"ok"}
    assert result["ok"].display_name == "Has Both"


def test_mentors_proxy_returns_empty_when_config_missing(monkeypatch, tmp_path: Path):
    """The MENTORS module-level proxy must not raise on `in` / `iter`
    when the YAML file is absent — the CLI uses it to display
    helpful 'run council init' errors instead of stack traces.

    We disable the bundled-config fallback so this exercises the
    truly-missing case (otherwise the proxy would fall back to the
    shipped config/mentors.yaml and return the example mentors)."""
    monkeypatch.setenv("COUNCIL_CONFIG", str(tmp_path / "nope.yaml"))
    # Re-import so the env change is picked up.
    import importlib

    import app.ingest.mentors as mentors_mod

    importlib.reload(mentors_mod)
    monkeypatch.setattr(mentors_mod, "_bundled_config_path", lambda: None)

    assert "anything" not in mentors_mod.MENTORS
    assert list(iter(mentors_mod.MENTORS)) == []
    assert mentors_mod.MENTORS.get("anything") is None
