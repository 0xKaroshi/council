"""Mentor registry — loaded from a user-editable YAML config.

The YAML file (default: `config/mentors.yaml`, override via the
`COUNCIL_CONFIG` env var) drives every mentor-specific decision in
the system: which Twitter handles to backfill, which blog domains
to crawl, what slug the per-mentor SQLite DB takes, what gets
displayed when council mode fans out a question.

Schema (one entry per mentor):

    mentors:
      - slug: paulgraham                         # required, kebab-or-snake-case
        display_name: Paul Graham                # required, human-friendly
        domain_focus: "startup essays, hacker culture"   # one-line LLM hint
        sources:
          - type: blog
            domain: paulgraham.com
            rss_url: http://www.aaronsw.com/2002/feeds/pgessays.rss   # optional
          - type: twitter
            handle: paulg                        # without the @
        source_priority:                         # optional; defaults below
          blog: 3        # canonical essays — boosted by retrieval
          twitter: 1
          podcast: 2

Re-load happens on every `load_mentors()` call so the YAML can be
edited live without restarting the CLI / server.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

log = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("config") / "mentors.yaml"

# Built-in priority defaults. Long-form authored content (essays,
# newsletters, podcasts-with-transcripts) gets the canonical slot
# (3), tweets get the noise-floor slot (1). retrieve()'s
# source_priority_boost multiplier (1.2x for priority 3) applies
# at retrieval time.
_DEFAULT_PRIORITIES: dict[str, int] = {
    "blog": 3,
    "substack": 3,
    "newsletter": 3,
    "podcast": 2,
    "youtube": 2,
    "twitter": 1,
}


@dataclass(frozen=True)
class SourceConfig:
    """One ingestion source for one mentor."""
    type: str                 # "twitter" | "blog" | "substack" | "newsletter" | "youtube"
    handle: str | None = None # twitter handle (no @)
    domain: str | None = None # blog / substack / newsletter domain
    rss_url: str | None = None  # explicit RSS override (skips discovery cascade)
    url: str | None = None      # canonical URL (e.g. youtube channel, podcast feed)


@dataclass(frozen=True)
class MentorConfig:
    """One mentor's full configuration."""
    slug: str
    display_name: str
    domain_focus: str = ""
    sources: tuple[SourceConfig, ...] = ()
    source_priority: dict[str, int] = field(default_factory=lambda: dict(_DEFAULT_PRIORITIES))

    # Convenience accessors used by the legacy single-source code paths.
    @property
    def twitter_handle(self) -> str | None:
        for s in self.sources:
            if s.type == "twitter" and s.handle:
                return s.handle
        return None

    @property
    def blog_domains(self) -> tuple[str, ...]:
        return tuple(s.domain for s in self.sources if s.type in ("blog", "substack") and s.domain)

    @property
    def blog_url(self) -> str | None:
        # Backwards-compat shim for code still expecting blog_url.
        for s in self.sources:
            if s.type in ("blog", "substack") and s.domain:
                return f"https://{s.domain}"
        return None

    def priority_for(self, source_type: str) -> int:
        return int(self.source_priority.get(source_type, _DEFAULT_PRIORITIES.get(source_type, 1)))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _config_path() -> Path:
    """Resolve the active config path. `COUNCIL_CONFIG` env wins; falls
    back to ./config/mentors.yaml relative to the current working dir."""
    raw = os.environ.get("COUNCIL_CONFIG")
    if raw:
        return Path(raw).expanduser()
    return _DEFAULT_CONFIG_PATH


def _bundled_config_path() -> Path | None:
    """Locate the bundled `config/mentors.yaml` (shipped via
    setuptools package-data) or return None.

    Mirrors the bundled-DB resolution strategy in `app.ingest.db`:
    importlib.resources first (works for installed wheels and pipx),
    fall back to walking up from this module file (works for editable
    installs and direct git checkouts).
    """
    try:
        from importlib.resources import files
        try:
            traversable = files("config").joinpath("mentors.yaml")
            if traversable.is_file():
                return Path(str(traversable))
        except (ModuleNotFoundError, FileNotFoundError, AttributeError):
            pass
    except ImportError:
        pass

    # __file__ → .../app/ingest/mentors.py  →  three .parent's = repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = repo_root / "config" / "mentors.yaml"
    if candidate.exists():
        return candidate
    return None


def _resolve_config_path() -> Path:
    """Pick the config path the loader will actually open. Prefers the
    user's path (cwd-relative or COUNCIL_CONFIG); falls back to the
    bundled default so `pipx install council && council status` works
    out of the box without any setup."""
    user_path = _config_path()
    if user_path.exists():
        return user_path
    bundled = _bundled_config_path()
    if bundled is not None:
        return bundled
    return user_path  # not found anywhere — load_mentors will raise


def _parse_one(entry: dict) -> MentorConfig:
    if "slug" not in entry:
        raise ValueError(f"mentor entry missing 'slug': {entry!r}")
    if "display_name" not in entry:
        raise ValueError(
            f"mentor entry {entry['slug']!r} missing 'display_name'"
        )
    sources_raw = entry.get("sources") or []
    sources = tuple(
        SourceConfig(
            type=str(s.get("type", "")).strip(),
            handle=s.get("handle"),
            domain=s.get("domain"),
            rss_url=s.get("rss_url"),
            url=s.get("url"),
        )
        for s in sources_raw
        if isinstance(s, dict)
    )
    priorities = {**_DEFAULT_PRIORITIES, **(entry.get("source_priority") or {})}
    return MentorConfig(
        slug=str(entry["slug"]).strip(),
        display_name=str(entry["display_name"]),
        domain_focus=str(entry.get("domain_focus", "")),
        sources=sources,
        source_priority=priorities,
    )


def load_mentors(path: Path | None = None) -> dict[str, MentorConfig]:
    """Read the YAML config and return slug → MentorConfig.

    Raises FileNotFoundError if no config can be located (neither
    user-supplied nor bundled) — callers should surface this with a
    "run `council init` first" hint rather than silently empty-dict.
    """
    p = path or _resolve_config_path()
    if not p.exists():
        raise FileNotFoundError(
            f"Mentor config not found at {p}. "
            f"Run `council init` to scaffold one, or set COUNCIL_CONFIG."
        )
    raw = yaml.safe_load(p.read_text()) or {}
    entries: Iterable[dict] = raw.get("mentors") or []
    out: dict[str, MentorConfig] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            log.warning("Skipping non-dict mentor entry: %r", entry)
            continue
        try:
            cfg = _parse_one(entry)
        except ValueError as exc:
            log.error("Invalid mentor entry, skipping: %s", exc)
            continue
        if cfg.slug in out:
            log.warning("Duplicate mentor slug %r — keeping first", cfg.slug)
            continue
        out[cfg.slug] = cfg
    return out


# Lazily-initialized module-level cache used by code paths that
# expect the old MENTORS dict shape. Re-reads on every `load_mentors()`
# call so YAML edits land without an import reload.
class _MentorsProxy(dict):
    def __getitem__(self, key):
        return load_mentors()[key]

    def __contains__(self, key):
        try:
            return key in load_mentors()
        except FileNotFoundError:
            return False

    def __iter__(self):
        try:
            return iter(load_mentors())
        except FileNotFoundError:
            return iter(())

    def keys(self):
        try:
            return load_mentors().keys()
        except FileNotFoundError:
            return {}.keys()

    def values(self):
        try:
            return load_mentors().values()
        except FileNotFoundError:
            return {}.values()

    def items(self):
        try:
            return load_mentors().items()
        except FileNotFoundError:
            return {}.items()

    def get(self, key, default=None):
        try:
            return load_mentors().get(key, default)
        except FileNotFoundError:
            return default


MENTORS: dict[str, MentorConfig] = _MentorsProxy()
