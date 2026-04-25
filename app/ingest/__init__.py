"""Ingestion pipeline package.

Phase 2, Step 1 — scaffold only. Dataclasses that move data between
stages live here so sources, chunker, dedupe, and db can import from
a single neutral location without pulling in each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawItem:
    """A single unit fetched from a source, before chunking.

    `body` is the raw text payload. `metadata` is source-specific
    (conversation_id for tweets, episode/speaker for transcripts, h2
    outline for blog posts, etc.) and opaque to the chunker contract —
    each strategy reads the keys it knows about.
    """

    source_type: str          # "twitter" | "substack" | "newsletter" | "blog" | "youtube"
    source_url: str
    date: str                 # ISO-8601 date or datetime string
    title: str | None         # None for sources that have no title (e.g. tweets)
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A post-chunked, dedup-hashed unit ready to insert into a
    per-mentor SQLite DB. `content_hash` is the dedupe key."""

    mentor_slug: str
    source_url: str
    source_type: str
    date: str
    speaker: str | None
    text: str
    content_hash: str
    # 1 = tweet, 2 = podcast, 3 = canonical (essay / newsletter).
    # Phase 5 retrieval reads this to boost canonical sources.
    source_priority: int = 1
