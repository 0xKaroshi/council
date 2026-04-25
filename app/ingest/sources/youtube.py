"""YouTube channel source — stub. Phase 2, Step 1 scaffold.

Step 4+ will likely delegate transcript fetching to Contendeo's
existing yt-dlp + Webshare plumbing rather than duplicate it here —
see TODO.md."""
from __future__ import annotations

from typing import AsyncIterator

from app.ingest import RawItem
from app.ingest.sources.base import Source


class YouTubeSource(Source):
    source_type = "youtube"

    async def fetch(self) -> AsyncIterator[RawItem]:
        raise NotImplementedError("YouTubeSource is a Step 2+ stub")
        yield  # pragma: no cover
