"""Newsletter source (generic) — stub. Phase 2, Step 1 scaffold only."""
from __future__ import annotations

from typing import AsyncIterator

from app.ingest import RawItem
from app.ingest.sources.base import Source


class NewsletterSource(Source):
    source_type = "newsletter"

    async def fetch(self) -> AsyncIterator[RawItem]:
        raise NotImplementedError("NewsletterSource is a Step 2+ stub")
        yield  # pragma: no cover
