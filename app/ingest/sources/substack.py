"""Substack source — stub. Phase 2, Step 1 scaffold only."""
from __future__ import annotations

from typing import AsyncIterator

from app.ingest import RawItem
from app.ingest.sources.base import Source


class SubstackSource(Source):
    source_type = "substack"

    async def fetch(self) -> AsyncIterator[RawItem]:
        raise NotImplementedError("SubstackSource is a Step 2+ stub")
        yield  # pragma: no cover — makes this an async generator
