"""Source abstract base class.

Every concrete source (Twitter, Substack, blog, YouTube, newsletter)
implements fetch() as an **async** generator yielding RawItem. Async
is the right default because every production source is network-bound;
paginated APIs want to issue requests and await them without blocking
the event loop.

The base class does NOT prescribe a constructor signature. Each source
needs different construction inputs (a twitter handle + bearer key vs.
a blog base URL vs. a YouTube channel ID), so subclasses define their
own `__init__`. The ingest CLI is the only call site, and it knows
how to wire each source's specific arguments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from app.ingest import RawItem


class Source(ABC):
    """Abstract source. Subclasses produce RawItem streams."""

    source_type: str = ""

    @abstractmethod
    def fetch(self) -> AsyncIterator[RawItem]:
        """Yield RawItem objects from this source.

        Implementations are expected to be async generators:

            async def fetch(self) -> AsyncIterator[RawItem]:
                async for item in ...:
                    yield item

        The return-type annotation is ``AsyncIterator[RawItem]``; the
        actual runtime object is an async generator, which satisfies
        that protocol.
        """
        raise NotImplementedError
