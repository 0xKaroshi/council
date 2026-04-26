"""Query normalization + embedding.

`normalize_query` collapses whitespace and strips edges. It's a no-op
otherwise — Phase 4 will hang HyDE (Hypothetical Document Embeddings)
or query rewriting off this seam.

`embed_query` calls OpenAI text-embedding-3-small for a single string
and returns the 1536-dim vector. Reuses `OpenAIEmbedder` so we get the
same exponential-backoff retry behavior as bulk ingest, at zero
duplicated code. The embedder is created/closed per call — query
volume is low enough that the connection overhead is negligible.
"""

from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    if not text:
        return ""
    return _WS_RE.sub(" ", text).strip()


async def embed_query(text: str) -> list[float]:
    """Embed one query string. Raises if OPENAI_API_KEY is not set."""
    from app.config import settings
    from app.embeddings.providers import OpenAIEmbedder

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot embed query for retrieval.")

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        batch_size=1,
    )
    try:
        batch = await embedder.embed([text])
        if not batch.vectors:
            raise RuntimeError("OpenAI returned an empty embedding for the query")
        return batch.vectors[0]
    finally:
        await embedder.aclose()
