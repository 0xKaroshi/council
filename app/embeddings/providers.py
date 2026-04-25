"""OpenAI embeddings via the official async SDK.

`OpenAIEmbedder.embed()` takes a pre-chunked list of strings (bounded
by `batch_size`) and returns a list of 1536-dim vectors plus the token
count reported by the API. Callers are responsible for slicing the
full corpus into `batch_size`-sized pieces; we keep the embedder
stateless across batches so retries don't accumulate partial state.

Retry policy: up to `max_retries` extra attempts on 429 / 5xx /
connection errors, with exponential backoff capped at 30 s. Other
OpenAI errors (400s, auth) bubble immediately — retrying won't help.

Cost: `text-embedding-3-small` is billed at $0.02 per 1M input tokens
(April 2026 pricing). The stats block surfaces the running total so
the CLI can show it in per-batch and final summary lines.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_BATCH_SIZE = 100
_COST_PER_1M_TOKENS = 0.02  # USD, text-embedding-3-small, April 2026
# OpenAI embedding endpoints reject inputs >8192 tokens with HTTP 400.
# We truncate to a slightly-conservative ceiling so oversized chunks
# (rare structural quirk in long-form blog posts) embed instead of
# breaking the whole batch.
_MAX_INPUT_TOKENS = 8000


@dataclass
class EmbedderStats:
    batches: int = 0
    vectors: int = 0
    tokens_embedded: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        return (self.tokens_embedded / 1_000_000) * _COST_PER_1M_TOKENS


@dataclass
class EmbeddingBatch:
    vectors: list[list[float]]
    tokens: int


class OpenAIEmbedder:
    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = _DEFAULT_MODEL,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_retries: int = 5,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if client is None and not api_key:
            raise ValueError("OpenAIEmbedder requires an api_key or a client")
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(api_key=api_key)
        self.stats = EmbedderStats()

    async def embed(self, texts: list[str]) -> EmbeddingBatch:
        """Embed a single batch. Caller must ensure len(texts) <= batch_size."""
        if not texts:
            return EmbeddingBatch(vectors=[], tokens=0)

        texts = [_truncate_to_token_limit(t) for t in texts]
        resp = await self._create_with_retry(texts)
        vectors = [item.embedding for item in resp.data]
        tokens = getattr(resp.usage, "total_tokens", 0) if resp.usage else 0

        self.stats.batches += 1
        self.stats.vectors += len(vectors)
        self.stats.tokens_embedded += tokens
        log.info(
            "embed: batch=%d vectors=%d tokens=%d cum_tokens=%d est_cost=$%.4f",
            self.stats.batches,
            len(vectors),
            tokens,
            self.stats.tokens_embedded,
            self.stats.estimated_cost_usd,
        )
        return EmbeddingBatch(vectors=vectors, tokens=tokens)

    async def _create_with_retry(self, texts: list[str]):
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
            except (RateLimitError, InternalServerError, APIConnectionError) as e:
                last_exc = e
                if attempt >= self.max_retries:
                    raise
                delay = min(30.0, 2 ** attempt)
                log.warning(
                    "embed: %s on attempt %d/%d; retrying in %.1fs",
                    type(e).__name__, attempt + 1, self.max_retries, delay,
                )
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

_TRUNCATION_ENC = None  # lazy: only built when we actually need to truncate


def _truncate_to_token_limit(text: str, max_tokens: int = _MAX_INPUT_TOKENS) -> str:
    """Trim `text` to at most `max_tokens` cl100k_base tokens. Used as
    a defensive guard against oversized chunks slipping through the
    chunker — OpenAI rejects the whole batch otherwise."""
    global _TRUNCATION_ENC
    if _TRUNCATION_ENC is None:
        import tiktoken
        _TRUNCATION_ENC = tiktoken.get_encoding("cl100k_base")
    tokens = _TRUNCATION_ENC.encode(text)
    if len(tokens) <= max_tokens:
        return text
    log.warning(
        "embed: truncating oversized chunk from %d → %d tokens",
        len(tokens), max_tokens,
    )
    return _TRUNCATION_ENC.decode(tokens[:max_tokens])
