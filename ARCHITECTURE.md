# Architecture

This document covers the technical shape of council. If you just
want to use it, `README.md` and `INSTALL.md` are enough; come
back here when something surprises you or you're extending it.

## Top-level shape

```
   config/mentors.yaml ──┐
                        │
   .env ─────────────────┼──> app.config.settings
                        │
                        ▼
                  app/cli.py (Click)        ── self-contained CLI
                        │
            ┌───────────┼─────────────┐
            ▼           ▼             ▼
        ingest()    embed()       retrieve()
            │           │             │
            ▼           ▼             ▼
         per-mentor SQLite DB         search() / council_retrieve()
         (chunks + chunks_vec + FTS)         │
                                             ▼
                                       formatted text → stdout
                                                     or → MCP client

   Optional second surface:  app/main.py  (FastAPI + FastMCP)
                             on top of the same retrieval core.
```

The CLI and the MCP server share every module under `app/`.
Anything queryable from one is queryable from the other; the
difference is just the transport.

## Storage layout

One SQLite file per mentor at `data/mentors/<slug>.db`. Three
tables in each:

- **`chunks`** — the canonical row, one per retrievable unit.
  Columns include `id`, `text`, `source_url`, `source_type`,
  `date` (ISO 8601), `content_hash` (SHA-256 over normalized
  text — the dedup key), and `source_priority` (1 = tweet,
  2 = podcast / long-form, 3 = canonical essay).
- **`chunks_fts`** — FTS5 virtual table mirroring `chunks.text`.
  Triggers keep it in sync on insert / update / delete.
- **`chunks_vec`** — `sqlite-vec` `vec0` virtual table holding
  one 1536-dim float vector per chunk. Created lazily on first
  Phase 3 use; missing chunks are detected via left-join in
  `app/embeddings/store.get_missing_chunks`.

The chunk schema is explicitly versioned via a `meta` table
(key = "schema_version") so future migrations can be
idempotent. `app/ingest/db.open_mentor_db` runs schema and
extension setup on every connect.

## Ingestion

```
RawItem  ── chunker ──▶ Chunk(s) ── upsert_chunks ──▶ chunks table
   ▲                                                    │
   │                                                    ▼
 Source.fetch()                                  content_hash dedup
   │                                                    │
 (twitter / blog / substack / ...)                      ▼
                                                same hash → skip
```

Each source type is an `app/ingest/sources/<type>.py` module that
implements `Source.fetch() -> AsyncIterator[RawItem]`. The Twitter
source paginates via TwitterAPI.io with cursor persistence (so
re-runs resume from the last `next_cursor`); the blog source
runs a discovery cascade (RSS → sitemap → sitemap-index) with
per-domain HTML caching and 1-second polite delays.

Chunkers (`chunk_tweets`, `chunk_blog_paragraphs`) are pure
functions: in → list of `Chunk`. Tweet threads chunk as one
unit (the conversation, not the individual tweet); blog posts
chunk paragraph-aware to a 700-token target with 100-token
overlap, prefixing each chunk with its post title and h2 so the
LLM gets the section context for free.

`source_priority` is set by the chunker based on source type:
canonical essays land at 3, tweets at 1. The retrieval boost
applies to anything at priority 3 — see "Retrieval" below.

## Embedding

`app/embeddings/providers.py` wraps OpenAI's
`text-embedding-3-small`. 100-chunk batches, exponential
backoff on 429/5xx, per-batch logging. Cost calculation uses the
documented $0.02 / 1M tokens rate; `OpenAIEmbedder.stats`
exposes the running totals so `scripts/embed.py` can print a
summary.

`app/embeddings/store.py` handles persistence:
`upsert_embeddings(db, [(chunk_id, vector), ...])` is
idempotent and transactional; `get_missing_chunks(db, limit)`
returns only the chunks without a vec0 row, making `council
embed --all` resume-safe across interrupted runs.

For very long blog posts, `_truncate_to_token_limit` defensively
caps any single chunk at 8000 cl100k tokens — the OpenAI
endpoint rejects payloads above 8192, and a single oversized
chunk can fail an entire 100-chunk batch.

## Retrieval

```
              query string
                  │
        normalize_query (whitespace + casing)
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
   embed_query           sanitize_fts5_query
   (OpenAI ~250ms)       (quote each token)
       │                     │
       ▼                     ▼
   search_vector         search_bm25
   (sqlite-vec KNN)      (FTS5 MATCH)
       │                     │
       └──────────┬──────────┘
                  ▼
        rrf_fuse(bm25, vec, k_rrf=60)
                  │
                  ▼
         (optional) recency_bias
                  │
                  ▼
       (optional) source_priority_boost  × 1.2 for priority 3
                  │
                  ▼
        top-k Snippets, hydrated from chunks
```

**Reciprocal Rank Fusion** is the load-bearing piece. Each rank-1
hit contributes `1 / (k_rrf + 1) = 1/61` to that doc's score;
overlapping documents (in both lists) accumulate scores from
both ranked lists and beat singletons cleanly.

**Source priority boost** is an optional 1.2× multiplier on
canonical-essay chunks (priority 3). Most user queries get
better signal from a few-paragraph essay snippet than from an
adjacent tweet; the boost lifts essays without erasing tweet
relevance.

**Recency bias** is a 0–10% linear boost over a 365-day window,
relative to the corpus's newest chunk. Off by default — opt in
with `--recency-bias` on `ask` / `convene`.

The query embedding cache is a deliberate non-feature. A typical
council session asks one or two questions; the cache complexity
isn't earned. If you start hammering the same query, embed the
result yourself and reuse it across runs.

## Council mode

```
     question string
         │
   embed_query (ONCE)
         │
         ▼
     shared 1536-dim vector
         │
   ┌─────┼──────┬──────────┐
   ▼     ▼      ▼          ▼
retrieve(M1, vec)  retrieve(M2, vec)  retrieve(M3, vec)
   ▲     ▲      ▲          ▲
   │     │      │          │
   └─────┴──────┴──────────┘
         asyncio.gather (genuinely parallel)
                │
                ▼
        labeled sections + mentor-prefixed citations
```

Two performance properties:

1. **One embed call, not N.** The OpenAI round-trip dominates
   wall time (~250ms vs ~10ms for the SQLite reads). The shared
   embedding makes that cost amortize across mentors.
2. **Parallel retrieve.** The three (or N) mentor retrievals run
   concurrently; total wall time floor is `embed_ms + max(per_mentor_ms)`,
   not `N * (embed_ms + per_mentor_ms)`.

Empirically: against three mentors with ~200ms-mocked retrievals,
council_retrieve completes in ~210ms total (vs ~600ms if it
were serial) — a **3× speedup** that scales with the mentor count.

**Per-mentor failure isolation.** If one mentor's `retrieve()`
raises (DB missing, sqlite-vec extension not loaded, corrupted
chunks_vec), that section becomes a one-line "<mentor>
unavailable" notice naming the survivors; the others render
normally; the tool returns a string, never raises. Only the
all-three-fail case returns a single "Council unavailable"
message.

The output is intentionally **just snippet sets**. The synthesis
— disagreement detection, recommendation, your-voice rewrite —
is the calling LLM's job. That's the line between this tool
(a retrieval surface) and a synthesis layer (the calling
agent's job).

## The user_context layer

`get_user_context` reads markdown files from a configurable
directory (default `./user_context/`) and returns them
concatenated. The set of files is dynamic — anything ending in
`.md` (and not `.example.md`) is loaded.

This is the seam where *your situation* enters the loop. The
mentor archives carry generic operator wisdom; your context
files carry the specific constraints, audience, brand, and
open questions that make a generic answer actionable. The
calling LLM is expected to read both — context first, then
mentor snippets — so the answer references your real situation
inline (`[brand.md]`, `[constraints.md]`).

Empty / missing files surface as explicit `(empty: ...)` /
`(missing: ...)` markers rather than silent omissions, so the
LLM can tell "context not provided" apart from "context
unavailable."

## MCP server (optional)

`app/main.py` is the FastAPI entry point; `app/mcp_server.py`
is the `FastMCP` instance with the four tools registered
(`ping`, `search`, `council_retrieve`, `get_user_context`).
The CLI never needs this; it exists for users who want to
expose their council over MCP for Claude Desktop / Claude.ai
integration.

OAuth is intentionally out of scope for v1. `MCP_AUTH_MODE=none`
ships with the FastAPI server with no auth — only safe behind
your own reverse proxy or on localhost. Adding OAuth back is a
documented extension point in `CONTRIBUTING.md`.

## What council is not

- Not a vector database. SQLite + sqlite-vec is the storage; the
  vector table is a query target, not an addressable service.
- Not a synthesis layer. Council retrieves; the calling LLM
  synthesizes. Building synthesis into the tool would make every
  output less inspectable.
- Not a multi-tenant system. Each install belongs to one human.
  Multi-user is a different shape and would warrant a different
  project, not a configuration flag here.
