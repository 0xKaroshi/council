# council

*Build your own private council from anyone's public archive.*

Council ingests selected operators' public content (blogs,
Substacks, Twitter timelines) into per-mentor SQLite archives,
embeds them, and exposes them through a CLI (or, optionally, an
MCP server) so you can ask any one mentor a question — or
**convene the whole council** on a multi-dimensional decision.

```bash
$ council ask paulgraham "what makes a startup hard?"
$ council convene "should I take VC funding?"
```

The repo ships with 2-3 months of pre-ingested example content
for Paul Graham, Naval Ravikant, and Patrick O'Shaughnessy so you
can verify the install path works before pointing it at your own
mentor lineup.

## Quick start

```bash
git clone https://github.com/<your-username>/council.git
cd council
pip install -e .                   # or: pipx install /path/to/council

# Set OpenAI key (only thing required to query the bundled examples)
echo "OPENAI_API_KEY=sk-..." > .env

# The bundled example archives (Paul Graham, Naval, Patrick) are
# visible immediately — no init or ingest needed:
council status
council ask paulgraham "what makes a startup hard?"
council convene "should I take VC funding?"
```

`council status` shows each mentor with a `(bundled)` tag while
you're querying the shipped example DBs. To customize the lineup
or pull fresh content into your own data directory:

```bash
council init                       # scaffolds config/ + user_context/
$EDITOR config/mentors.yaml        # add/remove mentors
echo "TWITTER_API_KEY=..." >> .env # if any mentor uses twitter

council ingest --all
council embed --all                # status now shows (user) per mentor
```

## Why this exists

Mentor advice is a category that scales unevenly. A blog post
reaches a million people; a one-on-one with the same operator
reaches one. The middle slot — *the conversation you actually
need*, with one specific operator about your specific situation —
isn't economically reachable for most decisions.

The half-solution is a corpus: ingest enough of an operator's
public output that you can query it the way a researcher queries
a primary source. The full solution is a *council* — three or
four operators with non-overlapping lenses, queried in parallel,
their disagreements as informative as their agreements.

Council is the shape of that second solution. Hybrid retrieval
(BM25 + dense vector + Reciprocal Rank Fusion) over per-mentor
SQLite archives means you keep more nuance than a chatbot
trained on top of the same corpus. Per-mentor labelling and
mentor-prefixed citations mean the synthesis tells you *which
operator said what* rather than collapsing them into a generic
voice.

It is not a substitute for the actual conversation. It is the
preparation work you'd do before the conversation, faster.

## Example mentors included

| Slug             | Display name              | Sources                                  |
|------------------|---------------------------|------------------------------------------|
| `paulgraham`     | Paul Graham               | paulgraham.com (essays)                  |
| `naval`          | Naval Ravikant            | nav.al (essays) + @naval (Twitter)       |
| `patrick_oshag`  | Patrick O'Shaughnessy     | joincolossus.com + @patrick_oshag        |

The corpora bundled in `examples/` are 2-3 months of recent
content (~13–203 chunks each, ~6MB per DB). They're enough to
verify the install path and get a feel for `council convene`
output; for serious use, re-ingest with the full archive.

To swap in your own mentors: edit `config/mentors.yaml` (see
`CONFIGURING_MENTORS.md`).

## How it works

**Ingestion.** Per source type (blog, Substack, Twitter), an
async source yields normalized `RawItem`s. A chunker splits
long-form content into 700-token paragraph-aware chunks (with a
title + h2 prefix so retrieved snippets carry their context); for
Twitter, threads are reassembled and chunked as one unit. Each
chunk is content-hashed for idempotent dedup, so re-ingesting an
unchanged corpus is a no-op.

**Embedding.** Chunks not already in `chunks_vec` are batched and
sent to OpenAI's `text-embedding-3-small` (1536-dim, $0.02 per 1M
tokens — a 5,000-chunk archive embeds for well under a dollar).
Vectors live in the same SQLite file as the chunks via
`sqlite-vec`'s vec0 virtual table; no separate vector store, no
network calls during retrieval.

**Retrieval.** Every query runs both BM25 (FTS5) and dense vector
(sqlite-vec KNN) in parallel via `asyncio.gather`, then fuses the
two ranked lists with Reciprocal Rank Fusion (k=60). Optional
`source_priority_boost` multiplies canonical-essay chunks
(priority 3) by 1.2× so essays outrank tweet noise at near-equal
relevance.

**Council mode.** `convene` embeds the query once, then fans the
shared embedding out to every configured mentor's retrieve()
in parallel. Three mentors with ~250ms per retrieve complete in
~250ms total, not 750ms. The output is N labeled sections with
mentor-prefixed citations (`[paulgraham_3]`, `[naval_1]`); the
calling LLM does the synthesis, not the tool.

Deep dive: `ARCHITECTURE.md`.

## Configuration

- `config/mentors.yaml` — the mentor lineup. See `CONFIGURING_MENTORS.md`.
- `user_context/` — your own situation files. See `USER_CONTEXT.md`.
- `.env` — API keys and runtime paths. See `.env.example`.

## Deployment modes

- **Local CLI (default).** Self-contained. Python 3.10+, an
  OpenAI key, optionally a TwitterAPI.io key. No server, no
  Docker, no OAuth. Most users want this.
- **MCP server (advanced).** Expose the council over MCP from
  your own subdomain. Integrate with Claude Desktop, Claude.ai,
  or any MCP client. OAuth is *not* shipped in v1 — for production
  deploys today, put the server behind a reverse proxy that
  handles auth. See `INSTALL.md`.

## License

Apache 2.0. See `LICENSE`.

## Contributing

Bug reports and source-type contributions welcome. Feature
requests are evaluated against the "is this generic-shaped?"
criterion — council is deliberately a thin reusable kernel, not
an opinionated platform. See `CONTRIBUTING.md`.

## Acknowledgements

Built by Karoshi (@0xKaroshi) as a generalized version of a
private decision-support system. The public version ships
generic; the curated version stays private.
