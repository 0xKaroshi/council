# Configuring mentors

The mentor lineup lives in `config/mentors.yaml` (overridable via
the `COUNCIL_CONFIG` env var). Edit the file, then re-run
`council ingest <slug>` and `council embed <slug>` to materialize
the new mentor's archive.

## Schema

```yaml
mentors:
  - slug: <required, kebab-or-snake-case>
    display_name: <required, human-friendly>
    domain_focus: <optional, one-line LLM hint>
    sources:
      - type: <blog | substack | twitter | newsletter | youtube>
        # type-specific fields below
    source_priority:
      blog: 3
      twitter: 1
```

### Required fields

- **`slug`** — the per-mentor SQLite filename (becomes
  `data/mentors/<slug>.db`) and the CLI argument. Stick to ASCII
  + underscores; kebab-case is fine but snake_case is what the
  rest of the codebase uses.
- **`display_name`** — what shows in `council list-mentors` and
  in the `convene` section headers.

### Optional fields

- **`domain_focus`** — one short sentence telling the LLM router
  what this mentor's archive is best at. Surfaces in the `search`
  and `convene` tool descriptions so the LLM can pick the right
  mentor for a question.
- **`sources`** — the ingestion sources for this mentor. A
  mentor with no sources is a name in the config; ingest will
  skip it (useful when staging a new entry before pulling
  content).
- **`source_priority`** — per-source-type priority for the
  retrieval boost. Defaults: blog/substack/newsletter = 3,
  podcast/youtube = 2, twitter = 1. Chunks with priority 3 get a
  1.2× score multiplier when `source_priority_boost=True` (the
  default in `search` and `convene`).

## Source types

### `blog`

```yaml
- type: blog
  domain: example.com                              # required
  rss_url: https://otherhost.com/feed.xml          # optional override
```

The blog source runs a discovery cascade:

  1. The explicit `rss_url` if provided (this is how Paul
     Graham's setup works — paulgraham.com has no native feed,
     so we point at Aaron Swartz's `pgessays.rss` mirror).
  2. `https://{domain}/index.xml` (Hugo / static-site RSS)
  3. `https://{domain}/feed` (Substack convention)
  4. `https://{domain}/feed/` (WordPress convention)
  5. `https://{domain}/rss.xml`
  6. `https://{domain}/sitemap.xml`
  7. `https://{domain}/sitemap_index.xml`

Discovery stops at the first endpoint that returns ≥30 URLs.
For sparse RSS feeds (Substack truncates to 20 most recent),
results are unioned with subsequent probes — `/feed` + `/sitemap.xml`
in concert often gives full archive coverage.

Per-post HTML is cached at `data/cache/blog/{domain}/{md5}.html`
so re-ingestion is free unless you pass `--refresh`.

### `substack`

```yaml
- type: substack
  domain: visualizevalue.substack.com
```

Functionally identical to `blog` for now (Substack is just a
hosted blog with predictable RSS + sitemap layout). Listed as a
separate type so the mentor config reads naturally and source
priority can be tuned per-type if you want Substack and "real"
blog content weighted differently.

### `twitter`

```yaml
- type: twitter
  handle: paulg                                    # without the @
```

Requires `TWITTER_API_KEY` set in `.env` (sign up at
https://twitterapi.io). Pricing: ~$0.15 per 1,000 tweets;
the ingest CLI accepts `--max-tweets N` and `--max-cost-usd X`
caps so you can bound the run.

Threads are reassembled by `conversationId` and chunked as one
unit (one chunk per thread, not per tweet). The reply filter
keeps originals + self-replies (i.e., the body of the mentor's
own threads) and drops conversational replies to other people.

The Twitter cursor is persisted in the meta table so re-running
`council ingest <slug> --source twitter` resumes from the last
fetched tweet.

### `newsletter`, `youtube`

Both source types are stubbed in the codebase. Implementing
them is straightforward — see `app/ingest/sources/blog.py` for
the shape (async generator yielding `RawItem`). Contributions
welcome.

## Worked example: adding Sam Altman

Suppose you want to add Sam Altman as a 4th mentor.

His archive: blog at `blog.samaltman.com` (WordPress, has RSS).
Twitter handle: `@sama`.

```yaml
# Append to config/mentors.yaml
  - slug: sama
    display_name: Sam Altman
    domain_focus: "AI strategy, fund-raising, leadership at scale"
    sources:
      - type: blog
        domain: blog.samaltman.com
      - type: twitter
        handle: sama
    source_priority:
      blog: 3
      twitter: 1
```

```bash
# Pull + embed
council ingest sama --source blog --max-posts 100
council ingest sama --source twitter --max-tweets 1000 --max-cost-usd 0.50
council embed sama

# Verify
council status
council ask sama "how do you decide which company to fund?"

# Now sama is included in council convene automatically
council convene "should I take VC funding?"
```

No code changes needed.

## Removing or disabling a mentor

To temporarily remove a mentor from `convene` without deleting
their archive:

```yaml
# Comment them out
#  - slug: paulgraham
#    display_name: Paul Graham
#    ...
```

The `data/mentors/<slug>.db` file stays on disk (so you don't
lose the corpus) but the mentor stops appearing in
`list-mentors` / `convene` / `search`. Uncomment to bring back.

To delete entirely: `rm data/mentors/<slug>.db` after removing
from the YAML.

## Common pitfalls

- **Substack RSS truncation.** Substack's `/feed` only returns
  the 20 most recent posts. The discovery cascade unions with
  `/sitemap.xml` to get full coverage, but verify with `council
  status` after first ingest.
- **Cloudflare-protected blogs.** Some blogs (especially Substack
  setups behind their own custom domain) reject bot user-agents.
  Council uses an explicit `User-Agent: council-bot/0.1
  (research; ...)`. If discovery fails with 403s, the blog
  doesn't permit machine readers and you'd need to accept
  manual export instead.
- **Twitter handle changes.** If a mentor changes their handle,
  the cursor in the `meta` table becomes stale. Pass `--restart`
  to start fresh, or update the YAML's `handle` field.
- **Empty corpora.** A slug with no source entries is valid YAML
  but produces an empty DB. `council status` shows 0 chunks;
  `convene` skips that mentor's section with "No matching
  snippets found".
