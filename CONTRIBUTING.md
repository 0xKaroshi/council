# Contributing

Bug reports, source-type contributions, and small focused PRs
welcome. Feature requests are evaluated against one criterion:

> Is this generic-shaped, or does it bake one user's preferences
> into the kernel?

Council is deliberately a thin reusable kernel — anything that
makes one specific deployment nicer at the cost of everyone
else's flexibility belongs in a fork.

## Ways to contribute

### Adding a new source type

The cleanest contribution. Today council ships:

- `twitter` (TwitterAPI.io)
- `blog` / `substack` (RSS / sitemap discovery + trafilatura)

Stubbed but unimplemented:

- `newsletter`
- `youtube`
- `podcast`

The shape is straightforward — see
`app/ingest/sources/blog.py` for a working reference. You need:

1. A new module under `app/ingest/sources/<type>.py` exporting a
   class that subclasses `Source` and implements
   `async def fetch(self) -> AsyncIterator[RawItem]`.
2. A chunker function in `app/ingest/chunker.py` if your source
   produces content shaped differently from blog or twitter.
3. CLI plumbing in `scripts/ingest.py` (just one new dispatch
   case in `main()`).
4. Tests in `tests/test_<type>_source.py`.

The existing tests (`test_twitter_source.py`,
`test_blog_source.py`) are good templates — both mock the
network entirely so you can run them on CI without real API keys.

### Adding a new mentor

Don't. Adding mentors is a config-only change (`config/mentors.yaml`)
and shouldn't go in PRs against the kernel. Your local mentor
lineup is yours; ship it in a fork or a sidecar config repo.

The exception: improving the **example mentors** bundled in
`examples/`. If Paul Graham's recent essays aren't ingesting
cleanly, or Naval's nav.al feed format changes, those are real
issues worth a PR.

### Adding OAuth to the MCP server

`app/main.py` ships without OAuth in v1. The original codebase
(mentor-brain, the private project this fork is generalized
from) had a working OAuth 2.1 / DCR / PKCE implementation
gated by an admin password. Bringing that back as an opt-in
(`MCP_AUTH_MODE=oauth`) is a documented extension point.

Sketch:

1. Re-introduce `app/auth/` with `oauth.py`, `middleware.py`,
   and `storage.py` (SQLite-backed token store).
2. Wire `AuthMiddleware` into `app/main.py` conditionally based
   on `settings.mcp_auth_mode`.
3. Add `OAUTH_SECRET_KEY` and `ADMIN_PASSWORD` to
   `app.config.Settings`.
4. Tests for the OAuth round-trip (token issuance, refresh,
   revocation).
5. Update `INSTALL.md` to document the OAuth deployment path.

The reason it didn't ship in v1: most users want CLI mode, and
the MCP-server-without-auth-behind-your-own-proxy path is
sufficient for those who want MCP. Adding OAuth complicates the
default install for everyone to serve the minority.

### Improving retrieval

The retrieval pipeline (BM25 + dense vector + RRF) is
deliberately conservative. Things that have been considered
but not shipped:

- **HyDE (hypothetical document embeddings).** A small LLM
  generates a hypothetical answer to the query; you embed that
  and search with it. Often improves recall for abstract
  questions. Worth experimenting with as an opt-in flag.
- **Cross-encoder re-ranking.** Re-rank the top-k with a cross-
  encoder (e.g. `sentence-transformers/ms-marco-MiniLM-L-6-v2`)
  for a precision boost. Adds a model dependency and ~50ms per
  query.
- **Query expansion.** Synonym expansion via WordNet or an LLM
  call. Diminishing returns once you have RRF + dense vectors.

PRs welcome for any of these as opt-in flags. Keep them off by
default — the conservative pipeline works well enough that
adding complexity needs an empirical justification.

## Running the test suite

```bash
pip install -e ".[dev]"
pytest tests/
```

The full suite runs in under 2 seconds and requires no network
or API keys (everything is mocked). If a test starts requiring
the network, that's a regression — the test runner has no
internet on CI.

## Style + format

`pyproject.toml` configures `ruff` with line length 100 and a
small lint set (`E`, `F`, `W`, `I`, `B`). Run:

```bash
ruff check .
ruff format .
```

Type hints encouraged but not required. Match existing module
style — terse docstrings, why-not-what comments, no
multi-paragraph block comments.

## Issue triage

- **Bugs**: please include the council version, Python version,
  the exact command that failed, and the full error output.
- **Feature requests**: please include the use case (not just
  the feature). "I want council to do X" is harder to evaluate
  than "I have a use case where Y happens and council can't
  handle it because Z."
- **Mentor-config questions**: not a bug. Read
  `CONFIGURING_MENTORS.md`. If it's still unclear, that's a
  docs issue — open it as such.

## What this project is NOT

- **A multi-tenant SaaS.** One install per human. Multi-user
  would be a different shape and would warrant a different
  project, not a flag here.
- **A chatbot.** The CLI / MCP surfaces are tools the calling
  agent (Claude, your own Python script, whatever) uses. The
  agent does the conversation; council does retrieval.
- **A managed service.** No hosted version, no support tiers, no
  pricing page. Run it yourself.

## License

Apache 2.0 — see `LICENSE`. Contributions land under the same
license; by submitting a PR you assert you have the right to
contribute the code under that license.
