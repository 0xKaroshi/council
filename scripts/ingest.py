"""Ingestion CLI.

Twitter is live (Step 2). Other sources stay stubbed until they ship.

Usage:
    python -m scripts.ingest <slug> --source twitter [--since 2023-01-01]
                                                      [--dry-run]
                                                      [--restart]
    python -m scripts.ingest <slug> --source <other>   # prints stub message

`--dry-run` fetches, chunks, and prints per-thread stats without
touching the mentor DB. `--restart` ignores any cursor stored in
the meta table and paginates from the top.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone

from app.ingest.mentors import MENTORS, MentorConfig

_SOURCE_TYPES = ("twitter", "substack", "newsletter", "blog", "youtube", "all")
_DEFAULT_SINCE = "2023-01-01"


def _parse_since(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _cursor_key(handle: str) -> str:
    return f"twitter_cursor_{handle}"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts.ingest",
        description="Ingest a mentor's corpus from one source type.",
    )
    parser.add_argument("mentor", help=f"Mentor slug (one of: {', '.join(MENTORS)})")
    parser.add_argument(
        "--source",
        required=True,
        choices=_SOURCE_TYPES,
        help="Which source to ingest from.",
    )
    parser.add_argument(
        "--since",
        default=_DEFAULT_SINCE,
        help=f"ISO-8601 cutoff; tweets older than this are skipped. Default: {_DEFAULT_SINCE}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and chunk, but do not write to the mentor DB.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore any stored pagination cursor; start from the top.",
    )
    parser.add_argument(
        "--max-tweets",
        type=int,
        default=20000,
        help=(
            "Safety cap. Aborts the run when this many tweets have been "
            "fetched, preserving everything yielded so far. Default: 20000."
        ),
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help=(
            "Optional cost ceiling in USD. Aborts the run when the estimated "
            "spend reaches this. Default: off."
        ),
    )
    # --- blog-source flags -------------------------------------------------
    parser.add_argument(
        "--domain",
        action="append",
        default=None,
        help=(
            "Blog domain to ingest (repeatable). Only meaningful with "
            "--source blog or --source all. If omitted, every domain in "
            "the mentor's `blog_domains` is processed."
        ),
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=500,
        help="Safety cap for blog post count per domain. Default: 500.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bust the per-post HTML cache and re-fetch every URL.",
    )
    args = parser.parse_args(argv)

    if args.mentor not in MENTORS:
        print(f"unknown mentor: {args.mentor!r}. known: {list(MENTORS)}", file=sys.stderr)
        return 2
    mentor = MENTORS[args.mentor]

    _configure_logging()

    if args.source == "twitter":
        return _dispatch_twitter(mentor, args)

    if args.source == "blog":
        return _dispatch_blog(mentor, args)

    if args.source == "all":
        # Blog first (per spec), then twitter. Either may abort with
        # non-zero; aggregate by surfacing the worst non-zero exit.
        rc_blog = _dispatch_blog(mentor, args)
        rc_tw = _dispatch_twitter(mentor, args)
        return rc_blog or rc_tw

    print(
        f"would ingest {args.mentor}/{args.source}, "
        f"but that source is still a stub"
    )
    return 0


def _dispatch_twitter(mentor: MentorConfig, args: argparse.Namespace) -> int:
    from app.config import settings

    if not mentor.twitter_handle:
        print(
            f"mentor {mentor.slug!r} has no twitter_handle configured",
            file=sys.stderr,
        )
        return 2
    if not settings.twitter_api_key:
        print(
            "TWITTER_API_KEY is not set; add it to .env before "
            "running --source twitter (sign up at https://twitterapi.io).",
            file=sys.stderr,
        )
        return 2
    return asyncio.run(_run_twitter(mentor, args))


def _dispatch_blog(mentor: MentorConfig, args: argparse.Namespace) -> int:
    # Build (domain, rss_url) pairs from the mentor's configured blog
    # sources so per-source overrides (rss_url, etc.) propagate.
    explicit = list(args.domain) if args.domain else []
    pairs: list[tuple[str, str | None]] = []
    if explicit:
        pairs = [(d, None) for d in explicit]
    else:
        for s in mentor.sources:
            if s.type in ("blog", "substack") and s.domain:
                pairs.append((s.domain, s.rss_url))
        if not pairs and mentor.blog_url:
            from urllib.parse import urlparse
            d = urlparse(mentor.blog_url).netloc
            if d:
                pairs.append((d, None))
    if not pairs:
        print(
            f"mentor {mentor.slug!r} has no blog sources configured. "
            "Pass --domain explicitly.",
            file=sys.stderr,
        )
        return 2

    final_rc = 0
    for d, rss in pairs:
        print(f"\n=== blog ingest: {mentor.slug} / {d} ===")
        rc = asyncio.run(_run_blog(mentor, args, domain=d, rss_url=rss))
        if rc and not final_rc:
            final_rc = rc
    return final_rc


async def _run_twitter(mentor: MentorConfig, args: argparse.Namespace) -> int:
    # Heavy imports happen here so neither stub-source invocations nor
    # fail-fast config errors need them installed.
    import tiktoken

    from app.config import settings
    from app.ingest import Chunk, RawItem
    from app.ingest.chunker import chunk_tweets
    from app.ingest.db import get_meta, open_mentor_db, set_meta, upsert_chunks
    from app.ingest.sources.twitter import (
        BatchInfo,
        TwitterBudgetExceeded,
        TwitterSource,
        TwitterSourceStats,
    )

    enc = tiktoken.get_encoding("cl100k_base")

    since = _parse_since(args.since)
    handle = mentor.twitter_handle

    db = open_mentor_db(mentor.slug)
    stored_cursor = None if args.restart else get_meta(db, _cursor_key(handle))

    async def _persist_cursor(info: BatchInfo) -> None:
        if args.dry_run:
            return
        if info.cursor:
            set_meta(db, _cursor_key(handle), info.cursor)

    source = TwitterSource(
        handle=handle,
        since=since,
        api_key=settings.twitter_api_key,
        cursor=stored_cursor,
        on_batch=_persist_cursor,
        max_tweets=args.max_tweets,
        max_cost_usd=args.max_cost_usd,
    )

    raw_items: list[RawItem] = []
    total_tokens = 0
    aborted_reason: str | None = None

    cost_cap_str = (
        "<off>" if args.max_cost_usd is None else f"${args.max_cost_usd:.2f}"
    )
    print(
        f"ingest: mentor={mentor.slug} source=twitter handle=@{handle} "
        f"since={since.isoformat()} dry_run={args.dry_run} "
        f"resume_cursor={'<set>' if stored_cursor else '<none>'} "
        f"max_tweets={args.max_tweets} max_cost_usd={cost_cap_str}"
    )

    try:
        async for item in source.fetch():
            raw_items.append(item)
            meta = item.metadata
            tokens = len(enc.encode(item.body))
            total_tokens += tokens
            print(
                f"  thread root={meta['root_tweet_id']} "
                f"tweets={meta['thread_length']} tokens={tokens} "
                f"likes={meta['like_count_total']} "
                f"rts={meta['retweet_count_total']} "
                f"date={item.date}"
            )
    except TwitterBudgetExceeded as exc:
        aborted_reason = str(exc)

    chunks: list[Chunk] = chunk_tweets(raw_items, mentor_slug=mentor.slug)

    if args.dry_run:
        inserted = 0
        print(
            f"dry-run: would upsert {len(chunks)} chunks "
            f"(skipped {mentor.slug}.db write)"
        )
    else:
        inserted = upsert_chunks(db, chunks)
        set_meta(db, "last_ingested_at", datetime.now(timezone.utc).isoformat())

    _print_summary(source.stats, chunk_count=len(chunks), inserted=inserted,
                   total_tokens=total_tokens)
    db.close()

    if aborted_reason is not None:
        print(f"\nABORT: {aborted_reason}", file=sys.stderr)
        print(
            "If this is intentional, raise --max-tweets or --max-cost-usd "
            "and re-run.",
            file=sys.stderr,
        )
        return 3
    return 0


def _print_summary(
    stats: TwitterSourceStats,
    *,
    chunk_count: int,
    inserted: int,
    total_tokens: int,
) -> None:
    print("\n=== ingest summary ===")
    print(f"  api_calls:          {stats.api_calls}")
    print(f"  tweets_fetched:     {stats.tweets_fetched}")
    print(f"  replies_dropped:    {stats.replies_dropped}")
    print(f"  tweets_kept:        {stats.tweets_kept}")
    print(f"  threads_emitted:    {stats.threads_emitted}")
    print(f"  chunks_built:       {chunk_count}")
    print(f"  chunks_inserted:    {inserted}")
    print(f"  total_tokens:       {total_tokens}")
    print(f"  estimated_cost:     ${stats.estimated_cost_usd:.4f}")


# ---------------------------------------------------------------------------
# Blog runner
# ---------------------------------------------------------------------------

async def _run_blog(
    mentor: MentorConfig,
    args: argparse.Namespace,
    *,
    domain: str,
    rss_url: str | None = None,
) -> int:
    import tiktoken

    from app.ingest import Chunk, RawItem
    from app.ingest.chunker import chunk_blog_paragraphs
    from app.ingest.db import open_mentor_db, set_meta, upsert_chunks
    from app.ingest.sources.blog import BlogBudgetExceeded, BlogSource

    enc = tiktoken.get_encoding("cl100k_base")
    since = _parse_since(args.since)

    db = open_mentor_db(mentor.slug)

    cost_cap_str = (
        "<off>"
        if args.max_cost_usd is None
        else f"${args.max_cost_usd:.2f}"
    )
    print(
        f"ingest: mentor={mentor.slug} source=blog domain={domain} "
        f"since={since.isoformat()} dry_run={args.dry_run} "
        f"refresh={args.refresh} max_posts={args.max_posts} "
        f"max_cost_usd={cost_cap_str}"
    )

    source = BlogSource(
        domain=domain,
        since=since,
        rss_url=rss_url,
        max_posts=args.max_posts,
        max_cost_usd=args.max_cost_usd,
        refresh=args.refresh,
    )

    raw_items: list[RawItem] = []
    aborted_reason: str | None = None
    try:
        async for item in source.fetch():
            raw_items.append(item)
            n_blocks = len(item.metadata.get("blocks") or [])
            print(
                f"  post date={item.date[:10] if item.date else 'unknown'} "
                f"blocks={n_blocks} title={(item.title or '')[:60]!r} "
                f"url={item.source_url}"
            )
    except BlogBudgetExceeded as exc:
        aborted_reason = str(exc)

    chunks: list[Chunk] = []
    total_tokens = 0
    for item in raw_items:
        for ch in chunk_blog_paragraphs(item, mentor_slug=mentor.slug):
            chunks.append(ch)
            total_tokens += len(enc.encode(ch.text))

    if args.dry_run:
        inserted = 0
        print(
            f"dry-run: would upsert {len(chunks)} chunks across "
            f"{len(raw_items)} posts (skipped {mentor.slug}.db write)"
        )
    else:
        inserted = upsert_chunks(db, chunks)
        set_meta(
            db,
            f"blog_last_ingested_{domain}",
            datetime.now(timezone.utc).isoformat(),
        )

    print("\n=== blog ingest summary ===")
    print(f"  domain:             {domain}")
    print(f"  discovery_method:   {source.stats.discovery_method or '(none)'}")
    print(f"  posts_discovered:   {source.stats.posts_discovered}")
    print(f"  posts_below_since:  {source.stats.posts_below_since}")
    print(f"  posts_emitted:      {source.stats.posts_emitted}")
    print(f"  cache_hits:         {source.stats.cache_hits}")
    print(f"  cache_misses:       {source.stats.cache_misses}")
    print(f"  fetch_errors:       {source.stats.fetch_errors}")
    print(f"  extract_errors:     {source.stats.extract_errors}")
    print(f"  chunks_built:       {len(chunks)}")
    print(f"  chunks_inserted:    {inserted}")
    print(f"  total_tokens:       {total_tokens}")
    db.close()

    if aborted_reason is not None:
        print(f"\nABORT: {aborted_reason}", file=sys.stderr)
        print(
            "If this is intentional, raise --max-posts and re-run.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
