"""Embedding CLI.

Usage:
    python -m scripts.embed <slug> [--limit N] [--dry-run]

Resume-safe: re-running picks up exactly the chunks that don't yet
have a vector in `chunks_vec`. `--dry-run` stops after estimating
cost + missing-chunk count so you can sanity-check before spending.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from app.ingest.mentors import MENTORS, MentorConfig


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts.embed",
        description="Embed mentor chunks into the sqlite-vec chunks_vec table.",
    )
    parser.add_argument("mentor", help=f"Mentor slug (one of: {', '.join(MENTORS)})")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max chunks to embed this run. Default: all missing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report chunk count + estimated cost; do not call OpenAI.",
    )
    args = parser.parse_args(argv)

    if args.mentor not in MENTORS:
        print(f"unknown mentor: {args.mentor!r}. known: {list(MENTORS)}", file=sys.stderr)
        return 2
    mentor = MENTORS[args.mentor]

    _configure_logging()
    return asyncio.run(_run(mentor, args))


async def _run(mentor: MentorConfig, args: argparse.Namespace) -> int:
    # Heavy imports are deferred so `--help` and unknown-mentor errors
    # work without needing openai / sqlite-vec / tiktoken installed.
    import tiktoken

    from app.config import settings
    from app.embeddings.providers import OpenAIEmbedder, _COST_PER_1M_TOKENS
    from app.embeddings.store import get_missing_chunks, upsert_embeddings
    from app.ingest.db import open_mentor_db

    enc = tiktoken.get_encoding("cl100k_base")
    db = open_mentor_db(mentor.slug)

    total_chunks: int = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    missing = get_missing_chunks(db, limit=args.limit)
    already = total_chunks - len(missing) if args.limit is None else None

    est_tokens = sum(len(enc.encode(text)) for _, text in missing)
    est_cost = (est_tokens / 1_000_000) * _COST_PER_1M_TOKENS

    header = (
        f"embed: mentor={mentor.slug} chunks_total={total_chunks} "
        f"chunks_missing={len(missing)} "
        f"limit={args.limit if args.limit is not None else '<all>'} "
        f"dry_run={args.dry_run}"
    )
    print(header)
    print(
        f"  estimate: input_tokens~{est_tokens} "
        f"est_cost~${est_cost:.4f} (at ${_COST_PER_1M_TOKENS}/1M tokens)"
    )

    if args.dry_run:
        print(
            f"\ndry-run: would embed {len(missing)} chunks "
            f"for ~${est_cost:.4f} — no API call made."
        )
        db.close()
        return 0

    if not missing:
        print("nothing to embed — all chunks already have vectors")
        db.close()
        return 0

    if not settings.openai_api_key:
        print(
            "OPENAI_API_KEY is not set; populate .env before running "
            "without --dry-run.",
            file=sys.stderr,
        )
        db.close()
        return 2

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        batch_size=100,
    )
    embedded = 0
    start = time.time()
    try:
        for i in range(0, len(missing), embedder.batch_size):
            slice_ = missing[i : i + embedder.batch_size]
            ids = [cid for cid, _ in slice_]
            texts = [txt for _, txt in slice_]
            result = await embedder.embed(texts)
            upsert_embeddings(db, list(zip(ids, result.vectors)))
            embedded += len(ids)
    finally:
        await embedder.aclose()
    wall = time.time() - start

    print("\n=== embed summary ===")
    print(f"  mentor:             {mentor.slug}")
    print(f"  chunks_total:       {total_chunks}")
    print(f"  chunks_embedded:    {embedded}")
    if already is not None:
        print(f"  chunks_skipped:     {already}    (already embedded)")
    print(f"  batches:            {embedder.stats.batches}")
    print(f"  tokens_embedded:    {embedder.stats.tokens_embedded}")
    print(f"  estimated_cost:     ${embedder.stats.estimated_cost_usd:.4f}")
    print(f"  wall_time_seconds:  {wall:.1f}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
