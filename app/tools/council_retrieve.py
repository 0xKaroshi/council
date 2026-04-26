"""council_retrieve — fan-out hybrid retrieval across all configured
mentors in parallel.

Architecture:

  1. Load the mentor config (slug → MentorConfig).
  2. Embed the query once via OpenAI text-embedding-3-small.
  3. asyncio.gather one retrieve() per mentor, each receiving the
     SHARED embedding so we don't pay the OpenAI round-trip N times.
  4. Format results as a structured text block with one labeled
     section per mentor; mentor-prefixed snippet numbers (`[paulgraham_1]`,
     `[naval_2]`, etc.) so the calling LLM can cite cleanly.
  5. If any single mentor's retrieval raises, swap that section for a
     "<mentor> unavailable" notice and continue. If every mentor
     raises, return a single "Council unavailable" line. Empty /
     whitespace question short-circuits with the canonical message.

Synthesis (disagreement detection, recommendation, lens-comparison)
is the calling LLM's job — this tool returns N independently-
retrieved snippet pools, nothing more. The single-mentor `search`
tool's description guides the LLM to pick that surface for
single-mentor questions; council_retrieve is for genuinely multi-
lens questions only.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from app.ingest.mentors import MentorConfig, load_mentors
from app.retrieval import Snippet, retrieve
from app.retrieval.query import embed_query, normalize_query

log = logging.getLogger(__name__)

TOOL_NAME = "council_retrieve"


def build_tool_description() -> str:
    """LLM-facing description, generated from the live mentor config so
    it stays in sync when mentors are added or removed."""
    try:
        mentors = load_mentors()
    except FileNotFoundError:
        return (
            "Fan a question out to every configured mentor in parallel "
            "and return labeled snippet sets for synthesis. (No mentors "
            "are configured yet — run `council init`.)"
        )
    names = [cfg.display_name for cfg in mentors.values()] or ["(no mentors configured)"]
    name_clause = ", ".join(names[:-1]) + f", and {names[-1]}" if len(names) > 1 else names[0]
    return (
        f"Query every configured mentor's archive ({name_clause}) in "
        "parallel and return one labeled snippet set per mentor. "
        "Use when a decision benefits from multiple operator lenses. "
        "Returns independently-retrieved snippet pools without "
        "synthesizing them; the calling LLM is expected to write the "
        "disagreement map and recommendation. For single-mentor "
        "questions where the user names a specific mentor, use the "
        "`search` tool with the appropriate mentor_slug instead."
    )


TOOL_DESCRIPTION = build_tool_description()

_MIN_K = 3
_MAX_K = 15
_DEFAULT_K = 8

_MSG_NO_QUESTION = "No question provided."
_MSG_NO_MENTORS = (
    "Council unavailable: no mentors are configured. "
    "Run `council init` to scaffold config/mentors.yaml."
)
_MSG_ALL_UNAVAILABLE = "Council unavailable: every configured mentor archive failed to respond."


def _label_prefix(slug: str) -> str:
    """Citation prefix for a slug. Lower-case ASCII so `[paulgraham_3]`
    is unambiguous in the synthesis. Falls back to first letters when
    the slug is something exotic."""
    cleaned = re.sub(r"[^a-z0-9]", "", slug.lower())
    return cleaned or "m"


def _header(cfg: MentorConfig) -> str:
    focus = f" — {cfg.domain_focus}" if cfg.domain_focus else ""
    return f"{cfg.display_name.upper()} ({cfg.display_name}{focus})"


async def council_retrieve(
    question: str,
    k_per_mentor: int = _DEFAULT_K,
    recency_bias: bool = False,
) -> str:
    """Fan-out retrieval across every configured mentor.

    Args:
        question: The user's question, phrased naturally.
        k_per_mentor: Snippets per mentor. Clamped to [3, 15].
        recency_bias: Same semantics as `search`, applied uniformly
            across all mentors.
    """
    question = (question or "").strip()
    if not question:
        return _MSG_NO_QUESTION

    try:
        mentors = load_mentors()
    except FileNotFoundError:
        return _MSG_NO_MENTORS

    if not mentors:
        return _MSG_NO_MENTORS

    k_per_mentor = max(_MIN_K, min(_MAX_K, int(k_per_mentor)))

    # Embed once and reuse across every retrieve() call. If embedding
    # itself fails (e.g. OPENAI_API_KEY missing), the council can't
    # function — surface the same all-unavailable message.
    t_total = time.perf_counter()
    normalized = normalize_query(question)
    try:
        shared_embedding = await embed_query(normalized)
    except Exception:
        log.exception("council_retrieve: query embed failed")
        return _MSG_ALL_UNAVAILABLE
    embed_ms = (time.perf_counter() - t_total) * 1000

    ordered = list(mentors.values())

    async def _one_mentor(cfg: MentorConfig) -> tuple[str, list[Snippet] | BaseException, float]:
        t = time.perf_counter()
        try:
            snippets = await retrieve(
                mentor_slug=cfg.slug,
                query=question,
                k=k_per_mentor,
                recency_bias=recency_bias,
                source_priority_boost=True,
                query_embedding=shared_embedding,
            )
            return cfg.slug, snippets, (time.perf_counter() - t) * 1000
        except BaseException as exc:  # noqa: BLE001 — degrade per-mentor, don't crash council
            return cfg.slug, exc, (time.perf_counter() - t) * 1000

    results = await asyncio.gather(*[_one_mentor(c) for c in ordered])
    by_slug = {slug: (payload, dur_ms) for slug, payload, dur_ms in results}

    total_ms = (time.perf_counter() - t_total) * 1000
    log.info(
        "council_retrieve total=%.0fms embed=%.0fms %s",
        total_ms,
        embed_ms,
        " ".join(
            f"{cfg.slug}={by_slug[cfg.slug][1]:.0f}ms"
            f"{'/ERR' if isinstance(by_slug[cfg.slug][0], BaseException) else ''}"
            for cfg in ordered
        ),
    )

    # If every mentor raised, surface one degraded message.
    if all(isinstance(by_slug[c.slug][0], BaseException) for c in ordered):
        for c in ordered:
            log.error("council_retrieve: %s failed: %r", c.slug, by_slug[c.slug][0])
        return _MSG_ALL_UNAVAILABLE

    survivor_names = [
        c.display_name for c in ordered if not isinstance(by_slug[c.slug][0], BaseException)
    ]

    lines: list[str] = [f'Council retrieved snippets for: "{question}"', ""]

    for cfg in ordered:
        payload, _dur_ms = by_slug[cfg.slug]
        lines.append(f"=== {_header(cfg)} ===")

        if isinstance(payload, BaseException):
            log.exception(
                "council_retrieve: %s retrieve() failed",
                cfg.slug,
                exc_info=(type(payload), payload, payload.__traceback__),
            )
            others_clause = " + ".join(survivor_names) if survivor_names else "no other mentors"
            lines.append(
                f"{cfg.display_name}'s archive is currently unavailable. "
                f"Council proceeds with {others_clause} only."
            )
            lines.append("")
            continue

        snippets: list[Snippet] = payload  # type: ignore[assignment]
        if not snippets:
            lines.append(f"No matching snippets found in {cfg.display_name}'s archive.")
            lines.append("")
            continue

        prefix = _label_prefix(cfg.slug)
        for i, s in enumerate(snippets, start=1):
            date_str = s.date[:10] if s.date else "unknown"
            lines.append(f"[{prefix}_{i}] (date: {date_str}, score: {s.score:.3f})")
            lines.append(s.text)
            lines.append(f"Source: {s.source_url}")
            lines.append("")

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)
