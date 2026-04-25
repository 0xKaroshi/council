"""Generic single-mentor search — replaces mentor-specific tools.

In council the mentor lineup is config-driven, so a single MCP tool
that accepts the mentor slug as an input parameter is the natural
shape. The LLM router picks `mentor_slug` from the configured set;
the tool description enumerates which slugs are available so the
router has explicit triggers.
"""
from __future__ import annotations

import logging

from app.ingest.mentors import load_mentors
from app.retrieval import retrieve

log = logging.getLogger(__name__)

TOOL_NAME = "search"


def build_tool_description() -> str:
    """Build the LLM-facing description from the live mentor config so
    new mentors automatically appear in the router prompt."""
    try:
        mentors = load_mentors()
    except FileNotFoundError:
        return (
            "Search a mentor's archive for relevant snippets. "
            "(No mentors are configured yet — run `council init` to scaffold "
            "config/mentors.yaml.)"
        )
    if not mentors:
        return "Search a mentor's archive. No mentors are currently configured."
    lines = [
        "Hybrid retrieval (BM25 + dense vector + RRF fusion) over one "
        "configured mentor's archive. Available mentors:"
    ]
    for cfg in mentors.values():
        focus = f" — {cfg.domain_focus}" if cfg.domain_focus else ""
        lines.append(f"  - `{cfg.slug}` ({cfg.display_name}){focus}")
    lines.append(
        "Pick the slug that best matches the user's question. For "
        "multi-lens questions, prefer the `convene` tool which fans "
        "out across all mentors in parallel."
    )
    return "\n".join(lines)


# Lazily computed at registration time — see app/mcp_server.py.
TOOL_DESCRIPTION = build_tool_description()

_MIN_K = 1
_MAX_K = 20
_DEFAULT_K = 8

_MSG_NO_QUESTION = "No question provided."
_MSG_UNKNOWN_MENTOR = "Unknown mentor: {slug!r}. Configured: {available}."
_MSG_NO_MATCHES = "No matching snippets found in {name}'s archive."
_MSG_UNAVAILABLE = "{name}'s archive is currently unavailable. Please try again later."


async def search(
    mentor_slug: str,
    question: str,
    k: int = _DEFAULT_K,
    recency_bias: bool = False,
) -> str:
    """Retrieve snippets from one mentor's archive and format them for
    the LLM to read.

    Args:
        mentor_slug: Which mentor to search. Must be a slug from
            config/mentors.yaml.
        question: The user's question, phrased naturally.
        k: Number of snippets to return. Clamped to [1, 20].
        recency_bias: If true, gently prefer more recent snippets.
    """
    question = (question or "").strip()
    if not question:
        return _MSG_NO_QUESTION

    try:
        mentors = load_mentors()
    except FileNotFoundError as exc:
        return str(exc)

    if mentor_slug not in mentors:
        return _MSG_UNKNOWN_MENTOR.format(
            slug=mentor_slug,
            available=sorted(mentors.keys()),
        )
    cfg = mentors[mentor_slug]

    k = max(_MIN_K, min(_MAX_K, int(k)))

    try:
        snippets = await retrieve(
            mentor_slug=mentor_slug,
            query=question,
            k=k,
            recency_bias=recency_bias,
            source_priority_boost=True,
        )
    except Exception:
        log.exception("search: retrieve() failed for %r", question)
        return _MSG_UNAVAILABLE.format(name=cfg.display_name)

    if not snippets:
        return _MSG_NO_MATCHES.format(name=cfg.display_name)

    lines: list[str] = [
        f"Found {len(snippets)} relevant snippets from {cfg.display_name}'s archive:",
        "",
    ]
    for i, s in enumerate(snippets, start=1):
        date_str = s.date[:10] if s.date else "unknown"
        kind = _kind_for(s.source_priority)
        lines.append(f"[{i}] ({kind}, date: {date_str}, score: {s.score:.3f})")
        lines.append(s.text)
        lines.append(f"Source: {s.source_url}")
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _kind_for(priority: int) -> str:
    if priority >= 3:
        return "essay"
    if priority == 2:
        return "long-form"
    return "tweet"
