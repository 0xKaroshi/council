"""Text-agnostic chunker.

Pure functions: each strategy takes a RawItem and returns a list of
Chunk objects. No I/O, no DB writes — callers stage the result before
deciding whether to persist.

Strategies:

1. **chunk_tweets** — thread-reassemble by conversation_id. One chunk
   per thread (so the full argument lands in retrieval as a unit, not
   140-char fragments).

2. **chunk_blog** — recursive splitter targeting 512 cl100k_base tokens
   with 15% overlap between neighbors. Prefixes each chunk with
   "{title} > {h2}: " so retrievals carry enough context to be useful
   in a ranked list.

3. **chunk_transcript** — 60–90 second speaker-filtered windows over
   a time-indexed transcript. Prefixes "From '{episode}' ({date}),
   {speaker}: " so citations read naturally.

tiktoken's cl100k_base encoder is the token counter (same one Claude
and GPT-4 use, close enough to real embedding tokenizers that the
512-token target ends up near the 500-token sweet spot for retrieval).
"""

from __future__ import annotations

from typing import Any, Iterable

import tiktoken

from app.ingest import Chunk, RawItem
from app.ingest.dedupe import content_hash

_ENC = tiktoken.get_encoding("cl100k_base")

_BLOG_TARGET_TOKENS = 512
_BLOG_OVERLAP_RATIO = 0.15
_BLOG_POST_TARGET_TOKENS = 700
_BLOG_POST_OVERLAP_TOKENS = 100
_TRANSCRIPT_MIN_SECONDS = 60
_TRANSCRIPT_MAX_SECONDS = 90


def _make_chunk(
    *,
    mentor_slug: str,
    item: RawItem,
    text: str,
    speaker: str | None,
    source_priority: int = 1,
) -> Chunk:
    return Chunk(
        mentor_slug=mentor_slug,
        source_url=item.source_url,
        source_type=item.source_type,
        date=item.date,
        speaker=speaker,
        text=text,
        content_hash=content_hash(text),
        source_priority=source_priority,
    )


# ---------------------------------------------------------------------------
# Tweets — one chunk per thread, keyed by conversation_id
# ---------------------------------------------------------------------------


def chunk_tweets(items: Iterable[RawItem], mentor_slug: str) -> list[Chunk]:
    """Reassemble threads by metadata['conversation_id'] and emit one
    chunk per thread. Tweets without a conversation_id are treated as
    their own one-tweet thread. Tweets within a thread are ordered by
    `date` ascending before being joined with newlines."""
    by_thread: dict[str, list[RawItem]] = {}
    for item in items:
        cid = str(item.metadata.get("conversation_id") or item.source_url)
        by_thread.setdefault(cid, []).append(item)

    chunks: list[Chunk] = []
    for _cid, thread in by_thread.items():
        thread.sort(key=lambda r: r.date)
        text = "\n\n".join(t.body.strip() for t in thread if t.body.strip())
        if not text:
            continue
        head = thread[0]
        chunks.append(
            _make_chunk(
                mentor_slug=mentor_slug,
                item=head,
                text=text,
                speaker=None,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Blog / newsletter — recursive 512-token chunks, 15% overlap
# ---------------------------------------------------------------------------


def _token_window(tokens: list[int], start: int, target: int) -> tuple[int, int]:
    end = min(len(tokens), start + target)
    return start, end


def chunk_blog(item: RawItem, mentor_slug: str) -> list[Chunk]:
    """Token-aware recursive chunker. Each emitted chunk is prefixed
    with "{title} > {h2}: " where h2 comes from metadata['h2'] if
    present, else the empty string (the "> : " collapses cleanly on
    absent sections at display time)."""
    title = item.title or "(untitled)"
    h2 = str(item.metadata.get("h2") or "")
    prefix = f"{title} > {h2}: " if h2 else f"{title}: "

    body = item.body.strip()
    if not body:
        return []

    tokens = _ENC.encode(body)
    if not tokens:
        return []

    overlap = max(1, int(_BLOG_TARGET_TOKENS * _BLOG_OVERLAP_RATIO))
    step = _BLOG_TARGET_TOKENS - overlap

    chunks: list[Chunk] = []
    start = 0
    while start < len(tokens):
        s, e = _token_window(tokens, start, _BLOG_TARGET_TOKENS)
        piece = _ENC.decode(tokens[s:e]).strip()
        if piece:
            text = prefix + piece
            chunks.append(
                _make_chunk(
                    mentor_slug=mentor_slug,
                    item=item,
                    text=text,
                    speaker=None,
                )
            )
        if e >= len(tokens):
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# Blog post (paragraph-aware) — for full essays from RSS / sitemap
# ---------------------------------------------------------------------------


def chunk_blog_paragraphs(
    item: RawItem,
    mentor_slug: str,
    *,
    target_tokens: int = _BLOG_POST_TARGET_TOKENS,
    overlap_tokens: int = _BLOG_POST_OVERLAP_TOKENS,
    source_priority: int = 3,
) -> list[Chunk]:
    """Paragraph-respecting chunker for long blog posts.

    Reads `item.metadata["blocks"]` — a list of
    {type: "heading"|"paragraph", level: int, text: str} produced by
    `app.ingest.sources.blog._markdown_to_blocks` — and walks them
    in order, accumulating paragraphs until the running token count
    crosses `target_tokens`. At that point it emits a chunk prefixed
    with the post title and the most recent heading
    ("{title} > {h2}: ..."), then begins the next chunk by re-using
    the trailing paragraphs of the prior chunk whose combined token
    count fits inside `overlap_tokens` (semantic overlap, not raw
    token-window overlap).

    Falls back to plain text chunking when the metadata is absent —
    keeps the function safe to call on items emitted by older
    sources or hand-built fixtures."""
    blocks: list[dict[str, Any]] = list(item.metadata.get("blocks") or [])
    title = item.title or "(untitled)"

    if not blocks:
        # No structured blocks — fall back to a single paragraph
        # over the raw body.
        body = (item.body or "").strip()
        if not body:
            return []
        blocks = [{"type": "paragraph", "level": 0, "text": body}]

    chunks: list[Chunk] = []

    current_heading: str = ""
    current_paragraphs: list[str] = []  # texts in current chunk
    current_tokens: int = 0

    def _emit() -> None:
        nonlocal current_paragraphs, current_tokens
        if not current_paragraphs:
            return
        body_text = "\n\n".join(current_paragraphs).strip()
        if not body_text:
            current_paragraphs = []
            current_tokens = 0
            return
        prefix = f"{title} > {current_heading}: " if current_heading else f"{title}: "
        chunks.append(
            _make_chunk(
                mentor_slug=mentor_slug,
                item=item,
                text=prefix + body_text,
                speaker=None,
                source_priority=source_priority,
            )
        )
        # Rebuild overlap tail for next chunk.
        tail: list[str] = []
        tail_tokens = 0
        for para in reversed(current_paragraphs):
            t = len(_ENC.encode(para))
            if tail_tokens + t > overlap_tokens and tail:
                break
            tail.insert(0, para)
            tail_tokens += t
        current_paragraphs = tail
        current_tokens = tail_tokens

    for block in blocks:
        kind = block.get("type")
        text = (block.get("text") or "").strip()
        if not text:
            continue

        if kind == "heading":
            level = int(block.get("level") or 2)
            # H2/H3 reset section context; deeper levels just
            # update the heading without flushing.
            if level <= 3:
                _emit()
                current_heading = text
            else:
                # Treat sub-sub-heading as bolded paragraph noise:
                # append into the running text without changing
                # section context.
                current_paragraphs.append(text)
                current_tokens += len(_ENC.encode(text))
            continue

        # paragraph
        para_tokens = len(_ENC.encode(text))
        if current_tokens + para_tokens > target_tokens and current_paragraphs:
            _emit()
        current_paragraphs.append(text)
        current_tokens += para_tokens

    _emit()
    return chunks


# ---------------------------------------------------------------------------
# Podcast / transcript — 60–90s speaker-filtered windows
# ---------------------------------------------------------------------------


def chunk_transcript(item: RawItem, mentor_slug: str) -> list[Chunk]:
    """Expect item.metadata['segments'] to be a list of dicts with
    keys {start, end, speaker, text}. Windows span `speaker_of_interest`
    utterances only (default metadata['speaker_of_interest'] = the
    mentor). Contiguous same-speaker segments are merged until the
    window duration hits _TRANSCRIPT_MAX_SECONDS or the speaker changes;
    windows shorter than _TRANSCRIPT_MIN_SECONDS are dropped unless
    they are the final segment of the transcript.

    Emits one chunk per window, prefixed with the citation header."""
    segments: list[dict] = item.metadata.get("segments") or []
    if not segments:
        return []

    speaker_of_interest = item.metadata.get("speaker_of_interest") or item.metadata.get("speaker")
    episode = item.title or "(untitled episode)"
    date = item.date

    chunks: list[Chunk] = []
    window_start: float | None = None
    window_end: float = 0.0
    window_speaker: str | None = None
    window_texts: list[str] = []

    def _flush(is_final: bool) -> None:
        nonlocal window_start, window_end, window_speaker, window_texts
        if window_start is None or not window_texts:
            window_start = None
            window_speaker = None
            window_texts = []
            return
        duration = window_end - window_start
        if duration < _TRANSCRIPT_MIN_SECONDS and not is_final:
            window_start = None
            window_speaker = None
            window_texts = []
            return
        body = " ".join(t.strip() for t in window_texts if t.strip()).strip()
        if body:
            text = f"From '{episode}' ({date}), {window_speaker}: {body}"
            chunks.append(
                _make_chunk(
                    mentor_slug=mentor_slug,
                    item=item,
                    text=text,
                    speaker=window_speaker,
                )
            )
        window_start = None
        window_speaker = None
        window_texts = []

    for seg in segments:
        spk = seg.get("speaker")
        if speaker_of_interest and spk != speaker_of_interest:
            _flush(is_final=False)
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = seg.get("text", "")

        if window_start is None:
            window_start = start
            window_end = end
            window_speaker = spk
            window_texts = [text]
            continue

        if spk != window_speaker or (end - window_start) > _TRANSCRIPT_MAX_SECONDS:
            _flush(is_final=False)
            window_start = start
            window_end = end
            window_speaker = spk
            window_texts = [text]
            continue

        window_end = end
        window_texts.append(text)

    _flush(is_final=True)
    return chunks
