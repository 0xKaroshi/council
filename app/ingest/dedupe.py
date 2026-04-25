"""Content hashing for dedup.

The hash is computed on *normalized* text (unicode NFKC, whitespace
collapsed, case-folded) so cosmetic re-encodings — smart quotes vs.
straight, double-spaces, tab vs. space — don't re-admit the same
content. The raw text remains unmodified in the chunk; only the
dedup key is derived from the normalized form.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = _WS_RE.sub(" ", t).strip()
    return t.casefold()


def content_hash(text: str) -> str:
    """SHA-256 hex digest of the normalized text. 64 chars, deterministic."""
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()
