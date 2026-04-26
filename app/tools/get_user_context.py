"""get_user_context — read the user's static context markdown files.

Context lives as plain markdown under the directory configured by
`COUNCIL_USER_CONTEXT_DIR` (default: `./user_context/`). Files are
user-editable and re-read on every tool call — no cache, no
retrieval, no DB. This is the always-include companion to the
`search` and `convene` tools: those surface what mentors say, this
surfaces the user's own situation so answers can be grounded
rather than generic.

The set of files is dynamic — anything ending in `.md` (and not
`.example`) under the context dir is treated as user context.
Empty / missing slots are surfaced explicitly so the LLM can tell
"context not provided yet" apart from "context unavailable."
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings

log = logging.getLogger(__name__)

TOOL_NAME = "get_user_context"
TOOL_DESCRIPTION = (
    "Read the user's own context markdown files (brand, business "
    "situation, audience, constraints, open strategic questions, "
    "anything else the user has put in their user_context/ dir). "
    "Use this whenever a mentor query needs to be grounded in the "
    "user's specific situation rather than answered generically. "
    'Pass `file` = a stem name (e.g. "brand") to read one file, '
    'or omit / pass "all" to read every file in the directory.'
)

_MSG_INVALID_FILE = (
    'Invalid file name. Pass either "all" or the stem of one of the '
    'files in the user_context directory (e.g. "brand" for brand.md). '
    "Files ending in .example are templates and are not loaded."
)
_MSG_DIR_UNAVAILABLE = (
    "User-context directory is not present at {path}. "
    "Run `council init` to scaffold it, or set COUNCIL_USER_CONTEXT_DIR."
)


def _list_files(directory: Path) -> list[Path]:
    """Real .md files (not .example templates), sorted by name."""
    if not directory.exists() or not directory.is_dir():
        return []
    out = sorted(p for p in directory.glob("*.md") if not p.name.endswith(".example.md"))
    return out


async def get_user_context(file: str | None = None) -> str:
    """Return one or all user-context files as markdown.

    Args:
        file: Stem of a single file (e.g. "brand" for brand.md), or
              "all" / None / empty for every file in the directory.
    """
    target = (file or "").strip() or "all"

    context_dir: Path = settings.user_context_dir
    if not context_dir.exists() or not context_dir.is_dir():
        log.error("get_user_context: directory missing: %s", context_dir)
        return _MSG_DIR_UNAVAILABLE.format(path=context_dir)

    files = _list_files(context_dir)
    available_stems = {p.stem for p in files}

    if target == "all":
        if not files:
            return (
                "(no user context files yet — copy templates "
                f"from {context_dir}/*.example.md, fill them in, "
                "and rename to .md)"
            )
        parts: list[str] = []
        for path in files:
            body = _read_one(path)
            parts.append(f"# {path.name}\n\n{body}")
        return "\n\n---\n\n".join(parts)

    if target not in available_stems:
        return _MSG_INVALID_FILE + f" Available: {sorted(available_stems) or '<none>'}."
    return _read_one(context_dir / f"{target}.md")


def _read_one(path: Path) -> str:
    """Render one file. Distinguish missing / empty / unreadable so
    the LLM can tell why a slot has no content."""
    if not path.exists():
        return f"(missing: {path.name} — file not yet created)"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        log.exception("get_user_context: read failed for %s", path)
        return f"(unavailable: {path.name} — read error logged)"
    stripped = text.strip()
    if not stripped:
        return f"(empty: {path.name} — context slot not yet populated)"
    return stripped
