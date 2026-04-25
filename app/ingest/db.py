"""Per-mentor SQLite database factory + schema + upsert helpers.

Each mentor gets its own file under data/mentors/<slug>.db. This keeps
concerns isolated (one bad ingest can't corrupt another mentor's index,
and per-mentor backups are trivial) and matches how retrieval will
eventually fan out — a query against a mentor slug opens that
mentor's DB and nothing else.

The `chunks_fts` virtual table is an external-content FTS5 index over
`chunks.text`; triggers keep it in sync on insert/update/delete so
lexical (BM25) search lands for free once we start ingesting.

`chunks_vec` (vec0) is initialized by `init_vector_tables()` and holds
one 1536-dim dense embedding per `chunks.id`. The sqlite-vec extension
is loaded at connection time; `open_mentor_db` calls
`init_vector_tables` on every open so both fresh DBs and pre-existing
ones (carried over from a previous ingest run) pick up the
virtual table lazily.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from app.config import settings
from app.ingest import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    source_url TEXT NOT NULL,
    source_type TEXT NOT NULL,
    date TEXT NOT NULL,
    speaker TEXT,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    source_priority INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_date ON chunks(date);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def _mentor_db_path(slug: str) -> Path:
    """User-data path. Always returned for write-mode opens regardless
    of whether the DB exists yet — `open_mentor_db` will create it."""
    return settings.data_dir / "mentors" / f"{slug}.db"


def _bundled_db_path(slug: str) -> Path | None:
    """Locate the bundled example DB for `slug`, or return None.

    Two resolution strategies, tried in order:

      1. importlib.resources — picks up DBs shipped via setuptools
         package-data (i.e. `pipx install council`, `pip install .`).
         The `examples/<slug>/` directories are sub-packages with
         `__init__.py` markers so importlib can discover them.
      2. Walk up from this module — picks up DBs at <repo-root>/examples
         when running from a git checkout (with or without
         `pip install -e .`).

    Both paths return a real `Path`; sqlite3 needs a filesystem path,
    not a Traversable.
    """
    try:
        from importlib.resources import files
        try:
            traversable = files(f"examples.{slug}").joinpath(f"{slug}.db")
            if traversable.is_file():
                return Path(str(traversable))
        except (ModuleNotFoundError, FileNotFoundError, AttributeError):
            pass
    except ImportError:
        pass

    # `__file__` is .../app/ingest/db.py — three .parent's up is the
    # repo (or installed-package) root.
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = repo_root / "examples" / slug / f"{slug}.db"
    if candidate.exists():
        return candidate
    return None


def _db_has_chunks(path: Path) -> bool:
    """True iff `path` is an openable SQLite file with at least one
    chunks row. Used to decide whether to fall back to the bundled
    DB — an empty user-data DB shouldn't shadow the bundled archive."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        conn = sqlite3.connect(str(path))
        try:
            n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            return int(n) > 0
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def resolve_mentor_db_path(slug: str, *, fallback_to_bundled: bool = False) -> tuple[Path, str]:
    """Return `(path, source)` for a mentor slug.

    `source` is one of:
      - "user"     — primary user-data path is present and non-empty
      - "bundled"  — falling back to the example DB shipped with the
                     package (only when fallback_to_bundled=True)
      - "missing"  — neither the user nor the bundled DB exists; the
                     returned path is the user-data location and will
                     be created on first write

    Read paths (`search`, `convene`, `status`) call with
    `fallback_to_bundled=True`. Write paths (`ingest`, `embed`)
    leave it False so they always materialize against user data —
    the bundled archive is read-only by design.
    """
    user_path = _mentor_db_path(slug)
    if _db_has_chunks(user_path):
        return user_path, "user"
    if fallback_to_bundled:
        bundled = _bundled_db_path(slug)
        if bundled is not None:
            return bundled, "bundled"
    return user_path, "missing" if not user_path.exists() else "user"


def open_mentor_db(
    slug: str, *, fallback_to_bundled: bool = False
) -> sqlite3.Connection:
    """Open (and lazily create) the SQLite DB for a mentor slug.

    `fallback_to_bundled=True` makes read-mode callers transparently
    use the example DB shipped with the package when the user hasn't
    ingested that mentor yet. Default is False — write-mode callers
    always materialize against user data.

    The connection is returned in autocommit mode with WAL enabled;
    callers should close it explicitly or use a context manager.
    """
    path, _source = resolve_mentor_db_path(slug, fallback_to_bundled=fallback_to_bundled)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    _migrate_chunks_source_priority(conn)
    init_vector_tables(conn)
    return conn


# ---------------------------------------------------------------------------
# Migrations — idempotent, run on every open
# ---------------------------------------------------------------------------

def _migrate_chunks_source_priority(conn: sqlite3.Connection) -> None:
    """Add `chunks.source_priority` to pre-existing DBs that were built
    before the column was introduced. 1=tweet, 2=podcast, 3=canonical
    (essay / newsletter). Unused today; Step 2 retrieval weights on it."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
    if "source_priority" in cols:
        return
    conn.execute(
        "ALTER TABLE chunks ADD COLUMN source_priority INTEGER NOT NULL DEFAULT 1"
    )


def init_vector_tables(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec on this connection and ensure `chunks_vec` exists.

    Idempotent: safe to call on every open. The extension load must
    happen per-connection (sqlite-vec registers virtual tables on a
    connection-local namespace)."""
    import sqlite_vec  # imported lazily so tests that don't exercise
                       # vector storage aren't forced to install it

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec "
        "USING vec0(embedding float[1536])"
    )


def upsert_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> int:
    """Insert new chunks, ignoring content-hash collisions (already
    ingested). `source_priority` is persisted from the Chunk
    (defaults to 1 for tweet-shaped sources; blog_post chunks use
    3). Returns the number of rows actually inserted."""
    if not chunks:
        return 0
    inserted = 0
    for c in chunks:
        cur = conn.execute(
            "INSERT OR IGNORE INTO chunks "
            "(source_url, source_type, date, speaker, text, content_hash, "
            " source_priority) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                c.source_url,
                c.source_type,
                c.date,
                c.speaker,
                c.text,
                c.content_hash,
                getattr(c, "source_priority", 1),
            ),
        )
        if cur.rowcount > 0:
            inserted += 1
    return inserted


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None
