"""sqlite-vec-backed embedding store.

Two call sites are all the vector layer needs today:
  - `get_missing_chunks(conn)` → resume-safe query for unembedded rows
  - `upsert_embeddings(conn, pairs)` → idempotent write, safe to re-run

The vec0 virtual table (`chunks_vec`) must already exist on the
connection. `app.ingest.db.init_vector_tables()` creates it from
`open_mentor_db()`; callers that open the DB some other way should
call that helper once before using this module.

Vectors are written as raw little-endian float32 blobs — that's what
sqlite-vec's `vec0` expects for INSERT. On read, vec0 returns the blob
back unchanged so we never need to unpack (Step 2 retrieval uses MATCH
against the virtual table, not raw blob comparisons).
"""
from __future__ import annotations

import sqlite3
import struct


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def get_missing_chunks(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
) -> list[tuple[int, str]]:
    """Return (chunk_id, text) for chunks not yet embedded, ordered by
    id ascending. `limit` caps the result for --limit CLI runs."""
    sql = (
        "SELECT id, text FROM chunks "
        "WHERE id NOT IN (SELECT rowid FROM chunks_vec) "
        "ORDER BY id ASC"
    )
    if limit is not None:
        sql += " LIMIT ?"
        rows = conn.execute(sql, (int(limit),)).fetchall()
    else:
        rows = conn.execute(sql).fetchall()
    return [(row["id"], row["text"]) for row in rows]


def upsert_embeddings(
    conn: sqlite3.Connection,
    pairs: list[tuple[int, list[float]]],
) -> int:
    """Write (chunk_id, vector) pairs to `chunks_vec`. Re-running with
    the same chunk_id overwrites the prior vector (delete-then-insert
    inside a single transaction) so partial/duplicate batches are safe.

    Returns the number of rows inserted."""
    if not pairs:
        return 0

    conn.execute("BEGIN")
    try:
        for chunk_id, vector in pairs:
            conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (chunk_id,))
            conn.execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                (chunk_id, _pack_vector(vector)),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return len(pairs)
