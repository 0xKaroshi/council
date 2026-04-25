"""sqlite-vec / vec0 wrapper.

KNN over the `chunks_vec` virtual table built by Phase 3 Step 1.
sqlite-vec's MATCH + `k = ?` syntax returns the K nearest neighbors
ordered by distance (ASC = closest first).

Distance metric is the vec0 default (L2). text-embedding-3-small
emits unit-length vectors, so L2 ordering is monotonically equivalent
to cosine ordering — same ranking, different scalar values.
"""
from __future__ import annotations

import sqlite3
import struct


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def search_vector(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 50,
) -> list[tuple[int, float]]:
    """Return [(chunk_id, distance)] ordered closest-first."""
    blob = _pack_vector(query_embedding)
    rows = conn.execute(
        "SELECT rowid, distance FROM chunks_vec "
        "WHERE embedding MATCH ? AND k = ? "
        "ORDER BY distance",
        (blob, int(limit)),
    ).fetchall()
    return [(row["rowid"], float(row["distance"])) for row in rows]
