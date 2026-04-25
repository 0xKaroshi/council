"""Reciprocal Rank Fusion.

RRF: combine multiple ranked result lists into one score:
    score(d) = sum over lists of  1 / (k_rrf + rank_in_list(d))
Items absent from a list contribute 0 from that list. The constant
k_rrf=60 is the canonical default from Cormack et al. (TREC 2009);
we keep it constant rather than tuning per-mentor.

Inputs are `[(chunk_id, raw_score)]`. Raw scores are ignored — only
their rank order in each list matters. The output is a single fused
list `[(chunk_id, combined_score)]` sorted descending by combined
score.
"""
from __future__ import annotations

_DEFAULT_K_RRF = 60


def rrf_fuse(
    bm25_results: list[tuple[int, float]],
    vec_results: list[tuple[int, float]],
    k_rrf: int = _DEFAULT_K_RRF,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k_rrf + rank)
    for rank, (chunk_id, _) in enumerate(vec_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
