import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger("semantic_engine.usecase.rrf")


def reciprocal_rank_fusion(
    dense_ranked: List[str],
    sparse_ranked: List[str],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Combines dense vector search and BM25 sparse rankings using Reciprocal Rank Fusion.

    Formula:
        RRF_Score(d) = sum( 1.0 / (k + rank_i(d)) )

    Args:
        dense_ranked: Document IDs ordered by dense vector similarity.
        sparse_ranked: Document IDs ordered by BM25 keyword score.
        k: Smoothing constant preventing top rankings from totally dominating (default 60).

    Returns:
        Sorted list of tuples (document_id, rrf_score) descending.
    """
    scores: Dict[str, float] = {}

    for rank, doc_id in enumerate(dense_ranked, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    for rank, doc_id in enumerate(sparse_ranked, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    logger.info("RRF fused %d dense candidates and %d sparse candidates into %d unique hits", len(dense_ranked), len(sparse_ranked), len(fused))
    return fused
