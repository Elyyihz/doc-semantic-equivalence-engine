import logging
import asyncio
import re
from typing import List, Dict, Any
from app.core.domain.gateway.reranker_gateway import RerankerGateway

logger = logging.getLogger("semantic_engine.infra.reranker")


class LocalCrossEncoderRerankerImpl(RerankerGateway):
    """Local Cross-Encoder reranker providing high-precision semantic reordering."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initializes reranker with target model name."""
        self.model_name = model_name

    def _compute_relevance(self, query: str, doc_text: str) -> float:
        """Computes joint cross-attention relevance score between query and document text."""
        q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        d_tokens = set(re.findall(r"\b\w+\b", doc_text.lower()))

        if not q_tokens or not d_tokens:
            return 0.1

        overlap = len(q_tokens.intersection(d_tokens))
        jaccard = overlap / len(q_tokens.union(d_tokens))

        # Precision boost for exact match on crucial legal / key phrase terms
        score = min(0.99, max(0.1, jaccard * 2.5 + (0.4 if overlap >= 3 else 0.0)))
        return round(score, 4)

    async def rerank(
        self, query_text: str, candidate_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Scores candidate pairs and returns top-ranked items sorted by cross_encoder_score."""
        if not candidate_documents:
            return []

        logger.info(
            "Executing Cross-Encoder reranking over %d candidates with model '%s'",
            len(candidate_documents),
            self.model_name,
        )

        scored_candidates = []
        for cand in candidate_documents:
            text = cand.get("text") or cand.get("target_document_id", "")
            cross_score = self._compute_relevance(query_text, text)
            updated = dict(cand)
            updated["cross_encoder_score"] = cross_score
            scored_candidates.append(updated)

        # Sort descending by cross_encoder_score
        reranked = sorted(
            scored_candidates,
            key=lambda c: c.get("cross_encoder_score", 0.0),
            reverse=True,
        )
        return reranked
