from abc import ABC, abstractmethod
from typing import List, Dict, Any


class RerankerGateway(ABC):
    """Abstract interface defining cross-encoder semantic reranking."""

    @abstractmethod
    async def rerank(
        self, query_text: str, candidate_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Scores (query, candidate) pairs using cross-attention and re-orders list.

        Args:
            query_text: Source document or clause text.
            candidate_documents: Candidate dictionaries (each with target_document_id and text/score).

        Returns:
            Reranked list of candidates with updated cross_encoder_score.
        """
        pass
