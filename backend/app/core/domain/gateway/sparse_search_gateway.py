from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any


class SparseSearchGateway(ABC):
    """Abstract interface for BM25 lexical keyword retrieval."""

    @abstractmethod
    def index_document(self, document_id: str, text: str) -> None:
        """Indexes document textual content for lexical BM25 token matching.

        Args:
            document_id: Unique document identifier.
            text: Raw document text content.
        """
        pass

    @abstractmethod
    def query_bm25(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Queries sparse index using BM25 token frequencies.

        Args:
            query_text: Search query.
            top_k: Maximum candidates to retrieve.

        Returns:
            List of tuples (document_id, bm25_score) sorted by relevance.
        """
        pass
