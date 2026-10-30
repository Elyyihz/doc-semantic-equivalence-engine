from abc import ABC, abstractmethod
from typing import List, Dict, Any
from app.core.domain.models.document import DocumentChunk, SimilarityScore


class VectorStoreGateway(ABC):
    """Abstract interface defining operations for vector storage and semantic retrieval."""

    @abstractmethod
    async def upsert_chunks(self, document_id: str, chunks: List[DocumentChunk]) -> bool:
        """Persists embedded document chunks into the vector collection.

        Args:
            document_id: Unique identifier of the document.
            chunks: List of DocumentChunk entities containing pre-calculated embeddings.

        Returns:
            bool indicating whether upsert succeeded.
        """
        pass

    @abstractmethod
    async def query_nearest_documents(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.70
    ) -> List[SimilarityScore]:
        """Queries nearest matching candidate documents based on cosine distance.

        Args:
            query_embedding: Vector representation of the target chunk/document.
            top_k: Maximum number of candidate matches.
            threshold: Minimum cosine similarity threshold.

        Returns:
            List of SimilarityScore instances ordered by similarity descending.
        """
        pass
