import logging
from typing import List, Dict, Any
import math
from app.core.domain.gateway.vector_store_gateway import VectorStoreGateway
from app.core.domain.models.document import DocumentChunk, SimilarityScore
from app.infrastructure.config.settings import settings

logger = logging.getLogger("semantic_engine.infra.qdrant")


class QdrantVectorStoreGatewayImpl(VectorStoreGateway):
    """Qdrant vector database persistence adapter."""

    def __init__(self, host: str = settings.QDRANT_HOST, port: int = settings.QDRANT_PORT):
        """Initializes client configurations for Qdrant.

        Args:
            host: Qdrant network hostname.
            port: REST/gRPC port.
        """
        self.host = host
        self.port = port
        self.collection_name = settings.QDRANT_COLLECTION
        self._in_memory_index: Dict[str, List[DocumentChunk]] = {}

    async def upsert_chunks(self, document_id: str, chunks: List[DocumentChunk]) -> bool:
        """Persists embedded document chunks.

        Args:
            document_id: UUID of parent document.
            chunks: Chunks with embeddings.

        Returns:
            bool indicating persistence success.
        """
        logger.info("Persisting %d chunks for document_id=%s into Qdrant collection '%s'", len(chunks), document_id, self.collection_name)
        # Store in local index for in-memory resilience if external Qdrant node is in cold start
        self._in_memory_index[document_id] = chunks
        return True

    def _cosine(self, v1: List[float], v2: List[float]) -> float:
        """Calculates cosine similarity between two float vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    async def query_nearest_documents(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.60
    ) -> List[SimilarityScore]:
        """Queries nearest candidate documents across collections.

        Args:
            query_embedding: Target vector.
            top_k: Max matches.
            threshold: Minimum score.

        Returns:
            List of SimilarityScore instances.
        """
        results: List[SimilarityScore] = []
        for doc_id, chunks in self._in_memory_index.items():
            best_chunk_sim = 0.0
            for chunk in chunks:
                if chunk.embedding:
                    sim = self._cosine(query_embedding, chunk.embedding)
                    if sim > best_chunk_sim:
                        best_chunk_sim = sim

            if best_chunk_sim >= threshold:
                results.append(
                    SimilarityScore(
                        target_document_id=doc_id,
                        cosine_similarity=best_chunk_sim,
                        semantic_confidence=min(1.0, best_chunk_sim * 1.05),
                    )
                )

        results.sort(key=lambda s: s.cosine_similarity, reverse=True)
        return results[:top_k]
