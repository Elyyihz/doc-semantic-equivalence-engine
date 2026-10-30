from abc import ABC, abstractmethod
from typing import List, Dict, Any


class LlmGateway(ABC):
    """Abstract interface defining LLM synthesis and embedding operations."""

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a chunk of text.

        Args:
            text: Input plain text.

        Returns:
            List of float values representing the embedding.
        """
        pass

    @abstractmethod
    async def evaluate_semantic_equivalence(
        self,
        source_text: str,
        candidate_text: str,
        domain_rules: List[str]
    ) -> Dict[str, Any]:
        """Synthesizes whether two text corpora represent equivalent legal/semantic commitments.

        Args:
            source_text: Ground truth or newly ingested document text.
            candidate_text: Retrieved document text to compare against.
            domain_rules: Specific contractual or institutional invariant rules.

        Returns:
            Dict containing:
                - status: "EQUIVALENT", "PARTIALLY_EQUIVALENT", or "DIVERGENT"
                - overall_score: float
                - explanation: str
                - divergence_points: List[str]
        """
        pass
