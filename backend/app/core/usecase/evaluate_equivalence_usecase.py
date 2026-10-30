import logging
from typing import Dict, Any
from app.core.domain.models.document import SemanticEquivalenceResult
from app.core.domain.enum.document_status import DocumentStatus
from app.core.usecase.semantic_graph.graph import DocumentEquivalenceGraphUseCase
from app.core.usecase.semantic_graph.state import SemanticGraphState

logger = logging.getLogger("semantic_engine.usecase.evaluate")


class EvaluateEquivalenceUseCase:
    """Application usecase orchestrating document equivalence processing."""

    def __init__(self, graph_use_case: DocumentEquivalenceGraphUseCase):
        """Initializes with the compiled LangGraph workflow.

        Args:
            graph_use_case: DocumentEquivalenceGraphUseCase instance.
        """
        self.graph_use_case = graph_use_case

    async def execute(
        self,
        document_id: str,
        filename: str,
        raw_bytes: bytes,
    ) -> SemanticEquivalenceResult:
        """Executes full asynchronous equivalence evaluation.

        Args:
            document_id: UUID of the uploaded document.
            filename: Original filename.
            raw_bytes: Binary payload.

        Returns:
            SemanticEquivalenceResult with scores, status, and rationales.
        """
        logger.info("Step started: Evaluating semantic equivalence for document_id=%s, file=%s", document_id, filename)

        initial_state: SemanticGraphState = {
            "document_id": document_id,
            "filename": filename,
            "raw_bytes": raw_bytes,
            "extracted_text": "",
            "ocr_confidence": 0.0,
            "chunks": [],
            "embeddings": [],
            "matched_candidates": [],
            "synthesis_result": None,
            "current_stage": "INITIALIZED",
            "execution_logs": ["Initialized pipeline"],
            "error": None,
        }

        final_state = await self.graph_use_case.execute(initial_state)

        if final_state.get("error"):
            logger.error("Decision taken: Returning failed equivalence result due to state error: %s", final_state["error"])
            return SemanticEquivalenceResult(
                source_document_id=document_id,
                status=DocumentStatus.FAILED,
                explanation=final_state["error"],
            )

        synthesis = final_state.get("synthesis_result") or {}
        status_str = synthesis.get("status", DocumentStatus.DIVERGENT.value)
        status_enum = DocumentStatus[status_str] if status_str in DocumentStatus.__members__ else DocumentStatus.DIVERGENT

        result = SemanticEquivalenceResult(
            source_document_id=document_id,
            candidate_document_id=final_state.get("matched_candidates", [{}])[0].get("target_document_id", "N/A") if final_state.get("matched_candidates") else "NONE",
            status=status_enum,
            overall_score=synthesis.get("overall_score", 0.0),
            explanation=synthesis.get("explanation", "Analysis completed without explicit divergence."),
            divergence_points=synthesis.get("divergence_points", []),
        )

        logger.info("Step concluded: Analysis concluded with status=%s, score=%.2f", result.status.value, result.overall_score)
        return result
