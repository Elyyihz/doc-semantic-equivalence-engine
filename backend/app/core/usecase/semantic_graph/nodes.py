import logging
from typing import Dict, Any, List
from app.core.domain.gateway.ocr_gateway import OcrGateway
from app.core.domain.gateway.vector_store_gateway import VectorStoreGateway
from app.core.domain.gateway.llm_gateway import LlmGateway
from app.core.domain.models.document import DocumentChunk
from app.core.domain.enum.processing_stage import ProcessingStage
from app.core.usecase.semantic_graph.state import SemanticGraphState

logger = logging.getLogger("semantic_engine.graph.nodes")


class SemanticGraphNodes:
    """Encapsulates node actions for the LangGraph StateGraph execution pipeline."""

    def __init__(
        self,
        ocr_gateway: OcrGateway,
        vector_store_gateway: VectorStoreGateway,
        llm_gateway: LlmGateway,
    ):
        """Initializes nodes with domain gateway dependencies.

        Args:
            ocr_gateway: OCR engine adapter.
            vector_store_gateway: Vector database adapter.
            llm_gateway: LLM inference runtime adapter.
        """
        self.ocr_gateway = ocr_gateway
        self.vector_store_gateway = vector_store_gateway
        self.llm_gateway = llm_gateway

    async def node_extract_ocr(self, state: SemanticGraphState) -> Dict[str, Any]:
        """Extracts text and layout tokens from document bytes using PaddleOCR.

        Args:
            state: Current graph state.

        Returns:
            Updated dictionary with extracted_text, ocr_confidence, and stage.
        """
        doc_id = state["document_id"]
        logger.info("Step started: Extracting document text via OCR for document_id=%s", doc_id)

        try:
            ocr_result = await self.ocr_gateway.extract_text(
                state["raw_bytes"], state["filename"]
            )
            confidence = ocr_result.get("confidence", 0.0)
            text = ocr_result.get("text", "")

            # Log decision based on OCR confidence
            if confidence < 0.65:
                logger.warning(
                    "Decision taken: Low OCR confidence (%.2f) detected for document_id=%s; flagging potential degradation",
                    confidence,
                    doc_id,
                )
            else:
                logger.info(
                    "Decision taken: High OCR confidence (%.2f) confirmed for document_id=%s",
                    confidence,
                    doc_id,
                )

            logger.info("Step concluded: OCR extraction completed successfully for document_id=%s", doc_id)
            return {
                "extracted_text": text,
                "ocr_confidence": confidence,
                "current_stage": ProcessingStage.CHUNK_EMBEDDING.value,
                "execution_logs": state["execution_logs"] + [f"OCR extraction finished with score {confidence:.2f}"],
            }
        except Exception as exc:
            logger.error("Decision taken: Aborting OCR pipeline due to parsing failure: %s", exc)
            return {
                "error": f"OCR extraction failed: {str(exc)}",
                "execution_logs": state["execution_logs"] + [f"OCR error: {str(exc)}"],
            }

    async def node_chunk_and_embed(self, state: SemanticGraphState) -> Dict[str, Any]:
        """Splits extracted text into semantic paragraphs and generates embeddings.

        Args:
            state: Current graph state.

        Returns:
            Updated state containing segmented chunks and dense vectors.
        """
        doc_id = state["document_id"]
        logger.info("Step started: Segmenting text and generating embeddings for document_id=%s", doc_id)

        raw_text = state.get("extracted_text", "")
        # Contextual splitting: paragraphs separated by double newlines or punctuation
        raw_paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 30]

        if not raw_paragraphs:
            raw_paragraphs = [raw_text.strip()] if raw_text.strip() else ["Document contains no text"]

        chunks: List[Dict[str, Any]] = []
        embeddings: List[List[float]] = []

        for idx, para in enumerate(raw_paragraphs):
            embedding = await self.llm_gateway.generate_embedding(para)
            chunks.append({
                "chunk_id": f"{doc_id}-chunk-{idx}",
                "text": para,
                "index": idx,
                "metadata": {"source_doc": doc_id, "paragraph_idx": idx},
            })
            embeddings.append(embedding)

        logger.info("Decision taken: Generated %d semantic chunks for document_id=%s", len(chunks), doc_id)

        # Upsert into vector repository
        domain_chunks = [
            DocumentChunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                index=c["index"],
                metadata=c["metadata"],
                embedding=embeddings[i],
            )
            for i, c in enumerate(chunks)
        ]
        await self.vector_store_gateway.upsert_chunks(doc_id, domain_chunks)

        logger.info("Step concluded: Text chunking and embedding successfully indexed for document_id=%s", doc_id)
        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "current_stage": ProcessingStage.VECTOR_SEARCH.value,
            "execution_logs": state["execution_logs"] + [f"Generated and persisted {len(chunks)} chunks"],
        }

    async def node_vector_search(self, state: SemanticGraphState) -> Dict[str, Any]:
        """Queries Qdrant vector index to identify nearest similar documents.

        Args:
            state: Current graph state.

        Returns:
            State updated with retrieved candidate documents.
        """
        doc_id = state["document_id"]
        logger.info("Step started: Executing vector similarity search for document_id=%s", doc_id)

        embeddings = state.get("embeddings", [])
        if not embeddings:
            logger.warning("Decision taken: No embeddings present; skipping nearest neighbor search")
            return {
                "matched_candidates": [],
                "current_stage": ProcessingStage.GRAPH_SYNTHESIS.value,
            }

        # Query using the primary lead paragraph vector
        lead_vector = embeddings[0]
        similarities = await self.vector_store_gateway.query_nearest_documents(
            query_embedding=lead_vector,
            top_k=5,
            threshold=0.60
        )

        candidates = [
            {
                "target_document_id": s.target_document_id,
                "cosine_similarity": s.cosine_similarity,
                "semantic_confidence": s.semantic_confidence,
            }
            for s in similarities
            if s.target_document_id != doc_id
        ]

        logger.info("Decision taken: Retrieved %d candidate matches from vector index", len(candidates))
        logger.info("Step concluded: Vector search completed for document_id=%s", doc_id)

        return {
            "matched_candidates": candidates,
            "current_stage": ProcessingStage.GRAPH_SYNTHESIS.value,
            "execution_logs": state["execution_logs"] + [f"Retrieved {len(candidates)} candidates"],
        }

    async def node_synthesize_equivalence(self, state: SemanticGraphState) -> Dict[str, Any]:
        """Evaluates semantic equivalence against nearest retrieved candidates via LLM.

        Args:
            state: Current graph state.

        Returns:
            State updated with synthesis result and final classification.
        """
        doc_id = state["document_id"]
        logger.info("Step started: Synthesizing equivalence evaluation via LLM for document_id=%s", doc_id)

        candidates = state.get("matched_candidates", [])
        source_text = state.get("extracted_text", "")

        if not candidates:
            logger.info("Decision taken: No existing candidates matched threshold; document is novel and unique")
            synthesis = {
                "status": "DIVERGENT",
                "overall_score": 0.0,
                "explanation": "No prior document in the repository demonstrated sufficient semantic similarity.",
                "divergence_points": ["Document presents novel terms with no existing equivalent."],
            }
        else:
            top_candidate = candidates[0]
            # Domain invariant rules for validation
            rules = [
                "Verify identical legal liability and indemnity terms",
                "Check SLA and operational commitment consistency",
                "Flag discrepancies in entity obligations or deadlines"
            ]
            synthesis = await self.llm_gateway.evaluate_semantic_equivalence(
                source_text=source_text,
                candidate_text=f"Candidate document reference {top_candidate['target_document_id']}",
                domain_rules=rules,
            )
            logger.info(
                "Decision taken: Synthesized status=%s with score=%.2f",
                synthesis.get("status"),
                synthesis.get("overall_score", 0.0),
            )

        logger.info("Step concluded: Equivalence synthesis completed for document_id=%s", doc_id)
        return {
            "synthesis_result": synthesis,
            "current_stage": ProcessingStage.CONCLUDED.value,
            "execution_logs": state["execution_logs"] + [f"Final status: {synthesis.get('status')}"],
        }
