from typing import TypedDict, Optional, Dict, Any, List
from app.core.domain.enum.document_status import DocumentStatus, ProcessingStage


class SemanticGraphState(TypedDict):
    """Execution state shared across all nodes within the LangGraph equivalence pipeline.

    Attributes:
        document_id: Ingested document UUID.
        filename: Original file name.
        raw_bytes: Raw binary bytes of the file.
        extracted_text: Textual content returned by OCR.
        ocr_confidence: Aggregate OCR confidence score [0.0 - 1.0].
        chunks: Segmented chunks with positional indices.
        embeddings: List of embedding vectors for each chunk.
        matched_candidates: Nearest candidate documents retrieved from Qdrant.
        synthesis_result: Consolidated equivalence decision and legal rationales.
        current_stage: Active stage in the state machine.
        execution_logs: Trace of operational log entries and routing decisions.
        error: Optional failure message if execution degrades.
    """

    document_id: str
    filename: str
    raw_bytes: bytes
    extracted_text: str
    ocr_confidence: float
    chunks: List[Dict[str, Any]]
    embeddings: List[List[float]]
    matched_candidates: List[Dict[str, Any]]
    synthesis_result: Optional[Dict[str, Any]]
    current_stage: str
    execution_logs: List[str]
    error: Optional[str]
