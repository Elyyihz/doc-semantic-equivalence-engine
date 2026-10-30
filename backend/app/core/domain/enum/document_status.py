from enum import Enum


class DocumentStatus(str, Enum):
    """Lifecycle status indicators for document ingestion and semantic comparison."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    EQUIVALENT = "EQUIVALENT"
    PARTIALLY_EQUIVALENT = "PARTIALLY_EQUIVALENT"
    DIVERGENT = "DIVERGENT"
    FAILED = "FAILED"


class ProcessingStage(str, Enum):
    """Execution stages inside the LangGraph semantic equivalence workflow."""

    OCR_EXTRACTION = "OCR_EXTRACTION"
    CHUNK_EMBEDDING = "CHUNK_EMBEDDING"
    VECTOR_SEARCH = "VECTOR_SEARCH"
    GRAPH_SYNTHESIS = "GRAPH_SYNTHESIS"
    CONCLUDED = "CONCLUDED"
