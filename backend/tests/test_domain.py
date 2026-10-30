import pytest
from app.core.domain.models.document import Document, DocumentChunk, SemanticEquivalenceResult
from app.core.domain.enum.document_status import DocumentStatus, ProcessingStage


def test_document_entity_instantiation():
    """Validates default initialization of the Document aggregate."""
    doc = Document(filename="terms_of_service.pdf", raw_bytes=b"sample content")
    assert doc.document_id is not None
    assert doc.filename == "terms_of_service.pdf"
    assert doc.status == DocumentStatus.PENDING
    assert doc.stage == ProcessingStage.OCR_EXTRACTION


def test_chunk_entity_structure():
    """Validates structure and attributes of DocumentChunk."""
    chunk = DocumentChunk(
        chunk_id="chk-001",
        text="Section 4: Confidentiality and non-disclosure obligations.",
        index=0,
        metadata={"page": 1},
        embedding=[0.1, 0.2, 0.3],
    )
    assert chunk.chunk_id == "chk-001"
    assert len(chunk.embedding) == 3


def test_semantic_equivalence_result_defaults():
    """Validates default instantiation of equivalence results."""
    res = SemanticEquivalenceResult(
        source_document_id="src-123",
        status=DocumentStatus.EQUIVALENT,
        overall_score=0.98,
        explanation="Clauses align completely with reference standard.",
    )
    assert res.analysis_id is not None
    assert res.overall_score == 0.98
    assert res.status == DocumentStatus.EQUIVALENT
