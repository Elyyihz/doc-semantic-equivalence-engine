import pytest
from app.infrastructure.gateway.paddle_ocr_impl import PaddleOcrGatewayImpl
from app.infrastructure.gateway.qdrant_vector_store_impl import QdrantVectorStoreGatewayImpl
from app.infrastructure.gateway.vllm_client_impl import VllmClientGatewayImpl
from app.core.usecase.semantic_graph.nodes import SemanticGraphNodes
from app.core.usecase.semantic_graph.graph import DocumentEquivalenceGraphUseCase
from app.core.usecase.evaluate_equivalence_usecase import EvaluateEquivalenceUseCase
from app.core.domain.enum.document_status import DocumentStatus


@pytest.mark.asyncio
async def test_full_graph_execution_happy_path():
    """Tests end-to-end execution of the LangGraph workflow with mocked gateway adapters."""
    ocr_gw = PaddleOcrGatewayImpl()
    qdrant_gw = QdrantVectorStoreGatewayImpl()
    vllm_gw = VllmClientGatewayImpl()

    nodes = SemanticGraphNodes(
        ocr_gateway=ocr_gw,
        vector_store_gateway=qdrant_gw,
        llm_gateway=vllm_gw,
    )
    graph_usecase = DocumentEquivalenceGraphUseCase(nodes)
    eval_usecase = EvaluateEquivalenceUseCase(graph_usecase)

    # Ingest baseline document first to establish vector memory
    await eval_usecase.execute(
        document_id="doc-baseline-001",
        filename="sla_baseline.pdf",
        raw_bytes=b"Standard service level agreement and 99.9% uptime commitment warranty.",
    )

    # Ingest candidate document
    result = await eval_usecase.execute(
        document_id="doc-candidate-002",
        filename="vendor_sla_v2.pdf",
        raw_bytes=b"Vendor SLA with warranty uptime and response obligations.",
    )

    assert result.analysis_id is not None
    assert result.source_document_id == "doc-candidate-002"
    assert result.status in [DocumentStatus.EQUIVALENT, DocumentStatus.PARTIALLY_EQUIVALENT, DocumentStatus.DIVERGENT]
    assert result.overall_score >= 0.0
