import json
import os
import pytest
from app.infrastructure.gateway.paddle_ocr_impl import PaddleOcrGatewayImpl
from app.infrastructure.gateway.qdrant_vector_store_impl import QdrantVectorStoreGatewayImpl
from app.infrastructure.gateway.vllm_client_impl import VllmClientGatewayImpl
from app.infrastructure.gateway.bm25_search_impl import BM25SearchGatewayImpl
from app.infrastructure.gateway.cross_encoder_reranker_impl import LocalCrossEncoderRerankerImpl
from app.core.usecase.semantic_graph.nodes import SemanticGraphNodes
from app.core.usecase.semantic_graph.graph import DocumentEquivalenceGraphUseCase
from app.core.usecase.evaluate_equivalence_usecase import EvaluateEquivalenceUseCase
from app.core.usecase.reciprocal_rank_fusion import reciprocal_rank_fusion
from app.core.domain.enum.document_status import DocumentStatus


@pytest.mark.asyncio
async def test_full_graph_execution_happy_path():
    """Tests end-to-end execution of the LangGraph workflow with mocked gateway adapters."""
    ocr_gw = PaddleOcrGatewayImpl()
    qdrant_gw = QdrantVectorStoreGatewayImpl()
    vllm_gw = VllmClientGatewayImpl()
    bm25_gw = BM25SearchGatewayImpl()
    reranker_gw = LocalCrossEncoderRerankerImpl()

    nodes = SemanticGraphNodes(
        ocr_gateway=ocr_gw,
        vector_store_gateway=qdrant_gw,
        llm_gateway=vllm_gw,
        sparse_gateway=bm25_gw,
        reranker_gateway=reranker_gw,
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


def test_bm25_sparse_search_and_rrf():
    """Validates Okapi BM25 keyword matching and Reciprocal Rank Fusion."""
    bm25 = BM25SearchGatewayImpl()
    bm25.index_document("doc-delaware", "This agreement is governed by the laws of the State of Delaware.")
    bm25.index_document("doc-california", "This contract is subject to California state jurisdiction.")
    bm25.index_document("doc-texas", "All arbitration shall occur exclusively within Texas.")

    hits = bm25.query_bm25("Delaware state laws", top_k=2)
    assert len(hits) >= 1
    assert hits[0][0] == "doc-delaware"

    # Test RRF
    dense = ["doc-texas", "doc-delaware"]
    sparse = ["doc-delaware", "doc-california"]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    assert fused[0][0] == "doc-delaware"


@pytest.mark.asyncio
async def test_cross_encoder_local_reranker():
    """Validates local cross-encoder reordering."""
    reranker = LocalCrossEncoderRerankerImpl()
    query = "Vendor shall indemnify Customer against IP infringement"
    candidates = [
        {"target_document_id": "cand-unrelated", "text": "Invoices are payable within 30 days."},
        {"target_document_id": "cand-indemnity", "text": "Customer indemnification for third-party patent and IP infringement."},
    ]

    reranked = await reranker.rerank(query, candidates)
    assert reranked[0]["target_document_id"] == "cand-indemnity"
    assert reranked[0]["cross_encoder_score"] > reranked[1]["cross_encoder_score"]


def test_benchmark_contract_equivalence_dataset():
    """Validates benchmark dataset parsing and clause coverage."""
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "contracts.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        clauses = json.load(f)

    assert len(clauses) == 20
    equivalent_count = sum(1 for c in clauses if c["expected_verdict"] == "EQUIVALENT")
    divergent_count = sum(1 for c in clauses if c["expected_verdict"] == "DIVERGENT")
    assert equivalent_count == 14
    assert divergent_count == 6
