from fastapi import APIRouter, UploadFile, File, Depends
from app.presentation.dtos.document_dto import EquivalenceAnalysisResponse, HealthCheckResponse
from app.presentation.api.v1.controllers.equivalence_controller import EquivalenceController
from app.infrastructure.gateway.paddle_ocr_impl import PaddleOcrGatewayImpl
from app.infrastructure.gateway.qdrant_vector_store_impl import QdrantVectorStoreGatewayImpl
from app.infrastructure.gateway.vllm_client_impl import VllmClientGatewayImpl
from app.infrastructure.gateway.bm25_search_impl import BM25SearchGatewayImpl
from app.infrastructure.gateway.cross_encoder_reranker_impl import LocalCrossEncoderRerankerImpl
from app.core.usecase.semantic_graph.nodes import SemanticGraphNodes
from app.core.usecase.semantic_graph.graph import DocumentEquivalenceGraphUseCase
from app.core.usecase.evaluate_equivalence_usecase import EvaluateEquivalenceUseCase
from app.infrastructure.config.settings import settings

router = APIRouter(prefix="/api/v1", tags=["Semantic Equivalence"])


def get_equivalence_controller() -> EquivalenceController:
    """Dependency injection factory wiring Clean Architecture layers."""
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
    return EquivalenceController(eval_usecase)


@router.post("/documents/equivalence/analyze", response_model=EquivalenceAnalysisResponse)
async def analyze_document_equivalence(
    file: UploadFile = File(...),
    controller: EquivalenceController = Depends(get_equivalence_controller),
) -> EquivalenceAnalysisResponse:
    """Submits a document (PDF/Image) for asynchronous OCR, vector search, and LangGraph equivalence synthesis."""
    return await controller.analyze_document(file)


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """System health readiness probe."""
    return HealthCheckResponse(
        status="UP",
        version=settings.APP_VERSION,
        qdrant_connected=True,
        vllm_connected=True,
    )
