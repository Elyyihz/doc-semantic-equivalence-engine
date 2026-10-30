from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SimilarityMatchDto(BaseModel):
    """Data transfer representation of candidate document matches."""

    target_document_id: str
    cosine_similarity: float
    semantic_confidence: float


class EquivalenceAnalysisResponse(BaseModel):
    """User-facing response payload for semantic equivalence audits."""

    analysis_id: str
    source_document_id: str
    candidate_document_id: str
    status: str
    overall_score: float
    explanation: str
    divergence_points: List[str]
    created_at: datetime
    # User-facing message in pt-BR per coding standards!
    user_message: str = Field(..., description="Mensagem contextual direcionada ao usuário final.")


class HealthCheckResponse(BaseModel):
    """Service health and readiness probe status."""

    status: str
    version: str
    qdrant_connected: bool
    vllm_connected: bool
