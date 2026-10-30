import logging
import uuid
from fastapi import UploadFile, HTTPException
from app.core.usecase.evaluate_equivalence_usecase import EvaluateEquivalenceUseCase
from app.presentation.dtos.document_dto import EquivalenceAnalysisResponse
from app.core.domain.enum.document_status import DocumentStatus

logger = logging.getLogger("semantic_engine.presentation.controller")


class EquivalenceController:
    """Handles HTTP requests and response mapping for semantic document equivalence."""

    def __init__(self, evaluate_use_case: EvaluateEquivalenceUseCase):
        """Initializes controller with usecase dependency.

        Args:
            evaluate_use_case: EvaluateEquivalenceUseCase instance.
        """
        self.evaluate_use_case = evaluate_use_case

    async def analyze_document(self, file: UploadFile) -> EquivalenceAnalysisResponse:
        """Processes an uploaded document and calculates semantic equivalence.

        Args:
            file: Multi-part uploaded file stream.

        Returns:
            EquivalenceAnalysisResponse with analysis metrics and user summary.

        Raises:
            HTTPException: If file is missing or processing encounters fatal errors.
        """
        if not file.filename:
            logger.warning("Decision taken: Rejecting request due to missing filename")
            raise HTTPException(status_code=400, detail="Nome do arquivo inválido ou ausente.")

        logger.info("Step started: Receiving uploaded file=%s for equivalence evaluation", file.filename)
        content = await file.read()

        if len(content) == 0:
            logger.warning("Decision taken: Rejecting upload due to empty file stream for file=%s", file.filename)
            raise HTTPException(status_code=400, detail="O arquivo enviado está vazio.")

        doc_id = str(uuid.uuid4())
        domain_result = await self.evaluate_use_case.execute(
            document_id=doc_id,
            filename=file.filename,
            raw_bytes=content,
        )

        # Map Portuguese user message for presentation
        if domain_result.status == DocumentStatus.EQUIVALENT:
            user_msg = "Documento equivalente identificado com alto grau de conformidade semântica."
        elif domain_result.status == DocumentStatus.PARTIALLY_EQUIVALENT:
            user_msg = "Equivalência parcial detectada. Existem cláusulas com divergências que requerem atenção."
        elif domain_result.status == DocumentStatus.DIVERGENT:
            user_msg = "Documento divergente ou inédito no repositório de contratos e termos."
        else:
            user_msg = "Falha durante o processamento de validação semântica."

        logger.info("Step concluded: Successfully responded analysis for document_id=%s", doc_id)

        return EquivalenceAnalysisResponse(
            analysis_id=domain_result.analysis_id,
            source_document_id=domain_result.source_document_id,
            candidate_document_id=domain_result.candidate_document_id,
            status=domain_result.status.value,
            overall_score=domain_result.overall_score,
            explanation=domain_result.explanation,
            divergence_points=domain_result.divergence_points,
            created_at=domain_result.created_at,
            user_message=user_msg,
        )
