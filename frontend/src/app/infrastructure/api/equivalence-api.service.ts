import { Injectable } from '@angular/core';
import { Observable, from } from 'rxjs';
import { SemanticEquivalenceAnalysis } from '../../domain/models/document-equivalence.model';

/**
 * Infrastructure HTTP client adapter for the Semantic Equivalence Backend.
 */
@Injectable({
  providedIn: 'root'
})
export class EquivalenceApiService {
  private readonly baseUrl = '/api/v1';

  /**
   * Uploads a document to the FastAPI backend for LangGraph processing.
   *
   * @param file - Binary file stream (PDF/Image)
   * @returns Observable resolving to SemanticEquivalenceAnalysis
   */
  analyzeDocument(file: File): Observable<SemanticEquivalenceAnalysis> {
    const formData = new FormData();
    formData.append('file', file);

    const promise = fetch(`${this.baseUrl}/documents/equivalence/analyze`, {
      method: 'POST',
      body: formData
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Falha na análise documental: código ${response.status}`);
        }
        return response.json();
      })
      .then(data => ({
        analysisId: data.analysis_id,
        sourceDocumentId: data.source_document_id,
        candidateDocumentId: data.candidate_document_id,
        status: data.status,
        overallScore: data.overall_score,
        explanation: data.explanation,
        divergencePoints: data.divergence_points || [],
        createdAt: data.created_at,
        userMessage: data.user_message
      }));

    return from(promise);
  }
}
