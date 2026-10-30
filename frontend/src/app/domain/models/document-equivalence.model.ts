/**
 * Status enumeration for document equivalence classification.
 */
export type EquivalenceStatus = 'EQUIVALENT' | 'PARTIALLY_EQUIVALENT' | 'DIVERGENT' | 'FAILED';

/**
 * Model representing the output contract from the semantic evaluation engine.
 */
export interface SemanticEquivalenceAnalysis {
  analysisId: string;
  sourceDocumentId: string;
  candidateDocumentId: string;
  status: EquivalenceStatus;
  overallScore: number;
  explanation: string;
  divergencePoints: string[];
  createdAt: string;
  userMessage: string;
}

/**
 * Model representing similarity graph edge relations.
 */
export interface SimilarityNode {
  id: string;
  label: string;
  score: number;
  status: EquivalenceStatus;
}
