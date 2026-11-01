import math
import re
import logging
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set
from app.core.domain.gateway.sparse_search_gateway import SparseSearchGateway

logger = logging.getLogger("semantic_engine.infra.bm25")


class BM25SearchGatewayImpl(SparseSearchGateway):
    """Pure Python implementation of Okapi BM25 sparse keyword ranking algorithm."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initializes Okapi BM25 parameters.

        Args:
            k1: Term frequency saturation parameter.
            b: Document length normalization factor.
        """
        self.k1 = k1
        self.b = b
        self.corpus: Dict[str, List[str]] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.term_doc_freq: Dict[str, int] = defaultdict(int)

    def _tokenize(self, text: str) -> List[str]:
        """Lowercases and extracts alphanumeric word tokens."""
        return re.findall(r"\b\w+\b", text.lower())

    def index_document(self, document_id: str, text: str) -> None:
        """Indexes document tokens into the BM25 corpus."""
        tokens = self._tokenize(text)
        if not tokens:
            return

        self.corpus[document_id] = tokens
        self.doc_lengths[document_id] = len(tokens)

        # Update document frequencies
        unique_terms = set(tokens)
        for term in unique_terms:
            self.term_doc_freq[term] += 1

        total_len = sum(self.doc_lengths.values())
        self.avg_doc_len = total_len / len(self.doc_lengths)
        logger.debug("Indexed document %s into BM25 (len=%d)", document_id, len(tokens))

    def query_bm25(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Calculates Okapi BM25 score for all indexed documents matching query tokens."""
        query_tokens = self._tokenize(query_text)
        if not query_tokens or not self.corpus:
            return []

        n_docs = len(self.corpus)
        scores: Dict[str, float] = defaultdict(float)

        for token in query_tokens:
            doc_freq = self.term_doc_freq.get(token, 0)
            if doc_freq == 0:
                continue

            # Standard Robertson-Spärck Jones IDF
            idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

            for doc_id, doc_tokens in self.corpus.items():
                tf = doc_tokens.count(token)
                if tf == 0:
                    continue

                doc_len = self.doc_lengths[doc_id]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
                scores[doc_id] += idf * (numerator / denominator)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]
