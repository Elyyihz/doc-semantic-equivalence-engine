from app.core.usecase.semantic_graph.state import SemanticGraphState


def routing_ocr_quality(state: SemanticGraphState) -> str:
    """Evaluates whether OCR extraction was sufficiently clear or degraded.

    Args:
        state: Active graph state.

    Returns:
        Edge branch identifier: 'proceed' or 'abort_error'.
    """
    if state.get("error") is not None:
        return "abort_error"
    if not state.get("extracted_text", "").strip():
        return "abort_error"
    return "proceed"


def routing_candidates_found(state: SemanticGraphState) -> str:
    """Routes based on whether vector candidates were identified.

    Args:
        state: Active graph state.

    Returns:
        Edge branch identifier: 'evaluate_synthesis' or 'finalize_unique'.
    """
    candidates = state.get("matched_candidates", [])
    if candidates and len(candidates) > 0:
        return "evaluate_synthesis"
    return "finalize_unique"
