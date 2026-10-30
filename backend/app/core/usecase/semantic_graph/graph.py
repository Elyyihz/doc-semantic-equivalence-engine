from langgraph.graph import StateGraph, END
from app.core.usecase.semantic_graph.state import SemanticGraphState
from app.core.usecase.semantic_graph.nodes import SemanticGraphNodes
from app.core.usecase.semantic_graph.edges import (
    routing_ocr_quality,
    routing_candidates_found,
)


class DocumentEquivalenceGraphUseCase:
    """Compiles and orchestrates the LangGraph state machine for semantic equivalence."""

    def __init__(self, nodes: SemanticGraphNodes):
        """Initializes the orchestrator with the graph node provider.

        Args:
            nodes: SemanticGraphNodes instance with injected gateway implementations.
        """
        self.nodes = nodes
        self.workflow = self._compile_graph()

    def _compile_graph(self):
        """Constructs the topology and compiled application for the StateGraph."""
        builder = StateGraph(SemanticGraphState)

        # 1. Register Nodes
        builder.add_node("node_extract_ocr", self.nodes.node_extract_ocr)
        builder.add_node("node_chunk_and_embed", self.nodes.node_chunk_and_embed)
        builder.add_node("node_vector_search", self.nodes.node_vector_search)
        builder.add_node("node_synthesize_equivalence", self.nodes.node_synthesize_equivalence)

        # 2. Entry point
        builder.set_entry_point("node_extract_ocr")

        # 3. Transitions
        builder.add_conditional_edges(
            "node_extract_ocr",
            routing_ocr_quality,
            {
                "proceed": "node_chunk_and_embed",
                "abort_error": END,
            },
        )

        builder.add_edge("node_chunk_and_embed", "node_vector_search")

        builder.add_edge("node_vector_search", "node_synthesize_equivalence")
        builder.add_edge("node_synthesize_equivalence", END)

        return builder.compile()

    async def execute(self, initial_state: SemanticGraphState) -> SemanticGraphState:
        """Executes the compiled LangGraph workflow.

        Args:
            initial_state: Initial state payload.

        Returns:
            Terminal state post-graph execution.
        """
        return await self.workflow.ainvoke(initial_state)
