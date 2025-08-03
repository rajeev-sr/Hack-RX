# /app/services/graph_service.py
from langgraph.graph import StateGraph, END
from app.services.nodes import (
    query_analysis_node,
    retrieval_node,
    rerank_node,
    web_search_node,
    generation_node,
    self_correction_node,
)
from app.state import AppState

async def execute_graph(query: str) -> dict:
    """
    Executes the LangGraph to process the query.
    """
    workflow = StateGraph(AppState)

    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generation_node)
    workflow.add_node("correct", self_correction_node)

    workflow.set_entry_point("query_analysis")

    workflow.add_edge("query_analysis", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_conditional_edges(
        "rerank",
        lambda state: "web_search" if state.get("needs_web_search") else "generate",
        {"web_search": "web_search", "generate": "generate"},
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_conditional_edges(
        "generate",
        lambda state: "correct" if state.get("needs_correction") else END,
        {"correct": "correct", END: END},
    )
    workflow.add_edge("correct", "generate")

    app_graph = workflow.compile()

    initial_state = {"original_query": query}
    final_state = await app_graph.ainvoke(initial_state)

    return final_state["final_response"]
