# /app/services/graph_service.py
from langgraph.graph import StateGraph, END
from app.services.nodes import (
    query_analysis_node,
    retrieval_node,
    rerank_node,
    generation_node,
    self_correction_node,
)
from app.state import AppState

def should_correct(state: AppState) -> str:
    """Determines if correction step is needed"""
    if state.get("needs_correction"):
        print("---DECISION: CORRECTION NEEDED---")
        return "correct"
    else :
        print("---DECISION: NO CORRECTION NEEDED---")
        return END
async def execute_graph(query: str) -> dict:
    """
    Executes the LangGraph to process the query.
    """
    workflow = StateGraph(AppState)

    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generation_node)
    workflow.add_node("correct", self_correction_node)

    workflow.set_entry_point("query_analysis")
    
    workflow.add_edge("query_analysis", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_conditional_edges(
        "generate",
        should_correct,
        {
            "correct": "correct",
            END: END,
        },
    )
    workflow.add_edge("correct", "generate")

    app_graph = workflow.compile()

    initial_state = {"original_query": query}
    final_state = await app_graph.ainvoke(initial_state)
    if not final_state.get("final_response"):
        raise Exception("Graph execution failed to produce a final response.")

    return final_state["final_response"]
