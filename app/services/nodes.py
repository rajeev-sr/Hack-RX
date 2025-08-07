# /app/services/nodes.py
from app.state import AppState
from app.services.llm_service import (
    analyze_query,
    rerank_documents,
    generate_initial_decision,
)
from app.services.retrival import retrieve_from_qdrant

async def query_analysis_node(state: AppState) -> AppState:
    """Analyzes the user's query."""
    analyzed_query = await analyze_query(state["original_query"])
    state["analyzed_query"] = analyzed_query
    print(f"==========Analyzed Query=======\n: {analyzed_query}")
    return state

async def retrieval_node(state: AppState) -> AppState:
    """Retrieves documents from Qdrant."""
    print("---NODE: RETRIEVING DOCUMENTS---")
    search_queries = state["analyzed_query"].get("search_queries", [])
    documents = await retrieve_from_qdrant(search_queries)
    state["retrieved_docs"] = documents
    print(f"Retrieved {len(documents)} documents.")
    # print(f"==========Retrieved Documents=======\n: {documents}")
    return state

async def rerank_node(state: AppState) -> AppState:
    """Re-ranks the retrieved documents for relevance."""
    print("---NODE: RERANKING DOCUMENTS---")
    reranked_docs = await rerank_documents(state["original_query"], state["retrieved_docs"])
    state["reranked_docs"] = reranked_docs
    return state


async def generation_node(state: AppState) -> AppState:
    """Generates the initial decision and a critique of that decision."""
    print("---NODE: GENERATING INITIAL DECISION---")
    decision, critique = await generate_initial_decision(
        state["analyzed_query"],
        state.get("reranked_docs", []),
        state.get("correction_feedback")
    )
    state["generated_decision"] = decision
    state["critique"] = critique
    state["needs_correction"] = critique.get("correction_needed", False)
    print(f"Critique: {critique}")
    print(f"Decision: {decision}")
    
    if not state["needs_correction"]:
        state["final_response"] = decision
        
    return state

async def correction_node(state: AppState) -> AppState:
    """Prepares feedback for a correction attempt."""
    print("---NODE: PREPARING CORRECTION---")
    critique = state.get("critique", {})
    state["correction_feedback"] = critique
    print(f"Correction feedback: {critique.get('feedback')}")
    state["needs_correction"] = False 
    return state
