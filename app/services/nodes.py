# /app/services/nodes.py
from app.state import AppState
from app.services.llm_service import (
    analyze_query,
    rerank_documents,
    search_web,
    generate_response,
    correct_response,
)
from app.services.retrival import retrieve_from_qdrant

async def query_analysis_node(state: AppState) -> AppState:
    """Analyzes the user's query."""
    analyzed_query = await analyze_query(state["original_query"])
    state["analyzed_query"] = analyzed_query
    return state

async def retrieval_node(state: AppState) -> AppState:
    """Retrieves documents from Qdrant."""
    documents = await retrieve_from_qdrant(state["analyzed_query"])
    state["retrieved_docs"] = documents
    return state

async def rerank_node(state: AppState) -> AppState:
    """Re-ranks the retrieved documents."""
    reranked_docs = await rerank_documents(state["original_query"], state["retrieved_docs"])
    state["reranked_docs"] = reranked_docs
    # Example condition for web search
    if not reranked_docs or len(reranked_docs) < 2:
        state["needs_web_search"] = True
    return state

async def web_search_node(state: AppState) -> AppState:
    """Performs a web search for additional context."""
    web_results = await search_web(state["analyzed_query"])
    state["web_results"] = web_results
    return state

async def generation_node(state: AppState) -> AppState:
    """Generates the final response."""
    response = await generate_response(
        state["original_query"], state.get("reranked_docs", []), state.get("web_results")
    )
    state["generated_response"] = response
    # Example condition for correction
    if "uncertain" in response.lower():
        state["needs_correction"] = True
    return state

async def self_correction_node(state: AppState) -> AppState:
    """Corrects the generated response."""
    corrected_response = await correct_response(state["generated_response"])
    state["final_response"] = corrected_response
    state["needs_correction"] = False  # End the loop
    return state
