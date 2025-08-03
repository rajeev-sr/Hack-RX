from typing import TypedDict, Optional, List

class AppState(TypedDict):
    original_query: str
    analyzed_query: Optional[dict]
    retrieved_docs: Optional[List[str]]
    reranked_docs: Optional[List[str]]
    web_results: Optional[str]
    generated_response: Optional[str]
    final_response: Optional[dict]
    needs_web_search: Optional[bool]
    needs_correction: Optional[bool]
