from typing import TypedDict, Optional, List, Dict, Any

class AppState(TypedDict):
    original_query: str
    analyzed_query: Optional[Dict]
    retrieved_docs: Optional[List[Dict[str, Any]]]
    reranked_docs: Optional[List[Dict[str, Any]]] 
    generated_decision: Optional[Dict]
    final_response: Optional[Dict]
    critique: Optional[Dict]
    needs_correction: Optional[bool]
    correction_feedback: Optional[Dict]
