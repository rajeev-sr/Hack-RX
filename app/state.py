from typing import TypedDict, Optional, List, Dict

class AppState(TypedDict):
    original_query: str
    analyzed_query: Optional[Dict]
    retrieved_docs: Optional[List[str]]
    reranked_docs: Optional[List[str]]
    generated_decision: Optional[Dict]
    final_response: Optional[Dict]
    critique: Optional[Dict]
    needs_correction: Optional[bool]
    correction_feedback: Optional[Dict]
