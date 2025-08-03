async def analyze_query(query: str) -> dict:
    # LLM call to analyze query
    return {"main_topic": "knee surgery", "location": "Pune"}

async def rerank_documents(query: str, docs: list) -> list:
    #re-ranking logic
    return docs

async def search_web(query: dict) -> str:
    # web search logic
    return "Web search results for knee surgery in Pune."

async def generate_response(query: str, docs: list, web_results: str) -> str:
    #LLM call to generate response
    return "Based on the documents, the knee surgery is approved."

async def correct_response(response: str) -> dict:
    # LLM call to correct response
    return {
        "decision": "Approved",
        "amount": "XXXX",
        "justification": "--------",
        "clauses": ["Clause 1", "Clause 2", "Clause 3"],
    }
