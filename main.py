from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services import graph_service


app = FastAPI(
    title="Insurance Claim Analysis",
    description="API for Insurance Claim Analysis",
)

# Helper function to convert ObjectId to string
@app.get("/")
def health_check():
    return {"message":"server is running"}

class QueryRequest(BaseModel):
    query: str

class DecisionResponse(BaseModel):
    decision: str
    amount: Optional[float] = None
    justification: str
    clauses: list[str]

@app.post("/process-query", response_model=DecisionResponse)
async def process_query(request: QueryRequest):
    """
    Processes a user's query and returns a decision.
    """
    try:
        result = await graph_service.execute_graph(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))