from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import Optional,List
from app.services.graph_service import execute_graph


app = FastAPI(
    title="Insurance Claim Analysis",
    description="API for Insurance Claim Analysis",
)

# Helper function to convert ObjectId to string
@app.get("/")
def health_check():
    return {"message":"server is running"}

class QueryRequest(BaseModel):
    jobId:str
    # url of the document
    documents:str
    questions: List[str]

class DecisionResponse(BaseModel):
    decision: str
    amount: Optional[float] = None
    justification: str
    clauses: list[str]

class ProcessResponse(BaseModel):
    answers: List[DecisionResponse]
    
@app.post("/process-query", response_model=ProcessResponse)
async def process_query(request: QueryRequest):
    if not all([request.jobId, request.documents, request.questions]):
        raise HTTPException(status_code=400, detail="jobId, documents (url), and a list of questions are required.")
    
    try:
        results = await execute_graph(request.jobId, request.documents, request.questions)
        return {"answers": results}
    except Exception as e:
        print(f"An error occurred during graph execution for jobId {request.jobId}: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

