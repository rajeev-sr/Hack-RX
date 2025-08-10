from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import Optional,List,Dict,Any
from app.services.graph_service import execute_graph
from app.celery_worker import process_document
from celery.result import AsyncResult
from datetime import datetime
import random
import string


app = FastAPI(
    title="Insurance Claim Analysis",
    description="API for Insurance Claim Analysis",
)
@app.get("/")
def health_check():
    return {"message":"server is running"}

class ProcessRequest(BaseModel):
    # jobId: str
    documents: str
    questions: List[str]

class Answer(BaseModel):
    decision: str
    details: Dict[str, Any]
    justification: str
    clauses: List[str]

class ProcessResponse(BaseModel):
    jobId: str
    answers: List[Answer]

class StatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any]
    
@app.post("/process", response_model=ProcessResponse)
async def process_document_and_query(request: ProcessRequest):
    if not all([request.documents, request.questions]):
        raise HTTPException(status_code=400, detail="jobId, documents (url), and a list of questions are required.")
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    jobId = f"{timestamp}-{random_suffix}"

    try:
        task = process_document.delay(jobId, request.documents, request.questions)
        return {"jobId": jobId, "answers": task.get()}
    except Exception as e:
        print(f"An error occurred during graph execution for jobId {jobId}: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/status/{task_id}", response_model=StatusResponse)
def get_job_status(task_id: str):
    task_result = AsyncResult(task_id, app=process_document.app)

    if not task_result:
        raise HTTPException(status_code=404, detail="Task not found.")

    if task_result.ready():
        if task_result.successful():
            return {
                "task_id": task_id,
                "status": "SUCCESS",
                "result": task_result.get()
            }
        else: 
            return {
                "task_id": task_id,
                "status": "FAILURE",
                "result": {"error": str(task_result.info)}
            }
    return {"task_id": task_id, "status": "PENDING", "result": None}

