from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field,RootModel
from langchain.chat_models import init_chat_model
from typing import List, Optional, Dict, Any, Union,Tuple
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from fastapi import HTTPException
load_dotenv()
class KeyEntities(RootModel[dict[str, Union[str, int, float]]]):
    pass

class AnalyzedQuery(BaseModel):
    """Structured representation of a user's insurance query.""" 
    domain: str = Field(..., description="The general domain of the query (e.g., 'Insurance', 'Legal Compliance', 'Human Resources', 'Contract Management').")
    
    key_entities: Optional[KeyEntities] = Field(..., description="A dictionary of key-value pairs extracted from the query, specific to the identified domain.")
    
    search_queries: List[str] = Field(..., description="A list of 3-5 detailed, rephrased questions for semantic search against a document database, tailored to the domain.")
    
    hypotheses: List[str] = Field(..., description="A list of potential outcomes or answers based on the query and common rules within the domain.")

class DocumentQueryResult(BaseModel):
    """Structured JSON response for a query against a document set."""
    decision: str = Field(..., description="The final, summary answer to the user's query (e.g., 'Approved', 'Compliant', 'Eligible', 'Termination Clause Found').")
    details: Dict[str, Any] = Field(..., description="A dictionary containing specific results, like an approved amount, compliance status, or contract details.")
    justification: str = Field(..., description="A clear, concise, step-by-step reasoning for the decision, explaining how the document clauses apply to the query's key entities.")
    clauses: List[str] = Field(..., description="A list of the specific clause IDs or text snippets from the documents that were used to make the decision.")

class DecisionCritique(BaseModel):
    """A critique of the query result to check for correctness."""
    correction_needed: bool = Field(..., description="Whether the original decision needs to be corrected.")
    confidence_score: float = Field(..., description="A score from 0.0 to 1.0 indicating confidence in the original decision.")
    feedback: str = Field(..., description="Detailed feedback for why a correction is or is not needed. If needed, suggest specific improvements.")
    
# Core LLM Functions

llm=init_chat_model(model_provider="openai",model="gpt-4.1")
async def analyze_query(query: str) -> dict:
    try:
        # LLM call to analyze query and translate to structured Json format
        SYSTEM_PROMPT = """
        You are a Domain-Aware Query Analyzer Assistant. Your job is to convert unstructured user queries into structured representations that downstream applications can use for accurate document retrieval, policy validation, contract management, etc.

        You MUST:
        1. Identify the domain of the query (e.g., Insurance, Legal Compliance, Human Resources, Contract Management).
        2. Extract key entities relevant to that domain in a structured dictionary format.
        3. Generate 3-5 diverse but relevant search queries that capture the core intent of the original query.
        4. Generate 2-4 likely hypotheses — assumptions or possible interpretations that the system might make **before** retrieving documents.

        Guidelines:
        - Tailor extracted entities and hypotheses to the identified domain.
        - Combine extractive and generative reasoning: Extract key facts AND generate potential queries and interpretations.
        - Handle ambiguity gracefully. If uncertain, make reasonable guesses and reflect them in the hypotheses.

        Output Schema (Python Pydantic):
        - domain: str — e.g., "Insurance"
        - key_entities: dict[str, str] — e.g., {{"age": "46", "procedure": "knee surgery", "location": "Pune"}}
        - search_queries: List[str] — e.g., ["Is knee surgery covered in 3-month policy?", ...]
        - hypotheses: List[str] — e.g., ["Waiting period might not be over", ...]

        Example Query:
        "46M, knee surgery, Pune, 3-month policy"

        Example Output:
        {{
        "domain": "Insurance",
        "key_entities": {{
            "age": "46M",
            "procedure": "knee surgery",
            "location": "Pune",
            "policy_duration": "3 months"
        }},
        "search_queries": [
            "Is knee surgery covered under a 3-month policy?",
            "Waiting period for knee surgery in health insurance",
            "Does a 3-month-old policy cover planned hospitalization?",
            "Coverage for knee surgery in Pune hospitals",
            "Knee surgery eligibility in short-term insurance plans"
        ],
        "hypotheses": [
            "The procedure may not be covered due to a waiting period.",
            "User is trying to check claim eligibility for upcoming surgery.",
            "Knee issue could be pre-existing and may affect approval.",
            "User likely seeks cashless hospital options in Pune."
        ]
        }}
        """
        prompt=ChatPromptTemplate.from_messages(
            [
                    ("system", SYSTEM_PROMPT),
                    ("human", "Translate the following query into a structured JSON format: {query}"),
            ]
        )
        structured_query = llm.with_structured_output(AnalyzedQuery, method="function_calling")
        chain=prompt | structured_query
        response=  await chain.ainvoke({"query": query})
        return response.model_dump()
    except Exception as e:
        print("Error in analyze_query:", e)
        raise HTTPException(status_code=500, detail=str(e))
async def generate_initial_decision(analyzed_query: dict, docs: list, feedback: Optional[dict] = None) -> Tuple[dict, dict]:
    """Generates and critiques a decision in a single, domain-agnostic step."""
    context = "\n\n".join(doc["content"] for doc in docs)
    correction_instruction=None
    if feedback:
        correction_instruction=f"""this is the second attempt, firts attempt was flawed Plz pay close attention to the feedback
        {feedback.get("feedback")}
        """
    system_prompt=ChatPromptTemplate.from_template(
        """
        you are multi-persona AI assistant.you will act as a meticulous Analyst for the {domain} domain and then as a skeptical Senior Auditor.
        
        Analyzed Query: {analyzed_query}
        Context: {context}
        Correction Instruction: {correction_instruction}
        -----------------------------------------------------
        PART 1: Analyst's Decision
        1. Fact Check: Use the `key_entities` from the Analyzed Query as your source of truth.
        2.  Clause Matching: For each key entity, find the most relevant document clause(s).
        3. Step-by-Step Reasoning: Evaluate each entity against its matched clause. Your reasoning must be specific to the {domain}domain.
        4. Synthesize Decision: Combine your evaluations into a final decision.
        5. Format Output: Provide your final decision as a JSON object conforming to the `DocumentQueryResult` schema. The `decision` and `details` fields must be relevant to the domain.
        ----------------------------------------------------------
        Part 2: Auditor's Critique
        
        Now, as the Senior Auditor, critically evaluate the Analyst's decision.
        1. Fact Verification: Did the Analyst correctly use all `key_entities`? Were any clauses misinterpreted?
        2. Logical Soundness: Is the `justification` logical and does it directly lead to the `decision`?
        3. Format Critique: Provide your critique as a JSON object conforming to the `DecisionCritique` schema.

        ----------------------------------------------------------

        Combined Final Output:
        Return a single JSON object with two keys: "decision" (the `DocumentQueryResult` JSON) and "critique" (the `DecisionCritique` JSON).
        """
    )
    class CombinedResponse(BaseModel):
        decision: DocumentQueryResult
        critique: DecisionCritique
         
    
    structured_llm = llm.with_structured_output(CombinedResponse)
    chain = system_prompt | structured_llm
    
    response = await chain.ainvoke({
        "domain": analyzed_query.get("domain"),
        "analyzed_query": str(analyzed_query),
        "context": context,
        "correction_instruction": correction_instruction
    })
    
    return response.decision.model_dump(), response.critique.model_dump()
async def rerank_documents(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-ranks retrieved documents based on relevance to the original query.
    
    Args:
        query: Original user query
        docs: List of documents retrieved from Qdrant
        
    Returns:
        Re-ranked list of documents
    """
    if not docs:
        print("No documents to rerank")
        return []
    
    try:
        print(f"Reranking {len(docs)} documents")
        if len(docs) < 2:
            return docs
            
        texts = [doc["content"] for doc in docs]
        
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[query, text] for text in texts]
        scores = reranker.predict(pairs)
        
        for i, doc in enumerate(docs):
            # Store both scores - original and reranked
            doc["metadata"]["original_score"] = doc["metadata"].get("score", 0.0)
            doc["metadata"]["rerank_score"] = float(scores[i])
            doc["metadata"]["score"] = float(scores[i])  
        # Sort
        reranked_docs = sorted(docs, key=lambda x: x["metadata"]["score"], reverse=True)
        
        print(f"Reranking complete. Top document score: {reranked_docs[0]['metadata']['score']:.4f}")
        return reranked_docs
        
    except Exception as e:
        print(f"Error reranking documents: {str(e)}")
        return docs
