from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field,RootModel
from langchain.chat_models import init_chat_model
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv
load_dotenv()
class KeyEntities(RootModel[dict[str, Union[str, int, float]]]):
    pass

class AnalyzedQuery(BaseModel):
    """Structured representation of a user's insurance query.""" 
    domain: str = Field(..., description="The general domain of the query (e.g., 'Insurance', 'Legal Compliance', 'Human Resources', 'Contract Management').")
    
    key_entities: Optional[KeyEntities] = Field(..., description="A dictionary of key-value pairs extracted from the query, specific to the identified domain.")
    
    search_queries: List[str] = Field(..., description="A list of 3-5 detailed, rephrased questions for semantic search against a document database, tailored to the domain.")
    
    hypotheses: List[str] = Field(..., description="A list of potential outcomes or answers based on the query and common rules within the domain.")

    
# Core LLM Functions
llm=init_chat_model(model_provider="openai",model="gpt-4.1-mini")


    
# Core LLM Functions
llm=init_chat_model(model_provider="openai",model_name="gpt-4.1")
async def analyze_query(query: str) -> dict:
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
- key_entities: dict[str, str] — e.g., {"age": "46", "procedure": "knee surgery", "location": "Pune"}
- search_queries: List[str] — e.g., ["Is knee surgery covered in 3-month policy?", ...]
- hypotheses: List[str] — e.g., ["Waiting period might not be over", ...]

Example Query:
"46M, knee surgery, Pune, 3-month policy"

Example Output:
{
  "domain": "Insurance",
  "key_entities": {
    "age": "46M",
    "procedure": "knee surgery",
    "location": "Pune",
    "policy_duration": "3 months"
  },
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
}
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
