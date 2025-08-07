from dotenv import load_dotenv
from logging import Logger
from typing import List, Dict, Any
from fastapi import logger
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
import os

load_dotenv()
QDRANT_URL=os.getenv("QDRANT_URL")
COLLECTION_NAME=os.getenv("QDRANT_COLLECTION_NAME")
EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL","text-embedding-3-large")
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "3"))

QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")

embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL)
async def retrieve_from_qdrant(search_queries: List[str]) -> List[Dict[str, Any]]:

    if not search_queries:
        print("No search queries provided")
        return []
    
    try:
        qdrant =Qdrant.from_existing_collection(
            path=None,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False,
        )
        print(f"Retrieving documents from Qdrant for {len(search_queries)} search queries")
        
        all_docs = []
        for query in search_queries:
            docs = qdrant.similarity_search_with_score(query, k=SEARCH_TOP_K)
            for doc,score in docs:
                all_docs.append({
                    "content": doc.page_content,
                    "metadata": {
                        **doc.metadata,
                        "score": score,
                        "query": query  
                    }
                })
        unique_docs = {}
        for doc in all_docs:
            doc_id = doc["metadata"].get("id", hash(doc["content"]))
            if doc_id not in unique_docs or doc["metadata"]["score"] > unique_docs[doc_id]["metadata"]["score"]:
                unique_docs[doc_id] = doc
        
        sorted_docs = sorted(unique_docs.values(), key=lambda x: x["metadata"]["score"], reverse=True)
        
        print(f"Retrieved {len(sorted_docs)} unique documents after deduplication")
        return sorted_docs
    except Exception as e:
        print(f"Error retrieving documents from Qdrant: {str(e)}")
        return []
    
    """"
    Retrieve relevant documents from Qdrant based on multiple search queries using LangChain.
    
    Args:
        search_queries: List of search query strings
        
    Returns:
        List of document dictionaries with content and metadata
    """
    if not search_queries:
        print("No search queries provided")
        return []
    
    try:
        qdrant =Qdrant.from_existing_collection(
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False,
            search_kwargs={"k": SEARCH_TOP_K}
        )
        Logger.info(f"Retrieving documents from Qdrant for {len(search_queries)} search queries")
        
        all_docs = []
        for query in search_queries:
            docs = qdrant.similarity_search_with_score(query, k=SEARCH_TOP_K)
            for doc,score in docs:
                all.docs.append({
                    "content": doc.page_content,
                    "metadata": {
                        **doc.metadata,
                        "score": score,
                        "query": query  
                    }
                })
        unique_docs = {}
        for doc in all_docs:
            doc_id = doc["metadata"].get("id", hash(doc["content"]))
            if doc_id not in unique_docs or doc["metadata"]["score"] > unique_docs[doc_id]["metadata"]["score"]:
                unique_docs[doc_id] = doc
        
        # Sort by relevance score
        sorted_docs = sorted(unique_docs.values(), key=lambda x: x["metadata"]["score"], reverse=True)
        
        logger.info(f"Retrieved {len(sorted_docs)} unique documents after deduplication")
        return sorted_docs
    except Exception as e:
        logger.error(f"Error retrieving documents from Qdrant: {str(e)}")
        return []