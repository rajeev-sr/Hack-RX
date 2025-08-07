from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import ParentDocumentRetriever

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4

from langchain_core.documents import Document
from app.components.data_preproceesing import load_document
import os
from dotenv import load_dotenv

load_dotenv()

# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_API_KEY="AIzaSyBNmR0wTij9XaX2_R5dmUIQuoGgKIlyxm8"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def result(url):
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        length_function=len,
        is_separator_regex=False,
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        length_function=len,
        is_separator_regex=False,
    )
    store = InMemoryStore()

    index = faiss.IndexFlatL2(len(gemini_embeddings.embed_query("hello world")))

    vectorstore = FAISS(
        embedding_function=gemini_embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        
    )

    documents = load_document(url,source="file")
    retriever.add_documents(documents=documents)
    res=retriever.invoke("tell me about knee surgery")
    return res


url="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

print(result(url))