import os
from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum

class Framework(Enum):
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph" 
    LLAMAINDEX = "llamaindex"
    CUSTOM = "custom"

class VectorStore(Enum):
    FAISS = "faiss"
    CHROMA = "chroma"
    WEAVIATE = "weaviate"
    PINECONE = "pinecone"

class RetrievalMethod(Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    BM25 = "bm25"

@dataclass
class RAGConfig:
    framework: Framework
    vector_store: VectorStore
    retrieval_method: RetrievalMethod
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-3.5-turbo"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    temperature: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework.value,
            "vector_store": self.vector_store.value,
            "retrieval_method": self.retrieval_method.value,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "temperature": self.temperature
        }

FRAMEWORK_OPTIONS = [f.value for f in Framework]
VECTOR_STORE_OPTIONS = [v.value for v in VectorStore]
RETRIEVAL_METHOD_OPTIONS = [r.value for r in RetrievalMethod]

EMBEDDING_MODELS = [
    "text-embedding-ada-002",
    "text-embedding-3-small", 
    "text-embedding-3-large",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"
]

LLM_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o"
]