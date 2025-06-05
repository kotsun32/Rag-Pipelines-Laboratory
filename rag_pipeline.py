import os
import time
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
from config import RAGConfig, Framework, VectorStore, RetrievalMethod

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = None
    
@dataclass
class QueryResult:
    answer: str
    retrieved_docs: List[Document]
    retrieval_time: float
    generation_time: float
    total_time: float
    config: Dict[str, Any]

class BaseRAGPipeline(ABC):
    def __init__(self, config: RAGConfig):
        self.config = config
        
    @abstractmethod
    def index_documents(self, documents: List[Document]) -> None:
        pass
    
    @abstractmethod
    def query(self, question: str) -> QueryResult:
        pass
    
    @abstractmethod
    def get_framework_info(self) -> str:
        pass

class CustomRAGPipeline(BaseRAGPipeline):
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.documents = []
        self.embeddings = []
        self.embedding_model = None
        self.vector_store = None
        self._setup_components()
    
    def _setup_components(self):
        if "sentence-transformers" in self.config.embedding_model:
            self.embedding_model = SentenceTransformer(self.config.embedding_model.split("/")[-1])
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            
        if self.config.vector_store == VectorStore.FAISS:
            import faiss
            self.vector_store = None  # Will initialize after indexing
        elif self.config.vector_store == VectorStore.CHROMA:
            import chromadb
            self.chroma_client = chromadb.Client()
            self.collection = None
    
    def index_documents(self, documents: List[Document]) -> None:
        self.documents = documents
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        if "sentence-transformers" in self.config.embedding_model:
            self.embeddings = self.embedding_model.encode(texts)
        else:
            # Use OpenAI embeddings
            response = openai.embeddings.create(
                input=texts,
                model=self.config.embedding_model
            )
            self.embeddings = [emb.embedding for emb in response.data]
        
        # Setup vector store
        if self.config.vector_store == VectorStore.FAISS:
            import faiss
            import numpy as np
            embeddings_array = np.array(self.embeddings).astype('float32')
            self.vector_store = faiss.IndexFlatIP(embeddings_array.shape[1])
            self.vector_store.add(embeddings_array)
        
        elif self.config.vector_store == VectorStore.CHROMA:
            self.collection = self.chroma_client.create_collection("rag_docs")
            self.collection.add(
                embeddings=self.embeddings,
                documents=texts,
                ids=[str(i) for i in range(len(texts))]
            )
    
    def _retrieve_documents(self, query: str) -> Tuple[List[Document], float]:
        start_time = time.time()
        
        # Generate query embedding
        if "sentence-transformers" in self.config.embedding_model:
            query_embedding = self.embedding_model.encode([query])
        else:
            response = openai.embeddings.create(
                input=[query],
                model=self.config.embedding_model
            )
            query_embedding = [response.data[0].embedding]
        
        # Retrieve similar documents
        if self.config.vector_store == VectorStore.FAISS:
            import numpy as np
            query_array = np.array(query_embedding).astype('float32')
            scores, indices = self.vector_store.search(query_array, self.config.top_k)
            retrieved_docs = [self.documents[idx] for idx in indices[0]]
        
        elif self.config.vector_store == VectorStore.CHROMA:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=self.config.top_k
            )
            retrieved_docs = [Document(content=doc) for doc in results['documents'][0]]
        
        retrieval_time = time.time() - start_time
        return retrieved_docs, retrieval_time
    
    def _generate_answer(self, query: str, context_docs: List[Document]) -> Tuple[str, float]:
        start_time = time.time()
        
        context = "\n\n".join([doc.content for doc in context_docs])
        prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""

        response = openai.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        generation_time = time.time() - start_time
        return answer, generation_time
    
    def query(self, question: str) -> QueryResult:
        total_start = time.time()
        
        # Retrieve relevant documents
        retrieved_docs, retrieval_time = self._retrieve_documents(question)
        
        # Generate answer
        answer, generation_time = self._generate_answer(question, retrieved_docs)
        
        total_time = time.time() - total_start
        
        return QueryResult(
            answer=answer,
            retrieved_docs=retrieved_docs,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            config=self.config.to_dict()
        )
    
    def get_framework_info(self) -> str:
        return f"Custom RAG Pipeline using {self.config.vector_store.value} vector store"

class LangChainRAGPipeline(BaseRAGPipeline):
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.embeddings import OpenAIEmbeddings
            from langchain.vectorstores import FAISS, Chroma
            from langchain.llms import OpenAI
            from langchain.chains import RetrievalQA
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            self.embeddings = OpenAIEmbeddings(model=config.embedding_model)
            self.vector_store = None
            self.qa_chain = None
        except ImportError as e:
            raise ImportError(f"LangChain dependencies not installed: {e}")
    
    def index_documents(self, documents: List[Document]) -> None:
        from langchain.vectorstores import FAISS, Chroma
        from langchain.schema import Document as LCDocument
        
        # Convert to LangChain documents
        lc_docs = [LCDocument(page_content=doc.content, metadata=doc.metadata or {}) 
                   for doc in documents]
        
        # Split documents
        split_docs = self.text_splitter.split_documents(lc_docs)
        
        # Create vector store
        if self.config.vector_store == VectorStore.FAISS:
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        elif self.config.vector_store == VectorStore.CHROMA:
            self.vector_store = Chroma.from_documents(split_docs, self.embeddings)
        
        # Create QA chain
        from langchain.llms import OpenAI
        from langchain.chains import RetrievalQA
        
        llm = OpenAI(model_name=self.config.llm_model, temperature=self.config.temperature)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": self.config.top_k})
        )
    
    def query(self, question: str) -> QueryResult:
        start_time = time.time()
        
        # Retrieve documents
        retrieval_start = time.time()
        retrieved_docs = self.vector_store.similarity_search(question, k=self.config.top_k)
        retrieval_time = time.time() - retrieval_start
        
        # Generate answer
        generation_start = time.time()
        answer = self.qa_chain.run(question)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        return QueryResult(
            answer=answer,
            retrieved_docs=[Document(content=doc.page_content, metadata=doc.metadata) 
                           for doc in retrieved_docs],
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            config=self.config.to_dict()
        )
    
    def get_framework_info(self) -> str:
        return f"LangChain RAG Pipeline with {self.config.vector_store.value}"

class LlamaIndexRAGPipeline(BaseRAGPipeline):
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        try:
            from llama_index import VectorStoreIndex, Document as LIDocument
            from llama_index.embeddings import OpenAIEmbedding
            from llama_index.llms import OpenAI
            
            self.index = None
            self.query_engine = None
        except ImportError as e:
            raise ImportError(f"LlamaIndex dependencies not installed: {e}")
    
    def index_documents(self, documents: List[Document]) -> None:
        from llama_index import VectorStoreIndex, Document as LIDocument, ServiceContext
        from llama_index.embeddings import OpenAIEmbedding
        from llama_index.llms import OpenAI
        
        # Convert to LlamaIndex documents
        li_docs = [LIDocument(text=doc.content, metadata=doc.metadata or {}) 
                   for doc in documents]
        
        # Setup service context
        embed_model = OpenAIEmbedding(model=self.config.embedding_model)
        llm = OpenAI(model=self.config.llm_model, temperature=self.config.temperature)
        
        service_context = ServiceContext.from_defaults(
            embed_model=embed_model,
            llm=llm,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Create index
        self.index = VectorStoreIndex.from_documents(
            li_docs, 
            service_context=service_context
        )
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.config.top_k
        )
    
    def query(self, question: str) -> QueryResult:
        start_time = time.time()
        
        # Query the index
        response = self.query_engine.query(question)
        
        total_time = time.time() - start_time
        
        # Extract retrieved documents from response
        retrieved_docs = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                retrieved_docs.append(Document(
                    content=node.node.text,
                    metadata=node.node.metadata
                ))
        
        return QueryResult(
            answer=str(response),
            retrieved_docs=retrieved_docs,
            retrieval_time=total_time * 0.6,  # Estimate
            generation_time=total_time * 0.4,  # Estimate
            total_time=total_time,
            config=self.config.to_dict()
        )
    
    def get_framework_info(self) -> str:
        return f"LlamaIndex RAG Pipeline"

def create_rag_pipeline(config: RAGConfig) -> BaseRAGPipeline:
    if config.framework == Framework.CUSTOM:
        return CustomRAGPipeline(config)
    elif config.framework == Framework.LANGCHAIN:
        return LangChainRAGPipeline(config)
    elif config.framework == Framework.LLAMAINDEX:
        return LlamaIndexRAGPipeline(config)
    elif config.framework == Framework.LANGGRAPH:
        # LangGraph implementation would go here
        raise NotImplementedError("LangGraph implementation coming soon")
    else:
        raise ValueError(f"Unknown framework: {config.framework}")