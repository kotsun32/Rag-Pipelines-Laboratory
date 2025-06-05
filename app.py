import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import time
import json

from config import (
    RAGConfig, Framework, VectorStore, RetrievalMethod,
    FRAMEWORK_OPTIONS, VECTOR_STORE_OPTIONS, RETRIEVAL_METHOD_OPTIONS,
    EMBEDDING_MODELS, LLM_MODELS
)
from rag_pipeline import create_rag_pipeline, Document, QueryResult

class RAGApp:
    def __init__(self):
        self.current_pipeline = None
        self.indexed_documents = []
        self.query_history = []
        
    def load_sample_documents(self) -> List[Document]:
        """Load sample documents for testing"""
        sample_docs = [
            Document(
                content="Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents.",
                metadata={"source": "AI_overview", "topic": "artificial_intelligence"}
            ),
            Document(
                content="Machine Learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
                metadata={"source": "ML_basics", "topic": "machine_learning"}
            ),
            Document(
                content="Deep Learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input. It has revolutionized fields like computer vision and natural language processing.",
                metadata={"source": "DL_intro", "topic": "deep_learning"}
            ),
            Document(
                content="Natural Language Processing (NLP) is a branch of artificial intelligence that deals with the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human language.",
                metadata={"source": "NLP_guide", "topic": "nlp"}
            ),
            Document(
                content="Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models with external knowledge retrieval. It allows LLMs to access and incorporate relevant information from external sources when generating responses.",
                metadata={"source": "RAG_explanation", "topic": "rag"}
            )
        ]
        return sample_docs
    
    def setup_pipeline(
        self,
        framework: str,
        vector_store: str,
        retrieval_method: str,
        embedding_model: str,
        llm_model: str,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        temperature: float
    ) -> str:
        """Setup RAG pipeline with given configuration"""
        try:
            config = RAGConfig(
                framework=Framework(framework),
                vector_store=VectorStore(vector_store),
                retrieval_method=RetrievalMethod(retrieval_method),
                embedding_model=embedding_model,
                llm_model=llm_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                temperature=temperature
            )
            
            self.current_pipeline = create_rag_pipeline(config)
            
            # Load and index sample documents
            self.indexed_documents = self.load_sample_documents()
            self.current_pipeline.index_documents(self.indexed_documents)
            
            return f"✅ Pipeline setup complete!\n\nFramework: {framework}\nVector Store: {vector_store}\nRetrieval: {retrieval_method}\nEmbedding Model: {embedding_model}\nLLM: {llm_model}\n\nIndexed {len(self.indexed_documents)} sample documents."
            
        except Exception as e:
            return f"❌ Error setting up pipeline: {str(e)}"
    
    def query_pipeline(self, question: str) -> tuple:
        """Query the current RAG pipeline"""
        if not self.current_pipeline:
            return "❌ Please setup a pipeline first!", "", ""
        
        if not question.strip():
            return "❌ Please enter a question!", "", ""
        
        try:
            result = self.query_pipeline_internal(question)
            
            # Store query history
            self.query_history.append(result)
            
            # Format retrieved documents
            retrieved_docs_text = "## Retrieved Documents:\n\n"
            for i, doc in enumerate(result.retrieved_docs, 1):
                retrieved_docs_text += f"**Document {i}:**\n{doc.content}\n\n"
            
            # Format metrics
            metrics_text = f"""## Performance Metrics:
            
**Retrieval Time:** {result.retrieval_time:.3f}s
**Generation Time:** {result.generation_time:.3f}s  
**Total Time:** {result.total_time:.3f}s
**Framework:** {result.config['framework']}
**Vector Store:** {result.config['vector_store']}
**Top-K:** {result.config['top_k']}
"""
            
            return result.answer, retrieved_docs_text, metrics_text
            
        except Exception as e:
            return f"❌ Error querying pipeline: {str(e)}", "", ""
    
    def query_pipeline_internal(self, question: str) -> QueryResult:
        """Internal method to query pipeline and return result object"""
        return self.current_pipeline.query(question)
    
    def compare_configurations(
        self,
        question: str,
        configs_json: str
    ) -> tuple:
        """Compare multiple pipeline configurations"""
        if not question.strip():
            return "❌ Please enter a question for comparison!", ""
        
        try:
            configs = json.loads(configs_json)
        except:
            configs = [
                {"framework": "custom", "vector_store": "faiss", "retrieval_method": "semantic"},
                {"framework": "langchain", "vector_store": "chroma", "retrieval_method": "semantic"},
            ]
        
        results = []
        comparison_data = []
        
        for i, config_dict in enumerate(configs):
            try:
                config = RAGConfig(**config_dict)
                pipeline = create_rag_pipeline(config)
                pipeline.index_documents(self.indexed_documents)
                
                result = pipeline.query(question)
                results.append(result)
                
                comparison_data.append({
                    "Configuration": f"Config {i+1}",
                    "Framework": config.framework.value,
                    "Vector Store": config.vector_store.value,
                    "Retrieval Time (s)": result.retrieval_time,
                    "Generation Time (s)": result.generation_time,
                    "Total Time (s)": result.total_time,
                    "Answer": result.answer[:100] + "..." if len(result.answer) > 100 else result.answer
                })
            except Exception as e:
                comparison_data.append({
                    "Configuration": f"Config {i+1}",
                    "Framework": config_dict.get("framework", "unknown"),
                    "Vector Store": config_dict.get("vector_store", "unknown"),
                    "Retrieval Time (s)": 0,
                    "Generation Time (s)": 0,
                    "Total Time (s)": 0,
                    "Answer": f"Error: {str(e)}"
                })
        
        # Create comparison table
        df = pd.DataFrame(comparison_data)
        
        # Create performance chart
        fig = px.bar(
            df, 
            x="Configuration", 
            y=["Retrieval Time (s)", "Generation Time (s)"],
            title="Performance Comparison Across Configurations",
            barmode="stack"
        )
        
        return df.to_string(index=False), fig
    
    def get_query_history(self) -> str:
        """Get formatted query history"""
        if not self.query_history:
            return "No queries yet!"
        
        history_text = "## Query History\n\n"
        for i, result in enumerate(self.query_history[-10:], 1):  # Show last 10 queries
            history_text += f"**Query {i}:**\n"
            history_text += f"Answer: {result.answer[:200]}{'...' if len(result.answer) > 200 else ''}\n"
            history_text += f"Total Time: {result.total_time:.3f}s\n"
            history_text += f"Framework: {result.config['framework']}\n\n"
        
        return history_text

# Initialize the app
app = RAGApp()

# Create Gradio interface
with gr.Blocks(title="RAG Pipeline Lab", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 RAG Pipeline Laboratory")
    gr.Markdown("Experiment with different RAG configurations and see how they affect performance!")
    
    with gr.Tab("🛠️ Pipeline Setup"):
        gr.Markdown("## Configure Your RAG Pipeline")
        
        with gr.Row():
            with gr.Column():
                framework_dropdown = gr.Dropdown(
                    choices=FRAMEWORK_OPTIONS,
                    value="custom",
                    label="Framework",
                    info="Choose the RAG framework"
                )
                
                vector_store_dropdown = gr.Dropdown(
                    choices=VECTOR_STORE_OPTIONS,
                    value="faiss",
                    label="Vector Store",
                    info="Choose the vector database"
                )
                
                retrieval_dropdown = gr.Dropdown(
                    choices=RETRIEVAL_METHOD_OPTIONS,
                    value="semantic",
                    label="Retrieval Method",
                    info="Choose the retrieval algorithm"
                )
                
                embedding_dropdown = gr.Dropdown(
                    choices=EMBEDDING_MODELS,
                    value="text-embedding-ada-002",
                    label="Embedding Model"
                )
            
            with gr.Column():
                llm_dropdown = gr.Dropdown(
                    choices=LLM_MODELS,
                    value="gpt-3.5-turbo",
                    label="LLM Model"
                )
                
                chunk_size_slider = gr.Slider(
                    minimum=200,
                    maximum=2000,
                    value=1000,
                    step=100,
                    label="Chunk Size"
                )
                
                chunk_overlap_slider = gr.Slider(
                    minimum=0,
                    maximum=500,
                    value=200,
                    step=50,
                    label="Chunk Overlap"
                )
                
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Top-K Retrieval"
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    label="Temperature"
                )
        
        setup_button = gr.Button("🚀 Setup Pipeline", variant="primary")
        setup_output = gr.Textbox(label="Setup Status", interactive=False)
        
        setup_button.click(
            fn=app.setup_pipeline,
            inputs=[
                framework_dropdown, vector_store_dropdown, retrieval_dropdown,
                embedding_dropdown, llm_dropdown, chunk_size_slider,
                chunk_overlap_slider, top_k_slider, temperature_slider
            ],
            outputs=setup_output
        )
    
    with gr.Tab("💬 Query Pipeline"):
        gr.Markdown("## Ask Questions")
        
        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What is artificial intelligence?",
                    lines=3
                )
                query_button = gr.Button("🔍 Query", variant="primary")
            
            with gr.Column():
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=5,
                    interactive=False
                )
        
        with gr.Row():
            retrieved_docs_output = gr.Textbox(
                label="Retrieved Documents",
                lines=8,
                interactive=False
            )
            metrics_output = gr.Textbox(
                label="Performance Metrics",
                lines=8,
                interactive=False
            )
        
        query_button.click(
            fn=app.query_pipeline,
            inputs=question_input,
            outputs=[answer_output, retrieved_docs_output, metrics_output]
        )
    
    with gr.Tab("📊 Comparison"):
        gr.Markdown("## Compare Different Configurations")
        
        comparison_question = gr.Textbox(
            label="Question for Comparison",
            placeholder="What is machine learning?",
            lines=2
        )
        
        configs_input = gr.Textbox(
            label="Configurations (JSON)",
            value='[{"framework": "custom", "vector_store": "faiss"}, {"framework": "custom", "vector_store": "chroma"}]',
            lines=5
        )
        
        compare_button = gr.Button("🔬 Compare Configurations", variant="primary")
        
        with gr.Row():
            comparison_table = gr.Textbox(label="Comparison Results", lines=10)
            comparison_chart = gr.Plot(label="Performance Chart")
        
        compare_button.click(
            fn=app.compare_configurations,
            inputs=[comparison_question, configs_input],
            outputs=[comparison_table, comparison_chart]
        )
    
    with gr.Tab("📈 History"):
        gr.Markdown("## Query History & Analytics")
        
        history_button = gr.Button("📋 Show Query History")
        history_output = gr.Textbox(label="Query History", lines=15, interactive=False)
        
        history_button.click(
            fn=app.get_query_history,
            outputs=history_output
        )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)