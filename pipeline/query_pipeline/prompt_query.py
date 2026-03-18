from hayhooks import BasePipelineWrapper  # <--- Import thêm cái này
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
# from haystack.components.builders import ChatPromptBuilder
# from haystack.dataclasses import ChatMessage
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack.components.joiners import DocumentJoiner

class QueryPipelineWrapper(BasePipelineWrapper):
    def setup(self, init_document_stores, user_document_store) -> None:
        query_pipeline = Pipeline()
        
        sparse_text_embedder = FastembedSparseTextEmbedder(model="Qdrant/bm42-all-minilm-l6-v2-attentions")
        dense_text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

        template = """You are an expert assistant. Your task is to answer the question based ONLY on the provided documents.
                
                CRITICAL INSTRUCTIONS:
                1. You MUST cite the source of your information.
                2. Use the format [file_name] at the end of the sentence or paragraph where you use that information from file.
                3. If the answer is not contained in the provided documents, politely say "I do not have enough information to answer this question" and DO NOT guess.

                Here are the relevant informations from User Uploads:
                {% for info in user_info %}
                    Source: [{{ info.meta.file_name }}]
                    Content: {{ info.content }}
                    ---
                {% endfor %}

                Here are the relevant informations from Internal Documents:
                {% for info in internal_info %}
                    Source: [{{ info.meta.file_name }}]
                    Content: {{ info.content }}
                    ---
                {% endfor %}

                Question: {{question}}
                """
        

        prompt_builder = PromptBuilder(template=template, required_variables=["question", "user_info", "internal_info"])
        generator = OllamaGenerator(model="qwen2.5:1.5b", url="http://localhost:11434")

        retriever_upload_data = QdrantHybridRetriever(document_store=user_document_store)
        retriever_initial_data = QdrantHybridRetriever(document_store=init_document_stores)
        
        query_pipeline.add_component("sparse_text_embedder", sparse_text_embedder)
        query_pipeline.add_component("dense_text_embedder", dense_text_embedder)
        query_pipeline.add_component("retriever_upload_data", retriever_upload_data)
        query_pipeline.add_component("retriever_initial_data", retriever_initial_data)
        query_pipeline.add_component("prompt_builder", prompt_builder)
        query_pipeline.add_component("llm", generator)
        query_pipeline.add_component("documents_joiner", DocumentJoiner())

        # Retrieval connections
        query_pipeline.connect("dense_text_embedder.embedding", "retriever_upload_data.query_embedding")
        query_pipeline.connect("dense_text_embedder.embedding", "retriever_initial_data.query_embedding")
        query_pipeline.connect("sparse_text_embedder.sparse_embedding", "retriever_upload_data.query_sparse_embedding")
        query_pipeline.connect("sparse_text_embedder.sparse_embedding", "retriever_initial_data.query_sparse_embedding")
        # LLM response
        query_pipeline.connect("retriever_upload_data.documents", "prompt_builder.user_info")
        query_pipeline.connect("retriever_initial_data.documents", "prompt_builder.internal_info")
        query_pipeline.connect("prompt_builder.prompt", "llm.prompt")

        # Documents joiner
        query_pipeline.connect("retriever_upload_data.documents", "documents_joiner.documents")
        query_pipeline.connect("retriever_initial_data.documents", "documents_joiner.documents")


        self.pipeline = query_pipeline


    def ask(self, question: str) -> str:
        """Hàm công khai để gọi từ FastAPI"""
        result = self.pipeline.run(
            {
                "dense_text_embedder": {"text": question},
                "sparse_text_embedder": {"text": question},
                "prompt_builder": {"question": question}
            }
        )
        
        # 1. LLM response
        answer = result["llm"]["replies"][0]
        
        # 2. Documents Joiner
        retrieved_docs = result["documents_joiner"]["documents"]
        
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "file_name": doc.meta.get("file_name", "Tài liệu không xác định"),
                "source_type": doc.meta.get("source_type", "unknown"),
                "content_snippet": doc.content + "..." # Trích 250 ký tự đầu tiên
            })
            
        return {
            "answer": answer,
            "sources": sources
        }