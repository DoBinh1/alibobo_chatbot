from hayhooks import BasePipelineWrapper  # <--- Import thêm cái này
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

class QueryPipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        query_pipeline = Pipeline()
        
        text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

        template = [
            ChatMessage.from_user(
                """You are an expert assistant. Your task is to answer the question based ONLY on the provided documents.
                
                CRITICAL INSTRUCTIONS:
                1. You MUST cite the source of your information.
                2. Use the format [file name] at the end of the sentence or paragraph where you use that information from file.
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
            )
        ]

        prompt_builder = ChatPromptBuilder(template=template, required_variables=["question", "user_info", "internal_info"])
        generator = OllamaGenerator(model="qwen2.5:1.5b", url="http://localhost:11434")

        retriever_upload_data = QdrantEmbeddingRetriever(document_store=QdrantDocumentStore(path="qdrant_vectordb", index="user_uploaded_data", embedding_dim=384))
        retriever_initial_data = QdrantEmbeddingRetriever(document_store=QdrantDocumentStore(path="qdrant_vectordb", index="initial_data", embedding_dim=384))

        query_pipeline.add_component("text_embedder", text_embedder)
        query_pipeline.add_component("retriever_upload_data", retriever_upload_data)
        query_pipeline.add_component("retriever_initial_data", retriever_initial_data)
        query_pipeline.add_component("prompt_builder", prompt_builder)
        query_pipeline.add_component("llm", generator)

        query_pipeline.connect("text_embedder.embedding", "retriever_upload_data.query_embedding")
        query_pipeline.connect("retriever_upload_data.documents", "prompt_builder.user_info")
        query_pipeline.connect("text_embedder.embedding", "retriever_initial_data.query_embedding")
        query_pipeline.connect("retriever_initial_data.documents", "prompt_builder.internal_info")
        query_pipeline.connect("prompt_builder.prompt", "llm.prompt")

        self.pipeline = query_pipeline


    def ask(self, question: str) -> str:
        """Hàm công khai để gọi từ FastAPI"""
        result = self.pipeline.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question}
            }
        )
        return result["llm"]["replies"][0]