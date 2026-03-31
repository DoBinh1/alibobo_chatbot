import ollama
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any
from hayhooks import BasePipelineWrapper 
from haystack import Pipeline, component, Document
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.joiners import DocumentJoiner

# ---------------------------------------------------------
# 1. CUSTOM COMPONENT: NOMIC QUERY EMBEDDER
# ---------------------------------------------------------
@component
class NomicQueryEmbedder:
    """Embedder chuyên dụng cho câu hỏi người dùng (Quy về vector 768 chiều)"""
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
        self.model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(self.device)
        self.model.eval()

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        # NOMIC BẮT BUỘC dùng tiền tố "search_query: " cho câu hỏi
        query_text = f"search_query: {text}"
        
        inputs = self.tokenizer(query_text, padding=True, truncation=True, max_length=8192, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**inputs)
            
        # Tính Mean Pooling
        token_embeddings = model_output[0]
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return {"embedding": embeddings[0].cpu().tolist()}

# ---------------------------------------------------------
# 2. CUSTOM COMPONENT: MULTIMODAL GENERATOR
# ---------------------------------------------------------
@component
class MultimodalOllamaGenerator:
    """Custom Generator tự động trích xuất ảnh từ Document để đưa vào Vision LLM"""
    def __init__(self, model: str = "moondream"):
        self.model = model

    # 1. SỬA LẠI: Khai báo thêm 'documents' ở đầu ra
    @component.output_types(replies=List[str], documents=List[Document])
    def run(self, prompt: str, documents: List[Document]):
        images = []
        
        for doc in documents:
            img_path = doc.meta.get("image_path")
            if img_path and img_path not in images: 
                images.append(img_path)
        
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            images=images if images else None,
            options={"temperature": 0.2}
        )
        
        # 2. SỬA LẠI: Trả về kèm theo cả danh sách documents gốc
        return {
            "replies": [response['response']], 
            "documents": documents
        }

# ---------------------------------------------------------
# 3. QUERY PIPELINE WRAPPER
# ---------------------------------------------------------
class QueryPipelineWrapper(BasePipelineWrapper):
    def setup(self, init_document_stores, user_document_store) -> None:
        query_pipeline = Pipeline()

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
        
        # Sử dụng các Custom Component
        query_embedder = NomicQueryEmbedder()
        generator = MultimodalOllamaGenerator(model="moondream")

        # Chuyển thành QdrantEmbeddingRetriever vì Nomic chỉ dùng Dense Vector (768 chiều)
        retriever_upload_data = QdrantEmbeddingRetriever(document_store=user_document_store)
        retriever_initial_data = QdrantEmbeddingRetriever(document_store=init_document_stores)
        documents_joiner = DocumentJoiner() # Gom tài liệu từ cả 2 nguồn
        
        query_pipeline.add_component("query_embedder", query_embedder)
        query_pipeline.add_component("retriever_upload_data", retriever_upload_data)
        query_pipeline.add_component("retriever_initial_data", retriever_initial_data)
        query_pipeline.add_component("prompt_builder", prompt_builder)
        query_pipeline.add_component("documents_joiner", documents_joiner)
        query_pipeline.add_component("llm", generator)

        # Retrieval connections: Từ Embedder truyền vector sang Retriever
        query_pipeline.connect("query_embedder.embedding", "retriever_upload_data.query_embedding")
        query_pipeline.connect("query_embedder.embedding", "retriever_initial_data.query_embedding")
        
        # Nối vào PromptBuilder để dựng text
        query_pipeline.connect("retriever_upload_data.documents", "prompt_builder.user_info")
        query_pipeline.connect("retriever_initial_data.documents", "prompt_builder.internal_info")
        
        # Nối vào Joiner để gom chung danh sách documents (cả ảnh và text)
        query_pipeline.connect("retriever_upload_data.documents", "documents_joiner.documents")
        query_pipeline.connect("retriever_initial_data.documents", "documents_joiner.documents")

        # Nối cả Prompt (Text) và Documents (để lấy link ảnh) vào Generator
        query_pipeline.connect("prompt_builder.prompt", "llm.prompt")
        query_pipeline.connect("documents_joiner.documents", "llm.documents")

        self.pipeline = query_pipeline

    def ask(self, question: str) -> dict:
        """Hàm công khai để gọi từ FastAPI"""
        result = self.pipeline.run(
            {
                "query_embedder": {"text": question},
                "prompt_builder": {"question": question}
            }
        )
        
        # 1. Câu trả lời từ LLM
        answer = result["llm"]["replies"][0]
        
        # 2. SỬA Ở ĐÂY: Lấy danh sách tài liệu từ "llm" thay vì "documents_joiner"
        retrieved_docs = result["llm"]["documents"]
        
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "file_name": doc.meta.get("file_name", "Tài liệu không xác định"),
                "source_type": doc.meta.get("source_type", "unknown"),
                "image_path": doc.meta.get("image_path", None),
                "content_snippet": doc.content[:250] + "..." 
            })
            
        return {
            "answer": answer,
            "sources": sources
        }