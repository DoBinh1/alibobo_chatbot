from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel
from haystack import Pipeline
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
import os
import shutil
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
import uvicorn
from haystack.dataclasses import ByteStream
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from vectorDB_pipeline.indexing_pipeline.InMemoryDocument_indexing import IndexingPipelineWrapper
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


class QueryRequest(BaseModel):
    question: str

#------ Process user-uploaded file ------
def process_file_to_memory(file_path: str, filename: str):
    """Hàm phụ trợ để biến 1 file thành ByteStream và đưa vào pipeline"""
    with open(file_path, "rb") as f:
        content = f.read()
    
    mime_type = "application/pdf" if filename.endswith(".pdf") else "text/plain"
    
    stream = ByteStream(
        data=content,
        mime_type=mime_type,
        meta={
            "file_name": filename,
            "source_type": "user_upload",
        }
    )
    upload_data_indexer.pipeline.run({"router": {"sources": [stream]}})

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Quét thư mục và nạp dữ liệu có sẵn khi vừa bật server
    print("Đang nạp dữ liệu cũ vào RAM...")
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        process_file_to_memory(file_path, filename)
    print("Hoàn tất khởi động!")
    yield

app = FastAPI(lifespan=lifespan)

#------ Tạo thư mục lưu trữ dữ liệu nếu chưa tồn tại ------
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


#------ Thiết lập Document Store và Query Pipeline ------
# document_store = InMemoryDocumentStore()

upload_file = []
    
# Đọc tất cả file trong thư mục upload và gắn nhãn "user_upload"
for filename in os.listdir(UPLOAD_DIR):
    file_path = os.path.join(UPLOAD_DIR, filename)
        
    with open(file_path, "rb") as f:
        content = f.read()
            
    mime_type = "application/pdf" if filename.endswith(".pdf") else "text/plain"
            
    stream = ByteStream(
        data=content,
        mime_type=mime_type,
        meta={
            "file_name": filename,
            "source_type": "user_upload",
        }
    )
    upload_file.append(stream)
    

upload_data_indexer = IndexingPipelineWrapper()
upload_data_indexer.setup()
upload_data_indexer.pipeline.run({"router": {"sources": upload_file}})

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

retriever_upload_data = InMemoryEmbeddingRetriever(upload_data_indexer.document_store)
retriever_initial_data = QdrantEmbeddingRetriever(document_store=QdrantDocumentStore(path="qdrant_data",embedding_dim=384,))

template = [
    ChatMessage.from_user(
        """You are an expert assistant. Your task is to answer the question based ONLY on the provided documents.
        
        CRITICAL INSTRUCTIONS:
        1. You MUST cite the source of your information.
        2. Use the format [file name] at the end of the sentence or paragraph where you use that information from file.
        3. If the answer is not contained in the provided documents, politely say "I do not have enough information to answer this question" and DO NOT guess.

        Here is the relevant information from User Uploads:
        {% for info in user_info %}
            Source: [{{ info.meta.file_name }}]
            Content: {{ info.content }}
            ---
        {% endfor %}

        Here is the relevant information from Internal Documents:
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
# Haystack Pipeline
generator = OllamaGenerator(
    model="qwen2.5:1.5b",
    url="http://localhost:11434"
)

query_pipeline = Pipeline()
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



# Định nghĩa Endpoint (API Route)
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 1. Lấy tên file và tạo đường dẫn lưu trữ
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # 2. Lưu dữ liệu từ request vào ổ cứng
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Xử lý file và đưa vào pipeline
        process_file_to_memory(file_path, file.filename)

        return {
            "status": "success",
            "message": "Đã lưu file thành công vào hệ thống!",
            "filename": file.filename,
            "saved_path": file_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Có lỗi xảy ra khi lưu file: {str(e)}")

@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    try:
        # Đưa câu hỏi của user vào pipeline
        result = query_pipeline.run(
            {
                "text_embedder": {"text": request.question},
                "prompt_builder": {"question": request.question}
            }
        )
        
        # Trích xuất câu trả lời
        answer = result["llm"]["replies"][0]
        
        return {
            "question": request.question,
            "answer": answer
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)