from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.concurrency import asynccontextmanager
import os
import shutil
import uvicorn
from haystack.dataclasses import ByteStream
from pipeline.indexing_pipeline.Qdrant_indexing import IndexingPipelineWrapper
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from pipeline.query_pipeline.prompt_query import QueryPipelineWrapper



#------ Thiết lập thư mục lưu trữ file upload -----

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

store_user = QdrantDocumentStore(
    path="qdrant_user_vectordb",
    index="Document", 
    embedding_dim=384, 
    use_sparse_embeddings=True
)

# 2. Database chứa dữ liệu gốc của hệ thống
store_initial = QdrantDocumentStore(
    path="qdrant_initial_vectordb",
    index="Document", 
    embedding_dim=384, 
    use_sparse_embeddings=True
)

#------ Process user-uploaded file ------
upload_data_indexer = IndexingPipelineWrapper()
upload_data_indexer.setup(document_store=store_user)

query_engine = QueryPipelineWrapper()
query_engine.setup(init_document_stores=store_initial, user_document_store=store_user)

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
    print("Đang nạp dữ liệu cũ ...")
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        process_file_to_memory(file_path, filename)
    print("Hoàn tất khởi động!")
    yield




app = FastAPI(lifespan=lifespan)

@app.post("/api/chat")
async def chat_and_upload(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None) 
):
    try:
        uploaded_filename = None
        
        # 1. Check file
        if file is not None and getattr(file, "filename", "") != "":
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            process_file_to_memory(file_path, file.filename)
            uploaded_filename = file.filename

        # 2. LLM response 
        response_data = query_engine.ask(question)
        
        return {
            "question": question,
            "answer": response_data["answer"],
            "sources": response_data["sources"],
            "attached_file": uploaded_filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# @app.post("/api/chat")
# async def chat_and_upload(
#     request: QueryRequest,
# ):
#     try:
#         uploaded_filename = None
        
#         # 1. Check file (Đảm bảo nó là UploadFile xịn chứ không phải chuỗi rỗng)
#         if file and isinstance(file, UploadFile) and file.filename:
#             file_path = os.path.join(UPLOAD_DIR, file.filename)
            
#             with open(file_path, "wb") as buffer:
#                 shutil.copyfileobj(file.file, buffer)
                
#             process_file_to_memory(file_path, file.filename)
#             uploaded_filename = file.filename

#         # 2. Xử lý câu hỏi (gọi AI)
#         response_data = query_engine.ask(question)
        
#         return {
#             "question": question,
#             "answer": response_data["answer"],
#             "sources": response_data["sources"],
#             "attached_file": uploaded_filename
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))




# @app.post("/api/upload")
# async def upload_file(file: UploadFile = File(...)):
#     try:
#         # 1. Lấy tên file và tạo đường dẫn lưu trữ
#         file_path = os.path.join(UPLOAD_DIR, file.filename)
        
#         # 2. Lưu dữ liệu từ request vào ổ cứng
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
            
#         # 3. Xử lý file và đưa vào pipeline
#         process_file_to_memory(file_path, file.filename)

#         return {
#             "status": "success",
#             "message": "Đã lưu file thành công vào hệ thống!",
#             "filename": file.filename,
#             "saved_path": file_path
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Có lỗi xảy ra khi lưu file: {str(e)}")

# @app.post("/api/ask")
# async def ask_question(request: QueryRequest):
#     try:
#         # Đưa câu hỏi của user vào pipeline
#         result = query_engine.pipeline.run(
#             {
#                 "text_embedder": {"text": request.question},
#                 "prompt_builder": {"question": request.question}
#             }
#         )
        
#         # Trích xuất câu trả lời
#         answer = result["llm"]["replies"][0]
        
#         return {
#             "question": request.question,
#             "answer": answer
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)