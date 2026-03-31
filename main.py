from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.concurrency import asynccontextmanager
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uvicorn
import asyncio

# Thay đổi import: Không cần ByteStream nữa
from pipeline.indexing_pipeline.Qdrant_indexing import IndexingPipelineWrapper
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from pipeline.query_pipeline.prompt_query import QueryPipelineWrapper

# ------ Thiết lập thư mục lưu trữ ------
UPLOAD_DIR = "uploaded_files"
IMAGE_DIR = "extracted_images" # Thư mục chứa ảnh do pymupdf4llm bóc ra

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ------ Khởi tạo Vector Database ------
# 1. Database chứa dữ liệu upload của user
store_user = QdrantDocumentStore(
    path="qdrant_user_vectordb",
    index="Document", 
    embedding_dim=768, 
    use_sparse_embeddings=False
)

# 2. Database chứa dữ liệu gốc của hệ thống
store_initial = QdrantDocumentStore(
    path="qdrant_initial_vectordb",
    index="Document", 
    embedding_dim=768, 
    use_sparse_embeddings=False
)

# ------ Khởi tạo Pipeline ------
upload_data_indexer = IndexingPipelineWrapper()
upload_data_indexer.setup(document_store=store_user)

query_engine = QueryPipelineWrapper()
query_engine.setup(init_document_stores=store_initial, user_document_store=store_user)

# ------ Hàm xử lý cốt lõi ------
def process_file_to_memory(file_path: str, filename: str):
    """Hàm phụ trợ đưa file vào pipeline RAG Đa phương thức"""
    print(f"Bắt đầu bóc tách và tạo vector cho: {filename}")
    
    # Kiến trúc mới chỉ cần truyền trực tiếp đường dẫn file vào trạm 'converter'
    upload_data_indexer.pipeline.run(
        {
            "converter": {
                "sources": [file_path]
            }
        }
    )
    print(f"Hoàn tất đưa vào VectorDB: {filename}")

# ------ Quản lý vòng đời (Lifespan) ------
async def index_existing_files():
    # Quét thư mục và nạp dữ liệu có sẵn (chạy nền, không block startup)
    all_files = os.listdir(UPLOAD_DIR)
    if not all_files:
        print("Không có file cũ để nạp.")
        return

    print(f"Đang nạp {len(all_files)} file cũ từ thư mục upload trong background...")
    for filename in all_files:
        file_path = os.path.join(UPLOAD_DIR, filename)
        try:
            await asyncio.to_thread(process_file_to_memory, file_path, filename)
        except Exception as ex:
            print(f"Lỗi khi nạp file cũ {filename}: {ex}")
    print("Hoàn tất nạp file cũ!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi động nhanh: tạo task background rồi trả về app ngay
    background_task = asyncio.create_task(index_existing_files())
    yield

    # Khi app tắt, chờ task hoàn thành (tùy lựa chọn)
    if not background_task.done():
        try:
            await background_task
        except asyncio.CancelledError:
            print("Background indexing đã bị hủy trong lúc shutdown.")

# ------ Khởi tạo FastAPI App ------
app = FastAPI(lifespan=lifespan)

# Mở kết nối thư mục tĩnh để frontend có thể hiển thị ảnh
app.mount("/static_images", StaticFiles(directory=IMAGE_DIR), name="static_images")

# ------ Các API Endpoints ------
@app.post("/api/chat")
async def chat_and_upload(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None) 
):
    try:
        uploaded_filename = None
        
        # 1. Kiểm tra và lưu file (nếu có)
        if file and file.filename:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            # Xử lý file bằng LLM và lưu vào Qdrant
            process_file_to_memory(file_path, file.filename)
            uploaded_filename = file.filename

        # 2. Xử lý câu hỏi (gọi AI)
        response_data = query_engine.ask(question)
        
        # 3. Xử lý đường dẫn ảnh cho Frontend
        sources = response_data["sources"]
        for source in sources:
            img_path = source.get("image_path")
            if img_path:
                # Biến đường dẫn vật lý: extracted_images/file/img.png 
                # Thành URL hợp lệ: /static_images/file/img.png
                img_url = img_path.replace("\\", "/").replace("extracted_images", "/static_images")
                source["image_url"] = img_url

        return {
            "question": question,
            "answer": response_data["answer"],
            "sources": sources,
            "attached_file": uploaded_filename
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc() # In chi tiết lỗi ra terminal để dễ debug
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)