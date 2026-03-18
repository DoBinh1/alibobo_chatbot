import os
from haystack.dataclasses import ByteStream
from indexing import IndexingPipelineWrapper


def seed_initial_data():
    print("Bắt đầu nạp dữ liệu ngữ cảnh ban đầu vào Qdrant...")
    
    # Khởi tạo pipeline của bạn
    indexer = IndexingPipelineWrapper()
    indexer.setup(index="initial_data")
    
    # Giả sử bạn để các file bài báo, file CSV ở thư mục này
    data_dir = "./initial_data" 
    sources = []
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        # Đọc file thành ByteStream và GẮN NHÃN "system_init"
        with open(file_path, "rb") as f:
            content = f.read()
            
        mime_type = "application/pdf" if filename.endswith(".pdf") else "text/plain"
            
        stream = ByteStream(
            data=content,
            mime_type=mime_type,
            meta={
                "file_name": filename,
                "source_type": "system_init",
            }
        )
        sources.append(stream)

    # Đẩy một lốc vào pipeline
    indexer.pipeline.run({"router": {"sources": sources}})
    print("Đã nạp xong dữ liệu ban đầu!")

if __name__ == "__main__":
    seed_initial_data()