import os
import shutil
from typing import List, Optional
from fastapi import UploadFile
from haystack import Pipeline, component, Document
from hayhooks import BasePipelineWrapper, log

from haystack.components.preprocessors import MarkdownHeaderSplitter, DocumentSplitter, DocumentCleaner
from haystack.components.joiners import DocumentJoiner
from haystack.components.writers import DocumentWriter

from pipeline.indexing_pipeline.document_processor import DocumentConverter, ContextualEnhancer

import torch
import torch.nn.functional as F
from PIL import Image
from typing import List
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor

import dataclasses

@component
class NomicMultimodalEmbedder:
    """
    Trạm 3: Embedder đa phương thức Nomic.
    Tự động phân loại: Text -> Nomic-Embed-Text | Ảnh -> Nomic-Embed-Vision
    Tất cả được quy về một không gian vector 768 chiều.
    """
    def __init__(self, device: str = None):
        # 1. Chọn phần cứng (Tự động dùng Card đồ họa nếu có, nếu không chạy CPU)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Đang tải Nomic Multimodal Models lên hệ thống ({self.device})...")
        
        # 2. Khởi tạo Model Văn Bản (Text)
        self.text_tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
        self.text_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(self.device)
        self.text_model.eval()

        # 3. Khởi tạo Model Hình Ảnh (Vision)
        self.image_processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        self.vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True).to(self.device)
        self.vision_model.eval()
        
        print("Đã tải xong Nomic Models! Sẵn sàng nhúng dữ liệu.")

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        embedded_docs = []
        
        for doc in documents:
            img_path = doc.meta.get("image_path")
            
            if img_path and os.path.exists(img_path):
                # ==========================================
                # XỬ LÝ NHÚNG VECTOR CHO ẢNH
                # ==========================================
                try:
                    image = Image.open(img_path).convert("RGB")
                    inputs = self.image_processor(image, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        img_emb = self.vision_model(**inputs).last_hidden_state
                        # Nomic Vision lấy token CLS (vị trí 0) làm vector đại diện cho ảnh
                        img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
                        

                    new_embedding = img_embeddings[0].cpu().tolist()
                    updated_doc = dataclasses.replace(doc, embedding=new_embedding)
                    embedded_docs.append(updated_doc)
                except Exception as e:
                    print(f"Lỗi khi nhúng ảnh {img_path}: {e}")
            else:
                # ==========================================
                # XỬ LÝ NHÚNG VECTOR CHO VĂN BẢN
                # ==========================================
                try:
                    # Model Nomic v1.5 BẮT BUỘC phải có tiền tố 'search_document: ' cho văn bản lưu vào Database
                    text_content = f"search_document: {doc.content}"
                    
                    inputs = self.text_tokenizer(text_content, padding=True, truncation=True, max_length=8192, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        text_emb = self.text_model(**inputs)
                    
                    # Nomic Text yêu cầu tính Mean Pooling cho toàn bộ token
                    embeddings = self._mean_pooling(text_emb, inputs['attention_mask'])
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                    new_embedding = embeddings[0].cpu().tolist()
                    updated_doc = dataclasses.replace(doc, embedding=new_embedding)
                    embedded_docs.append(updated_doc)
                except Exception as e:
                    print(f"Lỗi khi nhúng văn bản: {e}")
                    
        return {"documents": embedded_docs}
        
    def _mean_pooling(self, model_output, attention_mask):
        """Hàm toán nội bộ để tính trung bình các token sinh ra bởi model Text"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# ---------------------------------------------------------
# PIPELINE WRAPPER
# ---------------------------------------------------------
class IndexingPipelineWrapper(BasePipelineWrapper):
    def setup(self, document_store) -> None:
        indexing = Pipeline()
        
        # 1. THÊM CÁC COMPONENT
        indexing.add_component("converter", DocumentConverter())
        
        # Nhánh Text
        # indexing.add_component("cleaner", DocumentCleaner(
        #     remove_empty_lines=True,
        #     remove_extra_whitespaces=True,
        #     remove_repeated_substrings=False
        # ))
        indexing.add_component("md_splitter", MarkdownHeaderSplitter(keep_headers=True))
        indexing.add_component("sentence_splitter", DocumentSplitter(split_by="sentence", split_length=10, split_overlap=2))
        indexing.add_component("context_enhancer", ContextualEnhancer())
        
        # Gộp nhánh và Vector hóa đa phương thức
        indexing.add_component("joiner", DocumentJoiner())
        indexing.add_component("multimodal_embedder", NomicMultimodalEmbedder())
        indexing.add_component("writer", DocumentWriter(document_store=document_store, policy="overwrite"))

        # 2. KẾT NỐI PIPELINE
        # Nhánh Text: Cắt chia -> Gắn Context
        # indexing.connect("converter.text_docs", "cleaner.documents")
        # indexing.connect("cleaner.documents", "md_splitter.documents")
        indexing.connect("converter.text_docs", "md_splitter.documents")
        indexing.connect("md_splitter.documents", "sentence_splitter.documents")
        indexing.connect("sentence_splitter.documents", "context_enhancer.documents")
        
        # Gộp nhánh: Đưa text (đã có context) và ảnh (nguyên bản) vào Joiner
        indexing.connect("context_enhancer.enhanced_docs", "joiner.documents")
        indexing.connect("converter.image_docs", "joiner.documents")
        
        # Chạy qua Embedder Đa phương thức và Ghi xuống Qdrant
        indexing.connect("joiner.documents", "multimodal_embedder.documents")
        indexing.connect("multimodal_embedder.documents", "writer.documents")

        self.pipeline = indexing

    def run_api(self, files: Optional[List[UploadFile]] = None) -> dict:
        if not files:
            return {"message": "No files provided for indexing."}
        
        UPLOAD_DIR = "temp_uploaded_files"
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        saved_file_paths = []
        
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            log.debug(f"Saved file to hard drive: {file.filename}")
            saved_file_paths.append(file_path)
            
        if saved_file_paths:
            log.debug(f"Starting pipeline for {len(saved_file_paths)} files...")
            
            self.pipeline.run({
                "converter": {"sources": saved_file_paths}
            })
            
            for path in saved_file_paths:
                if os.path.exists(path):
                    os.remove(path)
            log.debug("Cleaned up temporary files.")

        return {"message": f"Successfully indexed {len(files)} files."}