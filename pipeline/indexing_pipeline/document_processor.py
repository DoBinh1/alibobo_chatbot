import os
import pymupdf4llm
import ollama 
from docx2pdf import convert as docx_convert
from typing import List
import dataclasses
from haystack import component, Document

@component
class DocumentConverter:
    """Trạm 1: Bóc tách, Tóm tắt từng trang và Tổng hợp ngữ cảnh tài liệu"""
    @component.output_types(text_docs=List[Document], image_docs=List[Document])
    def run(self, sources: List[str]):
        text_docs = []
        image_docs = []
        
        for file_path in sources:
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0] 
            ext = file_path.lower().split('.')[-1]
            pdf_path = file_path
            
            if ext == "docx":
                pdf_path = file_path.replace(".docx", "_temp.pdf")
                docx_convert(file_path, pdf_path)
                
            specific_img_folder = os.path.join("extracted_images", base_name)
            os.makedirs(specific_img_folder, exist_ok=True)
            
            # 1. Đọc dữ liệu và chia theo trang (page_chunks=True)
            page_data = pymupdf4llm.to_markdown(
                doc=pdf_path, 
                write_images=True, 
                image_path=specific_img_folder,
                page_chunks=True # Trả về list các dict theo từng trang
            )
            
            page_summaries = []
            full_markdown_text = ""
            
            # Xử lý LLM tóm tắt cho từng trang
            print(f"Đang tóm tắt {len(page_data)} trang của tài liệu {file_name}...")
            for idx, page in enumerate(page_data):
                page_text = page.get('text', '')
                # Gộp lại nguyên bản để không bị đứt đoạn nội dung giữa các trang
                full_markdown_text += page_text + "\n\n"
                
                if page_text.strip():
                    prompt_page = f"Briefly summarize the core content of the following document page in 1-2 sentences:\n\n{page_text}"
                    response = ollama.generate(
                        model='qwen2.5:1.5b', # Dùng model nhẹ để tóm tắt trang cho nhanh
                        prompt=prompt_page,
                        options={"num_predict": 100}
                    )
                    page_summaries.append(f"Trang {idx + 1}: {response['response'].strip()}")
            
            # Tổng hợp tóm tắt toàn bộ file từ các tóm tắt trang
            combined_page_summaries = "\n".join(page_summaries)
            prompt_full = f"Based on the page summaries below, write a comprehensive 3-5 sentence overview that covers the entire document:\n\n{combined_page_summaries}"
            
            full_summary_response = ollama.generate(
                model='qwen2.5:1.5b',
                prompt=prompt_full,
                options={"num_predict": 300}
            )
            document_summary = full_summary_response['response'].strip()
            
            # 2. Tạo Document nguyên bản (để chia Chunk) kèm Metadata là bối cảnh
            text_doc = Document(
                content=full_markdown_text, 
                meta={
                    "file_name": file_name,
                    "document_summary": document_summary # Lưu vào meta để nối vào chunk sau này
                }
            )
            text_docs.append(text_doc)
            
            # 3. Xử lý ảnh (Chỉ gắn metadata tài liệu, bỏ qua bước LLM đọc ảnh)
            for img_file in os.listdir(specific_img_folder):
                img_path = os.path.join(specific_img_folder, img_file)
                image_doc = Document(
                    content="", # Để trống content, Vision-Language Embedder sẽ tự đọc ảnh từ meta
                    meta={"file_name": file_name, "image_path": img_path}
                )
                image_docs.append(image_doc)

            if ext == "docx" and os.path.exists(pdf_path):
                os.remove(pdf_path)
                
        return {"text_docs": text_docs, "image_docs": image_docs}


@component
class ContextualEnhancer:
    """Trạm 2: Nhúng tóm tắt tài liệu vào đầu mỗi chunk"""
    @component.output_types(enhanced_docs=List[Document])
    def run(self, documents: List[Document]):
        enhanced_docs = []
        for doc in documents:
            file_name = doc.meta.get("file_name", "Tài liệu chưa rõ")
            headers = doc.meta.get("header", "Nội dung chung")
            
            # Lấy tóm tắt toàn bộ file đã được tạo từ Trạm 1 (Haystack splitter tự động kế thừa meta này)
            document_summary = doc.meta.get("document_summary", "")
            content = doc.content

            # Nối trực tiếp bối cảnh vào đầu chunk (Không cần gọi LLM nữa, tiết kiệm 100% thời gian)
            new_content = f"Source: {file_name}\nSection: {headers}\nDocument Context: {document_summary}\n\nDetailed Content:\n{content}"
            
            updated_doc = dataclasses.replace(doc, content=new_content)
            enhanced_docs.append(updated_doc)
            
        return {"enhanced_docs": enhanced_docs}