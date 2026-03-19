import os
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

def test_docling_quick(pdf_path: str):
    print("⏳ Đang bóc tách PDF (bao gồm cả Bảng, Công thức và Hình ảnh)...")

    # 1. Bật cấu hình lấy Hình ảnh (Công thức tự động được lấy dạng LaTeX)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.do_formula_enrichment = True  

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # 2. Xử lý file
    doc = converter.convert(pdf_path).document

    # 3. In thử 1000 ký tự Markdown đầu tiên (Công thức sẽ nằm trong cặp dấu $$)
    print("\n--- PREVIEW NỘI DUNG (MARKDOWN) ---")
    print(doc.export_to_markdown()[:50000]) 
    print("------------------------------------\n")

    # 4. Tìm và lưu tất cả hình ảnh/biểu đồ ra ổ cứng
    img_count = 0

    output_dir = "extracted_images"
    os.makedirs(output_dir, exist_ok=True)

    for item, _ in doc.iterate_items():
        if item.label in ["picture", "figure"]:
            img = item.get_image(doc)
            if img:
                img_count += 1
                img_name = os.path.join(output_dir, f"docling_image_{img_count}.png")
                img.save(img_name)
                print(f"✅ Đã cắt và lưu ảnh: {img_name}")

    print(f"\n🎯 Xong! Đã trích xuất được {img_count} hình ảnh.")

if __name__ == "__main__":
    # Đổi lại đường dẫn đúng với file của bạn
    test_docling_quick("uploaded_files/EAI.pdf")