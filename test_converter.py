import pymupdf4llm
import pathlib

def extract_the_easy_way(pdf_path: str):
    print("🚀 Bắt đầu bóc tách siêu tốc...")
    
    # 1. Tạo thư mục để nó tự động lưu ảnh vào đó
    image_dir = "extracted_images_easy"
    pathlib.Path(image_dir).mkdir(exist_ok=True)
    
    # 2. Dòng lệnh thần thánh: Lấy Markdown + Tự động lưu ảnh
    md_text = pymupdf4llm.to_markdown(
        doc=pdf_path,
        write_images=True,
        page_chunk=True,           # Bật công tắc chia trang
        image_path=image_dir,       # Trỏ vào thư mục vừa tạo
        image_format="png"          # Định dạng ảnh
    )
    
    # 3. Lưu file Markdown
    with open("ket_qua_sieu_nhanh.md", "w", encoding="utf-8") as f:
        f.write(md_text)
        
    print("✅ HOÀN TẤT TRONG TÍCH TẮC!")
    print(f"📄 File Markdown lưu tại: ket_qua_sieu_nhanh.md")
    print(f"🖼️ Hình ảnh đã được tự động cắt và lưu vào: {image_dir}/")

if __name__ == "__main__":
    extract_the_easy_way(r"D:\FastAPIgemini\uploaded_files\AM_littérature_review.pdf")