from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

# 1. Kết nối tới thư mục dữ liệu bạn vừa tạo
document_store = QdrantDocumentStore(
    path="qdrant_initial_vectordb",
    index="Document", 
    embedding_dim=384, 
    use_sparse_embeddings=True
)

# 2. Lấy tất cả tài liệu trong database ra
all_docs = document_store.filter_documents({"field": "meta.file_name", "operator": "==", "value": "Efficient_and_Explainable_Bearing_Condition_Monitoring_with_Decision_Tree_Based_Feature_Learning.pdf"})  

# 3. In kết quả ra màn hình
print(f"Tổng số tài liệu tìm thấy: {len(all_docs)}")
print("-" * 30)

for doc in all_docs:
    print(f"ID: {doc.id}")
    print(f"Nội dung: {doc.content}...")
    print(f"Metadata: {doc.meta}")
    print("-" * 30)