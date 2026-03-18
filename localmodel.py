from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from haystack import Pipeline
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
import os
import pandas as pd
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
import uvicorn

class QueryRequest(BaseModel):
    question: str

app = FastAPI(
)

document_store = InMemoryDocumentStore()

dataset = pd.read_csv("realistic_restaurant_reviews.csv")
reviews = [Document(content=row["Title"] + " " + row["Review"], meta={"rating": row["Rating"], "date": row["Date"]}) for _, row in dataset.iterrows()]

reviews_embeddings = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
# reviews_embeddings.warmup()

reviews_with_embeddings = reviews_embeddings.run(reviews)
document_store.write_documents(reviews_with_embeddings["documents"])

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

retriever = InMemoryEmbeddingRetriever(document_store)

template = [
    ChatMessage.from_user(
        """You are an exeprt in answering questions about a pizza restaurant

        Here are some relevant reviews:
        {% for review in reviews %}
            {{ review.content }}
        {% endfor %}


        Here is the question to answer: {{question}}
        """
    )
]

prompt_builder = ChatPromptBuilder(template=template)

# Haystack Pipeline
generator = OllamaGenerator(
    model="qwen2.5:1.5b", 
    url="http://localhost:11434"
)

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", text_embedder)
query_pipeline.add_component("retriever", retriever)
query_pipeline.add_component("prompt_builder", prompt_builder)
query_pipeline.add_component("llm", generator)

query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt_builder.reviews")
query_pipeline.connect("prompt_builder.prompt", "llm.prompt")

# Định nghĩa Endpoint (API Route)
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