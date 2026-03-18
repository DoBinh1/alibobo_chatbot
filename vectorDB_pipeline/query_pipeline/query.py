from haystack import Pipeline
from hayhooks import BasePipelineWrapper, log
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

class QueryPipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        query = Pipeline()
        query.add_component("text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
        query.add_component("retriever", QdrantEmbeddingRetriever(document_store=QdrantDocumentStore(path="qdrant_data",embedding_dim=384,)))
        query.connect("text_embedder", "retriever")

        self.pipeline = query

    def run_api(self, query: str) -> dict:
        log.debug(f"Running query pipeline with query: {query}")
        return self.pipeline.run({"text_embedder": {"text": query}})