from typing import List, Optional
from haystack import Pipeline
from haystack.dataclasses import ByteStream
from fastapi import UploadFile
from hayhooks import BasePipelineWrapper, log
from haystack.components.routers import FileTypeRouter
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder



class IndexingPipelineWrapper(BasePipelineWrapper):
    def setup(self,document_store) -> None:
        indexing = Pipeline()
        indexing.add_component("router", FileTypeRouter(mime_types=["application/pdf", "text/plain"]))
        indexing.add_component("pdf_converter", PyPDFToDocument())
        indexing.add_component("txt_converter", TextFileToDocument())
        indexing.add_component("joiner", DocumentJoiner())
        indexing.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=6, split_overlap=2))
        indexing.add_component("sparse_embedder", FastembedSparseDocumentEmbedder(model="Qdrant/bm42-all-minilm-l6-v2-attentions"))
        indexing.add_component("dense_embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
        indexing.add_component("writer", DocumentWriter(document_store=document_store, policy="overwrite"))

        indexing.connect("router.application/pdf", "pdf_converter.sources")
        indexing.connect("router.text/plain", "txt_converter.sources")
        indexing.connect("pdf_converter", "joiner")
        indexing.connect("txt_converter", "joiner")
        indexing.connect("joiner", "splitter")
        indexing.connect("splitter", "sparse_embedder")
        indexing.connect("splitter", "dense_embedder")
        indexing.connect("sparse_embedder", "writer")
        indexing.connect("dense_embedder", "writer")

        self.pipeline = indexing

    def run_api(self, files: Optional[List[UploadFile]] = None) -> dict:
        if not files:
            return {"message": "No files provided for indexing."}
        
        for file in files:
            content = file.file.read().decode("utf-8")
            log.debug(f"Indexing file: {file.filename}")

            self.pipeline.run({"router": {"files": [ByteStream(content.encode("utf-8"), filename=file.filename, mime_type=file.content_type)]}})
        
        else:
            log.debug("No files to index.")
        return {"message": f"Successfully indexed {len(files)} files."}