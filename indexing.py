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



class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        indexing = Pipeline()
        indexing.add_component("router", FileTypeRouter(mime_types=["application/pdf", "text/plain"]))
        indexing.add_component("pdf_converter", PyPDFToDocument())
        indexing.add_component("txt_converter", TextFileToDocument())
        indexing.add_component("joiner", DocumentJoiner())
        indexing.add_component("splitter", DocumentSplitter(split_by="word", split_length=300, split_overlap=30))
        indexing.add_component("embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
        document_store = QdrantDocumentStore(host='qdrant')
        indexing.add_component("writer", DocumentWriter(document_store=document_store))
        indexing.connect("router.application/pdf", "pdf_converter.sources")
        indexing.connect("router.text/plain", "txt_converter.sources")
        indexing.connect("pdf_converter", "joiner")
        indexing.connect("txt_converter", "joiner")
        indexing.connect("joiner", "splitter")
        indexing.connect("splitter", "embedder")
        indexing.connect("embedder", "writer")

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