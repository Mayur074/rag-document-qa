"""
src/rag_pipeline.py
--------------------
Core RAG pipeline: document ingestion, chunking, embedding,
vector store management, and retrieval-augmented generation.
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, DirectoryLoader
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR      = "data/chroma_db"
ANTHROPIC_MODEL  = "claude-sonnet-4-20250514"

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use the following retrieved context
to answer the question accurately and concisely. If the answer is not in the
context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
)


class RAGPipeline:
    """End-to-end RAG pipeline with persistent Chroma vector store."""

    def __init__(self, persist_dir: str = PERSIST_DIR):
        self.persist_dir = persist_dir
        self.embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain    = None

        if Path(persist_dir).exists():
            self._load_vectorstore()

    def ingest_documents(self, docs_path: str) -> int:
        """Load, chunk, embed and persist documents. Returns chunk count."""
        logger.info(f"Loading documents from: {docs_path}")

        loaders = [
            DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader),
        ]

        raw_docs = []
        for loader in loaders:
            try:
                raw_docs.extend(loader.load())
            except Exception as e:
                logger.warning(f"Loader error: {e}")

        if not raw_docs:
            raise ValueError(f"No documents found in {docs_path}")

        logger.info(f"Loaded {len(raw_docs)} documents")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)
        logger.info(f"Split into {len(chunks)} chunks")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        logger.info(f"Persisted vector store to {self.persist_dir}")
        self._build_qa_chain()
        return len(chunks)

    def _load_vectorstore(self):
        logger.info(f"Loading existing vector store from {self.persist_dir}")
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )
        self._build_qa_chain()

    def _build_qa_chain(self):
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized.")

        llm = ChatAnthropic(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20},
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": RAG_PROMPT},
            return_source_documents=True,
        )

    def query(self, question: str) -> dict:
        """Answer a question using retrieved context."""
        if not self.qa_chain:
            raise RuntimeError("Pipeline not initialized. Run ingest_documents() first.")

        result  = self.qa_chain.invoke({"query": question})
        sources = list({
            doc.metadata.get("source", "unknown")
            for doc in result["source_documents"]
        })

        return {
            "answer":           result["result"],
            "source_documents": result["source_documents"],
            "sources":          sources,
        }

    def get_stats(self) -> dict:
        if not self.vectorstore:
            return {"status": "not initialized"}
        count = self.vectorstore._collection.count()
        return {"chunks_stored": count, "persist_dir": self.persist_dir}
