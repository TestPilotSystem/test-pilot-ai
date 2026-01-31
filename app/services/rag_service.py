import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.config import settings


class VectorDBManager:
    """
    Singleton manager for ChromaDB connections.
    Maintains a persistent connection to avoid expensive reconnections on each request.
    """
    _instance = None
    _db = None
    _embeddings = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embeddings_model
            )
        return self._embeddings

    @property
    def db(self):
        if self._db is None:
            self._db = Chroma(
                persist_directory=settings.chroma_path,
                embedding_function=self.embeddings
            )
        return self._db

    def reset_connection(self):
        """Forces a new connection on next access. Use after reset_db."""
        self._db = None

    def add_documents(self, chunks):
        """Adds documents to the existing collection."""
        self.db.add_documents(chunks)
        return len(chunks)


vector_manager = VectorDBManager()


def process_pdf_to_vector_db(file_path: str, topic: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["topic"] = topic
        chunk.metadata["source"] = os.path.basename(file_path)

    return vector_manager.add_documents(chunks)


def search_in_vector_db(query: str, topic: str = None, k: int = None):
    if k is None:
        k = settings.default_search_k

    if topic:
        return vector_manager.db.similarity_search(
            query,
            k=k,
            filter={"topic": topic}
        )
    return vector_manager.db.similarity_search(query, k=k)


def reset_vector_db():
    vector_manager.db.delete_collection()
    vector_manager.reset_connection()
    return True