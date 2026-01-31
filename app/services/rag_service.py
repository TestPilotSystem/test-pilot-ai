import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import settings


embeddings_model = HuggingFaceEmbeddings(
    model_name=settings.embeddings_model
)

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

    db = Chroma.from_documents(
        chunks, 
        embeddings_model, 
        persist_directory=settings.chroma_path
    )
    
    return len(chunks)

def search_in_vector_db(query: str, topic: str, k: int = None):
    if k is None:
        k = settings.default_search_k
    
    db = Chroma(
        persist_directory=settings.chroma_path, 
        embedding_function=embeddings_model
    )
    
    results = db.similarity_search(
        query, 
        k=k, 
        filter={"topic": topic}
    )
    
    return results

def reset_vector_db():
    db = Chroma(persist_directory=settings.chroma_path, embedding_function=embeddings_model)
    db.delete_collection()
    return True