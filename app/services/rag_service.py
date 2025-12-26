import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "vector_db"

# Embbedings model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def process_pdf_to_vector_db(file_path: str, topic: str):
    # Load pdf file
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)

    # Metadata assignment
    for chunk in chunks:
        chunk.metadata["topic"] = topic
        chunk.metadata["source"] = os.path.basename(file_path)

    # Parse and store in Chroma vector database
    db = Chroma.from_documents(
        chunks, 
        embeddings_model, 
        persist_directory=CHROMA_PATH
    )
    
    return len(chunks)

def search_in_vector_db(query: str, topic: str, k: int = 4):
    # Connect to existing Chroma vector database
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings_model
    )
    
    # Search with topic filtering
    # 'k' is the number of chunks we want to retrieve
    # default is 4 but should be configurated based on result quality
    results = db.similarity_search(
        query, 
        k=k, 
        filter={"topic": topic}
    )
    
    return results