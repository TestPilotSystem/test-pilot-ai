import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.rag_service import process_pdf_to_vector_db
from app.services.rag_service import search_in_vector_db
from app.services.rag_service import reset_vector_db
from app.services.rag_service import get_all_topics
from app.services.llm_service import generate_test_from_chunks
from app.services.llm_service import generate_bulk_questions
from app.config import settings

router = APIRouter(prefix="/admin/ai", tags=["Admin AI"])

os.makedirs(settings.upload_dir, exist_ok=True)

@router.post("/upload-manual")
async def upload_manual(
    file: UploadFile = File(...),
    topic: str = Form(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

    # Build file path
    file_path = os.path.join(settings.upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        num_chunks = process_pdf_to_vector_db(file_path, topic)

        return {
            "message": f"Archivo '{file.filename}' procesado con éxito",
            "topic": topic,
            "chunks_created": num_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            
@router.get("/test-search")
async def test_search(query: str, topic: str):
    try:
        results = search_in_vector_db(query, topic)
        
        # Data formatting for json response
        formatted_results = [
            {
                "content": res.page_content,
                "metadata": res.metadata
            } for res in results
        ]
        
        return {
            "query": query,
            "topic": topic,
            "results_found": len(formatted_results),
            "data": formatted_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.delete("/reset-db")
async def reset_db():
    try:
        reset_vector_db()
        return {"message": "Base de datos vectorial reseteada con éxito"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
async def get_topics():
    """
    Returns all unique topics stored in the vector database.
    Useful for synchronizing topics with other services.
    """
    try:
        topics = get_all_topics()
        return {
            "topics": topics,
            "total": len(topics)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/generate-question")
async def get_test_question(topic: str):
    # Get relevant chunks from vector DB
    chunks = search_in_vector_db(query="conceptos principales y normas", topic=topic, k=10)
    
    if not chunks:
        return {"error": "No se encontró contenido para este tema"}
    
    # Generate question using LLM
    question_data = generate_test_from_chunks(chunks, topic)
    
    return question_data

import random

@router.get("/generate-full-test")
async def get_full_test(topic: str, num_questions: int = 10):
    # high k to get full context
    all_chunks = search_in_vector_db(query="normas generales", topic=topic, k=100)
    
    total_available = len(all_chunks)
    if total_available == 0:
        raise HTTPException(status_code=404, detail="No hay datos para este tema")

    random.shuffle(all_chunks)

    full_test = []
    # Adjustable number of questions per batch
    questions_per_batch = 5
    
    # Calculate number of batches needed
    batches = (num_questions // questions_per_batch) + (1 if num_questions % questions_per_batch != 0 else 0)

    for i in range(batches):
        # Circular indexing to avoid index errors
        start_idx = (i * 5) % total_available
        batch_chunks = all_chunks[start_idx : start_idx + 10]
        
        if not batch_chunks:
             batch_chunks = all_chunks[:10]

        current_count = min(questions_per_batch, num_questions - len(full_test))
        batch_result = generate_bulk_questions(batch_chunks, topic, current_count)
        full_test.extend(batch_result["preguntas"])

    return {"topic": topic, "total_questions": len(full_test), "test": full_test}