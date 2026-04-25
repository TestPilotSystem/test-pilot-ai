import os
import random
import shutil
import asyncio

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.rag_service import (
    process_pdf_to_vector_db,
    search_in_vector_db,
    reset_vector_db,
    get_all_topics
)
from app.services.llm_service import generate_test_from_chunks, generate_bulk_questions
from app.config import settings

router = APIRouter(prefix="/admin/ai", tags=["Admin AI"])

os.makedirs(settings.upload_dir, exist_ok=True)


@router.post("/upload-manual")
async def upload_manual(file: UploadFile = File(...), topic: str = Form(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

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
        formatted_results = [
            {"content": res.page_content, "metadata": res.metadata}
            for res in results
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
    try:
        topics = get_all_topics()
        return {"topics": topics, "total": len(topics)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate-question")
async def get_test_question(topic: str):
    chunks = search_in_vector_db(
        query="conceptos principales y normas",
        topic=topic,
        k=10
    )
    if not chunks:
        return {"error": "No se encontró contenido para este tema"}
    return generate_test_from_chunks(chunks, topic)


@router.get("/generate-full-test")
async def get_full_test(topic: str, num_questions: int = 10):
    # Bulk retrieval: skip multi-query and reranking for speed
    all_chunks = search_in_vector_db(
        query="normas generales conceptos clave",
        topic=topic,
        k=100,
        use_multi_query=False,
        use_rerank=False
    )

    if not all_chunks:
        raise HTTPException(status_code=404, detail="No hay datos para este tema")

    random.shuffle(all_chunks)
    total_available = len(all_chunks)

    questions_per_batch = 5
    batches_needed = (num_questions + questions_per_batch - 1) // questions_per_batch
    chunk_window = max(10, total_available // batches_needed)

    async def _run_batch(batch_chunks, count):
        return await asyncio.to_thread(generate_bulk_questions, batch_chunks, topic, count)

    questions_left = num_questions
    tasks = []
    for i in range(batches_needed):
        start = (i * chunk_window) % total_available
        end = min(start + chunk_window, total_available)
        batch_chunks = all_chunks[start:end] or all_chunks[:10]
        count = min(questions_per_batch, questions_left)
        questions_left -= count
        tasks.append(_run_batch(batch_chunks, count))

    # All batches run in parallel
    batch_results = await asyncio.gather(*tasks)

    full_test = []
    for result in batch_results:
        if isinstance(result, dict):
            full_test.extend(result.get("preguntas", []))
        elif hasattr(result, "preguntas"):
            full_test.extend(result.preguntas)

    return {"topic": topic, "total_questions": len(full_test), "test": full_test}
