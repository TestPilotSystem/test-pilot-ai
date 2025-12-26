import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.rag_service import process_pdf_to_vector_db
from app.services.rag_service import search_in_vector_db
from app.services.rag_service import reset_vector_db

router = APIRouter(prefix="/admin/ai", tags=["Admin AI"])

UPLOAD_DIR = "temp_uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-manual")
async def upload_manual(
    file: UploadFile = File(...),
    topic: str = Form(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

    # Build file path
    file_path = os.path.join(UPLOAD_DIR, file.filename)

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