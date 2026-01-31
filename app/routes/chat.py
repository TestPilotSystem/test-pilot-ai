from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_service import search_in_vector_db
from app.services.llm_service import chat_with_tutor

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    question: str
    topic: str = "general"


@router.post("/")
async def chat(request: ChatRequest):
    chunks = search_in_vector_db(query=request.question, topic=request.topic, k=6)
    
    if not chunks:
        raise HTTPException(status_code=404, detail="No hay material para este tema")
    
    response = chat_with_tutor(request.question, chunks)
    return {"question": request.question, "response": response}
