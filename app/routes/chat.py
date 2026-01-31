from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal, List
from app.services.rag_service import search_in_vector_db
from app.services.llm_service import chat_with_tutor

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    topic: str = "general"
    tone: Literal["formal", "informal", "conciso", "detallado"] = "formal"
    user_name: Optional[str] = None
    history: Optional[List[ChatMessage]] = None


@router.post("/")
async def chat(request: ChatRequest):
    chunks = search_in_vector_db(query=request.question, topic=request.topic, k=6)
    
    if not chunks:
        raise HTTPException(status_code=404, detail="No hay material para este tema")
    
    history_dicts = [msg.model_dump() for msg in request.history] if request.history else None
    response = chat_with_tutor(request.question, chunks, request.tone, request.user_name, history_dicts)
    return {"question": request.question, "response": response}
