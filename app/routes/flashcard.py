from fastapi import APIRouter, Header, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.auth_service import create_user_token, verify_token, check_flashcard_rate_limit
from app.services.flashcard_service import generate_flashcards

router = APIRouter(prefix="/flashcard", tags=["Flashcard"])


class AuthRequest(BaseModel):
    full_name: str
    dni: str


class GenerateRequest(BaseModel):
    topic: str


@router.post("/auth")
async def authenticate(request: AuthRequest, db: Session = Depends(get_db)):
    token = create_user_token(db, request.full_name, request.dni)
    return {"token": token}


@router.post("/generate")
async def generate(
    request: GenerateRequest,
    x_auth_token: str = Header(...),
    db: Session = Depends(get_db)
):
    user = verify_token(db, x_auth_token)
    check_flashcard_rate_limit(db, user["token_id"])

    try:
        result = generate_flashcards(request.topic)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
