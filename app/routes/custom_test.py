from fastapi import APIRouter, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.auth_service import create_user_token, verify_token, check_rate_limit
from app.services.custom_test_service import generate_custom_test

router = APIRouter(prefix="/custom-test", tags=["Custom Test"])


class AuthRequest(BaseModel):
    full_name: str
    dni: str


class GenerateRequest(BaseModel):
    student_stats: Optional[Dict[str, Any]] = None


@router.post("/auth")
async def authenticate(request: AuthRequest, db: Session = Depends(get_db)):
    token = create_user_token(db, request.full_name, request.dni)
    return {"token": token}


@router.post("/generate")
async def generate_test(
    request: GenerateRequest,
    x_auth_token: str = Header(...),
    db: Session = Depends(get_db)
):
    user = verify_token(db, x_auth_token)
    check_rate_limit(db, user["token_id"])

    profile = request.student_stats if request.student_stats else {}

    try:
        result = generate_custom_test(profile)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
