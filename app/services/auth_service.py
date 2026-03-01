import secrets
import datetime
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text


def create_user_token(db: Session, full_name: str, dni: str) -> str:
    result = db.execute(
        text("SELECT id FROM User WHERE dni = :dni"),
        {"dni": dni}
    )
    user = result.fetchone()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se encontró un usuario con ese DNI"
        )

    user_id = user[0]

    result = db.execute(
        text("SELECT token FROM ai_tokens WHERE user_id = :user_id"),
        {"user_id": user_id}
    )
    existing = result.fetchone()

    if existing:
        return existing[0]

    token = secrets.token_hex(16)
    db.execute(
        text("INSERT INTO ai_tokens (user_id, token) VALUES (:user_id, :token)"),
        {"user_id": user_id, "token": token}
    )
    db.commit()
    return token


def verify_token(db: Session, token: str):
    result = db.execute(
        text("""
            SELECT ai_tokens.id, ai_tokens.user_id, User.firstName, User.lastName, User.dni, User.email
            FROM ai_tokens
            JOIN User ON ai_tokens.user_id = User.id
            WHERE ai_tokens.token = :token
        """),
        {"token": token}
    )
    row = result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )

    return {
        "token_id": row[0],
        "user_id": row[1],
        "firstName": row[2],
        "lastName": row[3],
        "dni": row[4],
        "email": row[5]
    }


def check_rate_limit(db: Session, token_id: int):
    one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)

    result = db.execute(
        text("SELECT COUNT(*) FROM ai_requests WHERE token_id = :token_id AND created_at > :since"),
        {"token_id": token_id, "since": one_hour_ago}
    )
    count = result.scalar()

    if count >= 5:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 5 requests per hour."
        )

    db.execute(
        text("INSERT INTO ai_requests (token_id, created_at) VALUES (:token_id, :now)"),
        {"token_id": token_id, "now": datetime.datetime.now()}
    )
    db.commit()
