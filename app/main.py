from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import admin_ai
from app.config import settings

app = FastAPI(title="TestPilot AI Engine")

cors_origins = [origin.strip() for origin in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin_ai.router)

@app.get("/")
async def root():
    return {"status": "online", "message": "TestPilot AI Engine funcionando"}

@app.get("/health-check")
async def health():
    return {"model": "gemma2", "ollama_status": "ready"}