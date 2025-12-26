from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import admin_ai

app = FastAPI(title="TestPilot AI Engine")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route registration
app.include_router(admin_ai.router)

@app.get("/")
async def root():
    return {"status": "online", "message": "TestPilot AI Engine funcionando"}

@app.get("/health-check")
async def health():
    return {"model": "gemma2", "ollama_status": "ready"}