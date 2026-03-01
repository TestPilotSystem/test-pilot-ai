import pytest
from fastapi.testclient import TestClient
from app.main import app
import uuid

client = TestClient(app)

def test_auth_flow():
    unique_dni = f"12345678{uuid.uuid4().hex[:4]}"
    response = client.post("/custom-test/auth", json={
        "full_name": "Test User",
        "dni": unique_dni
    })
    assert response.status_code == 200
    assert "token" in response.json()
    token = response.json()["token"]
    
    response_2 = client.post("/custom-test/auth", json={
        "full_name": "Test User",
        "dni": unique_dni
    })
    assert response_2.status_code == 200
    assert response_2.json()["token"] == token

def test_rate_limit():
    unique_dni = f"87654321{uuid.uuid4().hex[:4]}"
    auth_resp = client.post("/custom-test/auth", json={
        "full_name": "Rate Limit User",
        "dni": unique_dni
    })
    token = auth_resp.json()["token"]
    headers = {"X-Auth-Token": token}
    
    # Mock LLM to avoid real calls during rate limit check
    with patch("app.routes.custom_test.generate_custom_test", return_value={"test": "mock"}):
        for _ in range(5):
            resp = client.post("/custom-test/generate", headers=headers, json={})
            if resp.status_code != 200:
                print(f"Failed at iteration, status: {resp.status_code}, detail: {resp.text}")

    resp_fail = client.post("/custom-test/generate", headers=headers, json={})
    assert resp_fail.status_code == 429
    assert "Rate limit exceeded" in resp_fail.json()["detail"]

from unittest.mock import patch

def test_generate_schema():
    unique_dni = f"11223344{uuid.uuid4().hex[:4]}"
    auth_resp = client.post("/custom-test/auth", json={
        "full_name": "Schema User",
        "dni": unique_dni
    })
    token = auth_resp.json()["token"]
    headers = {"X-Auth-Token": token}
    
    mock_response = {
        "preguntas": [
            {
                "pregunta": "Test Question",
                "opciones": ["A", "B", "C"],
                "respuesta_correcta": "A",
                "explicacion": "Because A",
                "tema": "General",
                "dificultad": "medio"
            }
        ] * 10
    }
    
    with patch("app.routes.custom_test.generate_custom_test", return_value=mock_response):
        response = client.post("/custom-test/generate", headers=headers, json={
            "student_stats": {"level": "beginner"}
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "preguntas" in data
        assert len(data["preguntas"]) == 10
        
        question = data["preguntas"][0]
        required_fields = ["pregunta", "opciones", "respuesta_correcta", "tema", "dificultad"]
        for field in required_fields:
            assert field in question
