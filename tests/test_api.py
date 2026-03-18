import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_read_runs():
    response = client.get("/api/runs")
    assert response.status_code == 200
    assert "runs" in response.json()

def test_read_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    assert "models" in response.json()
