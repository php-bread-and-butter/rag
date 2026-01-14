"""
Tests for main application endpoints
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_hello_world():
    """Test hello world endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Hello, World!"
    assert data["status"] == "success"


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "FastAPI Tutorial"
