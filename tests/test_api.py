"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check(self, client):
        """Test health check returns ok."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestQueryEndpoint:
    """Tests for query endpoint."""

    def test_query_empty(self, client):
        """Test query with empty database."""
        response = client.post("/api/v1/query", json={"query": "What is the revenue?"})

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_query_validation(self, client):
        """Test query validation."""
        response = client.post(
            "/api/v1/query",
            json={"query": ""},  # Empty query
        )

        assert response.status_code == 422  # Validation error


class TestDocumentEndpoints:
    """Tests for document endpoints."""

    def test_list_documents_empty(self, client):
        """Test listing documents when empty."""
        response = client.get("/api/v1/documents")

        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_stats(self, client):
        """Test getting system stats."""
        response = client.get("/api/v1/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "total_chunks" in data
