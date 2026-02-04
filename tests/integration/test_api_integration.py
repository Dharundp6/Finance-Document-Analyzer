"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from main import app

    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_returns_app_info(self, client):
        """Test that root endpoint returns application info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
