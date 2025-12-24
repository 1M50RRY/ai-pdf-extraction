"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client: TestClient):
        """Test root endpoint returns health status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_endpoint(self, client: TestClient):
        """Test /health endpoint returns health status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestUploadSampleEndpoint:
    """Tests for POST /upload-sample endpoint."""

    def test_upload_sample_rejects_non_pdf(self, client: TestClient):
        """Test that non-PDF files are rejected."""
        response = client.post(
            "/upload-sample",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_upload_sample_rejects_empty_file(self, client: TestClient):
        """Test that empty files are rejected."""
        response = client.post(
            "/upload-sample",
            files={"file": ("test.pdf", b"", "application/pdf")},
        )
        assert response.status_code == 400
        assert "Empty" in response.json()["detail"]

    def test_upload_sample_rejects_invalid_pdf(
        self, client: TestClient, invalid_file_bytes: bytes
    ):
        """Test that invalid PDF content is rejected."""
        response = client.post(
            "/upload-sample",
            files={"file": ("test.pdf", invalid_file_bytes, "application/pdf")},
        )
        # Should be 422 (Unprocessable Entity) for invalid PDF structure
        assert response.status_code == 422


class TestExtractBatchEndpoint:
    """Tests for POST /extract-batch-sync endpoint (legacy synchronous endpoint)."""

    def test_extract_batch_rejects_non_pdf(
        self, client: TestClient, sample_schema_json: str
    ):
        """Test that non-PDF files are rejected."""
        response = client.post(
            "/extract-batch-sync",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
            data={"confirmed_schema": sample_schema_json},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_extract_batch_rejects_invalid_schema(self, client: TestClient):
        """Test that invalid schema JSON is rejected."""
        response = client.post(
            "/extract-batch-sync",
            files={"file": ("test.pdf", b"%PDF-1.4 test", "application/pdf")},
            data={"confirmed_schema": "not valid json"},
        )
        assert response.status_code == 400
        assert "Invalid JSON" in response.json()["detail"]

    def test_extract_batch_rejects_malformed_schema(self, client: TestClient):
        """Test that malformed schema structure is rejected."""
        import json

        bad_schema = json.dumps({"name": "Test", "fields": []})  # Empty fields
        response = client.post(
            "/extract-batch-sync",
            files={"file": ("test.pdf", b"%PDF-1.4 test", "application/pdf")},
            data={"confirmed_schema": bad_schema},
        )
        assert response.status_code == 400
        assert "Invalid schema" in response.json()["detail"]


class TestAsyncBatchEndpoint:
    """Tests for POST /extract-batch async endpoint."""

    def test_extract_batch_requires_schema(self, client: TestClient):
        """Test that either schema_id or confirmed_schema is required."""
        response = client.post(
            "/extract-batch",
            files=[("files", ("test.pdf", b"%PDF-1.4 test", "application/pdf"))],
        )
        assert response.status_code == 400
        assert "schema" in response.json()["detail"].lower()

    def test_extract_batch_requires_files(
        self, client: TestClient, sample_schema_json: str
    ):
        """Test that at least one file is required."""
        response = client.post(
            "/extract-batch",
            data={"confirmed_schema": sample_schema_json},
        )
        assert response.status_code == 422  # FastAPI validation error

    def test_extract_batch_rejects_non_pdf_files(
        self, client: TestClient, sample_schema_json: str
    ):
        """Test that non-PDF files are rejected in batch."""
        response = client.post(
            "/extract-batch",
            files=[("files", ("test.txt", b"not a pdf", "text/plain"))],
            data={"confirmed_schema": sample_schema_json},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client: TestClient):
        """Test that CORS headers are returned for allowed origins."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # OPTIONS requests should be handled
        assert response.status_code in (200, 405)

    def test_cors_allows_localhost_3000(self, client: TestClient):
        """Test that localhost:3000 is allowed."""
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        assert response.status_code == 200
        # CORS headers should be present
        assert (
            response.headers.get("access-control-allow-origin")
            == "http://localhost:3000"
        )

