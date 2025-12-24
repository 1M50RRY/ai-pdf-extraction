"""Pytest configuration and fixtures."""

import io
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.backend.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """
    Create a minimal valid PDF for testing.

    This is a minimal PDF structure that should be recognized as a valid PDF.
    """
    # Minimal valid PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000214 00000 n 
trailer
<< /Size 5 /Root 1 0 R >>
startxref
306
%%EOF"""
    return pdf_content


@pytest.fixture
def invalid_file_bytes() -> bytes:
    """Create invalid (non-PDF) file bytes for testing."""
    return b"This is not a PDF file"


@pytest.fixture
def sample_schema_json() -> str:
    """Create a sample schema JSON string for testing."""
    import json

    return json.dumps(
        {
            "name": "Test Schema",
            "description": "A test schema for unit tests",
            "version": "1.0",
            "fields": [
                {
                    "name": "test_field",
                    "type": "string",
                    "description": "A test field",
                    "required": True,
                },
                {
                    "name": "amount",
                    "type": "currency",
                    "description": "An amount field",
                    "required": False,
                },
            ],
        }
    )

