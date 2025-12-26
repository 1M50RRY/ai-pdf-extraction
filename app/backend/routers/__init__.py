"""
Routers package for FastAPI endpoints.

Organized by domain:
- batches: Batch processing endpoints
- documents: Document content and repair endpoints
- history: History and extraction management
- schemas: Schema template management
- upload: Upload and schema discovery
"""

from . import batches, documents, history, schemas, upload

# Export extract_router for root-level routes
from .batches import extract_router

__all__ = ["batches", "documents", "history", "schemas", "upload", "extract_router"]

