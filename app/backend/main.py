"""
FastAPI application for PDF extraction service.

Provides endpoints for:
- Uploading sample PDFs for schema detection
- Managing schema templates (CRUD)
- Batch extraction with async processing
- Human-in-the-loop review and editing
"""

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Handle both package imports (when running as module) and standalone imports (uvicorn main:app)
try:
    from .models import HealthResponse
    from .routers import batches, documents, history, schemas, upload
    from .services.ai import AIServiceError, get_ai_service
    from .services.pdf_service import PDFConversionError, get_pdf_service
except ImportError:
    import sys
    from pathlib import Path
    # Add parent directory to path for standalone imports
    backend_dir = Path(__file__).parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from models import HealthResponse
    from routers import batches, documents, history, schemas, upload
    from services.ai import AIServiceError, get_ai_service
    from services.pdf_service import PDFConversionError, get_pdf_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logger.info("Starting PDF Extraction Service...")
    # Initialize services on startup
    get_pdf_service()
    get_ai_service()
    # Note: In production, use Alembic migrations instead of init_db()
    # init_db()
    logger.info("Services initialized successfully")
    yield
    logger.info("Shutting down PDF Extraction Service...")


# Create FastAPI application
app = FastAPI(
    title="PDF Extraction API",
    description="Production-grade PDF data extraction using AI",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React production (Docker)
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite development server
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Endpoints
# =============================================================================


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Root endpoint - health check."""
    return HealthResponse(status="healthy", message="PDF Extraction API is running")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", message="Service is healthy")


# =============================================================================
# Include Routers
# =============================================================================

app.include_router(schemas.router)
app.include_router(upload.router)
app.include_router(batches.extract_router)  # Root-level extract-batch routes
app.include_router(batches.router)  # /batches/* routes
app.include_router(history.router)
app.include_router(documents.router)


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(PDFConversionError)
async def pdf_conversion_error_handler(request, exc: PDFConversionError):
    """Handle PDF conversion errors."""
    from fastapi import status

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


@app.exception_handler(AIServiceError)
async def ai_service_error_handler(request, exc: AIServiceError):
    """Handle AI service errors."""
    from fastapi import status

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )
