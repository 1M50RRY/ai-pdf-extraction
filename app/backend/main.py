"""
FastAPI application for PDF extraction service.

Provides endpoints for:
- Uploading sample PDFs for schema detection
- Batch extraction with confirmed schemas
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import (
    ExtractBatchResponse,
    ExtractionResult,
    HealthResponse,
    SchemaDefinition,
    UploadSampleResponse,
)
from .services.ai_service import AIService, AIServiceError, get_ai_service
from .services.pdf_service import PDFConversionError, PDFService, get_pdf_service

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
    logger.info("Services initialized successfully")
    yield
    logger.info("Shutting down PDF Extraction Service...")


# Create FastAPI application
app = FastAPI(
    title="PDF Extraction API",
    description="Production-grade PDF data extraction using AI",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite development server
        "http://127.0.0.1:5173",
        # TODO: Add production domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Root endpoint - health check."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/upload-sample", response_model=UploadSampleResponse)
async def upload_sample(
    file: Annotated[UploadFile, File(description="PDF file to analyze")],
) -> UploadSampleResponse:
    """
    Upload a sample PDF for schema detection.

    Accepts a PDF, converts Page 1 to an image, and returns a suggested
    extraction schema based on AI analysis.

    Args:
        file: The PDF file to analyze.

    Returns:
        UploadSampleResponse with suggested schema.

    Raises:
        HTTPException: If file processing fails.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted",
        )

    # Validate content type
    if file.content_type and file.content_type != "application/pdf":
        logger.warning(
            "Unexpected content type: %s (expected application/pdf)",
            file.content_type,
        )

    try:
        # Read file content
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file provided",
            )

        logger.info("Processing PDF: %s (%d bytes)", file.filename, len(file_bytes))

        # Get services
        pdf_service = get_pdf_service()
        ai_service = get_ai_service()

        # Get page count
        try:
            page_count = pdf_service.get_page_count(file_bytes)
        except PDFConversionError:
            page_count = 1  # Default if count fails

        # Convert first page to image
        try:
            first_page_image = pdf_service.convert_first_page(file_bytes)
        except PDFConversionError as e:
            logger.error("PDF conversion failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e),
            )

        # Get AI suggested schema
        try:
            suggested_schema = await ai_service.suggest_schema(first_page_image)
        except AIServiceError as e:
            logger.error("AI schema suggestion failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"AI service error: {e}",
            )

        return UploadSampleResponse(
            message="PDF analyzed successfully",
            suggested_schema=suggested_schema,
            preview_available=True,
            page_count=page_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error processing PDF")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {e}",
        )
    finally:
        await file.close()


@app.post("/extract-batch", response_model=ExtractBatchResponse)
async def extract_batch(
    file: Annotated[UploadFile, File(description="PDF file to extract data from")],
    confirmed_schema: Annotated[
        str,
        Form(description="JSON string of confirmed SchemaDefinition"),
    ],
) -> ExtractBatchResponse:
    """
    Extract data from a PDF using a confirmed schema.

    Accepts a PDF and a confirmed extraction schema. Processes all pages
    and returns extraction results for each.

    Args:
        file: The PDF file to process.
        confirmed_schema: JSON string of the SchemaDefinition to use.

    Returns:
        ExtractBatchResponse with extraction results for all pages.

    Raises:
        HTTPException: If extraction fails.
    """
    import json

    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted",
        )

    # Parse schema
    try:
        schema_dict = json.loads(confirmed_schema)
        schema = SchemaDefinition.model_validate(schema_dict)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON in confirmed_schema: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid schema: {e}",
        )

    try:
        # Read file content
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file provided",
            )

        logger.info(
            "Extracting data from PDF: %s (%d bytes) with schema: %s",
            file.filename,
            len(file_bytes),
            schema.name,
        )

        # Get services
        pdf_service = get_pdf_service()
        ai_service = get_ai_service()

        # Convert all pages to images
        try:
            page_images = pdf_service.convert_pdf_to_images(file_bytes)
        except PDFConversionError as e:
            logger.error("PDF conversion failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e),
            )

        # Extract data from each page
        results: list[ExtractionResult] = []
        total_confidence = 0.0

        for i, image in enumerate(page_images, start=1):
            page_source = f"{file.filename} (page {i})"
            try:
                result = await ai_service.extract_data(image, schema, page_source)
                results.append(result)
                total_confidence += result.confidence
            except AIServiceError as e:
                logger.warning("Extraction failed for page %d: %s", i, e)
                # Add a failed result with warning
                results.append(
                    ExtractionResult(
                        source_file=page_source,
                        detected_schema=schema,
                        extracted_data={},
                        confidence=0.0,
                        warnings=[f"Extraction failed: {e}"],
                    )
                )

        # Calculate statistics
        total_pages = len(page_images)
        successful = sum(1 for r in results if r.confidence > 0)
        avg_confidence = total_confidence / total_pages if total_pages > 0 else 0.0

        return ExtractBatchResponse(
            results=results,
            total_pages=total_pages,
            successful_extractions=successful,
            average_confidence=round(avg_confidence, 3),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during extraction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {e}",
        )
    finally:
        await file.close()


# Custom exception handlers
@app.exception_handler(PDFConversionError)
async def pdf_conversion_error_handler(request, exc: PDFConversionError):
    """Handle PDF conversion errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc), "error_type": "pdf_conversion"},
    )


@app.exception_handler(AIServiceError)
async def ai_service_error_handler(request, exc: AIServiceError):
    """Handle AI service errors."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc), "error_type": "ai_service"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

