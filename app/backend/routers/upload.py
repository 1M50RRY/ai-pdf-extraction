"""
Router for upload and schema discovery endpoints.

Handles:
- Sample PDF upload for schema detection
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.params import Annotated

# Handle both package imports and standalone imports
try:
    from ..models import UploadSampleResponse
    from ..services.ai import AIService, AIServiceError, get_ai_service
    from ..services.pdf_service import PDFConversionError, get_pdf_service
except ImportError:
    from models import UploadSampleResponse
    from services.ai import AIService, AIServiceError, get_ai_service
    from services.pdf_service import PDFConversionError, get_pdf_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["upload"])


@router.post("/upload-sample", response_model=UploadSampleResponse)
async def upload_sample(
    file: Annotated[UploadFile, File(description="PDF file to analyze")],
) -> UploadSampleResponse:
    """
    Upload a sample PDF for schema detection.

    Accepts a PDF, converts Page 1 to an image, and returns a suggested
    extraction schema based on AI analysis.
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

    try:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file provided",
            )

        logger.info("Processing PDF: %s (%d bytes)", file.filename, len(file_bytes))

        pdf_service = get_pdf_service()
        ai_service = get_ai_service()

        try:
            page_count = pdf_service.get_page_count(file_bytes)
        except PDFConversionError:
            page_count = 1

        try:
            # Use adaptive sampling to get representative pages for schema discovery
            representative_pages = pdf_service.get_representative_pages(file_bytes, max_images=6)
            logger.info("Selected %d representative pages for schema discovery", len(representative_pages))
        except PDFConversionError as e:
            logger.error("PDF conversion failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e),
            )

        try:
            suggested_schema = await ai_service.suggest_schema(representative_pages)
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

