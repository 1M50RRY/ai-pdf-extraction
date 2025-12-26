"""
Router for document-related endpoints.

Handles:
- PDF content retrieval for preview
- Smart repair of extracted data
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

# Handle both package imports and standalone imports
try:
    from ..database import get_db
    from ..models import ExtractionDetailResponse, SchemaDefinition
    from ..models_db import Document, DocumentBatch, Extraction, SavedSchema
    from ..services.ai import AIService, get_ai_service
except ImportError:
    from database import get_db
    from models import ExtractionDetailResponse, SchemaDefinition
    from models_db import Document, DocumentBatch, Extraction, SavedSchema
    from services.ai import AIService, get_ai_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/{document_id}/content")
async def get_document_content(
    document_id: str,
    db: Session = Depends(get_db),
) -> Response:
    """
    Retrieve the PDF file content for a document.
    
    Used for PDF preview in the frontend, especially for historical batches
    where the local file blob URL is no longer available.
    
    Args:
        document_id: UUID of the document.
        db: Database session.
        
    Returns:
        PDF file content with appropriate content-type headers.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format",
        )
    
    document = db.query(Document).filter(Document.id == doc_uuid).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    
    if not document.file_content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF content not available for this document",
        )
    
    return Response(
        content=document.file_content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{document.filename}"',
        },
    )


@router.post("/{document_id}/smart-repair", response_model=ExtractionDetailResponse)
async def smart_repair_document(
    document_id: str,
    db: Session = Depends(get_db),
    ai_service: AIService = Depends(get_ai_service),
) -> ExtractionDetailResponse:
    """
    Use LLM to intelligently repair extracted data.
    
    Fixes:
    - Missing calculated values (Tax = Total - Subtotal, Due Date from Issue Date + terms)
    - OCR typos (l -> 1, O -> 0)
    - Math validation (qty * price vs total)
    - Format consistency (dates to YYYY-MM-DD, money to numeric)
    
    Args:
        document_id: UUID of the document.
        db: Database session.
        ai_service: AI service instance.
        
    Returns:
        Updated extraction details with repaired data.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format",
        )
    
    document = db.query(Document).filter(Document.id == doc_uuid).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    
    # Get the extraction for this document (use the first one if multiple)
    extraction = db.query(Extraction).filter(Extraction.document_id == doc_uuid).first()
    if not extraction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No extraction found for document {document_id}",
        )
    
    # Get current extracted data
    current_data = extraction.data.copy()
    
    # Get FULL schema from batch (REQUIRED for calculation engine)
    schema = None
    if document.batch_id:
        batch = db.query(DocumentBatch).filter(DocumentBatch.id == document.batch_id).first()
        if batch and batch.schema_id:
            saved_schema = db.query(SavedSchema).filter(SavedSchema.id == batch.schema_id).first()
            if saved_schema and saved_schema.structure:
                try:
                    schema = SchemaDefinition(**saved_schema.structure)
                    logger.info(
                        "Loaded schema for repair: '%s' with %d fields",
                        schema.name,
                        len(schema.fields),
                    )
                except Exception as e:
                    logger.warning("Failed to parse schema for repair: %s", e)
        else:
            logger.warning(
                "No schema_id found in batch %s - calculation engine will use basic mode",
                document.batch_id,
            )
    else:
        logger.warning(
            "Document %s has no batch_id - cannot load schema for calculation",
            document_id,
        )
    
    # Run LLM-based calculation engine (requires full schema for dynamic calculation)
    repaired_data = await ai_service.repair_data_with_llm(current_data, schema)
    
    # Check if any fields were repaired
    if repaired_data != current_data:
        # Update the extraction with repaired values
        extraction.data = repaired_data
        # Track repaired fields in manual_overrides
        repaired_fields = {}
        for k, v in repaired_data.items():
            if k not in current_data or current_data.get(k) != v:
                repaired_fields[k] = {
                    "smart_repaired": True,
                    "original": current_data.get(k),
                    "repaired": v,
                }
        
        if extraction.manual_overrides:
            extraction.manual_overrides.update(repaired_fields)
        else:
            extraction.manual_overrides = repaired_fields
        
        db.commit()
        db.refresh(extraction)
        
        logger.info(
            "Smart-repaired %d fields for document %s (extraction %s): %s",
            len(repaired_fields),
            document_id,
            extraction.id,
            list(repaired_fields.keys()),
        )
    
    return ExtractionDetailResponse(
        id=str(extraction.id),
        document_id=str(extraction.document_id),
        page_number=extraction.page_number,
        data=extraction.data,
        confidence=extraction.confidence,
        field_confidences=extraction.field_confidences or {},
        warnings=extraction.warnings or [],
        is_reviewed=extraction.is_reviewed,
        manual_overrides=extraction.manual_overrides,
        created_at=extraction.created_at.isoformat(),
        reviewed_at=extraction.reviewed_at.isoformat() if extraction.reviewed_at else None,
    )

