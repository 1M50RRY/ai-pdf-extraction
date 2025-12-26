"""
Router for history and extraction management endpoints.

Handles:
- Batch history listing
- Extraction detail retrieval
- Extraction updates (human corrections)
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

# Handle both package imports and standalone imports
try:
    from ..database import get_db
    from ..models import (
        BatchExtractionSummary,
        BatchHistoryResponse,
        BatchSummary,
        ExtractionDetailResponse,
        UpdateExtractionRequest,
    )
    from ..models_db import Document, DocumentBatch, Extraction, SavedSchema
except ImportError:
    from database import get_db
    from models import (
        BatchExtractionSummary,
        BatchHistoryResponse,
        BatchSummary,
        ExtractionDetailResponse,
        UpdateExtractionRequest,
    )
    from models_db import Document, DocumentBatch, Extraction, SavedSchema

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["history"])


@router.get("/batches", response_model=BatchHistoryResponse)
async def get_batch_history(
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
) -> BatchHistoryResponse:
    """
    Get history of all batch processing jobs.

    Args:
        db: Database session.
        limit: Maximum number of batches to return.
        offset: Number of batches to skip.

    Returns:
        List of batch summaries with processing stats.
    """
    # Query batches ordered by creation date (newest first)
    batches = (
        db.query(DocumentBatch)
        .order_by(DocumentBatch.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    
    total = db.query(DocumentBatch).count()

    summaries = []
    for batch in batches:
        # Get schema name if available
        schema_name = None
        schema_id_str = None
        if batch.schema_id:
            saved_schema = db.query(SavedSchema).filter(SavedSchema.id == batch.schema_id).first()
            if saved_schema:
                schema_name = saved_schema.name
            schema_id_str = str(batch.schema_id)

        # Determine status
        if batch.completed_at:
            batch_status = "completed"
        else:
            # Check if any documents are still processing
            try:
                from ..models_db import DocumentStatus
            except ImportError:
                from models_db import DocumentStatus
            processing_count = (
                db.query(Document)
                .filter(Document.batch_id == batch.id)
                .filter(Document.status == DocumentStatus.PROCESSING)
                .count()
            )
            batch_status = "processing" if processing_count > 0 else "pending"

        # Get full extraction data for each document
        extractions = []
        documents = db.query(Document).filter(Document.batch_id == batch.id).all()
        for doc in documents:
            # Get extraction for this document
            extraction = (
                db.query(Extraction)
                .filter(Extraction.document_id == doc.id)
                .first()
            )
            if extraction:
                extractions.append(
                    BatchExtractionSummary(
                        id=str(extraction.id),
                        document_id=str(doc.id),
                        filename=doc.filename,
                        extracted_data=extraction.data or {},
                        confidence=extraction.confidence,
                        field_confidences=extraction.field_confidences or {},
                        warnings=extraction.warnings or [],
                        is_reviewed=extraction.is_reviewed,
                    )
                )

        summaries.append(
            BatchSummary(
                id=str(batch.id),
                schema_name=schema_name,
                schema_id=schema_id_str,
                created_at=batch.created_at.isoformat(),
                completed_at=batch.completed_at.isoformat() if batch.completed_at else None,
                total_documents=batch.total_documents,
                successful_documents=batch.successful_documents,
                failed_documents=batch.failed_documents,
                status=batch_status,
                extractions=extractions,
            )
        )

    return BatchHistoryResponse(
        batches=summaries,
        total=total,
    )


@router.get("/extractions/{extraction_id}", response_model=ExtractionDetailResponse)
async def get_extraction(
    extraction_id: str,
    db: Session = Depends(get_db),
) -> ExtractionDetailResponse:
    """
    Get details of a specific extraction.

    Args:
        extraction_id: UUID of the extraction.
        db: Database session.

    Returns:
        Extraction details.
    """
    try:
        extraction_uuid = uuid.UUID(extraction_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid extraction ID format",
        )

    extraction = db.query(Extraction).filter(Extraction.id == extraction_uuid).first()
    if not extraction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extraction {extraction_id} not found",
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


@router.patch("/extractions/{extraction_id}", response_model=ExtractionDetailResponse)
async def update_extraction(
    extraction_id: str,
    request: UpdateExtractionRequest,
    db: Session = Depends(get_db),
) -> ExtractionDetailResponse:
    """
    Update extraction data (human correction).

    Merges the provided data with existing data and marks
    as manually reviewed.

    Args:
        extraction_id: UUID of the extraction.
        request: Fields to update.
        db: Database session.

    Returns:
        Updated extraction details.
    """
    try:
        extraction_uuid = uuid.UUID(extraction_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid extraction ID format",
        )

    extraction = db.query(Extraction).filter(Extraction.id == extraction_uuid).first()
    if not extraction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extraction {extraction_id} not found",
        )

    # Track what was changed
    old_data = extraction.data.copy()
    changed_fields = {}

    # Merge new data - ensure we preserve all existing keys
    merged_data = old_data.copy()
    for key, value in request.data.items():
        if key in merged_data and merged_data[key] != value:
            changed_fields[key] = {"old": merged_data[key], "new": value}
        merged_data[key] = value
    
    # Update the extraction data with merged result
    extraction.data = merged_data

    # Update manual overrides tracking
    if extraction.manual_overrides:
        extraction.manual_overrides.update(changed_fields)
    else:
        extraction.manual_overrides = changed_fields

    # Mark as reviewed
    extraction.is_reviewed = True
    extraction.reviewed_at = datetime.utcnow()

    db.commit()
    db.refresh(extraction)

    logger.info(
        "Updated extraction %s: changed %d fields",
        extraction_id,
        len(changed_fields),
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

