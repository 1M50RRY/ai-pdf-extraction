"""
FastAPI application for PDF extraction service.

Provides endpoints for:
- Uploading sample PDFs for schema detection
- Managing schema templates (CRUD)
- Batch extraction with async processing
- Human-in-the-loop review and editing
"""

import hashlib
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .database import get_db, init_db
from .models import (
    ApproveExtractionRequest,
    ApproveExtractionResponse,
    BatchStatusResponse,
    DocumentStatusResponse,
    ExtractBatchResponse,
    ExtractionDetailResponse,
    ExtractionResult,
    HealthResponse,
    SavedSchemaResponse,
    SaveSchemaRequest,
    SchemaDefinition,
    SchemaListResponse,
    StartBatchResponse,
    UpdateExtractionRequest,
    UploadSampleResponse,
)
from .models_db import (
    Document,
    DocumentBatch,
    DocumentStatus,
    Extraction,
    SavedSchema,
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
    return HealthResponse(status="healthy", version="2.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="2.0.0")


# =============================================================================
# Schema Registry Endpoints
# =============================================================================


@app.post("/schemas", response_model=SavedSchemaResponse, status_code=status.HTTP_201_CREATED)
async def create_schema(
    request: SaveSchemaRequest,
    db: Session = Depends(get_db),
) -> SavedSchemaResponse:
    """
    Save a schema as a reusable template.

    Args:
        request: Schema to save.
        db: Database session.

    Returns:
        The saved schema with generated ID.
    """
    schema_def = request.schema_definition

    # Check for duplicate name/version
    existing = (
        db.query(SavedSchema)
        .filter(
            SavedSchema.name == schema_def.name,
            SavedSchema.version == schema_def.version,
            SavedSchema.is_active == True,  # noqa: E712
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Schema '{schema_def.name}' version '{schema_def.version}' already exists",
        )

    # Create new schema record
    db_schema = SavedSchema(
        name=schema_def.name,
        description=schema_def.description,
        version=schema_def.version,
        structure=schema_def.model_dump(),
    )
    db.add(db_schema)
    db.commit()
    db.refresh(db_schema)

    logger.info("Created schema: %s (id=%s)", schema_def.name, db_schema.id)

    return SavedSchemaResponse(
        id=str(db_schema.id),
        name=db_schema.name,
        description=db_schema.description or "",
        version=db_schema.version,
        structure=SchemaDefinition.model_validate(db_schema.structure),
        created_at=db_schema.created_at.isoformat(),
        is_active=db_schema.is_active,
    )


@app.get("/schemas", response_model=SchemaListResponse)
async def list_schemas(
    db: Session = Depends(get_db),
    active_only: bool = True,
) -> SchemaListResponse:
    """
    List all saved schema templates.

    Args:
        db: Database session.
        active_only: Only return active schemas.

    Returns:
        List of saved schemas.
    """
    query = db.query(SavedSchema)
    if active_only:
        query = query.filter(SavedSchema.is_active == True)  # noqa: E712
    
    schemas = query.order_by(SavedSchema.created_at.desc()).all()

    return SchemaListResponse(
        schemas=[
            SavedSchemaResponse(
                id=str(s.id),
                name=s.name,
                description=s.description or "",
                version=s.version,
                structure=SchemaDefinition.model_validate(s.structure),
                created_at=s.created_at.isoformat(),
                is_active=s.is_active,
            )
            for s in schemas
        ],
        total=len(schemas),
    )


@app.get("/schemas/{schema_id}", response_model=SavedSchemaResponse)
async def get_schema(
    schema_id: str,
    db: Session = Depends(get_db),
) -> SavedSchemaResponse:
    """
    Get a specific schema by ID.

    Args:
        schema_id: UUID of the schema.
        db: Database session.

    Returns:
        The schema details.
    """
    try:
        schema_uuid = uuid.UUID(schema_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid schema ID format",
        )

    schema = db.query(SavedSchema).filter(SavedSchema.id == schema_uuid).first()
    if not schema:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schema {schema_id} not found",
        )

    return SavedSchemaResponse(
        id=str(schema.id),
        name=schema.name,
        description=schema.description or "",
        version=schema.version,
        structure=SchemaDefinition.model_validate(schema.structure),
        created_at=schema.created_at.isoformat(),
        is_active=schema.is_active,
    )


@app.delete("/schemas/{schema_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schema(
    schema_id: str,
    db: Session = Depends(get_db),
) -> None:
    """
    Soft delete a schema (mark as inactive).

    Args:
        schema_id: UUID of the schema.
        db: Database session.
    """
    try:
        schema_uuid = uuid.UUID(schema_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid schema ID format",
        )

    schema = db.query(SavedSchema).filter(SavedSchema.id == schema_uuid).first()
    if not schema:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schema {schema_id} not found",
        )

    schema.is_active = False
    db.commit()
    logger.info("Deactivated schema: %s", schema_id)


# =============================================================================
# Sample Upload Endpoint
# =============================================================================


@app.post("/upload-sample", response_model=UploadSampleResponse)
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
            first_page_image = pdf_service.convert_first_page(file_bytes)
        except PDFConversionError as e:
            logger.error("PDF conversion failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e),
            )

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


# =============================================================================
# Batch Extraction Endpoints
# =============================================================================


async def process_batch_documents(
    batch_id: uuid.UUID,
    file_data: list[tuple[str, bytes, str]],  # (filename, content, file_hash)
    schema: SchemaDefinition,
    db_url: str,
) -> None:
    """
    Background task to process documents in a batch.

    Args:
        batch_id: The batch ID.
        file_data: List of (filename, content, file_hash) tuples.
        schema: The schema to use for extraction.
        db_url: Database URL for creating a new session.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # Create a new session for the background task
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    pdf_service = get_pdf_service()
    ai_service = get_ai_service()

    try:
        batch = db.query(DocumentBatch).filter(DocumentBatch.id == batch_id).first()
        if not batch:
            logger.error("Batch %s not found", batch_id)
            return

        documents = db.query(Document).filter(Document.batch_id == batch_id).all()
        doc_map = {d.filename: d for d in documents}

        successful = 0
        failed = 0

        for filename, content, file_hash in file_data:
            doc = doc_map.get(filename)
            if not doc:
                logger.warning("Document %s not found in batch", filename)
                continue

            try:
                # Update status to processing
                doc.status = DocumentStatus.PROCESSING
                db.commit()

                # Convert PDF to images
                try:
                    page_images = pdf_service.convert_pdf_to_images(content)
                    doc.page_count = len(page_images)
                except PDFConversionError as e:
                    doc.status = DocumentStatus.FAILED
                    doc.error_message = f"PDF conversion failed: {e}"
                    doc.processed_at = datetime.utcnow()
                    db.commit()
                    failed += 1
                    continue

                # Extract data from each page
                total_confidence = 0.0
                page_warnings: list[str] = []

                for page_num, image in enumerate(page_images, start=1):
                    page_source = f"{filename} (page {page_num})"
                    try:
                        result = await ai_service.extract_data(image, schema, page_source)
                        
                        # Save extraction result
                        extraction = Extraction(
                            document_id=doc.id,
                            page_number=page_num,
                            data=result.extracted_data,
                            confidence=result.confidence,
                            warnings=result.warnings,
                        )
                        db.add(extraction)
                        total_confidence += result.confidence
                        page_warnings.extend(result.warnings)

                    except AIServiceError as e:
                        logger.warning("Extraction failed for %s: %s", page_source, e)
                        # Save failed extraction
                        extraction = Extraction(
                            document_id=doc.id,
                            page_number=page_num,
                            data={},
                            confidence=0.0,
                            warnings=[f"Extraction failed: {e}"],
                        )
                        db.add(extraction)
                        page_warnings.append(f"Page {page_num}: {e}")

                # Update document status
                doc.status = DocumentStatus.COMPLETED
                doc.processed_at = datetime.utcnow()
                db.commit()
                successful += 1

                logger.info(
                    "Processed document %s: %d pages, avg confidence %.2f",
                    filename,
                    len(page_images),
                    total_confidence / len(page_images) if page_images else 0,
                )

            except Exception as e:
                logger.exception("Error processing document %s", filename)
                doc.status = DocumentStatus.FAILED
                doc.error_message = str(e)
                doc.processed_at = datetime.utcnow()
                db.commit()
                failed += 1

        # Update batch statistics
        batch.successful_documents = successful
        batch.failed_documents = failed
        batch.completed_at = datetime.utcnow()
        db.commit()

        logger.info(
            "Batch %s completed: %d successful, %d failed",
            batch_id,
            successful,
            failed,
        )

    except Exception as e:
        logger.exception("Fatal error in batch processing: %s", e)
    finally:
        db.close()


@app.post("/extract-batch", response_model=StartBatchResponse)
async def extract_batch(
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File(description="PDF files to extract")],
    schema_id: Annotated[str | None, Form(description="ID of saved schema")] = None,
    confirmed_schema: Annotated[
        str | None,
        Form(description="JSON string of SchemaDefinition (if not using schema_id)"),
    ] = None,
    db: Session = Depends(get_db),
) -> StartBatchResponse:
    """
    Start batch extraction of multiple PDFs.

    Creates a batch record and processes documents asynchronously.
    Use GET /batches/{id}/status to monitor progress.

    Args:
        files: List of PDF files to process.
        schema_id: ID of a saved schema to use.
        confirmed_schema: Inline schema JSON (alternative to schema_id).
        db: Database session.

    Returns:
        Batch ID and initial status.
    """
    from .database import DATABASE_URL

    # Resolve schema
    schema: SchemaDefinition | None = None

    if schema_id:
        try:
            schema_uuid = uuid.UUID(schema_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid schema_id format",
            )

        db_schema = db.query(SavedSchema).filter(SavedSchema.id == schema_uuid).first()
        if not db_schema:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Schema {schema_id} not found",
            )
        schema = SchemaDefinition.model_validate(db_schema.structure)

    elif confirmed_schema:
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
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either schema_id or confirmed_schema must be provided",
        )

    # Validate files
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file is required",
        )

    # Read and validate all files
    file_data: list[tuple[str, bytes, str]] = []
    for file in files:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All files must have filenames",
            )
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Only PDF files accepted: {file.filename}",
            )

        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Empty file: {file.filename}",
            )

        file_hash = hashlib.sha256(content).hexdigest()
        file_data.append((file.filename, content, file_hash))
        await file.close()

    # Create batch record
    batch = DocumentBatch(
        schema_id=uuid.UUID(schema_id) if schema_id else None,
        total_documents=len(file_data),
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)

    # Create document records
    for filename, content, file_hash in file_data:
        doc = Document(
            batch_id=batch.id,
            filename=filename,
            file_hash=file_hash,
            file_size_bytes=len(content),
            status=DocumentStatus.PENDING,
        )
        db.add(doc)
    db.commit()

    logger.info(
        "Created batch %s with %d documents",
        batch.id,
        len(file_data),
    )

    # Start background processing
    background_tasks.add_task(
        process_batch_documents,
        batch.id,
        file_data,
        schema,
        DATABASE_URL,
    )

    return StartBatchResponse(
        batch_id=str(batch.id),
        message=f"Batch started with {len(file_data)} documents",
        total_documents=len(file_data),
        status="processing",
    )


@app.get("/batches/{batch_id}/status", response_model=BatchStatusResponse)
async def get_batch_status(
    batch_id: str,
    db: Session = Depends(get_db),
) -> BatchStatusResponse:
    """
    Get the status of a batch extraction.

    Args:
        batch_id: UUID of the batch.
        db: Database session.

    Returns:
        Batch status with progress and document details.
    """
    try:
        batch_uuid = uuid.UUID(batch_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid batch ID format",
        )

    batch = db.query(DocumentBatch).filter(DocumentBatch.id == batch_uuid).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_id} not found",
        )

    # Get all documents in batch
    documents = db.query(Document).filter(Document.batch_id == batch_uuid).all()

    # Calculate progress
    completed = sum(
        1 for d in documents if d.status in (DocumentStatus.COMPLETED, DocumentStatus.FAILED)
    )
    progress = (completed / batch.total_documents * 100) if batch.total_documents > 0 else 0

    # Determine overall status
    if batch.completed_at:
        overall_status = "completed"
    elif any(d.status == DocumentStatus.PROCESSING for d in documents):
        overall_status = "processing"
    elif all(d.status == DocumentStatus.PENDING for d in documents):
        overall_status = "pending"
    else:
        overall_status = "processing"

    # Get schema info
    schema_name = None
    if batch.schema_id:
        schema = db.query(SavedSchema).filter(SavedSchema.id == batch.schema_id).first()
        if schema:
            schema_name = schema.name

    # Build document status list
    doc_statuses = []
    for doc in documents:
        # Get extraction confidence if completed
        confidence = None
        warnings: list[str] = []
        if doc.status == DocumentStatus.COMPLETED:
            extractions = db.query(Extraction).filter(Extraction.document_id == doc.id).all()
            if extractions:
                confidence = sum(e.confidence for e in extractions) / len(extractions)
                for e in extractions:
                    warnings.extend(e.warnings or [])

        doc_statuses.append(
            DocumentStatusResponse(
                id=str(doc.id),
                filename=doc.filename,
                status=doc.status.value,
                confidence=round(confidence, 3) if confidence is not None else None,
                error_message=doc.error_message,
                warnings=warnings,
            )
        )

    return BatchStatusResponse(
        id=str(batch.id),
        status=overall_status,
        created_at=batch.created_at.isoformat(),
        completed_at=batch.completed_at.isoformat() if batch.completed_at else None,
        total_documents=batch.total_documents,
        completed_documents=completed,
        failed_documents=batch.failed_documents,
        progress_percent=round(progress, 1),
        documents=doc_statuses,
        schema_id=str(batch.schema_id) if batch.schema_id else None,
        schema_name=schema_name,
    )


@app.get("/batches/{batch_id}/results", response_model=ExtractBatchResponse)
async def get_batch_results(
    batch_id: str,
    db: Session = Depends(get_db),
) -> ExtractBatchResponse:
    """
    Get extraction results for a completed batch.

    Args:
        batch_id: UUID of the batch.
        db: Database session.

    Returns:
        Extraction results for all documents.
    """
    try:
        batch_uuid = uuid.UUID(batch_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid batch ID format",
        )

    batch = db.query(DocumentBatch).filter(DocumentBatch.id == batch_uuid).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_id} not found",
        )

    # Get schema
    if batch.schema_id:
        db_schema = db.query(SavedSchema).filter(SavedSchema.id == batch.schema_id).first()
        if db_schema:
            schema = SchemaDefinition.model_validate(db_schema.structure)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Schema not found for batch",
            )
    else:
        # If no saved schema, we need to reconstruct from first extraction
        # This is a fallback for inline schemas
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch was created without a saved schema_id. Use /batches/{id}/status to get results.",
        )

    # Get all documents and extractions
    documents = db.query(Document).filter(Document.batch_id == batch_uuid).all()
    results: list[ExtractionResult] = []
    total_confidence = 0.0
    successful = 0

    for doc in documents:
        extractions = db.query(Extraction).filter(Extraction.document_id == doc.id).all()
        for extraction in extractions:
            result = ExtractionResult(
                source_file=f"{doc.filename} (page {extraction.page_number})",
                detected_schema=schema,
                extracted_data=extraction.data,
                confidence=extraction.confidence,
                warnings=extraction.warnings or [],
            )
            results.append(result)
            total_confidence += extraction.confidence
            if extraction.confidence > 0:
                successful += 1

    avg_confidence = total_confidence / len(results) if results else 0.0

    return ExtractBatchResponse(
        results=results,
        total_pages=len(results),
        successful_extractions=successful,
        average_confidence=round(avg_confidence, 3),
    )


# =============================================================================
# Human-in-the-Loop Endpoints
# =============================================================================


@app.get("/extractions/{extraction_id}", response_model=ExtractionDetailResponse)
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
        warnings=extraction.warnings or [],
        is_reviewed=extraction.is_reviewed,
        manual_overrides=extraction.manual_overrides,
        created_at=extraction.created_at.isoformat(),
        reviewed_at=extraction.reviewed_at.isoformat() if extraction.reviewed_at else None,
    )


@app.patch("/extractions/{extraction_id}", response_model=ExtractionDetailResponse)
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

    # Merge new data
    for key, value in request.data.items():
        if key in old_data and old_data[key] != value:
            changed_fields[key] = {"old": old_data[key], "new": value}
        extraction.data[key] = value

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
        warnings=extraction.warnings or [],
        is_reviewed=extraction.is_reviewed,
        manual_overrides=extraction.manual_overrides,
        created_at=extraction.created_at.isoformat(),
        reviewed_at=extraction.reviewed_at.isoformat() if extraction.reviewed_at else None,
    )


@app.post("/batches/{batch_id}/approve", response_model=ApproveExtractionResponse)
async def approve_batch(
    batch_id: str,
    request: ApproveExtractionRequest | None = None,
    db: Session = Depends(get_db),
) -> ApproveExtractionResponse:
    """
    Mark all extractions in a batch as reviewed.

    Args:
        batch_id: UUID of the batch.
        request: Optional reviewer info.
        db: Database session.

    Returns:
        Approval status.
    """
    try:
        batch_uuid = uuid.UUID(batch_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid batch ID format",
        )

    batch = db.query(DocumentBatch).filter(DocumentBatch.id == batch_uuid).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_id} not found",
        )

    # Get all documents in batch
    documents = db.query(Document).filter(Document.batch_id == batch_uuid).all()
    doc_ids = [d.id for d in documents]

    # Update all extractions
    now = datetime.utcnow()
    reviewed_by = request.reviewed_by if request else None

    count = (
        db.query(Extraction)
        .filter(Extraction.document_id.in_(doc_ids))
        .update(
            {
                Extraction.is_reviewed: True,
                Extraction.reviewed_at: now,
                Extraction.reviewed_by: reviewed_by,
            },
            synchronize_session="fetch",
        )
    )

    db.commit()

    logger.info("Approved %d extractions in batch %s", count, batch_id)

    return ApproveExtractionResponse(
        message=f"Approved {count} extractions",
        approved_count=count,
        batch_id=batch_id,
    )


# =============================================================================
# Legacy Endpoint (Synchronous - kept for backwards compatibility)
# =============================================================================


@app.post("/extract-batch-sync", response_model=ExtractBatchResponse)
async def extract_batch_sync(
    file: Annotated[UploadFile, File(description="PDF file to extract data from")],
    confirmed_schema: Annotated[
        str,
        Form(description="JSON string of confirmed SchemaDefinition"),
    ],
) -> ExtractBatchResponse:
    """
    [LEGACY] Synchronous batch extraction.

    Processes a single PDF synchronously and returns results immediately.
    For new integrations, use POST /extract-batch with async processing.
    """
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

        pdf_service = get_pdf_service()
        ai_service = get_ai_service()

        try:
            page_images = pdf_service.convert_pdf_to_images(file_bytes)
        except PDFConversionError as e:
            logger.error("PDF conversion failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e),
            )

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
                results.append(
                    ExtractionResult(
                        source_file=page_source,
                        detected_schema=schema,
                        extracted_data={},
                        confidence=0.0,
                        warnings=[f"Extraction failed: {e}"],
                    )
                )

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


# =============================================================================
# Exception Handlers
# =============================================================================


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
