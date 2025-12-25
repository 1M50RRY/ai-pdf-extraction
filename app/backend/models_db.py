"""
SQLAlchemy database models for the PDF Extraction application.

This module defines the ORM models for persisting schemas, documents,
and extraction results to PostgreSQL.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Handle both package imports (FastAPI) and standalone imports (Alembic)
try:
    from .database import Base
except ImportError:
    from database import Base


class DocumentStatus(enum.Enum):
    """Status of a document in the extraction pipeline."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SavedSchema(Base):
    """
    Persisted extraction schema.
    
    Stores the schema definition (fields, types, validation rules)
    that can be reused across multiple extraction batches.
    """
    
    __tablename__ = "saved_schemas"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    version: Mapped[str] = mapped_column(
        String(50),
        default="1.0",
    )
    structure: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="Full SchemaDefinition as JSON",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    
    # Relationships
    batches: Mapped[list["DocumentBatch"]] = relationship(
        "DocumentBatch",
        back_populates="schema",
        cascade="all, delete-orphan",
    )
    
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_schema_name_version"),
    )
    
    def __repr__(self) -> str:
        return f"<SavedSchema(id={self.id}, name='{self.name}', version='{self.version}')>"


class DocumentBatch(Base):
    """
    A batch of documents processed together.
    
    Groups multiple documents that are extracted using the same schema
    in a single processing session.
    """
    
    __tablename__ = "document_batches"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    schema_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("saved_schemas.id", ondelete="SET NULL"),
        nullable=True,
    )
    name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )
    total_documents: Mapped[int] = mapped_column(
        default=0,
    )
    successful_documents: Mapped[int] = mapped_column(
        default=0,
    )
    failed_documents: Mapped[int] = mapped_column(
        default=0,
    )
    
    # Relationships
    schema: Mapped[SavedSchema | None] = relationship(
        "SavedSchema",
        back_populates="batches",
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document",
        back_populates="batch",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<DocumentBatch(id={self.id}, total={self.total_documents})>"


class Document(Base):
    """
    A single document (PDF) in the extraction pipeline.
    
    Tracks the status and metadata of individual documents
    within a batch.
    """
    
    __tablename__ = "documents"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    batch_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_batches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )
    file_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="SHA-256 hash for deduplication",
    )
    file_size_bytes: Mapped[int | None] = mapped_column(
        nullable=True,
    )
    page_count: Mapped[int | None] = mapped_column(
        nullable=True,
    )
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus),
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    upload_date: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )
    file_content: Mapped[bytes | None] = mapped_column(
        LargeBinary,
        nullable=True,
        comment="Stored PDF file content for retrieval",
    )
    
    # Relationships
    batch: Mapped[DocumentBatch] = relationship(
        "DocumentBatch",
        back_populates="documents",
    )
    extractions: Mapped[list["Extraction"]] = relationship(
        "Extraction",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}', status={self.status.value})>"


class Extraction(Base):
    """
    Extraction result for a single document page.
    
    Stores the AI-extracted data, confidence scores, and
    any manual corrections made by users.
    """
    
    __tablename__ = "extractions"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(
        default=1,
        nullable=False,
    )
    data: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="Extracted field values as JSON",
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )
    field_confidences: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        default=None,
        comment="Per-field confidence scores (0.0 to 1.0)",
    )
    warnings: Mapped[list] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )
    is_reviewed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Has a human reviewed this extraction?",
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )
    reviewed_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    manual_overrides: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Fields that were manually corrected",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    
    # Relationships
    document: Mapped[Document] = relationship(
        "Document",
        back_populates="extractions",
    )
    
    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_extraction_document_page"),
    )
    
    def __repr__(self) -> str:
        return f"<Extraction(id={self.id}, confidence={self.confidence:.2f}, reviewed={self.is_reviewed})>"

