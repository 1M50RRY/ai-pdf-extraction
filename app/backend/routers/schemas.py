"""
Router for schema template management endpoints.

Handles:
- Creating schema templates
- Listing schemas
- Getting schema details
- Deleting schemas
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

# Handle both package imports and standalone imports
try:
    from ..database import get_db
    from ..models import (
        SaveSchemaRequest,
        SavedSchemaResponse,
        SchemaDefinition,
        SchemaListResponse,
    )
    from ..models_db import SavedSchema
except ImportError:
    from database import get_db
    from models import (
        SaveSchemaRequest,
        SavedSchemaResponse,
        SchemaDefinition,
        SchemaListResponse,
    )
    from models_db import SavedSchema

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/schemas", tags=["schemas"])


@router.post("", response_model=SavedSchemaResponse, status_code=status.HTTP_201_CREATED)
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


@router.get("", response_model=SchemaListResponse)
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


@router.get("/{schema_id}", response_model=SavedSchemaResponse)
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


@router.delete("/{schema_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schema(
    schema_id: str,
    db: Session = Depends(get_db),
):
    """
    Soft-delete a schema (mark as inactive).

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

    # Soft delete (mark as inactive)
    schema.is_active = False
    db.commit()

    logger.info("Deleted schema: %s (id=%s)", schema.name, schema_id)

