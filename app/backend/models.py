"""
Pydantic models for the PDF extraction pipeline.

Defines strict types for schema definitions, field specifications,
and extraction results with validation.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class FieldType(str, Enum):
    """Supported field types for extraction."""

    STRING = "string"  # Also serves as catch-all for mixed text/data
    CURRENCY = "currency"
    DATE = "date"
    NUMBER = "number"
    BOOLEAN = "boolean"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    PERCENTAGE = "percentage"


class FieldDefinition(BaseModel):
    """
    Definition of a single field to extract from a document.

    Attributes:
        name: Unique identifier for the field (snake_case recommended).
        type: The expected data type for validation and parsing.
        description: Human-readable description to guide AI extraction.
        required: Whether this field must be present in the document.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique field identifier",
        examples=["invoice_number", "total_amount"],
    )
    type: FieldType = Field(
        ...,
        description="Expected data type for the field",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Description to guide AI extraction",
        examples=["The invoice number, usually at the top of the document"],
    )
    required: bool = Field(
        default=True,
        description="Whether this field must be present",
    )

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        """Ensure field name is a valid identifier."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Field name must contain only alphanumeric characters, underscores, or hyphens"
            )
        return v.lower().replace("-", "_")


class SchemaDefinition(BaseModel):
    """
    Complete schema defining all fields to extract from a document type.

    Attributes:
        name: Human-readable name for this schema.
        description: Description of what document type this schema handles.
        fields: List of field definitions to extract.
        version: Schema version for tracking changes.
        validation_rules: Optional list of mathematical validation rules.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Schema name",
        examples=["Invoice Schema", "Receipt Schema"],
    )
    description: str = Field(
        default="",
        max_length=1000,
        description="Description of the document type",
    )
    fields: list[FieldDefinition] = Field(
        ...,
        min_length=1,
        description="List of fields to extract",
    )
    version: str = Field(
        default="1.0",
        description="Schema version",
    )
    validation_rules: list[str] = Field(
        default_factory=list,
        description=(
            "Mathematical validation rules using field names. "
            "Format: 'field_a == field_b + field_c' or 'field_x == field_y - field_z'. "
            "Used to verify numerical relationships in extracted data."
        ),
        examples=[
            "total_amount == subtotal + tax",
            "net_pay == gross_pay - deductions",
            "balance == credits - debits",
        ],
    )

    @field_validator("fields")
    @classmethod
    def validate_unique_field_names(
        cls, v: list[FieldDefinition]
    ) -> list[FieldDefinition]:
        """Ensure all field names are unique."""
        names = [f.name for f in v]
        if len(names) != len(set(names)):
            raise ValueError("All field names must be unique within a schema")
        return v

    @field_validator("validation_rules")
    @classmethod
    def validate_rule_format(cls, v: list[str]) -> list[str]:
        """
        Validate that rules follow a valid expression format.

        Allows complex expressions like:
        - "total == subtotal + tax"
        - "grand_total == (subtotal + shipping) * (1 + tax_rate)"
        - "margin == (revenue - cost) / revenue"
        """
        import re

        # More flexible pattern: must have == and valid expression chars
        # Allows: identifiers, operators (+, -, *, /, ==), parentheses, numbers, spaces
        valid_chars_pattern = re.compile(
            r"^[a-z_][a-z0-9_]*\s*==\s*[\w\s+\-*/().]+$",
            re.IGNORECASE,
        )
        validated = []
        for rule in v:
            rule = rule.strip()
            if rule and "==" in rule and valid_chars_pattern.match(rule):
                validated.append(rule)
        return validated


class ExtractionResult(BaseModel):
    """
    Complete result of a document extraction operation.

    This model matches the required JSON structure exactly:
    {
        "source_file": str,
        "detected_schema": SchemaDefinition,
        "extracted_data": dict,
        "confidence": float (0.0 to 1.0),
        "warnings": List[str]
    }
    """

    source_file: str = Field(
        ...,
        description="Original filename of the processed document",
    )
    detected_schema: SchemaDefinition = Field(
        ...,
        description="The schema used for extraction",
    )
    extracted_data: dict[str, Any] = Field(
        ...,
        description="Dynamic dictionary of extracted field values",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence score for the extraction (0.0 to 1.0)",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="List of warnings encountered during extraction",
    )

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        """Round confidence to 3 decimal places."""
        return round(v, 3)


class UploadSampleRequest(BaseModel):
    """Request model for uploading a sample PDF."""

    # File is handled via Form/File, this is for any additional metadata
    pass


class UploadSampleResponse(BaseModel):
    """Response model for the upload-sample endpoint."""

    message: str = Field(..., description="Status message")
    suggested_schema: SchemaDefinition = Field(
        ...,
        description="AI-suggested schema based on document analysis",
    )
    preview_available: bool = Field(
        default=False,
        description="Whether a preview image is available",
    )
    page_count: int = Field(
        ...,
        ge=1,
        description="Total number of pages in the PDF",
    )


class ExtractBatchRequest(BaseModel):
    """Request model for batch extraction."""

    confirmed_schema: SchemaDefinition = Field(
        ...,
        description="The confirmed schema to use for extraction",
    )


class ExtractBatchResponse(BaseModel):
    """Response model for the extract-batch endpoint."""

    results: list[ExtractionResult] = Field(
        ...,
        description="List of extraction results, one per page",
    )
    total_pages: int = Field(..., ge=1, description="Total pages processed")
    successful_extractions: int = Field(
        ...,
        ge=0,
        description="Number of successful extractions",
    )
    average_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average confidence across all extractions",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")

