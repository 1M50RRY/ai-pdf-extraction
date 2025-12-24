"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from app.backend.models import (
    ExtractionResult,
    FieldDefinition,
    FieldType,
    SchemaDefinition,
)


class TestFieldDefinition:
    """Tests for FieldDefinition model."""

    def test_valid_field_definition(self):
        """Test creating a valid field definition."""
        field = FieldDefinition(
            name="invoice_number",
            type=FieldType.STRING,
            description="The invoice number",
        )
        assert field.name == "invoice_number"
        assert field.type == FieldType.STRING
        assert field.required is True  # Default

    def test_field_name_normalization(self):
        """Test that field names are normalized to lowercase with underscores."""
        field = FieldDefinition(
            name="Invoice-Number",
            type=FieldType.STRING,
            description="Test",
        )
        assert field.name == "invoice_number"

    def test_invalid_field_name_characters(self):
        """Test that invalid characters in field name raise error."""
        with pytest.raises(ValidationError):
            FieldDefinition(
                name="field@name!",
                type=FieldType.STRING,
                description="Test",
            )

    def test_empty_field_name_rejected(self):
        """Test that empty field name is rejected."""
        with pytest.raises(ValidationError):
            FieldDefinition(
                name="",
                type=FieldType.STRING,
                description="Test",
            )

    def test_all_field_types(self):
        """Test that all field types are valid."""
        for field_type in FieldType:
            field = FieldDefinition(
                name=f"test_{field_type.value}",
                type=field_type,
                description=f"Testing {field_type.value}",
            )
            assert field.type == field_type


class TestSchemaDefinition:
    """Tests for SchemaDefinition model."""

    def test_valid_schema_definition(self):
        """Test creating a valid schema definition."""
        schema = SchemaDefinition(
            name="Invoice Schema",
            description="For invoices",
            fields=[
                FieldDefinition(
                    name="invoice_number",
                    type=FieldType.STRING,
                    description="Invoice ID",
                ),
            ],
        )
        assert schema.name == "Invoice Schema"
        assert len(schema.fields) == 1
        assert schema.version == "1.0"  # Default

    def test_schema_requires_at_least_one_field(self):
        """Test that schema requires at least one field."""
        with pytest.raises(ValidationError):
            SchemaDefinition(
                name="Empty Schema",
                fields=[],
            )

    def test_schema_rejects_duplicate_field_names(self):
        """Test that duplicate field names are rejected."""
        with pytest.raises(ValidationError):
            SchemaDefinition(
                name="Duplicate Fields",
                fields=[
                    FieldDefinition(
                        name="amount",
                        type=FieldType.CURRENCY,
                        description="Amount 1",
                    ),
                    FieldDefinition(
                        name="amount",
                        type=FieldType.CURRENCY,
                        description="Amount 2",
                    ),
                ],
            )

    def test_schema_with_validation_rules(self):
        """Test schema with validation rules."""
        schema = SchemaDefinition(
            name="Invoice",
            fields=[
                FieldDefinition(name="subtotal", type=FieldType.CURRENCY, description="Subtotal"),
                FieldDefinition(name="tax", type=FieldType.CURRENCY, description="Tax"),
                FieldDefinition(name="total", type=FieldType.CURRENCY, description="Total"),
            ],
            validation_rules=["total == subtotal + tax"],
        )
        assert len(schema.validation_rules) == 1
        assert schema.validation_rules[0] == "total == subtotal + tax"

    def test_schema_validation_rules_default_empty(self):
        """Test that validation_rules defaults to empty list."""
        schema = SchemaDefinition(
            name="Simple",
            fields=[
                FieldDefinition(name="field1", type=FieldType.STRING, description="Field 1"),
            ],
        )
        assert schema.validation_rules == []

    def test_schema_validation_rules_filters_invalid(self):
        """Test that invalid rule formats are filtered out."""
        schema = SchemaDefinition(
            name="Test",
            fields=[
                FieldDefinition(name="a", type=FieldType.NUMBER, description="A"),
                FieldDefinition(name="b", type=FieldType.NUMBER, description="B"),
            ],
            validation_rules=[
                "a == b + c",  # Valid format
                "invalid rule without equals",  # Invalid - filtered
                "",  # Empty - filtered
            ],
        )
        # Only valid rules should remain
        assert len(schema.validation_rules) == 1
        assert schema.validation_rules[0] == "a == b + c"


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_valid_extraction_result(self):
        """Test creating a valid extraction result."""
        schema = SchemaDefinition(
            name="Test Schema",
            fields=[
                FieldDefinition(
                    name="test_field",
                    type=FieldType.STRING,
                    description="Test",
                ),
            ],
        )
        result = ExtractionResult(
            source_file="test.pdf",
            detected_schema=schema,
            extracted_data={"test_field": "value"},
            confidence=0.95,
            warnings=[],
        )
        assert result.source_file == "test.pdf"
        assert result.confidence == 0.95
        assert result.extracted_data["test_field"] == "value"

    def test_confidence_bounds(self):
        """Test that confidence must be between 0 and 1."""
        schema = SchemaDefinition(
            name="Test",
            fields=[
                FieldDefinition(
                    name="field",
                    type=FieldType.STRING,
                    description="Test",
                ),
            ],
        )

        # Test lower bound
        with pytest.raises(ValidationError):
            ExtractionResult(
                source_file="test.pdf",
                detected_schema=schema,
                extracted_data={},
                confidence=-0.1,
            )

        # Test upper bound
        with pytest.raises(ValidationError):
            ExtractionResult(
                source_file="test.pdf",
                detected_schema=schema,
                extracted_data={},
                confidence=1.1,
            )

    def test_confidence_rounding(self):
        """Test that confidence is rounded to 3 decimal places."""
        schema = SchemaDefinition(
            name="Test",
            fields=[
                FieldDefinition(
                    name="field",
                    type=FieldType.STRING,
                    description="Test",
                ),
            ],
        )
        result = ExtractionResult(
            source_file="test.pdf",
            detected_schema=schema,
            extracted_data={},
            confidence=0.123456789,
        )
        assert result.confidence == 0.123

    def test_extraction_result_json_structure(self):
        """Test that ExtractionResult matches the required JSON structure."""
        schema = SchemaDefinition(
            name="Test",
            fields=[
                FieldDefinition(
                    name="field",
                    type=FieldType.STRING,
                    description="Test",
                ),
            ],
        )
        result = ExtractionResult(
            source_file="test.pdf",
            detected_schema=schema,
            extracted_data={"field": "value"},
            confidence=0.9,
            warnings=["Test warning"],
        )
        data = result.model_dump()

        # Verify required keys
        assert "source_file" in data
        assert "detected_schema" in data
        assert "extracted_data" in data
        assert "confidence" in data
        assert "warnings" in data

        # Verify types
        assert isinstance(data["source_file"], str)
        assert isinstance(data["detected_schema"], dict)
        assert isinstance(data["extracted_data"], dict)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["warnings"], list)

