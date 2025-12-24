"""
AI service for document schema detection and data extraction.

Uses OpenAI GPT-4o with structured outputs via Pydantic models.
"""

import base64
import io
import logging
import os
from typing import Any

from PIL import Image

from ..models import (
    ExtractionResult,
    FieldDefinition,
    FieldType,
    SchemaDefinition,
)

logger = logging.getLogger(__name__)


class AIServiceError(Exception):
    """Raised when AI service operations fail."""

    pass


class AIService:
    """
    Service for AI-powered document analysis and extraction.

    Uses OpenAI's GPT-4o model with vision capabilities for:
    - Analyzing document structure and suggesting extraction schemas
    - Extracting data according to confirmed schemas
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        """
        Initialize the AI service.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: OpenAI model to use (must support vision).
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise AIServiceError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
                )
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError as e:
                raise AIServiceError(
                    "openai library not installed. Run: pip install openai"
                ) from e
        return self._client

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for API."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def suggest_schema(self, image: Image.Image) -> SchemaDefinition:
        """
        Analyze a document image and suggest an extraction schema.

        Args:
            image: PIL Image of the document page.

        Returns:
            Suggested SchemaDefinition based on document analysis.
        """
        # TODO: Implement actual OpenAI call when ready
        # For now, return a mock schema for development
        logger.info("Generating suggested schema (mock mode)")
        return self._get_mock_schema()

    async def extract_data(
        self,
        image: Image.Image,
        schema: SchemaDefinition,
        source_file: str,
    ) -> ExtractionResult:
        """
        Extract data from a document image according to a schema.

        Args:
            image: PIL Image of the document page.
            schema: The schema defining what fields to extract.
            source_file: Original filename for the result.

        Returns:
            ExtractionResult with extracted data.
        """
        # TODO: Implement actual OpenAI call when ready
        # For now, return mock extraction for development
        logger.info("Extracting data (mock mode) for: %s", source_file)
        return self._get_mock_extraction(schema, source_file)

    def _get_mock_schema(self) -> SchemaDefinition:
        """
        Return a mock invoice schema for development.

        This will be replaced with actual AI analysis.
        """
        return SchemaDefinition(
            name="Invoice Schema",
            description="Standard invoice document with common fields",
            version="1.0",
            fields=[
                FieldDefinition(
                    name="invoice_number",
                    type=FieldType.STRING,
                    description="The unique invoice identifier, usually at the top",
                    required=True,
                ),
                FieldDefinition(
                    name="invoice_date",
                    type=FieldType.DATE,
                    description="The date the invoice was issued",
                    required=True,
                ),
                FieldDefinition(
                    name="due_date",
                    type=FieldType.DATE,
                    description="Payment due date",
                    required=False,
                ),
                FieldDefinition(
                    name="vendor_name",
                    type=FieldType.STRING,
                    description="Name of the company/person issuing the invoice",
                    required=True,
                ),
                FieldDefinition(
                    name="subtotal",
                    type=FieldType.CURRENCY,
                    description="Subtotal before tax",
                    required=False,
                ),
                FieldDefinition(
                    name="tax_amount",
                    type=FieldType.CURRENCY,
                    description="Tax amount",
                    required=False,
                ),
                FieldDefinition(
                    name="total_amount",
                    type=FieldType.CURRENCY,
                    description="Total amount due including tax",
                    required=True,
                ),
            ],
        )

    def _get_mock_extraction(
        self, schema: SchemaDefinition, source_file: str
    ) -> ExtractionResult:
        """
        Return mock extraction data for development.

        This will be replaced with actual AI extraction.
        """
        # Generate mock data based on schema fields
        mock_data: dict[str, Any] = {}
        warnings: list[str] = []

        for field in schema.fields:
            if field.type == FieldType.STRING:
                mock_data[field.name] = f"MOCK-{field.name.upper()}-001"
            elif field.type == FieldType.CURRENCY:
                mock_data[field.name] = "$1,234.56"
            elif field.type == FieldType.DATE:
                mock_data[field.name] = "2024-01-15"
            elif field.type == FieldType.NUMBER:
                mock_data[field.name] = 42
            elif field.type == FieldType.BOOLEAN:
                mock_data[field.name] = True
            elif field.type == FieldType.EMAIL:
                mock_data[field.name] = "mock@example.com"
            elif field.type == FieldType.PHONE:
                mock_data[field.name] = "+1-555-123-4567"
            elif field.type == FieldType.ADDRESS:
                mock_data[field.name] = "123 Mock Street, Test City, TC 12345"
            elif field.type == FieldType.PERCENTAGE:
                mock_data[field.name] = "15%"
            else:
                mock_data[field.name] = None
                if field.required:
                    warnings.append(f"Could not extract required field: {field.name}")

        # Add a development warning
        warnings.append("DEVELOPMENT MODE: Using mock data. Connect OpenAI for real extraction.")

        return ExtractionResult(
            source_file=source_file,
            detected_schema=schema,
            extracted_data=mock_data,
            confidence=0.85,  # Mock confidence
            warnings=warnings,
        )

    async def suggest_schema_real(self, image: Image.Image) -> SchemaDefinition:
        """
        Actual OpenAI implementation for schema suggestion.

        Call this once OpenAI integration is ready.
        """
        base64_image = self._image_to_base64(image)

        system_prompt = """You are a document analysis expert. Analyze the provided document image and suggest a data extraction schema.

Identify all extractable fields and categorize them by type:
- string: Text fields (names, IDs, descriptions)
- currency: Money amounts
- date: Dates in any format
- number: Numeric values
- boolean: Yes/No, True/False fields
- email: Email addresses
- phone: Phone numbers
- address: Physical addresses
- percentage: Percentage values

Return a structured schema with field names (snake_case), types, and descriptions."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this document and suggest an extraction schema.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
                max_tokens=2000,
            )

            # Parse response and create schema
            # This is a simplified version - production would use structured outputs
            content = response.choices[0].message.content
            logger.info("AI schema suggestion received: %s", content[:200])

            # For now, return mock - full implementation would parse the response
            return self._get_mock_schema()

        except Exception as e:
            logger.exception("OpenAI API call failed")
            raise AIServiceError(f"Schema suggestion failed: {e}") from e


# Singleton instance for convenience
_ai_service: AIService | None = None


def get_ai_service() -> AIService:
    """Get or create the AI service singleton."""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService()
    return _ai_service

