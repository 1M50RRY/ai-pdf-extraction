"""
AI service package for document schema detection and data extraction.

This package provides modular AI functionality split into:
- discovery: Schema discovery from document images
- extraction: Data extraction with confidence scoring
- repair: LLM-based data repair and calculation
- validation: Data validation and normalization

The AIService class provides a backward-compatible interface that delegates
to these modules.
"""

import logging
import os
from typing import Any

from PIL import Image

# Handle both package imports and standalone imports
try:
    from ...models import ExtractionResult, SchemaDefinition
except ImportError:
    from models import ExtractionResult, SchemaDefinition

from .discovery import discover_schema as _discover_schema
from .exceptions import AIServiceError
from .extraction import (
    _build_extraction_prompt as _build_extraction_prompt_func,
    calculate_confidence_from_logprobs,
    extract_data as _extract_data,
)
from .repair import repair_data_with_llm as _repair_data_with_llm
from .validation import (
    ValidationResult,
    _evaluate_validation_rule,
    _extract_field_names_from_rule,
    _parse_validation_rule,
    parse_currency,
    parse_date,
    validate_extracted_data,
)

logger = logging.getLogger(__name__)

# Export public functions and classes
__all__ = [
    "AIService",
    "AIServiceError",
    "ValidationResult",
    "discover_schema",
    "extract_data",
    "repair_data_with_llm",
    "validate_extracted_data",
    "parse_currency",
    "parse_date",
    "calculate_confidence_from_logprobs",
    "get_ai_service",
    # Private functions exported for tests
    "_evaluate_validation_rule",
    "_parse_validation_rule",
    "_extract_field_names_from_rule",
]


# =============================================================================
# AIService Class (Backward Compatibility Wrapper)
# =============================================================================


class AIService:
    """
    Service for AI-powered document analysis and extraction.

    Uses OpenAI's GPT-4.1 model with vision capabilities for:
    - Analyzing document structure and suggesting extraction schemas
    - Extracting data according to confirmed schemas with confidence scoring

    This class now delegates to modular functions for maintainability.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1",
        use_mock: bool = False,
    ):
        """
        Initialize the AI service.

        Args:
            api_key: OpenAI API key. If None, reads from config/environment.
            model: OpenAI model to use (must support vision).
            use_mock: If True, return mock data instead of calling OpenAI.
        """
        # Load API key from settings if not provided
        if api_key is None:
            try:
                from ...config import get_settings

                settings = get_settings()
                api_key = settings.openai_api_key
            except ImportError:
                # Fallback to direct env var if config not available
                api_key = os.getenv("OPENAI_API_KEY")

        self.api_key = api_key
        self.model = model
        self.use_mock = use_mock or not self.api_key
        self._client = None

        if self.use_mock:
            logger.warning(
                "AI Service running in MOCK MODE. Set OPENAI_API_KEY in .env for real extraction."
            )

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

    async def discover_schema(
        self, images: list[Image.Image] | Image.Image
    ) -> SchemaDefinition:
        """
        Analyze document images and discover an extraction schema.

        Delegates to the discovery module.

        Args:
            images: Single PIL Image or list of PIL Images (representative pages).

        Returns:
            SchemaDefinition based on AI analysis.
        """
        return await _discover_schema(
            images,
            client=self.client,
            model=self.model,
            use_mock=self.use_mock,
            get_mock_schema=self._get_mock_schema if self.use_mock else None,
        )

    async def extract_data(
        self,
        images: list[Image.Image] | Image.Image,
        schema: SchemaDefinition,
        source_file: str,
    ) -> ExtractionResult:
        """
        Extract data from document images according to a schema.

        Delegates to the extraction module.

        Args:
            images: Single PIL Image or list of PIL Images to extract from.
            schema: The schema defining what fields to extract.
            source_file: Original filename for the result.

        Returns:
            ExtractionResult with extracted data and confidence score.
        """
        return await _extract_data(
            images,
            schema,
            source_file,
            client=self.client,
            model=self.model,
            use_mock=self.use_mock,
            get_mock_extraction=self._get_mock_extraction if self.use_mock else None,
        )

    async def repair_data_with_llm(
        self,
        extracted_data: dict[str, Any],
        schema: SchemaDefinition | None = None,
    ) -> dict[str, Any]:
        """
        Dynamic Calculation Engine: Use LLM to complete and calculate missing values.

        Delegates to the repair module.

        Args:
            extracted_data: The full extracted data dictionary to repair/complete.
            schema: The full schema definition with field descriptions.

        Returns:
            Fully populated and calculated data dictionary.
        """
        return await _repair_data_with_llm(
            extracted_data,
            client=self.client,
            model=self.model,
            schema=schema,
            use_mock=self.use_mock,
        )

    async def suggest_schema(
        self, images: list[Image.Image] | Image.Image
    ) -> SchemaDefinition:
        """
        Alias for discover_schema for backwards compatibility.

        Args:
            images: Single PIL Image or list of PIL Images (representative pages).

        Returns:
            SchemaDefinition based on AI analysis.
        """
        return await self.discover_schema(images)

    def _build_extraction_prompt(self, schema: SchemaDefinition) -> str:
        """
        Build a dynamic extraction prompt from the schema.

        Delegates to the extraction module.

        Args:
            schema: The schema to build the prompt for.

        Returns:
            The extraction prompt string.
        """
        return _build_extraction_prompt_func(schema)

    def _get_mock_schema(self) -> SchemaDefinition:
        """Return a mock invoice schema for development."""
        try:
            from ...models import FieldDefinition, FieldType
        except ImportError:
            from models import FieldDefinition, FieldType

        return SchemaDefinition(
            name="Invoice Schema",
            description="Mock schema for development - detected as standard invoice with billing information",
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
                    name="vendor_address",
                    type=FieldType.ADDRESS,
                    description="Address of the vendor",
                    required=False,
                ),
                FieldDefinition(
                    name="customer_name",
                    type=FieldType.STRING,
                    description="Name of the customer being billed",
                    required=True,
                ),
                FieldDefinition(
                    name="subtotal",
                    type=FieldType.CURRENCY,
                    description="Subtotal before tax and discounts",
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
                FieldDefinition(
                    name="payment_terms",
                    type=FieldType.STRING,
                    description="Payment terms (e.g., Net 30)",
                    required=False,
                ),
            ],
            validation_rules=[
                "total_amount == subtotal + tax_amount",
            ],
        )

    def _get_mock_extraction(
        self, schema: SchemaDefinition, source_file: str
    ) -> ExtractionResult:
        """Return mock extraction data for development."""
        try:
            from ...models import FieldType
        except ImportError:
            from models import FieldType

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

        warnings.append(
            "DEVELOPMENT MODE: Using mock data. Set OPENAI_API_KEY for real extraction."
        )

        return ExtractionResult(
            source_file=source_file,
            detected_schema=schema,
            extracted_data=mock_data,
            confidence=0.85,
            warnings=warnings,
        )


# =============================================================================
# Singleton Factory
# =============================================================================

_ai_service: AIService | None = None


def get_ai_service() -> AIService:
    """Get or create the AI service singleton."""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService()
    return _ai_service


# =============================================================================
# Module-level function exports (for direct use without AIService class)
# =============================================================================

# Re-export module functions for convenience
discover_schema = _discover_schema
extract_data = _extract_data
repair_data_with_llm = _repair_data_with_llm

