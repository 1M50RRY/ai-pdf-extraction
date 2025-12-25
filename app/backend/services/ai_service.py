"""
AI service for document schema detection and data extraction.

Uses OpenAI GPT-4o with structured outputs via Pydantic models.
Implements high-sophistication prompting with Chain-of-Thought reasoning
and logprobs-based confidence scoring.
"""

import base64
import io
import json
import logging
import math
import os
import re
from datetime import datetime
from typing import Any

from PIL import Image
from price_parser import Price
from pydantic import BaseModel, Field
from simpleeval import NameNotDefined, SimpleEval

# Handle both package imports and standalone imports
try:
    from ..models import (
        ExtractionResult,
        FieldDefinition,
        FieldType,
        SchemaDefinition,
    )
except ImportError:
    from models import (
        ExtractionResult,
        FieldDefinition,
        FieldType,
        SchemaDefinition,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# AI Response Models (for Structured Outputs)
# =============================================================================


class DiscoveredField(BaseModel):
    """A field discovered during schema analysis."""

    name: str = Field(
        ...,
        description="Semantic field name in snake_case (e.g., invoice_number, total_amount)",
    )
    type: str = Field(
        ...,
        description=(
            "Field type: string, currency, date, number, boolean, email, phone, "
            "address, percentage. Use 'string' for mixed text/numbers or any non-standard data."
        ),
    )
    description: str = Field(
        ...,
        description="Clear description of what this field represents and where to find it",
    )
    required: bool = Field(
        default=True,
        description="Whether this field is critical/required for the document type",
    )


class DiscoveryResponse(BaseModel):
    """
    Response model for schema discovery.

    Uses Chain-of-Thought reasoning to classify documents
    and identify extraction fields.
    """

    document_type: str = Field(
        ...,
        description="Classified document type (e.g., 'Invoice', 'Balance Sheet', 'Resume', 'Medical Record', 'Tax Form')",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this document type was identified and the key visual/textual indicators",
    )
    schema_definition: list[DiscoveredField] = Field(
        ...,
        min_length=1,
        description="List of ALL business-relevant data points found in the document",
    )
    validation_rules: list[str] = Field(
        default_factory=list,
        description=(
            "Mathematical validation rules expressing numerical relationships between fields. "
            "Format: 'result_field == operand1 + operand2' or 'result_field == operand1 - operand2'. "
            "Examples: 'total_amount == subtotal + vat', 'net_pay == gross_salary - tax_deduction'. "
            "Only include rules for clear mathematical relationships visible in the document."
        ),
    )


class ExtractionResponse(BaseModel):
    """Response model for data extraction."""

    extracted_fields: dict[str, Any] = Field(
        ...,
        description="Dictionary of field_name -> extracted_value",
    )
    extraction_notes: str = Field(
        default="",
        description="Any notes about the extraction process or ambiguities encountered",
    )


# =============================================================================
# Validation Utilities
# =============================================================================


class ValidationResult:
    """Result of data validation."""

    def __init__(self):
        self.warnings: list[str] = []
        self.validated_data: dict[str, Any] = {}


def parse_currency(value: Any) -> float | None:
    """
    Parse a currency string to float using price-parser.

    Handles all international formats automatically:
    - "$1,234.56", "€1.234,56", "1000 USD", "£500.00"
    - "1.000,00 €" (European), "¥1,234" (Japanese)

    Uses the robust price-parser library for production-grade parsing.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    try:
        price = Price.fromstring(value)
        if price.amount_float is not None:
            return price.amount_float

        # Fallback: try parsing as plain number if price-parser fails
        # This handles cases like "1234.56" without currency symbol
        cleaned = re.sub(r"[^\d.,\-]", "", value)
        if cleaned:
            # Handle European format
            if "," in cleaned and "." in cleaned:
                if cleaned.rfind(",") > cleaned.rfind("."):
                    cleaned = cleaned.replace(".", "").replace(",", ".")
                else:
                    cleaned = cleaned.replace(",", "")
            elif "," in cleaned:
                parts = cleaned.split(",")
                if len(parts) == 2 and len(parts[1]) == 2:
                    cleaned = cleaned.replace(",", ".")
                else:
                    cleaned = cleaned.replace(",", "")
            return float(cleaned)
        return None

    except (ValueError, AttributeError):
        return None


def parse_date(value: Any) -> str | None:
    """
    Parse various date formats to YYYY-MM-DD.

    Returns None if parsing fails.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    # Common date formats to try
    formats = [
        "%Y-%m-%d",  # 2024-01-15
        "%m/%d/%Y",  # 01/15/2024
        "%d/%m/%Y",  # 15/01/2024
        "%m-%d-%Y",  # 01-15-2024
        "%d-%m-%Y",  # 15-01-2024
        "%B %d, %Y",  # January 15, 2024
        "%b %d, %Y",  # Jan 15, 2024
        "%d %B %Y",  # 15 January 2024
        "%d %b %Y",  # 15 Jan 2024
        "%Y/%m/%d",  # 2024/01/15
        "%m.%d.%Y",  # 01.15.2024
        "%d.%m.%Y",  # 15.01.2024
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def validate_extracted_data(
    data: dict[str, Any],
    schema: SchemaDefinition,
) -> ValidationResult:
    """
    Validate extracted data against the schema.

    Performs:
    - Type checking (dates, currencies, etc.)
    - Null checks for required fields
    - Math checks for financial totals

    Args:
        data: The extracted data dictionary.
        schema: The schema definition.

    Returns:
        ValidationResult with warnings and validated data.
    """
    result = ValidationResult()
    result.validated_data = data.copy()

    # Build field lookup from schema
    schema_field_names = [f.name for f in schema.fields]
    data_field_names = list(data.keys())
    field_map = {f.name: f for f in schema.fields}

    # Debug logging for schema/data mismatch diagnosis
    logger.info(
        "Validating data against schema '%s'. Schema fields: %s. Data fields: %s",
        schema.name,
        schema_field_names,
        data_field_names,
    )

    # Check for fields in data not in schema (informational only, no warnings)
    extra_fields = set(data_field_names) - set(schema_field_names)
    if extra_fields:
        logger.debug("Data contains extra fields not in schema: %s (trusting data keys)", extra_fields)

    # Check for fields in schema not in data (informational only, no warnings)
    # We trust the data keys - this may happen due to renaming
    missing_fields = set(schema_field_names) - set(data_field_names)
    if missing_fields:
        logger.debug("Schema fields not in data: %s (trusting data keys)", missing_fields)

    # Track currency fields for math checks
    currency_values: dict[str, float] = {}

    for field_name, field_def in field_map.items():
        # Skip if field is not in data (trust data keys, no warning for missing)
        if field_name not in data:
            continue
            
        value = data.get(field_name)

        # Relaxed null/empty check - ONLY warn for explicitly None or empty string ""
        # Do NOT warn for whitespace, empty lists, or missing keys (trust data)
        is_explicitly_empty = value is None or value == ""

        if is_explicitly_empty:
            if field_def.required:
                result.warnings.append(
                    f"Required field '{field_name}' has empty value"
                )
            result.validated_data[field_name] = None
            continue

        # Type-specific validation
        if field_def.type == FieldType.DATE:
            parsed = parse_date(value)
            if parsed is None:
                result.warnings.append(
                    f"Field '{field_name}' has invalid date format: '{value}'. Expected YYYY-MM-DD."
                )
            else:
                result.validated_data[field_name] = parsed

        elif field_def.type == FieldType.CURRENCY:
            parsed = parse_currency(value)
            if parsed is None:
                result.warnings.append(
                    f"Field '{field_name}' has invalid currency format: '{value}'"
                )
            else:
                currency_values[field_name] = parsed
                # Keep original string but note the parsed value
                result.validated_data[field_name] = value

        elif field_def.type == FieldType.NUMBER:
            try:
                if isinstance(value, str):
                    # Remove commas and parse
                    cleaned = value.replace(",", "").strip()
                    parsed = float(cleaned) if "." in cleaned else int(cleaned)
                    result.validated_data[field_name] = parsed
                elif not isinstance(value, (int, float)):
                    result.warnings.append(
                        f"Field '{field_name}' expected number, got: '{value}'"
                    )
            except ValueError:
                result.warnings.append(
                    f"Field '{field_name}' has invalid number format: '{value}'"
                )

        elif field_def.type == FieldType.BOOLEAN:
            if isinstance(value, bool):
                pass  # Already correct
            elif isinstance(value, str):
                lower = value.lower().strip()
                if lower in ("true", "yes", "y", "1", "on"):
                    result.validated_data[field_name] = True
                elif lower in ("false", "no", "n", "0", "off"):
                    result.validated_data[field_name] = False
                else:
                    result.warnings.append(
                        f"Field '{field_name}' has ambiguous boolean value: '{value}'"
                    )

        elif field_def.type == FieldType.EMAIL:
            if isinstance(value, str) and "@" not in value:
                result.warnings.append(
                    f"Field '{field_name}' appears to be invalid email: '{value}'"
                )

        elif field_def.type == FieldType.PERCENTAGE:
            # Normalize percentage strings
            if isinstance(value, str):
                cleaned = value.replace("%", "").strip()
                try:
                    float(cleaned)  # Validate it's a number
                except ValueError:
                    result.warnings.append(
                        f"Field '{field_name}' has invalid percentage format: '{value}'"
                    )

    # Math checks using dynamic validation rules from schema
    _perform_math_checks(currency_values, schema.validation_rules, result)

    return result


def _evaluate_validation_rule_simpleeval(
    rule: str,
    numeric_values: dict[str, float],
) -> tuple[bool, str, str | None]:
    """
    Evaluate a validation rule using simpleeval for safe expression parsing.

    Supports any mathematical expression the AI discovers, including:
    - Simple: "total == subtotal + tax"
    - Complex: "grand_total == (subtotal + shipping) * (1 + tax_rate)"
    - Division: "margin == (revenue - cost) / revenue"

    Uses defense-in-depth: functions={} to prevent any function calls.

    Args:
        rule: The validation rule string (e.g., "total == a + b").
        numeric_values: Dictionary of field_name -> numeric value.

    Returns:
        Tuple of (success, message, failed_rule or None).
        Success is True if rule passes or cannot be evaluated.
    """
    rule = rule.strip()
    if "==" not in rule:
        return (True, f"Invalid rule format (no ==): {rule}", None)

    parts = rule.split("==", 1)
    if len(parts) != 2:
        return (True, f"Invalid rule format: {rule}", None)

    left_side = parts[0].strip()
    right_side = parts[1].strip()

    # Create a safe evaluator with:
    # - numeric values as context
    # - Explicit allowlist of safe math functions (defense-in-depth)
    evaluator = SimpleEval()
    evaluator.names = numeric_values

    # Safe functions allowlist - only mathematical operations
    # This prevents access to dangerous builtins like os.system, eval, exec, etc.
    evaluator.functions = {
        "sum": sum,
        "round": round,
        "abs": abs,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "pow": pow,
        "len": len,
    }

    # Try to evaluate both sides
    try:
        left_value = evaluator.eval(left_side)
    except NameNotDefined as e:
        # Field not in extracted data, skip validation
        return (True, f"Field not found for rule '{rule}': {e}", None)
    except Exception as e:
        return (True, f"Could not evaluate left side of '{rule}': {e}", None)

    try:
        right_value = evaluator.eval(right_side)
    except NameNotDefined as e:
        # Field not in extracted data, skip validation
        return (True, f"Field not found for rule '{rule}': {e}", None)
    except Exception as e:
        return (True, f"Could not evaluate right side of '{rule}': {e}", None)

    # Allow 1% tolerance for rounding errors
    if left_value == 0 and right_value == 0:
        return (True, f"Rule passed: {rule}", None)

    tolerance = max(abs(left_value) * 0.01, abs(right_value) * 0.01, 0.02)
    if abs(left_value - right_value) <= tolerance:
        return (True, f"Rule passed: {rule}", None)
    else:
        return (
            False,
            f"Math validation failed: {rule} "
            f"(left={left_value:.2f}, right={right_value:.2f}, diff={abs(left_value - right_value):.2f})",
            rule,
        )


# Legacy function signatures for backwards compatibility
def _parse_validation_rule(rule: str) -> tuple[str, list[tuple[str, str]]] | None:
    """Legacy parser - kept for backwards compatibility with tests."""
    rule = rule.strip()
    if "==" not in rule:
        return None

    parts = rule.split("==")
    if len(parts) != 2:
        return None

    result_field = parts[0].strip()
    expression = parts[1].strip()

    operands: list[tuple[str, str]] = []
    tokens = re.split(r"(\s*[+\-*/]\s*)", expression)
    current_op = "+"

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token in ("+", "-", "*", "/"):
            current_op = token
        elif re.match(r"^[a-z_][a-z0-9_]*$", token, re.IGNORECASE):
            operands.append((current_op, token))

    if not operands:
        return None

    return (result_field, operands)


def _evaluate_validation_rule(
    rule: str,
    numeric_values: dict[str, float],
) -> tuple[bool, str, float | None, float | None]:
    """Legacy evaluator - delegates to simpleeval-based implementation."""
    success, message, failed_rule = _evaluate_validation_rule_simpleeval(
        rule, numeric_values
    )
    # Return format compatible with old signature
    return (success, message, None, None)


def _perform_math_checks(
    numeric_values: dict[str, float],
    validation_rules: list[str],
    result: ValidationResult,
) -> None:
    """
    Perform math checks using simpleeval for dynamic expression evaluation.

    This is 100% dynamic - supports any mathematical expression including:
    - Parentheses: "(a + b) * c"
    - All operators: +, -, *, /
    - Complex formulas: "(revenue - cost) / revenue"

    Uses simpleeval for safe, sandboxed expression evaluation.

    Args:
        numeric_values: Dictionary of field_name -> parsed numeric value.
        validation_rules: List of rule strings from the schema.
        result: ValidationResult to append warnings to.
    """
    if not validation_rules:
        return

    for rule in validation_rules:
        success, message, failed_rule = _evaluate_validation_rule_simpleeval(
            rule, numeric_values
        )
        if not success:
            result.warnings.append(message)
            logger.warning("Validation rule failed: %s", message)


def calculate_confidence_from_logprobs(
    logprobs_data: list[Any] | None,
) -> float:
    """
    Calculate confidence score from logprobs using geometric mean.

    The geometric mean of probabilities gives a balanced measure of
    overall extraction confidence.

    Args:
        logprobs_data: List of token logprob objects from OpenAI.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    if not logprobs_data:
        return 0.75  # Default confidence if no logprobs

    # Extract logprob values
    log_probs = []
    for token_data in logprobs_data:
        if hasattr(token_data, "logprob") and token_data.logprob is not None:
            log_probs.append(token_data.logprob)

    if not log_probs:
        return 0.75

    # Calculate geometric mean via arithmetic mean of logs
    # Geometric mean = exp(mean(log(p))) = exp(mean(logprob))
    avg_logprob = sum(log_probs) / len(log_probs)

    # Convert from log probability to probability
    # Clamp to avoid overflow/underflow
    avg_logprob = max(avg_logprob, -10)  # Minimum ~0.00005 probability

    confidence = math.exp(avg_logprob)

    # Clamp to valid range
    return max(0.0, min(1.0, confidence))


# =============================================================================
# AI Service Error
# =============================================================================


class AIServiceError(Exception):
    """Raised when AI service operations fail."""

    pass


# =============================================================================
# Main AI Service
# =============================================================================


class AIService:
    """
    Service for AI-powered document analysis and extraction.

    Uses OpenAI's GPT-4o model with vision capabilities for:
    - Analyzing document structure and suggesting extraction schemas
    - Extracting data according to confirmed schemas with confidence scoring
    """

    # System prompts
    DISCOVERY_SYSTEM_PROMPT = """You are a Senior Data Architect specializing in document digitization.
Your goal is to analyze documents and design optimal database schemas for data extraction.

## Your Analysis Process (Chain-of-Thought):

1. **Visual Layout Analysis**: Examine the document's structure, headers, sections, tables, and formatting.
2. **Document Classification**: Identify the document type based on visual and textual indicators.
3. **Key Data Point Identification**: Identify ALL business-relevant data points found in the document. Do not limit the count.
4. **Numerical Relationship Detection**: Identify mathematical relationships between numeric fields.

## Field Naming Rules:
- Use semantic, descriptive names in snake_case (e.g., invoice_number, account_holder_name)
- NEVER use generic names like field_1, data_2, value_a
- Names should clearly indicate what the data represents

## Field Type Selection:
- string: Text data (names, IDs, descriptions, titles)
- currency: Money amounts (prices, totals, fees)
- date: Dates in any format (will be normalized to YYYY-MM-DD)
- number: Numeric values (quantities, counts, percentages as numbers)
- boolean: Yes/No, True/False, checkboxes
- email: Email addresses
- phone: Phone numbers
- address: Physical/mailing addresses
- percentage: Percentage values (e.g., "15%", "0.15")

IMPORTANT: For mixed text/numbers or any data that doesn't fit the above types, strictly use "string".
Do NOT invent new types. "string" is the catch-all for any text or mixed data.

## Validation Rules (IMPORTANT):
When you detect numerical relationships in the document (like totals that sum up components), 
output them as validation_rules using the EXACT field names you chose.

### VALIDATION RULES SYNTAX:
- Use standard Python expression syntax.
- Use == for equality comparison.
- Operators: +, -, *, /, parentheses ()
- Available Functions: sum(), round(x, 2), abs(), min(), max(), sqrt(), log(), len()

### Examples:
- Simple: "total == subtotal + tax"
- Complex: "grand_total == (subtotal + shipping) * (1 + tax_rate)"
- Division: "margin == (revenue - cost) / revenue"
- With functions: "total == round(subtotal * tax_rate, 2)"
- Absolute: "variance == abs(budget - actual)"

### Do NOT use:
- Excel syntax (like "SUM(A1:A5)" or "=A1+B1")
- SQL syntax
- Any functions not listed above

### Real-world examples:
- If you see "Subtotal: $100, VAT: $20, Total: $120", output: "total == subtotal + vat"
- If you see "Gross Pay: $5000, Deductions: $1000, Net Pay: $4000", output: "net_pay == gross_pay - deductions"
- If you see "Credits: $500, Debits: $300, Balance: $200", output: "balance == credits - debits"

Use the EXACT field names from your schema_definition. Only include rules for relationships 
that are clearly visible in the document. If no mathematical relationships exist, return an empty list.

## Critical Rules:
- Do NOT assume the document is an invoice
- Analyze what you SEE, not what you expect
- If it's a form, extract the form field labels and values
- If it's a report, extract the key metrics and findings
- If it's a resume, extract professional information
- Extract ALL business-relevant fields, not just a subset
- Prioritize fields that would be most valuable in a database"""

    EXTRACTION_SYSTEM_PROMPT = """You are a precise Data Entry Clerk with exceptional attention to detail.
Your task is to extract specific data fields from a document image AND estimate your confidence for each field.

## Extraction Rules:

1. **Strict Adherence**: Only extract the fields specified. Do not add extra fields.
2. **Accuracy Over Guessing**: If a value is unclear or not present, return null.
3. **Preserve Original Format**: Keep dates, currencies, and numbers as they appear in the document.
4. **No Assumptions**: Do not infer or calculate values unless explicitly stated in the document.

## Confidence Scoring:
For EVERY field you extract, provide a confidence score (0.0 to 1.0):
- 1.0: Perfectly clear, no ambiguity
- 0.8-0.99: Very confident, minor formatting uncertainty
- 0.5-0.79: Somewhat confident, value partially visible or slightly unclear
- 0.1-0.49: Low confidence, significant uncertainty or guessing
- 0.0: Field not found, returning null

## Important Guidelines:
- For currency fields: Include the currency symbol if visible (e.g., "$1,234.56")
- For dates: Transcribe as shown, the system will normalize
- For empty/missing fields: Return null with confidence 0.0
- For ambiguous values: Choose the most likely interpretation and reflect uncertainty in confidence score

Return data in the EXACT JSON format specified in the user prompt."""

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
                from ..config import get_settings
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

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for API."""
        buffer = io.BytesIO()
        # Resize if too large (max 2048px on longest side for efficiency)
        max_size = 2048
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        image.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # =========================================================================
    # Task 1: Smart Schema Discovery
    # =========================================================================

    async def discover_schema(self, image: Image.Image) -> SchemaDefinition:
        """
        Analyze a document image and discover an extraction schema.

        Uses Chain-of-Thought prompting to:
        1. Classify the document type
        2. Identify key data points
        3. Generate a semantic schema

        Args:
            image: PIL Image of the document page.

        Returns:
            SchemaDefinition based on AI analysis.
        """
        if self.use_mock:
            logger.info("Discovering schema (MOCK MODE)")
            return self._get_mock_schema()

        logger.info("Discovering schema using GPT-4o structured outputs")
        base64_image = self._image_to_base64(image)

        try:
            # Use beta.chat.completions.parse for structured outputs
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.DISCOVERY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this document image carefully.\n\n"
                                    "1. Examine the visual layout and text to classify the document type.\n"
                                    "2. Identify ALL business-relevant data points that should be extracted. "
                                    "Do not limit the count.\n"
                                    "3. Choose semantic field names in snake_case.\n"
                                    "4. If you detect numerical relationships (like totals summing up), "
                                    "output validation_rules using the exact field names you chose.\n\n"
                                    "Provide your analysis with reasoning."
                                ),
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
                response_format=DiscoveryResponse,
                # No max_tokens limit - let the model use its full capacity
            )

            # Extract the parsed response
            discovery = response.choices[0].message.parsed

            if discovery is None:
                logger.error("OpenAI returned no parsed response")
                raise AIServiceError("Failed to parse schema discovery response")

            logger.info(
                "Discovered document type: %s (reasoning: %s...)",
                discovery.document_type,
                discovery.reasoning[:100],
            )

            # Convert to SchemaDefinition
            return self._convert_discovery_to_schema(discovery)

        except Exception as e:
            logger.exception("Schema discovery failed")
            raise AIServiceError(f"Schema discovery failed: {e}") from e

    def _convert_discovery_to_schema(
        self, discovery: DiscoveryResponse
    ) -> SchemaDefinition:
        """Convert DiscoveryResponse to SchemaDefinition."""
        fields = []
        for discovered_field in discovery.schema_definition:
            # Map string type to FieldType enum
            try:
                field_type = FieldType(discovered_field.type.lower())
            except ValueError:
                logger.warning(
                    "Unknown field type '%s', defaulting to string",
                    discovered_field.type,
                )
                field_type = FieldType.STRING

            fields.append(
                FieldDefinition(
                    name=discovered_field.name,
                    type=field_type,
                    description=discovered_field.description,
                    required=discovered_field.required,
                )
            )

        # Get field names for validating rules
        field_names = {f.name for f in fields}

        # Filter validation rules to only include those with valid field references
        valid_rules = []
        for rule in discovery.validation_rules:
            # Extract field names from rule
            rule_fields = re.findall(r"[a-z_][a-z0-9_]*", rule, re.IGNORECASE)
            # Check if all referenced fields exist in schema
            if all(f in field_names for f in rule_fields if f not in ("==",)):
                valid_rules.append(rule)
            else:
                logger.warning(
                    "Skipping validation rule with unknown fields: %s", rule
                )

        if valid_rules:
            logger.info("Discovered %d validation rules: %s", len(valid_rules), valid_rules)

        return SchemaDefinition(
            name=f"{discovery.document_type} Schema",
            description=discovery.reasoning,
            fields=fields,
            version="1.0",
            validation_rules=valid_rules,
        )

    # =========================================================================
    # Task 2: Strict Data Extraction
    # =========================================================================

    async def extract_data(
        self,
        image: Image.Image,
        schema: SchemaDefinition,
        source_file: str,
    ) -> ExtractionResult:
        """
        Extract data from a document image according to a schema.

        Uses:
        - Dynamic prompt construction based on schema
        - Logprobs for confidence scoring (geometric mean)
        - Post-processing validation

        Args:
            image: PIL Image of the document page.
            schema: The schema defining what fields to extract.
            source_file: Original filename for the result.

        Returns:
            ExtractionResult with extracted data and confidence score.
        """
        if self.use_mock:
            logger.info("Extracting data (MOCK MODE) for: %s", source_file)
            return self._get_mock_extraction(schema, source_file)

        # DEBUG: Log exact schema fields being used (no caching - rebuilt each time)
        field_names = [f.name for f in schema.fields]
        print(f"DEBUG AI_SERVICE: Extracting with fields: {field_names}")
        logger.info(
            "DEBUG AI_SERVICE: Extracting data for '%s' using schema '%s' with fields: %s",
            source_file,
            schema.name,
            field_names,
        )
        
        base64_image = self._image_to_base64(image)

        # Build dynamic extraction prompt (rebuilt fresh for each extraction - NO CACHING)
        extraction_prompt = self._build_extraction_prompt(schema)
        
        # DEBUG: Log first 500 chars of prompt to verify field names
        logger.debug("DEBUG AI_SERVICE: Extraction prompt preview: %s...", extraction_prompt[:500])

        try:
            # Use standard chat.completions.create with json_object for logprobs access
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": extraction_prompt},
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
                response_format={"type": "json_object"},
                # No max_tokens limit - let the model use its full capacity
                logprobs=True,
                top_logprobs=1,
            )

            # Parse the JSON response
            content = response.choices[0].message.content
            if not content:
                raise AIServiceError("Empty response from OpenAI")

            try:
                response_data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse extraction response: %s", content[:500])
                raise AIServiceError(f"Invalid JSON in extraction response: {e}") from e

            # Extract data and field confidences from new response format
            # Handle both new format (with extracted_data/field_confidences) and legacy format
            if "extracted_data" in response_data and "field_confidences" in response_data:
                extracted_data = response_data["extracted_data"]
                field_confidences = response_data.get("field_confidences", {})
                logger.info("Using new per-field confidence format")
            else:
                # Legacy format: response is just the data object
                extracted_data = response_data
                field_confidences = {}
                logger.info("Using legacy extraction format (no per-field confidences)")

            # Calculate global confidence from logprobs (fallback/comparison)
            logprobs_data = None
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                logprobs_data = response.choices[0].logprobs.content

            logprobs_confidence = calculate_confidence_from_logprobs(logprobs_data)

            # Use average of field confidences if available, else fall back to logprobs
            if field_confidences:
                valid_confidences = [c for c in field_confidences.values() if isinstance(c, (int, float))]
                if valid_confidences:
                    avg_field_confidence = sum(valid_confidences) / len(valid_confidences)
                    # Blend with logprobs confidence (weight field confidence higher)
                    confidence = 0.7 * avg_field_confidence + 0.3 * logprobs_confidence
                else:
                    confidence = logprobs_confidence
            else:
                confidence = logprobs_confidence

            logger.info(
                "Extraction confidence: %.3f (logprobs: %.3f, field avg: %.3f)",
                confidence,
                logprobs_confidence,
                sum(field_confidences.values()) / len(field_confidences) if field_confidences else 0.0,
            )

            # Validate and post-process
            validation = validate_extracted_data(extracted_data, schema)

            return ExtractionResult(
                source_file=source_file,
                detected_schema=schema,
                extracted_data=validation.validated_data,
                confidence=confidence,
                warnings=validation.warnings,
                field_confidences=field_confidences,
            )

        except AIServiceError:
            raise
        except Exception as e:
            logger.exception("Data extraction failed")
            raise AIServiceError(f"Data extraction failed: {e}") from e

    def _build_extraction_prompt(self, schema: SchemaDefinition) -> str:
        """Build a dynamic extraction prompt from the schema.
        
        NOTE: This is rebuilt fresh for EVERY extraction - no caching.
        The field names come directly from the schema parameter passed in.
        """
        field_names = [f.name for f in schema.fields]
        print(f"DEBUG _build_extraction_prompt: Building prompt for fields: {field_names}")
        
        field_descriptions = []
        for field in schema.fields:
            required_marker = " (REQUIRED)" if field.required else " (optional)"
            field_descriptions.append(
                f"- **{field.name}** ({field.type.value}){required_marker}: {field.description}"
            )

        fields_text = "\n".join(field_descriptions)

        return f"""Extract the following fields from this document image.

## Fields to Extract:
{fields_text}

## Response Format (MUST follow this exact structure):
Return a JSON object with TWO keys:
1. `extracted_data`: Object with the field values
2. `field_confidences`: Object with confidence scores (0.0-1.0) for each field

For any field that cannot be found or is unclear, set its value to null with confidence 0.0.
Do not add any fields that are not in the list above.

Example response structure:
{{
  "extracted_data": {{
    "{field_names[0] if field_names else 'field_name'}": "extracted value or null",
    ...
  }},
  "field_confidences": {{
    "{field_names[0] if field_names else 'field_name'}": 0.95,
    ...
  }}
}}

Field names to extract: {json.dumps(field_names)}"""

    # =========================================================================
    # Legacy method for backwards compatibility
    # =========================================================================

    async def suggest_schema(self, image: Image.Image) -> SchemaDefinition:
        """
        Alias for discover_schema for backwards compatibility.

        Args:
            image: PIL Image of the document page.

        Returns:
            SchemaDefinition based on AI analysis.
        """
        return await self.discover_schema(image)

    # =========================================================================
    # Mock Data for Development
    # =========================================================================

    def _get_mock_schema(self) -> SchemaDefinition:
        """Return a mock invoice schema for development."""
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
