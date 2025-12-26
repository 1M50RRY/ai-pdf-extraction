"""
Schema discovery functionality for analyzing documents and suggesting extraction schemas.

Uses Chain-of-Thought prompting to classify document types and identify extraction fields.
"""

import base64
import io
import logging
import re
from typing import Any, Callable

from PIL import Image
from pydantic import BaseModel, Field

# Handle both package imports and standalone imports
try:
    from ...models import FieldDefinition, FieldType, SchemaDefinition
except ImportError:
    from models import FieldDefinition, FieldType, SchemaDefinition

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


# =============================================================================
# Discovery System Prompt
# =============================================================================

DISCOVERY_SYSTEM_PROMPT = """You are a Senior Data Architect specializing in document digitization.
Your goal is to analyze documents and design optimal database schemas for data extraction.

## Important: Representative Sample Analysis
You are analyzing a **representative sample** of pages from a document (Start, Middle, End).
1. Look for Schema Definitions across ALL these pages.
2. If you see a Data Table on a middle page, define a corresponding `Array` field.
3. Do not assume the schema is limited to the cover page.
4. Consider that important data structures (tables, lists) may appear in the middle sections.

## Your Analysis Process (Chain-of-Thought):

1. **Visual Layout Analysis**: Examine the document's structure, headers, sections, tables, and formatting across ALL provided pages.
2. **Document Classification**: Identify the document type based on visual and textual indicators from all pages.
3. **Key Data Point Identification**: Identify ALL business-relevant data points found across the document. Do not limit the count.
4. **Numerical Relationship Detection**: Identify mathematical relationships between numeric fields.

## Field Naming Rules:
- Use semantic, descriptive names in snake_case (e.g., invoice_number, account_holder_name)
- NEVER use generic names like field_1, data_2, value_a
- Names should clearly indicate what the data represents

## Field Type Selection:
- string: Text data (names, IDs, descriptions, titles)
- array: Tables or lists of items (e.g., invoice line items, transaction lists, product catalogs)
- currency: Money amounts (prices, totals, fees)
- date: Dates in any format (will be normalized to YYYY-MM-DD)
- number: Numeric values (quantities, counts, percentages as numbers)
- boolean: Yes/No, True/False, checkboxes
- email: Email addresses
- phone: Phone numbers
- address: Physical/mailing addresses
- percentage: Percentage values (e.g., "15%", "0.15")

## CRITICAL: Table/List Detection
If you detect a table or list of items (e.g., invoice line items, transaction history, product list), you MUST:
1. Define it as a SINGLE field with `type: array`
2. Name this field semantically (e.g., `line_items`, `transactions`, `products`)
3. In the description, explicitly list the columns/sub-fields to extract (e.g., "List of invoice line items containing: description, quantity, unit_price, total")
4. Do NOT create separate fields for each column - the entire table is ONE array field

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

### CRITICAL CONSTRAINT:
- ONLY generate validation rules for fields at the ROOT level of the document (e.g., `total`, `subtotal`, `tax`)
- Do NOT generate validation rules for fields INSIDE array fields (e.g., do NOT create rules for `line_items[0].unit_price`)
- Validation rules should only reference document-level summary fields, not nested array item fields

### Do NOT use:
- Excel syntax (like "SUM(A1:A5)" or "=A1+B1")
- SQL syntax
- Any functions not listed above
- References to nested array fields (e.g., "line_items[0].price")

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


# =============================================================================
# Helper Functions
# =============================================================================


def _image_to_base64(image: Image.Image) -> str:
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


def _convert_discovery_to_schema(discovery: DiscoveryResponse) -> SchemaDefinition:
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


# =============================================================================
# Main Discovery Function
# =============================================================================


async def discover_schema(
    images: list[Image.Image] | Image.Image,
    client: Any,  # OpenAI client
    model: str = "gpt-4.1",
    use_mock: bool = False,
    get_mock_schema: Callable[[], SchemaDefinition] | None = None,
) -> SchemaDefinition:
    """
    Analyze document images and discover an extraction schema.

    Uses Chain-of-Thought prompting to:
    1. Classify the document type
    2. Identify key data points across all pages
    3. Generate a semantic schema

    Args:
        images: Single PIL Image or list of PIL Images (representative pages).
        client: OpenAI client instance.
        model: Model name to use.
        use_mock: If True, return mock schema instead of calling OpenAI.
        get_mock_schema: Function to get mock schema (for testing).

    Returns:
        SchemaDefinition based on AI analysis.
    """
    if use_mock and get_mock_schema:
        logger.info("Discovering schema (MOCK MODE)")
        return get_mock_schema()

    # Normalize to list
    if isinstance(images, Image.Image):
        images = [images]

    logger.info("Discovering schema using GPT-4.1 structured outputs (%d pages)", len(images))

    # Convert all images to base64
    base64_images = [_image_to_base64(img) for img in images]

    try:
        # Build content with multiple images
        content = [
            {
                "type": "text",
                "text": (
                    f"Analyze these {len(images)} representative pages from a document carefully.\n\n"
                    "1. Examine the visual layout and text across ALL pages to classify the document type.\n"
                    "2. Identify ALL business-relevant data points that should be extracted across all pages. "
                    "Do not limit the count.\n"
                    "3. **CRITICAL: If you detect a table or list of items on ANY page (e.g., invoice line items, transaction history), "
                    "define it as a SINGLE field with type 'array'. Name it semantically (e.g., 'line_items'). "
                    "In the description, explicitly list the columns to extract (e.g., 'List of items containing: description, quantity, unit_price, total').**\n"
                    "4. Choose semantic field names in snake_case.\n"
                    "5. If you detect numerical relationships at the document summary level (like totals summing up), "
                    "output validation_rules using the exact field names you chose. "
                    "Do NOT create rules for fields inside array fields.\n\n"
                    "Remember: Important data structures may appear in the middle pages, not just the first page.\n\n"
                    "Provide your analysis with reasoning."
                ),
            },
        ]

        # Add all images
        for i, base64_img in enumerate(base64_images):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_img}",
                    "detail": "high",
                },
            })

        # Use beta.chat.completions.parse for structured outputs
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": content,
                },
            ],
            response_format=DiscoveryResponse,
            # No max_tokens limit - let the model use its full capacity
        )

        # Extract the parsed response
        discovery = response.choices[0].message.parsed

        if discovery is None:
            logger.error("OpenAI returned no parsed response")
            from .exceptions import AIServiceError
            raise AIServiceError("Failed to parse schema discovery response")

        logger.info(
            "Discovered document type: %s (reasoning: %s...)",
            discovery.document_type,
            discovery.reasoning[:100],
        )

        # Convert to SchemaDefinition
        return _convert_discovery_to_schema(discovery)

    except Exception as e:
        logger.exception("Schema discovery failed")
        from .exceptions import AIServiceError
        raise AIServiceError(f"Schema discovery failed: {e}") from e

