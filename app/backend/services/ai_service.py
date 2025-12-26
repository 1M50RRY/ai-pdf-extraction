"""
AI service for document schema detection and data extraction.

Uses OpenAI GPT-4.1 with structured outputs via Pydantic models.
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

    # Step 1: Create normalized_data map for case-insensitive matching
    # Format: {normalized_key: value}
    normalized_data: dict[str, Any] = {}
    for k, v in data.items():
        norm_key = k.strip().lower()
        # If duplicate normalized keys, keep the first one (or prefer exact match)
        if norm_key not in normalized_data:
            normalized_data[norm_key] = v

    # Step 2: Use set() to deduplicate warnings
    warnings_set: set[str] = set()

    # Step 3: Track currency values for math checks
    currency_values: dict[str, float] = {}

    # Step 4: Iterate through schema fields
    for field in schema.fields:
        # Normalize field name for lookup
        norm_field_name = field.name.strip().lower()
        
        # Check if key exists in normalized_data (trust AI if key is missing)
        if norm_field_name not in normalized_data:
            # Key is totally missing - DO NOT warn (AI likely renamed it or skipped it)
            logger.debug("Field '%s' not found in extracted data (trusting AI)", field.name)
            continue
        
        # Key exists - get the value
        value = normalized_data[norm_field_name]
        
        # Debug logging for each field
        logger.info(
            "Validating field '%s': Found value '%s' (type: %s)",
            field.name,
            value,
            type(value).__name__ if value is not None else "None",
        )
        
        # Step 5: Only warn if value is explicitly None or "" (empty string)
        # Trust the AI: if key is missing, don't warn
        if value is None or value == "":
            if field.required:
                warnings_set.add(f"Required field '{field.name}' has empty value")
            # Set to None in validated_data
            result.validated_data[field.name] = None
            continue

        # Step 6: Type-specific validation
        if field.type == FieldType.ARRAY:
            # Validate array fields - ensure it's a list
            if not isinstance(value, list):
                warnings_set.add(
                    f"Field '{field.name}' expected array/list, got: {type(value).__name__}"
                )
                result.validated_data[field.name] = [] if value is None else [value]
            else:
                # Filter out None/null items from arrays (fix "List Stutter")
                filtered_array = [x for x in value if x is not None]
                result.validated_data[field.name] = filtered_array
                if len(filtered_array) < len(value):
                    logger.debug(
                        "Filtered %d null items from array field '%s' (%d -> %d items)",
                        len(value) - len(filtered_array),
                        field.name,
                        len(value),
                        len(filtered_array),
                    )
                else:
                    logger.debug("Validated array field '%s' with %d items", field.name, len(filtered_array))
            continue

        elif field.type == FieldType.DATE:
            # Try to parse, but don't warn on failure - prefer raw data over no data
            parsed = parse_date(value)
            if parsed is not None:
                result.validated_data[field.name] = parsed
            else:
                # Keep the original value even if parsing failed (relaxed validation)
                result.validated_data[field.name] = value

        elif field.type == FieldType.CURRENCY:
            parsed = parse_currency(value)
            if parsed is None:
                warnings_set.add(
                    f"Field '{field.name}' has invalid currency format: '{value}'"
                )
            else:
                # Use schema field name for currency_values
                currency_values[field.name] = parsed
                # Keep original string but note the parsed value
                result.validated_data[field.name] = value

        elif field.type == FieldType.NUMBER:
            try:
                if isinstance(value, str):
                    # Remove commas and parse
                    cleaned = value.replace(",", "").strip()
                    parsed = float(cleaned) if "." in cleaned else int(cleaned)
                    result.validated_data[field.name] = parsed
                elif not isinstance(value, (int, float)):
                    warnings_set.add(
                        f"Field '{field.name}' expected number, got: '{value}'"
                    )
            except ValueError:
                warnings_set.add(
                    f"Field '{field.name}' has invalid number format: '{value}'"
                )

        elif field.type == FieldType.BOOLEAN:
            if isinstance(value, bool):
                pass  # Already correct
            elif isinstance(value, str):
                lower = value.lower().strip()
                if lower in ("true", "yes", "y", "1", "on"):
                    result.validated_data[field.name] = True
                elif lower in ("false", "no", "n", "0", "off"):
                    result.validated_data[field.name] = False
                else:
                    warnings_set.add(
                        f"Field '{field.name}' has ambiguous boolean value: '{value}'"
                    )

        elif field.type == FieldType.EMAIL:
            if isinstance(value, str) and "@" not in value:
                warnings_set.add(
                    f"Field '{field.name}' appears to be invalid email: '{value}'"
                )

        elif field.type == FieldType.PERCENTAGE:
            # Normalize percentage strings
            if isinstance(value, str):
                cleaned = value.replace("%", "").strip()
                try:
                    float(cleaned)  # Validate it's a number
                except ValueError:
                    warnings_set.add(
                        f"Field '{field.name}' has invalid percentage format: '{value}'"
                    )

    # Step 7: Math checks using dynamic validation rules from schema
    # Pass extracted_data to filter out rules referencing nested array fields
    # Pass warnings_set so math warnings are included in the final warnings list
    _perform_math_checks(currency_values, schema.validation_rules, warnings_set, data)

    # Step 8: Convert warnings set to list (deduplicated)
    result.warnings = list(warnings_set)

    return result


def _clean_null_from_arrays(data: dict[str, Any] | list[Any] | Any) -> dict[str, Any] | list[Any] | Any:
    """
    Recursively clean None/null values from arrays in extracted data.
    
    Fixes "List Stutter" where arrays start with null (e.g., [null, {...}]).
    
    Args:
        data: The data structure to clean (dict, list, or scalar).
        
    Returns:
        Cleaned data structure with nulls removed from arrays.
    """
    if isinstance(data, dict):
        # Recursively clean all values in the dict
        return {k: _clean_null_from_arrays(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Filter out None/null items from arrays
        filtered = [item for item in data if item is not None]
        # Recursively clean remaining items
        return [_clean_null_from_arrays(item) for item in filtered]
    else:
        # Scalar value - return as-is
        return data


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
    warnings_set: set[str],
    extracted_data: dict[str, Any] | None = None,
) -> None:
    """
    Perform math checks using simpleeval for dynamic expression evaluation.

    This is 100% dynamic - supports any mathematical expression including:
    - Parentheses: "(a + b) * c"
    - All operators: +, -, *, /
    - Complex formulas: "(revenue - cost) / revenue"

    Uses simpleeval for safe, sandboxed expression evaluation.
    
    Safety: Filters out rules that reference fields not in the root of extracted_data
    to prevent crashes from nested array fields.

    Args:
        numeric_values: Dictionary of field_name -> parsed numeric value.
        validation_rules: List of rule strings from the schema.
        warnings_set: Set to add warnings to (for deduplication).
        extracted_data: The full extracted data dict to check for field existence.
    """
    if not validation_rules:
        return

    # Get root-level field names from extracted_data for safety check
    root_fields = set(extracted_data.keys()) if extracted_data else set()
    
    for rule in validation_rules:
        # Safety check: Extract field names from rule and verify they exist at root level
        # This prevents crashes if AI generates rules for nested array fields
        rule_field_names = _extract_field_names_from_rule(rule)
        
        # Filter out rules that reference fields not in root (likely nested array fields)
        if root_fields and rule_field_names:
            missing_fields = rule_field_names - root_fields
            if missing_fields:
                logger.debug(
                    "Skipping validation rule '%s' - references fields not in root: %s (likely nested array fields)",
                    rule,
                    missing_fields,
                )
                continue
        
        success, message, failed_rule = _evaluate_validation_rule_simpleeval(
            rule, numeric_values
        )
        if not success:
            warnings_set.add(message)
            logger.warning("Validation rule failed: %s", message)


def _extract_field_names_from_rule(rule: str) -> set[str]:
    """
    Extract field names from a validation rule string.
    
    This is a simple heuristic to find potential field names in expressions.
    Looks for valid Python identifiers that could be field names.
    
    Args:
        rule: The validation rule string (e.g., "total == subtotal + tax")
        
    Returns:
        Set of potential field names found in the rule.
    """
    # Remove operators and parentheses to isolate identifiers
    # This is a simple heuristic - not perfect but good enough for safety checks
    cleaned = re.sub(r'[+\-*/()=<>!&| ]+', ' ', rule)
    tokens = cleaned.split()
    
    field_names = set()
    for token in tokens:
        # Check if token looks like a field name (snake_case identifier)
        if re.match(r'^[a-z_][a-z0-9_]*$', token, re.IGNORECASE):
            # Skip known functions
            if token.lower() not in ('sum', 'round', 'abs', 'min', 'max', 'sqrt', 'log', 'len', 'pow', 'log10'):
                field_names.add(token)
    
    return field_names


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

    Uses OpenAI's GPT-4.1 model with vision capabilities for:
    - Analyzing document structure and suggesting extraction schemas
    - Extracting data according to confirmed schemas with confidence scoring
    """

    # System prompts
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

    EXTRACTION_SYSTEM_PROMPT = """You are a precise Data Entry Clerk with exceptional attention to detail.
Your task is to extract specific data fields from a document image AND estimate your confidence for each field.

## Extraction Rules:

1. **Strict Adherence**: Only extract the fields specified. Do not add extra fields.
2. **Accuracy Over Guessing**: If a value is unclear or not present, return null. DO NOT HALLUCINATE.
3. **Preserve Original Format**: Keep dates, currencies, and numbers as they appear in the document.
4. **No Assumptions**: Do not infer or calculate values unless explicitly stated in the document.

## Critical: Headers and Footers
Pay special attention to **Headers and Footers** for:
- Organization Names (issuing_organization, company_name, etc.)
- Addresses (mailing_address, business_address, etc.)
- Websites (website, url, company_website, etc.)
- Contact Information (phone, email, etc.)

These fields are often located in document headers or footers, not just in the main body.

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
- For empty/missing fields: Return null with confidence 0.0. DO NOT HALLUCINATE OR MAKE UP VALUES.
- For ambiguous values: Choose the most likely interpretation and reflect uncertainty in confidence score
- If a field is not found after checking headers, body, and footers: Return null, do not guess.

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

    async def discover_schema(self, images: list[Image.Image] | Image.Image) -> SchemaDefinition:
        """
        Analyze document images and discover an extraction schema.

        Uses Chain-of-Thought prompting to:
        1. Classify the document type
        2. Identify key data points across all pages
        3. Generate a semantic schema

        Args:
            images: Single PIL Image or list of PIL Images (representative pages).

        Returns:
            SchemaDefinition based on AI analysis.
        """
        if self.use_mock:
            logger.info("Discovering schema (MOCK MODE)")
            return self._get_mock_schema()

        # Normalize to list
        if isinstance(images, Image.Image):
            images = [images]
        
        logger.info("Discovering schema using GPT-4.1 structured outputs (%d pages)", len(images))
        
        # Convert all images to base64
        base64_images = [self._image_to_base64(img) for img in images]

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
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.DISCOVERY_SYSTEM_PROMPT},
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
        images: list[Image.Image] | Image.Image,
        schema: SchemaDefinition,
        source_file: str,
    ) -> ExtractionResult:
        """
        Extract data from document images according to a schema.

        Uses:
        - Dynamic prompt construction based on schema
        - Logprobs for confidence scoring (geometric mean)
        - Post-processing validation
        - Chunking for large documents (>10 pages)

        Args:
            images: Single PIL Image or list of PIL Images to extract from.
            schema: The schema defining what fields to extract.
            source_file: Original filename for the result.

        Returns:
            ExtractionResult with extracted data and confidence score.
        """
        if self.use_mock:
            logger.info("Extracting data (MOCK MODE) for: %s", source_file)
            return self._get_mock_extraction(schema, source_file)

        # Normalize to list
        if isinstance(images, Image.Image):
            images = [images]
        
        total_pages = len(images)
        logger.info(
            "Extracting data from %d page(s) for '%s' using schema '%s'",
            total_pages,
            source_file,
            schema.name,
        )

        # DEBUG: Log exact schema fields being used (no caching - rebuilt each time)
        field_names = [f.name for f in schema.fields]
        print(f"DEBUG AI_SERVICE: Extracting with fields: {field_names}")
        logger.info(
            "DEBUG AI_SERVICE: Extracting data for '%s' using schema '%s' with fields: %s",
            source_file,
            schema.name,
            field_names,
        )
        
        # For extraction, we want ALL pages if <= 10, otherwise chunk
        if total_pages <= 10:
            # Process all pages in a single request
            raw_result = await self._extract_from_images(images, schema, source_file)
            
            # Validate ONCE after extraction (post-merge equivalent for single request)
            validation = validate_extracted_data(raw_result.extracted_data, schema)
            
            return ExtractionResult(
                source_file=raw_result.source_file,
                detected_schema=raw_result.detected_schema,
                extracted_data=validation.validated_data,
                confidence=raw_result.confidence,
                warnings=validation.warnings,
                field_confidences=raw_result.field_confidences,
            )
        else:
            # Process in chunks of 5 pages, then merge results
            # Validation happens in _merge_extraction_results (post-merge)
            logger.info("Large document (%d pages): processing in chunks of 5", total_pages)
            return await self._extract_from_images_chunked(images, schema, source_file)

    async def _extract_from_images(
        self,
        images: list[Image.Image],
        schema: SchemaDefinition,
        source_file: str,
    ) -> ExtractionResult:
        """Extract data from a list of images (single request)."""
        # Convert all images to base64
        base64_images = [self._image_to_base64(img) for img in images]

        # Build dynamic extraction prompt (rebuilt fresh for each extraction - NO CACHING)
        extraction_prompt = self._build_extraction_prompt(schema)
        
        # DEBUG: Log first 500 chars of prompt to verify field names
        logger.debug("DEBUG AI_SERVICE: Extraction prompt preview: %s...", extraction_prompt[:500])

        try:
            # Build content with multiple images
            content = [
                {"type": "text", "text": extraction_prompt},
            ]
            
            # Add all images
            for base64_img in base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_img}",
                        "detail": "high",
                    },
                })
            
            # Use standard chat.completions.create with json_object for logprobs access
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": content,
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

            # Calculate global confidence using strict math: average of all non-None field confidences
            # Round to 3 decimal places for consistency
            if field_confidences:
                # Get all non-None field scores
                field_scores = [
                    v for k, v in field_confidences.items()
                    if v is not None
                ]
                
                if field_scores:
                    confidence = round(sum(field_scores) / len(field_scores), 3)
                    logger.info(
                        "Global confidence: %.3f (from %d field scores, strict average)",
                        confidence,
                        len(field_scores),
                    )
                else:
                    # No valid scores found - return 0.0
                    confidence = 0.0
                    logger.warning("No valid field confidence scores found, confidence set to 0.0")
            else:
                # Fallback: calculate from logprobs if field confidences not available
                logprobs_data = None
                if response.choices[0].logprobs and response.choices[0].logprobs.content:
                    logprobs_data = response.choices[0].logprobs.content
                confidence = calculate_confidence_from_logprobs(logprobs_data)
                logger.warning("Field confidences not available, using logprobs-based confidence")

            logger.info(
                "Extraction confidence: %.3f (from %d field confidences, avg: %.3f)",
                confidence,
                len(field_confidences) if field_confidences else 0,
                sum(field_confidences.values()) / len(field_confidences) if field_confidences and len(field_confidences) > 0 else 0.0,
            )

            # Post-process: Clean nulls from arrays recursively (fix "List Stutter")
            cleaned_data = _clean_null_from_arrays(extracted_data)
            
            # DO NOT validate here - validation happens post-merge to avoid false warnings
            # Return raw extracted data (validation will happen after merging if chunked,
            # or in extract_data for single requests)

            return ExtractionResult(
                source_file=source_file,
                detected_schema=schema,
                extracted_data=cleaned_data,  # Return cleaned but unvalidated data
                confidence=confidence,
                warnings=[],  # No warnings yet - validation happens post-merge
                field_confidences=field_confidences,
            )

        except AIServiceError:
            raise
        except Exception as e:
            logger.exception("Data extraction failed")
            raise AIServiceError(f"Data extraction failed: {e}") from e

    async def _extract_from_images_chunked(
        self,
        images: list[Image.Image],
        schema: SchemaDefinition,
        source_file: str,
    ) -> ExtractionResult:
        """Extract data from images in chunks of 5, then merge results."""
        chunk_size = 5
        all_results: list[ExtractionResult] = []
        
        # Process in chunks
        for i in range(0, len(images), chunk_size):
            chunk = images[i:i + chunk_size]
            chunk_num = (i // chunk_size) + 1
            total_chunks = (len(images) + chunk_size - 1) // chunk_size
            logger.info(
                "Processing chunk %d/%d (%d pages) for '%s'",
                chunk_num,
                total_chunks,
                len(chunk),
                source_file,
            )
            
            chunk_result = await self._extract_from_images(chunk, schema, source_file)
            all_results.append(chunk_result)
        
        # Merge results
        return self._merge_extraction_results(all_results, schema, source_file)
    
    def _merge_extraction_results(
        self,
        results: list[ExtractionResult],
        schema: SchemaDefinition,
        source_file: str,
    ) -> ExtractionResult:
        """Merge multiple extraction results into one, appending array fields."""
        if not results:
            raise AIServiceError("No results to merge")
        
        if len(results) == 1:
            return results[0]
        
        # Start with first result
        merged_data = results[0].extracted_data.copy()
        merged_field_confidences: dict[str, list[float]] = {}
        all_confidences: list[float] = []
        
        # Initialize field confidences from first result
        for field_name, conf in results[0].field_confidences.items():
            merged_field_confidences[field_name] = [conf]
        
        all_confidences.append(results[0].confidence)
        # Do NOT include warnings from individual chunks - they may be false positives
        # Warnings will be generated post-merge based on the final merged data
        
        # Merge subsequent results
        for result in results[1:]:
            all_confidences.append(result.confidence)
            # Do NOT include warnings from individual chunks - they may be false positives
            # Warnings will be generated post-merge based on the final merged data
            
            # Merge field confidences
            for field_name, conf in result.field_confidences.items():
                if field_name not in merged_field_confidences:
                    merged_field_confidences[field_name] = []
                merged_field_confidences[field_name].append(conf)
            
            # Merge extracted data
            for field_name, value in result.extracted_data.items():
                # Check if this field is an array type
                field_def = next((f for f in schema.fields if f.name == field_name), None)
                is_array = field_def and field_def.type == FieldType.ARRAY
                
                if is_array and isinstance(value, list):
                    # Append array items
                    if field_name in merged_data:
                        if isinstance(merged_data[field_name], list):
                            merged_data[field_name].extend(value)
                        else:
                            merged_data[field_name] = [merged_data[field_name]] + value
                    else:
                        merged_data[field_name] = value
                else:
                    # For non-array fields, keep the latest value (or merge intelligently)
                    # For now, prefer non-null values
                    if value is not None and value != "":
                        if field_name not in merged_data or merged_data[field_name] is None or merged_data[field_name] == "":
                            merged_data[field_name] = value
        
        # Calculate average field confidences
        final_field_confidences: dict[str, float] = {}
        for field_name, conf_list in merged_field_confidences.items():
            valid_confs = [c for c in conf_list if isinstance(c, (int, float)) and 0 <= c <= 1]
            if valid_confs:
                final_field_confidences[field_name] = sum(valid_confs) / len(valid_confs)
        
        # Calculate overall confidence as average (rounded to 3 decimal places)
        overall_confidence = round(
            sum(all_confidences) / len(all_confidences) if all_confidences else 0.0,
            3
        )
        
        # Post-process: Clean nulls from arrays in merged data
        cleaned_merged_data = _clean_null_from_arrays(merged_data)
        
        # CRITICAL: Validate ONCE on the final merged data (post-merge)
        # This prevents false warnings from individual chunks that are missing fields
        # that appear in other chunks
        validation = validate_extracted_data(cleaned_merged_data, schema)
        
        # Use only the warnings from post-merge validation (ignore chunk warnings)
        unique_warnings = list(set(validation.warnings))
        
        logger.info(
            "Merged %d extraction results: %d fields, confidence=%.3f",
            len(results),
            len(merged_data),
            overall_confidence,
        )
        
        return ExtractionResult(
            source_file=source_file,
            detected_schema=schema,
            extracted_data=validation.validated_data,
            confidence=overall_confidence,
            warnings=unique_warnings,
            field_confidences=final_field_confidences,
        )

    def _build_extraction_prompt(self, schema: SchemaDefinition) -> str:
        """Build a dynamic extraction prompt from the schema.
        
        NOTE: This is rebuilt fresh for EVERY extraction - no caching.
        The field names come directly from the schema parameter passed in.
        """
        field_names = [f.name for f in schema.fields]
        print(f"DEBUG _build_extraction_prompt: Building prompt for fields: {field_names}")
        
        field_descriptions = []
        array_fields = []
        for field in schema.fields:
            required_marker = " (REQUIRED)" if field.required else " (optional)"
            field_descriptions.append(
                f"- **{field.name}** ({field.type.value}){required_marker}: {field.description}"
            )
            if field.type == FieldType.ARRAY:
                array_fields.append(field.name)

        fields_text = "\n".join(field_descriptions)
        
        # Build array extraction instructions
        array_instructions = ""
        if array_fields:
            array_instructions = f"""

## CRITICAL: Array Field Extraction
The following fields are ARRAY types (tables/lists): {', '.join(array_fields)}
- For each array field, you MUST extract EVERY row in the table/list
- Do NOT stop after the first row - capture ALL rows
- Return a JSON array of objects, where each object represents one row
- Example structure:
  {{
    "extracted_data": {{
      "line_items": [
        {{"description": "Item 1", "quantity": 2, "unit_price": 10.00, "total": 20.00}},
        {{"description": "Item 2", "quantity": 1, "unit_price": 15.00, "total": 15.00}},
        {{"description": "Item 3", "quantity": 3, "unit_price": 5.00, "total": 15.00}}
      ]
    }},
    "field_confidences": {{
      "line_items": 0.95
    }}
  }}
- Ensure ALL rows are captured, not just the first one"""

        return f"""Extract the following fields from this document image.

## Fields to Extract:
{fields_text}{array_instructions}

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

    async def repair_data_with_llm(
        self,
        extracted_data: dict[str, Any],
        schema: SchemaDefinition | None = None,
    ) -> dict[str, Any]:
        """
        Dynamic Calculation Engine: Use LLM to complete and calculate missing values.
        
        This is a sophisticated calculation engine that:
        - Analyzes the full schema to identify missing fields
        - Infers formulas from field names and descriptions
        - Calculates missing values from available data
        - Cross-references values across the dataset
        - Aggressively completes missing data based on patterns
        
        Args:
            extracted_data: The full extracted data dictionary to repair/complete.
            schema: The full schema definition with field descriptions (REQUIRED for calculation).
            
        Returns:
            Fully populated and calculated data dictionary.
        
        Raises:
            AIServiceError: If the repair operation fails.
        """
        if self.use_mock:
            logger.info("Repairing data (MOCK MODE) - returning original data")
            return extracted_data
        
        if not schema:
            logger.warning("No schema provided for repair - inferring schema from data")
            # Fallback: Create a minimal schema from the extracted data keys
            # This allows the calculation engine to work even without a saved schema
            inferred_fields = []
            for key, value in extracted_data.items():
                # Infer field type from value
                if isinstance(value, (int, float)):
                    field_type = FieldType.NUMBER
                elif isinstance(value, bool):
                    field_type = FieldType.BOOLEAN
                elif isinstance(value, list):
                    field_type = FieldType.ARRAY
                elif isinstance(value, str):
                    # Try to infer more specific types
                    if "@" in value and "." in value:
                        field_type = FieldType.EMAIL
                    else:
                        field_type = FieldType.STRING
                else:
                    field_type = FieldType.STRING
                
                inferred_fields.append(
                    FieldDefinition(
                        name=key,
                        type=field_type,
                        description=f"Inferred field: {key}",
                        required=False,
                    )
                )
            
            if not inferred_fields:
                # If no data at all, create a minimal placeholder
                inferred_fields.append(
                    FieldDefinition(
                        name="placeholder",
                        type=FieldType.STRING,
                        description="Placeholder field",
                        required=False,
                    )
                )
            
            schema = SchemaDefinition(
                name="Inferred Schema",
                description="Schema inferred from extracted data",
                fields=inferred_fields,
                version="1.0",
            )
            logger.info("Inferred schema with %d fields from extracted data", len(inferred_fields))
        
        REPAIR_SYSTEM_PROMPT = """You are an expert Data Analyst and Mathematician. Your goal is to **COMPLETE** the dataset by calculating missing values and enforcing logical consistency.

**Your Instructions:**

1. **Analyze the Schema:** Look at every field in the provided Schema, especially those that are currently `null` in the JSON.

2. **Infer Formulas:** If a field is named `average_attendance_rate` or `total_cost`, YOU MUST CALCULATE IT based on the other data available in the JSON (e.g., average the items in the `monthly_data` array, or sum the `line_items`).

3. **Cross-Reference:** If a value is missing in one section but present in another (e.g., 'Date' is in the footer but missing in the header), copy it over.

4. **Aggressive Completion:** If you see a pattern (e.g., a list of years), fill in the obvious gaps.

5. **Mathematical Operations:**
   - Calculate totals from components (e.g., `total = subtotal + tax`)
   - Calculate averages from arrays (e.g., `average_attendance = sum(monthly_attendance) / count(monthly_attendance)`)
   - Derive missing dates (e.g., `due_date = issue_date + payment_terms_days`)
   - Compute percentages and ratios from available data

6. **Data Consistency:**
   - Fix OCR typos (e.g., 'l' -> '1', 'O' -> '0')
   - Normalize formats (dates to YYYY-MM-DD, currencies to numeric)
   - Ensure mathematical relationships hold (e.g., line item totals = qty * price)

**CRITICAL:** Return ONLY the fully populated, mathematically corrected JSON object. No markdown, no explanations, no commentary. Just the complete JSON."""

        try:
            # Convert extracted_data to JSON string
            json_data = json.dumps(extracted_data, indent=2)
            
            # Convert schema to a readable format
            schema_fields = []
            for field in schema.fields:
                field_info = {
                    "name": field.name,
                    "type": field.type.value,
                    "description": field.description,
                    "required": field.required,
                }
                schema_fields.append(field_info)
            
            schema_definition = json.dumps({
                "name": schema.name,
                "description": schema.description,
                "fields": schema_fields,
            }, indent=2)
            
            user_prompt = f"""**Input Data:**
{json_data}

**Target Schema:**
{schema_definition}

Complete and calculate all missing values based on the schema and available data. Return the fully populated JSON object."""
            
            logger.info(
                "Calling LLM calculation engine: %d fields in schema, %d fields in data",
                len(schema.fields),
                len(extracted_data),
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                # No max_tokens limit - let the model use its full capacity
            )
            
            content = response.choices[0].message.content
            if not content:
                raise AIServiceError("Empty response from OpenAI calculation engine")
            
            try:
                repaired_data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse calculation response: %s", content[:500])
                raise AIServiceError(f"Invalid JSON in calculation response: {e}") from e
            
            # Count how many fields were added/calculated
            original_keys = set(extracted_data.keys())
            repaired_keys = set(repaired_data.keys())
            new_fields = repaired_keys - original_keys
            updated_fields = [
                k for k in original_keys
                if k in repaired_data and extracted_data.get(k) != repaired_data.get(k)
            ]
            
            logger.info(
                "Calculation engine completed: %d total fields (%d new, %d updated)",
                len(repaired_data),
                len(new_fields),
                len(updated_fields),
            )
            
            if new_fields:
                logger.info("New calculated fields: %s", list(new_fields))
            if updated_fields:
                logger.info("Updated fields: %s", updated_fields[:10])  # Log first 10
            
            return repaired_data
            
        except AIServiceError:
            raise
        except Exception as e:
            logger.exception("Data calculation engine failed")
            raise AIServiceError(f"Data calculation engine failed: {e}") from e

    # =========================================================================
    # Legacy method for backwards compatibility
    # =========================================================================

    async def suggest_schema(self, images: list[Image.Image] | Image.Image) -> SchemaDefinition:
        """
        Alias for discover_schema for backwards compatibility.

        Args:
            images: Single PIL Image or list of PIL Images (representative pages).

        Returns:
            SchemaDefinition based on AI analysis.
        """
        return await self.discover_schema(images)

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
