"""
Validation and data normalization utilities for extracted data.

Handles:
- Type validation (dates, currency, numbers, booleans)
- Math validation using dynamic rules
- Data cleaning (null removal from arrays)
"""

import logging
import math
import re
from typing import Any

from simpleeval import NameNotDefined, SimpleEval

# Handle both package imports and standalone imports
try:
    from ...models import FieldDefinition, FieldType, SchemaDefinition
except ImportError:
    from models import FieldDefinition, FieldType, SchemaDefinition

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of data validation."""

    def __init__(self):
        self.validated_data: dict[str, Any] = {}
        self.warnings: list[str] = []


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
        from price_parser import Price

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
    if not value:
        return None

    # Try ISO format first (YYYY-MM-DD)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value

    # Try US format (MM/DD/YYYY)
    match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", value)
    if match:
        month, day, year = match.groups()
        try:
            from datetime import datetime

            dt = datetime(int(year), int(month), int(day))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    # Try European format (DD/MM/YYYY)
    match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", value)
    if match:
        day, month, year = match.groups()
        try:
            from datetime import datetime

            dt = datetime(int(year), int(month), int(day))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    # Try written formats
    try:
        from dateutil import parser

        dt = parser.parse(value)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, ImportError):
        return None


def _clean_null_from_arrays(
    data: dict[str, Any] | list[Any] | Any,
) -> dict[str, Any] | list[Any] | Any:
    """
    Recursively remove None/null values from arrays in the data structure.

    This fixes the "List Stutter" bug where arrays contain null values.
    """
    if isinstance(data, dict):
        return {k: _clean_null_from_arrays(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Filter out None/null items
        filtered = [x for x in data if x is not None]
        # Recursively clean nested structures
        return [_clean_null_from_arrays(item) for item in filtered]
    else:
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

    # Simple parser for expressions like "a + b - c"
    # Split by operators and extract field names
    operators = ["+", "-", "*", "/"]
    components = []
    current = ""
    for char in expression:
        if char in operators:
            if current.strip():
                components.append(("+", current.strip()))
            current = ""
        else:
            current += char
    if current.strip():
        components.append(("+", current.strip()))

    return (result_field, components)


def _evaluate_validation_rule(
    rule: str, values: dict[str, float]
) -> tuple[bool, str, float | None, float | None]:
    """
    Legacy evaluation function - kept for backwards compatibility.

    Returns (success, message, expected_value, actual_value).
    """
    success, message, _ = _evaluate_validation_rule_simpleeval(rule, values)
    return (success, message, None, None)


def _extract_field_names_from_rule(rule: str) -> set[str]:  # Exported for tests
    """
    Extract field names from a validation rule string.

    This is a simple heuristic to find potential field names in expressions.
    Looks for valid Python identifiers that could be field names.

    Args:
        rule: The validation rule string (e.g., "total == subtotal + tax")

    Returns:
        Set of potential field names found in the rule.
    """
    # Find all valid Python identifiers (field names)
    # Exclude Python keywords and built-in functions
    python_keywords = {
        "and",
        "or",
        "not",
        "in",
        "is",
        "if",
        "else",
        "for",
        "while",
        "def",
        "class",
        "import",
        "from",
        "as",
        "return",
        "True",
        "False",
        "None",
        "sum",
        "round",
        "abs",
        "min",
        "max",
        "sqrt",
        "log",
        "len",
    }

    # Match valid Python identifiers
    identifiers = re.findall(r"[a-z_][a-z0-9_]*", rule, re.IGNORECASE)
    # Filter out keywords and operators
    field_names = {
        ident.lower()
        for ident in identifiers
        if ident.lower() not in python_keywords
        and ident not in ("==", "=", "+", "-", "*", "/", "(", ")", "[", "]")
    }

    return field_names


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


def validate_extracted_data(
    data: dict[str, Any], schema: SchemaDefinition
) -> ValidationResult:
    """
    Validate extracted data against a schema.

    Performs:
    - Type checking (dates, currency, numbers)
    - Null/empty checks for required fields
    - Math validation using dynamic rules from schema

    Args:
        data: The extracted data dictionary.
        schema: The schema to validate against.

    Returns:
        ValidationResult with validated_data and warnings.
    """
    result = ValidationResult()

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

