"""
Backward-compatible wrapper for AI service.

This module maintains backward compatibility by re-exporting
everything from the new services.ai package.
"""

# Re-export everything from the new package for backward compatibility
from .ai import (
    AIService,
    AIServiceError,
    ValidationResult,
    _evaluate_validation_rule,
    _extract_field_names_from_rule,
    _parse_validation_rule,
    calculate_confidence_from_logprobs,
    discover_schema,
    extract_data,
    get_ai_service,
    parse_currency,
    parse_date,
    repair_data_with_llm,
    validate_extracted_data,
)

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
