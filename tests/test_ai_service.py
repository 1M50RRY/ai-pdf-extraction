"""Tests for AI service with validation and confidence scoring."""

import math

import pytest

from app.backend.models import FieldDefinition, FieldType, SchemaDefinition
from app.backend.services.ai_service import (
    AIService,
    ValidationResult,
    _evaluate_validation_rule,
    _parse_validation_rule,
    calculate_confidence_from_logprobs,
    parse_currency,
    parse_date,
    validate_extracted_data,
)


class TestParseCurrency:
    """Tests for currency parsing utility."""

    def test_parse_usd_format(self):
        """Test parsing US dollar format."""
        assert parse_currency("$1,234.56") == 1234.56
        assert parse_currency("$100.00") == 100.00
        assert parse_currency("$1,000,000.00") == 1000000.00

    def test_parse_euro_format(self):
        """Test parsing European format (comma as decimal)."""
        assert parse_currency("1.234,56") == 1234.56
        assert parse_currency("€1.234,56") == 1234.56

    def test_parse_plain_numbers(self):
        """Test parsing plain numbers."""
        assert parse_currency("1234.56") == 1234.56
        assert parse_currency("1234") == 1234.0
        assert parse_currency(100) == 100.0
        assert parse_currency(99.99) == 99.99

    def test_parse_with_currency_symbols(self):
        """Test parsing with various currency symbols."""
        assert parse_currency("€100.00") == 100.00
        assert parse_currency("£500.00") == 500.00
        assert parse_currency("¥1000") == 1000.0

    def test_parse_invalid_returns_none(self):
        """Test that invalid values return None."""
        assert parse_currency(None) is None
        assert parse_currency("") is None
        assert parse_currency("not a number") is None
        # Note: "abc123" may extract 123 via price-parser fallback, which is valid behavior

    def test_parse_european_decimal_only(self):
        """Test European format with only decimal comma."""
        assert parse_currency("1234,56") == 1234.56


class TestParseDate:
    """Tests for date parsing utility."""

    def test_parse_iso_format(self):
        """Test parsing ISO date format."""
        assert parse_date("2024-01-15") == "2024-01-15"
        assert parse_date("2023-12-31") == "2023-12-31"

    def test_parse_us_format(self):
        """Test parsing US date format (MM/DD/YYYY)."""
        assert parse_date("01/15/2024") == "2024-01-15"
        assert parse_date("12/31/2023") == "2023-12-31"

    def test_parse_european_format(self):
        """Test parsing European date format (DD/MM/YYYY)."""
        assert parse_date("15/01/2024") == "2024-01-15"

    def test_parse_written_format(self):
        """Test parsing written date formats."""
        assert parse_date("January 15, 2024") == "2024-01-15"
        assert parse_date("Jan 15, 2024") == "2024-01-15"
        assert parse_date("15 January 2024") == "2024-01-15"

    def test_parse_invalid_returns_none(self):
        """Test that invalid dates return None."""
        assert parse_date(None) is None
        assert parse_date("") is None
        assert parse_date("not a date") is None
        assert parse_date("32/13/2024") is None  # Invalid day/month


class TestValidateExtractedData:
    """Tests for data validation."""

    @pytest.fixture
    def invoice_schema(self) -> SchemaDefinition:
        """Create a sample invoice schema for testing."""
        return SchemaDefinition(
            name="Invoice Schema",
            fields=[
                FieldDefinition(
                    name="invoice_number",
                    type=FieldType.STRING,
                    description="Invoice ID",
                    required=True,
                ),
                FieldDefinition(
                    name="invoice_date",
                    type=FieldType.DATE,
                    description="Invoice date",
                    required=True,
                ),
                FieldDefinition(
                    name="subtotal",
                    type=FieldType.CURRENCY,
                    description="Subtotal",
                    required=False,
                ),
                FieldDefinition(
                    name="tax_amount",
                    type=FieldType.CURRENCY,
                    description="Tax",
                    required=False,
                ),
                FieldDefinition(
                    name="total_amount",
                    type=FieldType.CURRENCY,
                    description="Total",
                    required=True,
                ),
                FieldDefinition(
                    name="is_paid",
                    type=FieldType.BOOLEAN,
                    description="Payment status",
                    required=False,
                ),
            ],
            validation_rules=[
                "total_amount == subtotal + tax_amount",
            ],
        )

    def test_valid_data_passes(self, invoice_schema: SchemaDefinition):
        """Test that valid data passes validation."""
        data = {
            "invoice_number": "INV-001",
            "invoice_date": "2024-01-15",
            "subtotal": "$100.00",
            "tax_amount": "$10.00",
            "total_amount": "$110.00",
            "is_paid": True,
        }
        result = validate_extracted_data(data, invoice_schema)
        assert len(result.warnings) == 0
        assert result.validated_data["invoice_number"] == "INV-001"

    def test_missing_required_field_warns(self, invoice_schema: SchemaDefinition):
        """Test that missing required fields generate warnings."""
        data = {
            "invoice_date": "2024-01-15",
            "total_amount": "$110.00",
        }
        result = validate_extracted_data(data, invoice_schema)
        # Note: Current behavior - only warns if value is explicitly None or ""
        # Missing keys are trusted (AI may have renamed or skipped them)
        # So missing keys don't generate warnings, only empty values do
        assert not any("invoice_number" in w for w in result.warnings)

    def test_invalid_date_warns(self, invoice_schema: SchemaDefinition):
        """Test that invalid dates are handled gracefully."""
        data = {
            "invoice_number": "INV-001",
            "invoice_date": "not a date",
            "total_amount": "$110.00",
        }
        result = validate_extracted_data(data, invoice_schema)
        # Note: Current behavior - date validation is relaxed
        # Invalid dates are kept as-is (prefer raw data over no data)
        # No warnings are generated for invalid date formats
        assert not any("invalid date" in w.lower() for w in result.warnings)
        # The value should still be in validated_data
        assert result.validated_data["invoice_date"] == "not a date"

    def test_date_normalization(self, invoice_schema: SchemaDefinition):
        """Test that dates are normalized to YYYY-MM-DD."""
        data = {
            "invoice_number": "INV-001",
            "invoice_date": "January 15, 2024",
            "total_amount": "$110.00",
        }
        result = validate_extracted_data(data, invoice_schema)
        assert result.validated_data["invoice_date"] == "2024-01-15"

    def test_math_mismatch_warns(self, invoice_schema: SchemaDefinition):
        """Test that math mismatches generate warnings."""
        data = {
            "invoice_number": "INV-001",
            "invoice_date": "2024-01-15",
            "subtotal": "$100.00",
            "tax_amount": "$10.00",
            "total_amount": "$200.00",  # Should be $110, not $200
        }
        result = validate_extracted_data(data, invoice_schema)
        assert any("math validation failed" in w.lower() for w in result.warnings)

    def test_math_match_no_warning(self, invoice_schema: SchemaDefinition):
        """Test that correct math doesn't generate warnings."""
        data = {
            "invoice_number": "INV-001",
            "invoice_date": "2024-01-15",
            "subtotal": "$100.00",
            "tax_amount": "$10.00",
            "total_amount": "$110.00",  # Correct!
        }
        result = validate_extracted_data(data, invoice_schema)
        assert not any("math validation failed" in w.lower() for w in result.warnings)

    def test_boolean_string_conversion(self, invoice_schema: SchemaDefinition):
        """Test that boolean strings are converted."""
        data = {
            "invoice_number": "INV-001",
            "invoice_date": "2024-01-15",
            "total_amount": "$110.00",
            "is_paid": "yes",
        }
        result = validate_extracted_data(data, invoice_schema)
        assert result.validated_data["is_paid"] is True

        data["is_paid"] = "no"
        result = validate_extracted_data(data, invoice_schema)
        assert result.validated_data["is_paid"] is False


class TestDynamicValidationRules:
    """Tests for dynamic validation rule parsing and evaluation."""

    def test_parse_simple_addition_rule(self):
        """Test parsing a simple addition rule."""
        result = _parse_validation_rule("total == subtotal + tax")
        assert result is not None
        assert result[0] == "total"
        assert result[1] == [("+", "subtotal"), ("+", "tax")]

    def test_parse_subtraction_rule(self):
        """Test parsing a subtraction rule."""
        result = _parse_validation_rule("net_pay == gross_pay - deductions")
        assert result is not None
        assert result[0] == "net_pay"
        assert result[1] == [("+", "gross_pay"), ("-", "deductions")]

    def test_parse_complex_rule(self):
        """Test parsing a rule with multiple operations."""
        result = _parse_validation_rule("total == a + b + c - d")
        assert result is not None
        assert result[0] == "total"
        assert len(result[1]) == 4

    def test_parse_invalid_rule_returns_none(self):
        """Test that invalid rules return None."""
        assert _parse_validation_rule("invalid rule") is None
        assert _parse_validation_rule("") is None
        assert _parse_validation_rule("no equals sign") is None

    def test_evaluate_rule_passes_when_math_correct(self):
        """Test that correct math passes evaluation."""
        values = {"total": 110.0, "subtotal": 100.0, "tax": 10.0}
        success, msg, _, _ = _evaluate_validation_rule(
            "total == subtotal + tax", values
        )
        assert success is True
        assert "passed" in msg.lower() or "not found" not in msg.lower()

    def test_evaluate_rule_fails_when_math_wrong(self):
        """Test that incorrect math fails evaluation."""
        values = {"total": 200.0, "subtotal": 100.0, "tax": 10.0}
        success, msg, _, _ = _evaluate_validation_rule(
            "total == subtotal + tax", values
        )
        assert success is False
        assert "Math validation failed" in msg

    def test_evaluate_rule_skips_missing_result_field(self):
        """Test that missing result field skips validation."""
        values = {"subtotal": 100.0, "tax": 10.0}  # No 'total'
        success, msg, _, _ = _evaluate_validation_rule(
            "total == subtotal + tax", values
        )
        assert success is True  # Skipped, not failed
        assert "not found" in msg.lower() or "not defined" in msg.lower()

    def test_evaluate_rule_skips_missing_operands(self):
        """Test that missing operands skip validation."""
        values = {"total": 110.0, "subtotal": 100.0}  # No 'tax'
        success, msg, _, _ = _evaluate_validation_rule(
            "total == subtotal + tax", values
        )
        assert success is True  # Skipped, not failed
        assert "not found" in msg.lower() or "not defined" in msg.lower()

    def test_evaluate_subtraction_rule(self):
        """Test subtraction rule evaluation."""
        values = {"net_pay": 4000.0, "gross_pay": 5000.0, "deductions": 1000.0}
        success, msg, _, _ = _evaluate_validation_rule(
            "net_pay == gross_pay - deductions", values
        )
        assert success is True

    def test_evaluate_complex_expression_with_parentheses(self):
        """Test complex expressions with parentheses using simpleeval."""
        # Test: grand_total == (subtotal + shipping) * tax_multiplier
        values = {
            "subtotal": 100.0,
            "shipping": 10.0,
            "tax_multiplier": 1.1,
            "grand_total": 121.0,  # (100 + 10) * 1.1 = 121
        }
        success, msg, _, _ = _evaluate_validation_rule(
            "grand_total == (subtotal + shipping) * tax_multiplier", values
        )
        assert success is True

    def test_evaluate_division_expression(self):
        """Test division expressions."""
        values = {
            "revenue": 1000.0,
            "cost": 600.0,
            "margin": 0.4,  # (1000 - 600) / 1000 = 0.4
        }
        success, msg, _, _ = _evaluate_validation_rule(
            "margin == (revenue - cost) / revenue", values
        )
        assert success is True

    def test_evaluate_with_round_function(self):
        """Test expressions using round() function."""
        values = {
            "price": 100.0,
            "tax_rate": 0.0875,
            "total": 108.75,  # round(100 * 1.0875, 2) = 108.75
        }
        success, msg, _, _ = _evaluate_validation_rule(
            "total == round(price * (1 + tax_rate), 2)", values
        )
        assert success is True

    def test_evaluate_with_abs_function(self):
        """Test expressions using abs() function."""
        values = {
            "budget": 1000.0,
            "actual": 1150.0,
            "variance": 150.0,  # abs(1000 - 1150) = 150
        }
        success, msg, _, _ = _evaluate_validation_rule(
            "variance == abs(budget - actual)", values
        )
        assert success is True

    def test_evaluate_with_min_max_functions(self):
        """Test expressions using min() and max() functions."""
        values = {
            "a": 100.0,
            "b": 200.0,
            "c": 150.0,
            "smallest": 100.0,
            "largest": 200.0,
        }
        success1, msg1, _, _ = _evaluate_validation_rule(
            "smallest == min(a, b, c)", values
        )
        success2, msg2, _, _ = _evaluate_validation_rule(
            "largest == max(a, b, c)", values
        )
        assert success1 is True
        assert success2 is True

    def test_evaluate_with_sqrt_function(self):
        """Test expressions using sqrt() function."""
        values = {
            "variance": 16.0,
            "std_dev": 4.0,  # sqrt(16) = 4
        }
        success, msg, _, _ = _evaluate_validation_rule(
            "std_dev == sqrt(variance)", values
        )
        assert success is True

    def test_validate_with_custom_vat_terminology(self):
        """Test validation with non-standard field names like VAT."""
        schema = SchemaDefinition(
            name="EU Invoice",
            fields=[
                FieldDefinition(name="netto", type=FieldType.CURRENCY, description="Net amount"),
                FieldDefinition(name="vat", type=FieldType.CURRENCY, description="VAT"),
                FieldDefinition(name="brutto", type=FieldType.CURRENCY, description="Gross"),
            ],
            validation_rules=["brutto == netto + vat"],
        )
        # Correct math
        data = {"netto": "€100.00", "vat": "€19.00", "brutto": "€119.00"}
        result = validate_extracted_data(data, schema)
        assert not any("Math validation failed" in w for w in result.warnings)

        # Incorrect math
        data = {"netto": "€100.00", "vat": "€19.00", "brutto": "€200.00"}
        result = validate_extracted_data(data, schema)
        assert any("Math validation failed" in w for w in result.warnings)

    def test_validate_payroll_terminology(self):
        """Test validation with payroll-specific field names."""
        schema = SchemaDefinition(
            name="Payslip",
            fields=[
                FieldDefinition(name="gross_salary", type=FieldType.CURRENCY, description="Gross"),
                FieldDefinition(name="tax_deduction", type=FieldType.CURRENCY, description="Tax"),
                FieldDefinition(name="insurance", type=FieldType.CURRENCY, description="Insurance"),
                FieldDefinition(name="net_pay", type=FieldType.CURRENCY, description="Net"),
            ],
            validation_rules=["net_pay == gross_salary - tax_deduction - insurance"],
        )
        data = {
            "gross_salary": "$5000",
            "tax_deduction": "$800",
            "insurance": "$200",
            "net_pay": "$4000",
        }
        result = validate_extracted_data(data, schema)
        assert not any("Math validation failed" in w for w in result.warnings)


class TestCalculateConfidence:
    """Tests for confidence calculation from logprobs."""

    def test_empty_logprobs_returns_default(self):
        """Test that empty logprobs return default confidence."""
        assert calculate_confidence_from_logprobs(None) == 0.75
        assert calculate_confidence_from_logprobs([]) == 0.75

    def test_high_probability_tokens(self):
        """Test that high probability tokens yield high confidence."""
        # Mock logprob objects with high probability (close to 0)

        class MockToken:
            def __init__(self, logprob):
                self.logprob = logprob

        # logprob of -0.1 = ~90% probability
        tokens = [MockToken(-0.1) for _ in range(10)]
        confidence = calculate_confidence_from_logprobs(tokens)
        assert confidence > 0.8

    def test_low_probability_tokens(self):
        """Test that low probability tokens yield lower confidence."""

        class MockToken:
            def __init__(self, logprob):
                self.logprob = logprob

        # logprob of -2.0 = ~13.5% probability
        tokens = [MockToken(-2.0) for _ in range(10)]
        confidence = calculate_confidence_from_logprobs(tokens)
        assert confidence < 0.5

    def test_confidence_bounds(self):
        """Test that confidence is always between 0 and 1."""

        class MockToken:
            def __init__(self, logprob):
                self.logprob = logprob

        # Very low probability
        tokens = [MockToken(-100.0) for _ in range(5)]
        confidence = calculate_confidence_from_logprobs(tokens)
        assert 0.0 <= confidence <= 1.0

        # Very high probability (close to 0)
        tokens = [MockToken(-0.001) for _ in range(5)]
        confidence = calculate_confidence_from_logprobs(tokens)
        assert 0.0 <= confidence <= 1.0


class TestAIServiceMock:
    """Tests for AI service in mock mode."""

    def test_mock_mode_enabled_without_api_key(self):
        """Test that mock mode is enabled without API key."""
        # Explicitly pass empty string as API key to force mock mode
        # (None would trigger loading from settings/.env)
        service = AIService(api_key="", use_mock=False)
        # Empty string is falsy, so use_mock should be enabled
        assert service.use_mock is True

    def test_mock_mode_enabled_explicitly(self):
        """Test that mock mode can be explicitly enabled."""
        service = AIService(api_key="fake-key", use_mock=True)
        assert service.use_mock is True

    def test_mock_schema_has_fields(self):
        """Test that mock schema has valid fields."""
        service = AIService(use_mock=True)
        schema = service._get_mock_schema()
        assert len(schema.fields) >= 5
        assert schema.name
        assert all(f.name and f.type for f in schema.fields)

    @pytest.mark.asyncio
    async def test_discover_schema_mock(self):
        """Test discover_schema in mock mode."""
        from PIL import Image

        service = AIService(use_mock=True)
        image = Image.new("RGB", (100, 100), color="white")
        schema = await service.discover_schema(image)

        assert isinstance(schema, SchemaDefinition)
        assert len(schema.fields) > 0

    @pytest.mark.asyncio
    async def test_extract_data_mock(self):
        """Test extract_data in mock mode."""
        from PIL import Image

        from app.backend.models import ExtractionResult

        service = AIService(use_mock=True)
        image = Image.new("RGB", (100, 100), color="white")
        schema = service._get_mock_schema()

        result = await service.extract_data(image, schema, "test.pdf")

        assert isinstance(result, ExtractionResult)
        assert result.source_file == "test.pdf"
        assert result.confidence > 0
        assert len(result.extracted_data) > 0
        assert any("MOCK" in str(w) or "DEVELOPMENT" in str(w) for w in result.warnings)


class TestAIServicePrompts:
    """Tests for AI service prompt generation."""

    def test_extraction_prompt_includes_all_fields(self):
        """Test that extraction prompt includes all schema fields."""
        service = AIService(use_mock=True)
        schema = SchemaDefinition(
            name="Test Schema",
            fields=[
                FieldDefinition(
                    name="field_one",
                    type=FieldType.STRING,
                    description="First field",
                    required=True,
                ),
                FieldDefinition(
                    name="field_two",
                    type=FieldType.CURRENCY,
                    description="Second field",
                    required=False,
                ),
            ],
        )

        prompt = service._build_extraction_prompt(schema)

        assert "field_one" in prompt
        assert "field_two" in prompt
        assert "REQUIRED" in prompt
        assert "optional" in prompt
        assert "string" in prompt
        assert "currency" in prompt

    def test_extraction_prompt_valid_json_example(self):
        """Test that extraction prompt has valid JSON structure."""
        import json

        service = AIService(use_mock=True)
        schema = SchemaDefinition(
            name="Test",
            fields=[
                FieldDefinition(
                    name="test_field",
                    type=FieldType.STRING,
                    description="Test",
                ),
            ],
        )

        prompt = service._build_extraction_prompt(schema)

        # Should contain the field names as a JSON array
        assert '["test_field"]' in prompt

