"""
Data repair and calculation functionality using LLM.

Dynamic Calculation Engine that completes missing values and enforces logical consistency.
"""

import json
import logging
from typing import Any

# Handle both package imports and standalone imports
try:
    from ...models import FieldDefinition, FieldType, SchemaDefinition
except ImportError:
    from models import FieldDefinition, FieldType, SchemaDefinition

from .exceptions import AIServiceError

logger = logging.getLogger(__name__)


# =============================================================================
# Repair System Prompt
# =============================================================================

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


# =============================================================================
# Main Repair Function
# =============================================================================


async def repair_data_with_llm(
    extracted_data: dict[str, Any],
    client: Any,  # OpenAI client
    model: str = "gpt-4.1",
    schema: SchemaDefinition | None = None,
    use_mock: bool = False,
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
        client: OpenAI client instance.
        model: Model name to use.
        schema: The full schema definition with field descriptions (REQUIRED for calculation).
        use_mock: If True, return original data without calling OpenAI.
            
    Returns:
        Fully populated and calculated data dictionary.
    
    Raises:
        AIServiceError: If the repair operation fails.
    """
    if use_mock:
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
        
        response = client.chat.completions.create(
            model=model,
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

