"""
Data extraction functionality for extracting structured data from document images.

Uses OpenAI GPT-4.1 with per-field confidence scoring and chunked processing for large documents.
"""

import base64
import io
import json
import logging
import math
from typing import Any, Callable

from PIL import Image

# Handle both package imports and standalone imports
try:
    from ...models import ExtractionResult, FieldType, SchemaDefinition
except ImportError:
    from models import ExtractionResult, FieldType, SchemaDefinition

from .exceptions import AIServiceError
from .validation import _clean_null_from_arrays, validate_extracted_data

logger = logging.getLogger(__name__)


# =============================================================================
# Extraction System Prompt
# =============================================================================

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


def calculate_confidence_from_logprobs(logprobs_data: list[Any] | None) -> float:
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


def _build_extraction_prompt(schema: SchemaDefinition) -> str:  # Exported for tests
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


# =============================================================================
# Main Extraction Functions
# =============================================================================


async def _extract_from_images(
    images: list[Image.Image],
    schema: SchemaDefinition,
    source_file: str,
    client: Any,  # OpenAI client
    model: str = "gpt-4.1",
) -> ExtractionResult:
    """Extract data from a list of images (single request)."""
    # Convert all images to base64
    base64_images = [_image_to_base64(img) for img in images]

    # Build dynamic extraction prompt (rebuilt fresh for each extraction - NO CACHING)
    extraction_prompt = _build_extraction_prompt(schema)
    
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
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
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


def _merge_extraction_results(
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


async def extract_data(
    images: list[Image.Image] | Image.Image,
    schema: SchemaDefinition,
    source_file: str,
    client: Any,  # OpenAI client
    model: str = "gpt-4.1",
    use_mock: bool = False,
    get_mock_extraction: Callable[[SchemaDefinition, str], ExtractionResult] | None = None,
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
        client: OpenAI client instance.
        model: Model name to use.
        use_mock: If True, return mock extraction instead of calling OpenAI.
        get_mock_extraction: Function to get mock extraction (for testing).

    Returns:
        ExtractionResult with extracted data and confidence score.
    """
    if use_mock and get_mock_extraction:
        logger.info("Extracting data (MOCK MODE) for: %s", source_file)
        return get_mock_extraction(schema, source_file)

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
        raw_result = await _extract_from_images(images, schema, source_file, client, model)
        
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
        return await _extract_from_images_chunked(images, schema, source_file, client, model)


async def _extract_from_images_chunked(
    images: list[Image.Image],
    schema: SchemaDefinition,
    source_file: str,
    client: Any,  # OpenAI client
    model: str = "gpt-4.1",
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
        
        chunk_result = await _extract_from_images(chunk, schema, source_file, client, model)
        all_results.append(chunk_result)
    
    # Merge results
    return _merge_extraction_results(all_results, schema, source_file)

