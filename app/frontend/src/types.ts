// Types matching the backend Pydantic models

export type FieldType =
  | "string"
  | "currency"
  | "date"
  | "number"
  | "boolean"
  | "email"
  | "phone"
  | "address"
  | "percentage";

export interface FieldDefinition {
  name: string;
  type: FieldType;
  description: string;
  required: boolean;
}

export interface SchemaDefinition {
  name: string;
  description: string;
  fields: FieldDefinition[];
  version: string;
  validation_rules: string[];
}

export interface ExtractionResult {
  source_file: string;
  detected_schema: SchemaDefinition;
  extracted_data: Record<string, unknown>;
  confidence: number;
  warnings: string[];
}

export interface UploadSampleResponse {
  message: string;
  suggested_schema: SchemaDefinition;
  preview_available: boolean;
  page_count: number;
}

export interface ExtractBatchResponse {
  results: ExtractionResult[];
  total_pages: number;
  successful_extractions: number;
  average_confidence: number;
}

// UI State types
export type AppStep = "upload" | "schema" | "batch" | "results";

export interface BatchProgress {
  current: number;
  total: number;
  status: "idle" | "processing" | "complete" | "error";
  message?: string;
}

