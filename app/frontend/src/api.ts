import axios from "axios";
import type {
  ExtractBatchResponse,
  SchemaDefinition,
  UploadSampleResponse,
} from "./types";

// API base URL - uses Vite proxy in development
const API_BASE = "/api";

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    Accept: "application/json",
  },
});

// =============================================================================
// Types for new endpoints
// =============================================================================

export interface SavedSchema {
  id: string;
  name: string;
  description: string;
  version: string;
  structure: SchemaDefinition;
  created_at: string;
  is_active: boolean;
}

export interface SchemaListResponse {
  schemas: SavedSchema[];
  total: number;
}

export interface StartBatchResponse {
  batch_id: string;
  message: string;
  total_documents: number;
  status: string;
}

export interface DocumentStatus {
  id: string;
  filename: string;
  status: string;
  confidence: number | null;
  error_message: string | null;
  warnings: string[];
  extraction_id: string | null;
  extracted_data: Record<string, unknown> | null;
  field_confidences: Record<string, number> | null;
  is_reviewed: boolean;
}

export interface BatchStatusResponse {
  id: string;
  status: string;
  created_at: string;
  completed_at: string | null;
  total_documents: number;
  completed_documents: number;
  failed_documents: number;
  progress_percent: number;
  documents: DocumentStatus[];
  schema_id: string | null;
  schema_name: string | null;
}

export interface ExtractionDetail {
  id: string;
  document_id: string;
  page_number: number;
  data: Record<string, unknown>;
  confidence: number;
  warnings: string[];
  is_reviewed: boolean;
  manual_overrides: Record<string, unknown> | null;
  created_at: string;
  reviewed_at: string | null;
}

export interface ApproveResponse {
  message: string;
  approved_count: number;
  batch_id: string;
}

// =============================================================================
// Schema Registry Endpoints
// =============================================================================

/**
 * Save a schema as a reusable template
 */
export async function saveSchema(schema: SchemaDefinition): Promise<SavedSchema> {
  const response = await api.post<SavedSchema>("/schemas", {
    schema: schema,
  });
  return response.data;
}

/**
 * List all saved schema templates
 */
export async function listSchemas(): Promise<SchemaListResponse> {
  const response = await api.get<SchemaListResponse>("/schemas");
  return response.data;
}

/**
 * Get a specific schema by ID
 */
export async function getSchema(schemaId: string): Promise<SavedSchema> {
  const response = await api.get<SavedSchema>(`/schemas/${schemaId}`);
  return response.data;
}

/**
 * Delete (deactivate) a schema
 */
export async function deleteSchema(schemaId: string): Promise<void> {
  await api.delete(`/schemas/${schemaId}`);
}

// =============================================================================
// Sample Upload
// =============================================================================

/**
 * Upload a sample PDF for schema detection
 */
export async function uploadSample(file: File): Promise<UploadSampleResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post<UploadSampleResponse>(
    "/upload-sample",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );

  return response.data;
}

// =============================================================================
// Async Batch Processing
// =============================================================================

/**
 * Start async batch extraction of multiple PDFs
 * Returns a batch ID for status polling
 * 
 * IMPORTANT: Always sends confirmed_schema (user's current/edited schema).
 * The schemaId is only for tracking purposes - the confirmed_schema is the source of truth.
 */
export async function startBatchExtraction(
  files: File[],
  schema: SchemaDefinition,
  schemaId?: string
): Promise<StartBatchResponse> {
  const formData = new FormData();
  
  // Add all files
  files.forEach((file) => {
    formData.append("files", file);
  });

  // ALWAYS send the schema from the request - this is the user's potentially edited version
  // The confirmed_schema takes priority on the backend (user edits win over saved templates)
  formData.append("confirmed_schema", JSON.stringify(schema));
  console.log("DEBUG: Sending schema to backend with fields:", schema.fields.map(f => f.name));
  
  // Also send schema_id for tracking/linking purposes (but confirmed_schema is source of truth)
  if (schemaId) {
    formData.append("schema_id", schemaId);
  }

  const response = await api.post<StartBatchResponse>(
    "/extract-batch",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );

  return response.data;
}

/**
 * Get batch processing status
 */
export async function getBatchStatus(batchId: string): Promise<BatchStatusResponse> {
  const response = await api.get<BatchStatusResponse>(`/batches/${batchId}/status`);
  return response.data;
}

/**
 * Get batch extraction results
 */
export async function getBatchResults(batchId: string): Promise<ExtractBatchResponse> {
  const response = await api.get<ExtractBatchResponse>(`/batches/${batchId}/results`);
  return response.data;
}

/**
 * Get document PDF content for preview
 */
export function getDocumentContentUrl(documentId: string): string {
  return `${API_BASE}/documents/${documentId}/content`;
}

/**
 * Poll for batch completion
 */
export async function pollBatchUntilComplete(
  batchId: string,
  onProgress?: (status: BatchStatusResponse) => void,
  pollInterval: number = 1000,
  maxAttempts: number = 300
): Promise<BatchStatusResponse> {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const status = await getBatchStatus(batchId);
    onProgress?.(status);

    if (status.status === "completed" || status.progress_percent >= 100) {
      return status;
    }

    // Wait before next poll
    await new Promise((resolve) => setTimeout(resolve, pollInterval));
  }

  throw new Error("Batch processing timed out");
}

// =============================================================================
// Legacy Sync Batch Processing (backwards compatibility)
// =============================================================================

/**
 * Extract data from PDFs using a confirmed schema (synchronous, legacy)
 */
export async function extractBatch(
  files: File[],
  schema: SchemaDefinition,
  onProgress?: (current: number, total: number) => void
): Promise<ExtractBatchResponse[]> {
  const results: ExtractBatchResponse[] = [];

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const formData = new FormData();
    formData.append("file", file);
    formData.append("confirmed_schema", JSON.stringify(schema));

    const response = await api.post<ExtractBatchResponse>(
      "/extract-batch-sync",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    results.push(response.data);
    onProgress?.(i + 1, files.length);
  }

  return results;
}

// =============================================================================
// Human-in-the-Loop Endpoints
// =============================================================================

/**
 * Get extraction details
 */
export async function getExtraction(extractionId: string): Promise<ExtractionDetail> {
  const response = await api.get<ExtractionDetail>(`/extractions/${extractionId}`);
  return response.data;
}

/**
 * Update extraction data (human correction)
 */
export async function updateExtraction(
  extractionId: string,
  data: Record<string, unknown>
): Promise<ExtractionDetail> {
  const response = await api.patch<ExtractionDetail>(`/extractions/${extractionId}`, {
    data,
  });
  return response.data;
}

/**
 * Approve all extractions in a batch
 */
export async function approveBatch(
  batchId: string,
  reviewedBy?: string
): Promise<ApproveResponse> {
  const response = await api.post<ApproveResponse>(`/batches/${batchId}/approve`, {
    reviewed_by: reviewedBy,
  });
  return response.data;
}

// =============================================================================
// Health Check
// =============================================================================

/**
 * Health check
 */
export async function healthCheck(): Promise<boolean> {
  try {
    const response = await api.get("/health");
    return response.data.status === "healthy";
  } catch {
    return false;
  }
}

// =============================================================================
// Batch History
// =============================================================================

export interface BatchExtractionSummary {
  id: string;
  document_id: string;
  filename: string;
  extracted_data: Record<string, unknown>;
  confidence: number;
  field_confidences: Record<string, number>;
  warnings: string[];
  is_reviewed: boolean;
}

export interface BatchSummary {
  id: string;
  schema_name: string | null;
  schema_id: string | null;
  created_at: string;
  completed_at: string | null;
  total_documents: number;
  successful_documents: number;
  failed_documents: number;
  status: string;
  extractions: BatchExtractionSummary[];
}

export interface BatchHistoryResponse {
  batches: BatchSummary[];
  total: number;
}

/**
 * Get batch processing history
 */
export async function getBatchHistory(
  limit: number = 50,
  offset: number = 0
): Promise<BatchHistoryResponse> {
  const response = await api.get<BatchHistoryResponse>(
    `/batches?limit=${limit}&offset=${offset}`
  );
  return response.data;
}

export default api;
