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

  // Add schema (either by ID or inline)
  if (schemaId) {
    formData.append("schema_id", schemaId);
  } else {
    formData.append("confirmed_schema", JSON.stringify(schema));
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

export default api;
