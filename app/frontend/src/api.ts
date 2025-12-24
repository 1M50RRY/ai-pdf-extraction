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

/**
 * Extract data from PDFs using a confirmed schema
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
      "/extract-batch",
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

