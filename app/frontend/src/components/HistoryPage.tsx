import { useEffect, useState, useCallback } from "react";
import {
  ArrowLeft,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  FileText,
  ChevronRight,
  History,
} from "lucide-react";
import { getBatchHistory, getBatchStatus, getSchema, type BatchSummary } from "../api";
import { EditableResultsTable, type EditableExtractionResult } from "./EditableResultsTable";
import type { SchemaDefinition } from "../types";

interface HistoryPageProps {
  onBack: () => void;
}

function formatDate(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function StatusBadge({ status }: { status: string }) {
  const config = {
    completed: {
      icon: CheckCircle,
      bg: "bg-emerald-500/20",
      text: "text-emerald-400",
      label: "Completed",
    },
    processing: {
      icon: Loader2,
      bg: "bg-indigo-500/20",
      text: "text-indigo-400",
      label: "Processing",
      animate: true,
    },
    pending: {
      icon: Clock,
      bg: "bg-amber-500/20",
      text: "text-amber-400",
      label: "Pending",
    },
    failed: {
      icon: XCircle,
      bg: "bg-red-500/20",
      text: "text-red-400",
      label: "Failed",
    },
  }[status] || {
    icon: Clock,
    bg: "bg-slate-500/20",
    text: "text-slate-400",
    label: status,
  };

  const Icon = config.icon;

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${config.bg} ${config.text}`}
    >
      <Icon className={`w-3.5 h-3.5 ${config.animate ? "animate-spin" : ""}`} />
      {config.label}
    </span>
  );
}

export function HistoryPage({ onBack }: HistoryPageProps) {
  const [batches, setBatches] = useState<BatchSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Selected batch state
  const [selectedBatch, setSelectedBatch] = useState<BatchSummary | null>(null);
  const [selectedSchema, setSelectedSchema] = useState<SchemaDefinition | null>(null);
  const [selectedResults, setSelectedResults] = useState<EditableExtractionResult[]>([]);
  const [loadingBatch, setLoadingBatch] = useState(false);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getBatchHistory(100, 0);
      setBatches(response.batches);
    } catch (err) {
      console.error("Failed to fetch history:", err);
      setError("Failed to load batch history");
    } finally {
      setLoading(false);
    }
  };

  const handleSelectBatch = useCallback(async (batch: BatchSummary) => {
    setLoadingBatch(true);
    setError(null);

    try {
      // Get batch status with full document data
      const batchStatus = await getBatchStatus(batch.id);
      
      // Load schema if available
      let schema: SchemaDefinition | null = null;
      if (batch.schema_id) {
        try {
          const savedSchema = await getSchema(batch.schema_id);
          schema = savedSchema.structure;
        } catch (schemaErr) {
          console.warn("Could not load schema:", schemaErr);
        }
      }

      // Convert to editable results
      const editableResults: EditableExtractionResult[] = batchStatus.documents
        .filter((doc) => doc.status === "completed" && doc.extracted_data)
        .map((doc) => ({
          id: doc.extraction_id || doc.id,
          document_id: doc.id,
          source_file: doc.filename,
          page_number: 1,
          extracted_data: doc.extracted_data || {},
          confidence: doc.confidence || 0,
          field_confidences: doc.field_confidences,
          warnings: doc.warnings,
          is_reviewed: false,
          manual_overrides: null,
        }));

      setSelectedBatch(batch);
      setSelectedSchema(schema);
      setSelectedResults(editableResults);
    } catch (err) {
      console.error("Failed to load batch:", err);
      setError("Failed to load batch details");
    } finally {
      setLoadingBatch(false);
    }
  }, []);

  const handleBackToList = () => {
    setSelectedBatch(null);
    setSelectedSchema(null);
    setSelectedResults([]);
  };

  // If viewing a specific batch
  if (selectedBatch && selectedSchema) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <button
            onClick={handleBackToList}
            className="flex items-center gap-2 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to History
          </button>
          <div className="flex items-center gap-3">
            <StatusBadge status={selectedBatch.status} />
            <span className="text-sm text-slate-400">
              {formatDate(selectedBatch.created_at)}
            </span>
          </div>
        </div>

        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
          <h3 className="text-lg font-medium text-slate-200">
            {selectedBatch.schema_name || "Untitled Batch"}
          </h3>
          <p className="text-sm text-slate-400 mt-1">
            {selectedBatch.total_documents} documents • {selectedBatch.successful_documents} successful
            {selectedBatch.failed_documents > 0 && (
              <span className="text-red-400"> • {selectedBatch.failed_documents} failed</span>
            )}
          </p>
        </div>

        <EditableResultsTable
          results={selectedResults}
          schema={selectedSchema}
          batchId={selectedBatch.id}
        />
      </div>
    );
  }

  // History list view
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-slate-400 hover:text-slate-200 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Extraction
        </button>
        <button
          onClick={fetchHistory}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-2 text-slate-400 hover:text-slate-200 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <History className="w-4 h-4" />
          )}
          Refresh
        </button>
      </div>

      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-slate-100">
          Extraction History
        </h2>
        <p className="text-slate-400 mt-2">
          View and manage past extraction batches
        </p>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-12">
          <Loader2 className="w-10 h-10 text-indigo-400 animate-spin mb-4" />
          <p className="text-slate-400">Loading history...</p>
        </div>
      ) : error ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <XCircle className="w-10 h-10 text-red-400 mb-4" />
          <p className="text-red-300 font-medium">{error}</p>
          <button
            onClick={fetchHistory}
            className="mt-4 text-indigo-400 hover:text-indigo-300 text-sm"
          >
            Try again
          </button>
        </div>
      ) : batches.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <FileText className="w-12 h-12 text-slate-600 mb-4" />
          <p className="text-slate-400 font-medium">No extraction history yet</p>
          <p className="text-sm text-slate-500 mt-1">
            Your batch extractions will appear here
          </p>
        </div>
      ) : (
        <div className="grid gap-4">
          {batches.map((batch) => (
            <button
              key={batch.id}
              onClick={() => handleSelectBatch(batch)}
              disabled={loadingBatch}
              className="w-full flex items-center justify-between p-5 bg-slate-800/50 hover:bg-slate-800 border border-slate-700 hover:border-indigo-500/50 rounded-xl transition-all group text-left"
            >
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center">
                  <FileText className="w-7 h-7 text-slate-400 group-hover:text-indigo-400 transition-colors" />
                </div>
                <div>
                  <div className="flex items-center gap-3">
                    <span className="font-semibold text-slate-200 group-hover:text-white transition-colors">
                      {batch.schema_name || "Untitled Batch"}
                    </span>
                    <StatusBadge status={batch.status} />
                  </div>
                  <div className="flex items-center gap-4 mt-1.5 text-sm text-slate-400">
                    <span className="flex items-center gap-1">
                      <FileText className="w-4 h-4" />
                      {batch.total_documents} documents
                    </span>
                    <span>•</span>
                    <span>
                      <span className="text-emerald-400">{batch.successful_documents} success</span>
                      {batch.failed_documents > 0 && (
                        <span className="text-red-400 ml-2">
                          {batch.failed_documents} failed
                        </span>
                      )}
                    </span>
                    <span>•</span>
                    <span>{formatDate(batch.created_at)}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {loadingBatch && (
                  <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
                )}
                <ChevronRight className="w-5 h-5 text-slate-500 group-hover:text-indigo-400 transition-colors" />
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

