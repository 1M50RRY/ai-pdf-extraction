import { useState, useCallback, useEffect } from "react";
import {
  flexRender,
} from "@tanstack/react-table";
import {
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { updateExtraction, approveBatch, getBatchStatus, autoCalculateDocument } from "../api";
import type { SchemaDefinition } from "../types";
import { useTableConfig } from "./results/hooks/useTableConfig";
import { TableToolbar } from "./results/TableToolbar";
import { ValidationSummary } from "./results/ValidationSummary";

// Extended extraction result with editing metadata
export interface EditableExtractionResult {
  id: string;  // Extraction ID from database
  document_id: string;
  source_file: string;
  page_number: number;
  extracted_data: Record<string, unknown>;
  confidence: number;
  field_confidences: Record<string, number> | null;  // Per-field confidence for cell highlighting
  warnings: string[];
  is_reviewed: boolean;
  manual_overrides: Record<string, unknown> | null;
}

interface EditableResultsTableProps {
  results: EditableExtractionResult[];
  schema: SchemaDefinition;
  batchId: string;
  onRowClick?: (result: EditableExtractionResult) => void;
  onDataUpdate?: (results: EditableExtractionResult[]) => void;
}

export function EditableResultsTable({
  results,
  schema,
  batchId,
  onRowClick,
  onDataUpdate,
}: EditableResultsTableProps) {
  const [localResults, setLocalResults] = useState(results);
  const [isApproving, setIsApproving] = useState(false);
  const [approvalStatus, setApprovalStatus] = useState<"idle" | "success" | "error">("idle");
  const [isAutoCalculating, setIsAutoCalculating] = useState(false);
  const [calculatingDocumentIds, setCalculatingDocumentIds] = useState<Set<string>>(new Set());

  // Sync local state with incoming results prop (e.g., after refresh from backend)
  useEffect(() => {
    setLocalResults(results);
  }, [results]);

  // Handle cell update - CRITICAL: Must persist to backend and update local state
  const handleCellUpdate = useCallback(
    async (extractionId: string, fieldName: string, value: unknown) => {
      console.log("Saving...", { extractionId, fieldName, value });

      try {
        // Call API to persist the change - MUST await
        await updateExtraction(extractionId, { [fieldName]: value });
        console.log("API update successful, refetching from server to verify persistence");

        // Refetch batch status from server to prove persistence
        try {
          const batchStatus = await getBatchStatus(batchId);
          const updatedDoc = batchStatus.documents.find(
            (d) => d.extraction_id === extractionId
          );
          if (updatedDoc) {
            // Update local state with server data
            setLocalResults((prev) => {
              const updated = prev.map((r) => {
                if (r.id === extractionId) {
                  return {
                    ...r,
                    extracted_data: updatedDoc.extracted_data || r.extracted_data,
                    is_reviewed: updatedDoc.is_reviewed || false,
                    field_confidences: updatedDoc.field_confidences || r.field_confidences,
                  };
                }
                return r;
              });
              onDataUpdate?.(updated);
              return updated;
            });
            console.log("Refetched from server - persistence verified");
          }
        } catch (refetchErr) {
          console.warn("Refetch failed, using local update:", refetchErr);
          // Fallback to local update if refetch fails
          setLocalResults((prev) => {
            const updated = prev.map((r) => {
              if (r.id === extractionId) {
                return {
                  ...r,
                  extracted_data: { ...r.extracted_data, [fieldName]: value },
                  is_reviewed: true,
                  manual_overrides: {
                    ...(r.manual_overrides || {}),
                    [fieldName]: { old: r.extracted_data[fieldName], new: value },
                  },
                };
              }
              return r;
            });
            onDataUpdate?.(updated);
            return updated;
          });
        }
      } catch (err) {
        console.error("Update failed:", err);
        throw err; // Re-throw to let SmartCell handle the error
      }
    },
    [onDataUpdate, batchId]
  );

  // Handle dynamic calculation engine for all rows
  const handleAutoCalculate = async () => {
    setIsAutoCalculating(true);
    const documentIds = localResults.map((r) => r.document_id);
    setCalculatingDocumentIds(new Set(documentIds));

    try {
      // Process all documents in parallel
      const promises = documentIds.map(async (docId) => {
        try {
          const updated = await autoCalculateDocument(docId);
          return { docId, success: true, data: updated };
        } catch (err) {
          console.error(`Auto-calculate failed for document ${docId}:`, err);
          return { docId, success: false, error: err };
        }
      });

      await Promise.all(promises);

      // Refetch batch status to get updated data
      try {
        const batchStatus = await getBatchStatus(batchId);

        // Update local state with calculated values
        const updatedResults = localResults.map((r) => {
          const updatedDoc = batchStatus.documents.find(
            (d) => d.extraction_id === r.id
          );
          if (updatedDoc && updatedDoc.extracted_data) {
            return {
              ...r,
              extracted_data: updatedDoc.extracted_data,
              field_confidences: updatedDoc.field_confidences || r.field_confidences,
            };
          }
          return r;
        });

        setLocalResults(updatedResults);
        onDataUpdate?.(updatedResults);
      } catch (refetchErr) {
        console.warn("Failed to refetch batch status after auto-calculate:", refetchErr);
      }
    } catch (err) {
      console.error("Auto-calculate failed:", err);
    } finally {
      setIsAutoCalculating(false);
      setCalculatingDocumentIds(new Set());
    }
  };

  // Handle approve all
  const handleApproveAll = async () => {
    setIsApproving(true);
    setApprovalStatus("idle");

    try {
      await approveBatch(batchId);
      setApprovalStatus("success");

      // Update local state
      setLocalResults((prev) =>
        prev.map((r) => ({ ...r, is_reviewed: true }))
      );
    } catch (err) {
      console.error("Approval failed:", err);
      setApprovalStatus("error");
    } finally {
      setIsApproving(false);
    }
  };

  // Use the extracted table configuration hook
  const { table } = useTableConfig({
    results: localResults,
    schema,
    onRowClick,
    onCellUpdate: handleCellUpdate,
    calculatingDocumentIds,
  });

  // Helper to format cell value for CSV (handles arrays)
  const formatCellForCSV = (value: unknown): string => {
    if (value === null || value === undefined) {
      return "";
    }
    if (Array.isArray(value)) {
      // Format array as readable string: "Item 1 | Item 2 | Item 3"
      return value
        .map((item) => {
          if (typeof item === "object" && item !== null) {
            // For objects, create a compact representation
            return JSON.stringify(item).replace(/,/g, "; ");
          }
          return String(item);
        })
        .join(" | ");
    }
    if (typeof value === "object") {
      // For non-array objects, stringify
      return JSON.stringify(value);
    }
    return String(value);
  };

  const exportToCSV = () => {
    const headers = [
      "source_file",
      "confidence",
      "warnings",
      "is_reviewed",
      ...schema.fields.map((f) => f.name),
      ...schema.fields.map((f) => `${f.name}_confidence`), // Add confidence columns
    ];
    const rows = localResults.map((result) => [
      result.source_file,
      result.confidence,
      result.warnings.join("; "),
      result.is_reviewed,
      ...schema.fields.map((f) => formatCellForCSV(result.extracted_data[f.name])),
      ...schema.fields.map((f) => {
        const conf = result.field_confidences?.[f.name];
        return conf !== undefined ? String(Math.round(conf * 100)) : "";
      }),
    ]);

    const csv = [
      headers.join(","),
      ...rows.map((row) =>
        row.map((cell) => {
          const cellStr = String(cell);
          // Escape quotes and wrap in quotes
          return `"${cellStr.replace(/"/g, '""')}"`;
        }).join(",")
      ),
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `extraction_results_${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportToJSON = () => {
    const avgConfidence =
      localResults.reduce((sum, r) => sum + r.confidence, 0) / localResults.length;

    const exportData = {
      schema: schema,
      batch_id: batchId,
      exported_at: new Date().toISOString(),
      total_documents: localResults.length,
      average_confidence: avgConfidence,
      results: localResults,
    };

    const json = JSON.stringify(exportData, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `extraction_results_${new Date().toISOString().split("T")[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Calculate stats
  const reviewedCount = localResults.filter((r) => r.is_reviewed).length;

  return (
    <div className="space-y-6">
      {/* Toolbar with Stats and Actions */}
      <TableToolbar
        results={localResults}
        schema={schema}
        batchId={batchId}
        isAutoCalculating={isAutoCalculating}
        isApproving={isApproving}
        approvalStatus={approvalStatus}
        reviewedCount={reviewedCount}
        totalCount={localResults.length}
        onAutoCalculate={handleAutoCalculate}
        onApproveAll={handleApproveAll}
        onExportCSV={exportToCSV}
        onExportJSON={exportToJSON}
      />

      {/* Edit Instructions */}
      <ValidationSummary />

      {/* Table */}
      <div className="bg-slate-800/50 rounded-xl overflow-hidden border border-slate-700">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              {table.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id} className="bg-slate-800">
                  {headerGroup.headers.map((header) => (
                    <th
                      key={header.id}
                      className="text-left px-4 py-3 text-sm font-medium text-slate-400 cursor-pointer hover:text-slate-200 transition-colors"
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      <div className="flex items-center gap-1">
                        {flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                        {header.column.getIsSorted() === "asc" && (
                          <ChevronUp className="w-4 h-4" />
                        )}
                        {header.column.getIsSorted() === "desc" && (
                          <ChevronDown className="w-4 h-4" />
                        )}
                      </div>
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {table.getRowModel().rows.map((row) => (
                <tr
                  key={row.id}
                  className="hover:bg-slate-700/30 transition-colors"
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="px-4 py-3">
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
