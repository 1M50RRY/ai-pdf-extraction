import { useState, useMemo, useCallback, useEffect } from "react";
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
  type ColumnDef,
} from "@tanstack/react-table";
import {
  AlertTriangle,
  Check,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Download,
  Edit3,
  Eye,
  FileText,
  Info,
  Loader2,
  ChevronRight,
} from "lucide-react";
import { updateExtraction, approveBatch } from "../api";
import type { SchemaDefinition } from "../types";
import { SmartCell } from "./SmartCell";

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

function ConfidenceBadge({ confidence }: { confidence: number }) {
  const percentage = Math.round(confidence * 100);

  let bgColor = "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
  if (confidence < 0.5) {
    bgColor = "bg-red-500/20 text-red-300 border-red-500/30";
  } else if (confidence < 0.8) {
    bgColor = "bg-amber-500/20 text-amber-300 border-amber-500/30";
  }

  return (
    <span
      className={`inline-block px-2 py-0.5 rounded text-xs font-medium border ${bgColor}`}
    >
      {percentage}%
    </span>
  );
}

function WarningsTooltip({ warnings }: { warnings: string[] }) {
  const [isOpen, setIsOpen] = useState(false);

  if (warnings.length === 0) return null;

  return (
    <div className="relative inline-block">
      <button
        onMouseEnter={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
        className="p-1 text-amber-400 hover:bg-amber-400/20 rounded transition-colors"
      >
        <AlertTriangle className="w-4 h-4" />
      </button>

      {isOpen && (
        <div className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-72">
          <div className="bg-slate-800 border border-slate-600 rounded-lg shadow-xl p-3">
            <div className="flex items-center gap-2 mb-2">
              <Info className="w-4 h-4 text-amber-400" />
              <span className="text-sm font-medium text-amber-300">
                {warnings.length} Warning(s)
              </span>
            </div>
            <ul className="space-y-1 text-xs text-slate-300">
              {warnings.map((warning, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-slate-500">•</span>
                  <span>{warning}</span>
                </li>
              ))}
            </ul>
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-full">
              <div className="border-8 border-transparent border-t-slate-600" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

interface EditableCellProps {
  value: unknown;
  extractionId: string;
  fieldName: string;
  hasOverride: boolean;
  confidence: number;
  onUpdate: (extractionId: string, fieldName: string, value: unknown) => Promise<void>;
}

// Component to render array values
function ArrayCell({ value }: { value: unknown[] }) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!Array.isArray(value) || value.length === 0) {
    return <span className="text-sm text-slate-400">Empty array</span>;
  }

  return (
    <div className="max-w-[300px]">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1 text-sm text-indigo-400 hover:text-indigo-300 transition-colors"
      >
        <ChevronRight
          className={`w-3 h-3 transition-transform ${isExpanded ? "rotate-90" : ""}`}
        />
        <span>{value.length} item{value.length !== 1 ? "s" : ""}</span>
      </button>
      {isExpanded && (
        <div className="mt-2 ml-4 space-y-2 max-h-48 overflow-y-auto">
          {value.map((item, idx) => (
            <div
              key={idx}
              className="bg-slate-700/50 rounded p-2 text-xs font-mono text-slate-300"
            >
              <pre className="whitespace-pre-wrap break-words">
                {JSON.stringify(item, null, 2)}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function EditableCell({
  value,
  extractionId,
  fieldName,
  hasOverride,
  confidence,
  onUpdate,
}: EditableCellProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(
    Array.isArray(value) ? JSON.stringify(value) : String(value ?? "")
  );
  const [isSaving, setIsSaving] = useState(false);

  // Debug logging
  console.log("EditableCell - Row Data:", { extractionId, fieldName, value, confidence });

  const handleBlur = async () => {
    // Parse the value appropriately
    let parsedValue: unknown = editValue;

    // Try to parse as JSON if it looks like JSON
    if (editValue.trim().startsWith("{") || editValue.trim().startsWith("[")) {
      try {
        parsedValue = JSON.parse(editValue);
      } catch {
        // If parsing fails, keep as string
        parsedValue = editValue;
      }
    }

    // Only update if value actually changed
    const currentValueStr = Array.isArray(value)
      ? JSON.stringify(value)
      : String(value ?? "");

    if (editValue !== currentValueStr) {
      setIsSaving(true);
      try {
        // Call the API to persist the change
        await onUpdate(extractionId, fieldName, parsedValue);
        console.log("Successfully updated field:", fieldName, "to:", parsedValue);
      } catch (err) {
        console.error("Failed to update:", err);
        setEditValue(currentValueStr); // Revert on error
      } finally {
        setIsSaving(false);
      }
    }
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleBlur();
    } else if (e.key === "Escape") {
      setEditValue(
        Array.isArray(value) ? JSON.stringify(value) : String(value ?? "")
      );
      setIsEditing(false);
    }
  };

  // Confidence-based cell coloring with background
  let cellClass = "text-sm text-slate-300";
  let bgClass = "";
  if (confidence >= 0.9) {
    cellClass = "text-sm text-emerald-300";
  } else if (confidence < 0.5) {
    cellClass = "text-sm text-red-400 font-medium";
    bgClass = "bg-red-500/10";
  } else if (confidence < 0.7) {
    cellClass = "text-sm text-amber-400";
    bgClass = "bg-amber-500/10";
  } else if (confidence < 0.8) {
    cellClass = "text-sm text-amber-300";
    bgClass = "bg-amber-500/5";
  }

  // Handle array values - show special renderer
  if (Array.isArray(value) && !isEditing) {
    return (
      <div className={`relative ${bgClass} px-2 py-1 rounded`}>
        <ArrayCell value={value} />
        {confidence < 0.8 && (
          <span
            className="absolute top-0 right-0 text-xs text-slate-500"
            title={`Confidence: ${Math.round(confidence * 100)}%`}
          >
            {Math.round(confidence * 100)}%
          </span>
        )}
      </div>
    );
  }

  if (isEditing) {
    return (
      <div className="relative">
        <textarea
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          autoFocus
          disabled={isSaving}
          rows={Array.isArray(value) ? 4 : 1}
          className="w-full bg-slate-700 border-2 border-indigo-500 rounded px-2 py-1 text-sm text-slate-200 focus:outline-none font-mono"
        />
        {isSaving && (
          <div className="absolute right-2 top-1/2 -translate-y-1/2">
            <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      className={`relative group cursor-text ${cellClass} ${bgClass} max-w-[200px] whitespace-nowrap overflow-x-auto scrollbar-hide px-2 py-1 rounded`}
      onClick={() => setIsEditing(true)}
      title={`Confidence: ${Math.round(confidence * 100)}%`}
    >
      {/* Override indicator */}
      {hasOverride && (
        <div
          className="absolute -left-1 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-indigo-500 z-10"
          title="Manually edited"
        />
      )}

      <div className="flex items-center gap-1">
        <span className={hasOverride ? "pl-2" : ""}>
          {value !== null && value !== undefined ? String(value) : "—"}
        </span>
        <Edit3 className="w-3 h-3 text-slate-500 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" />
      </div>

      {/* Confidence tooltip indicator */}
      {confidence < 0.8 && (
        <span className="absolute top-0 right-0 text-xs text-slate-500 opacity-0 group-hover:opacity-100">
          {Math.round(confidence * 100)}%
        </span>
      )}
    </div>
  );
}

export function EditableResultsTable({
  results,
  schema,
  batchId,
  onRowClick,
  onDataUpdate,
}: EditableResultsTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [localResults, setLocalResults] = useState(results);
  const [isApproving, setIsApproving] = useState(false);
  const [approvalStatus, setApprovalStatus] = useState<"idle" | "success" | "error">("idle");

  // Sync local state with incoming results prop (e.g., after refresh from backend)
  useEffect(() => {
    setLocalResults(results);
  }, [results]);

  // Handle cell update - CRITICAL: Must persist to backend and update local state
  const handleCellUpdate = useCallback(
    async (extractionId: string, fieldName: string, value: unknown) => {
      console.log("handleCellUpdate called:", { extractionId, fieldName, value });

      try {
        // Call API to persist the change
        await updateExtraction(extractionId, { [fieldName]: value });
        console.log("API update successful, updating local state");

        // Update local state IMMEDIATELY after successful API call
        setLocalResults((prev) => {
          const updated = prev.map((r) => {
            if (r.id === extractionId) {
              const updatedResult = {
                ...r,
                extracted_data: { ...r.extracted_data, [fieldName]: value },
                is_reviewed: true,
                manual_overrides: {
                  ...(r.manual_overrides || {}),
                  [fieldName]: { old: r.extracted_data[fieldName], new: value },
                },
              };
              console.log("Updated result:", updatedResult);
              return updatedResult;
            }
            return r;
          });

          // Notify parent component of the update
          onDataUpdate?.(updated);
          return updated;
        });
      } catch (err) {
        console.error("Update failed:", err);
        throw err; // Re-throw to let EditableCell handle the error
      }
    },
    [onDataUpdate]
  );

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

  const columns = useMemo((): ColumnDef<EditableExtractionResult, unknown>[] => {
    const cols: ColumnDef<EditableExtractionResult, unknown>[] = [
      {
        id: "actions",
        header: "",
        cell: (info) => (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onRowClick?.(info.row.original);
            }}
            className="p-1.5 text-slate-400 hover:text-indigo-400 hover:bg-indigo-500/20 rounded-lg transition-colors"
            title="View details"
          >
            <Eye className="w-4 h-4" />
          </button>
        ),
      },
      {
        accessorKey: "source_file",
        header: "Source File",
        cell: (info) => (
          <div className="flex items-center gap-2 max-w-[180px] whitespace-nowrap overflow-x-auto scrollbar-hide">
            <FileText className="w-4 h-4 text-slate-500 flex-shrink-0" />
            <span className="text-sm font-mono text-slate-300 truncate">
              {info.getValue() as string}
            </span>
            {info.row.original.is_reviewed && (
              <span title="Reviewed" className="flex-shrink-0">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
              </span>
            )}
          </div>
        ),
      },
      {
        accessorKey: "confidence",
        header: "Confidence",
        cell: (info) => <ConfidenceBadge confidence={info.getValue() as number} />,
      },
      {
        accessorKey: "warnings",
        header: "Status",
        cell: (info) => {
          const warnings = info.getValue() as string[];
          return warnings.length > 0 ? (
            <WarningsTooltip warnings={warnings} />
          ) : (
            <span className="text-emerald-400 text-xs">✓ OK</span>
          );
        },
      },
    ];

    // Add editable columns for each field in schema
    schema.fields.forEach((field) => {
      cols.push({
        id: field.name,
        accessorFn: (row) => row.extracted_data[field.name],
        header: field.name.replace(/_/g, " ").toUpperCase(),
        cell: (info) => {
          const value = info.getValue();
          const row = info.row.original;

          // Debug logging
          console.log("Row Data:", {
            id: row.id,
            field: field.name,
            value,
            field_confidences: row.field_confidences,
            confidence: row.confidence,
          });

          const hasOverride = row.manual_overrides?.[field.name] !== undefined;
          // Use per-field confidence if available, else fall back to global
          const fieldConfidence = row.field_confidences?.[field.name] ?? row.confidence;

          // For arrays, use SmartCell for display (non-editable for now)
          if (Array.isArray(value)) {
            return (
              <div className="relative">
                <SmartCell
                  value={value}
                  confidence={fieldConfidence}
                  fieldName={field.name}
                />
                {hasOverride && (
                  <div
                    className="absolute -left-1 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-indigo-500 z-10"
                    title="Manually edited"
                  />
                )}
              </div>
            );
          }

          return (
            <EditableCell
              value={value}
              extractionId={row.id}
              fieldName={field.name}
              hasOverride={hasOverride}
              confidence={fieldConfidence}
              onUpdate={handleCellUpdate}
            />
          );
        },
      });
    });

    return cols;
  }, [schema.fields, handleCellUpdate, onRowClick]);

  const table = useReactTable({
    data: localResults,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
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
  const avgConfidence =
    localResults.reduce((sum, r) => sum + r.confidence, 0) / localResults.length;
  const warningCount = localResults.filter((r) => r.warnings.length > 0).length;
  const reviewedCount = localResults.filter((r) => r.is_reviewed).length;

  return (
    <div className="space-y-6">
      {/* Stats Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div>
            <p className="text-sm text-slate-400">Total Extractions</p>
            <p className="text-2xl font-bold text-slate-100">
              {localResults.length}
            </p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Avg. Confidence</p>
            <p className="text-2xl font-bold text-slate-100">
              {Math.round(avgConfidence * 100)}%
            </p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Reviewed</p>
            <p
              className={`text-2xl font-bold ${reviewedCount === localResults.length
                ? "text-emerald-400"
                : "text-amber-400"
                }`}
            >
              {reviewedCount}/{localResults.length}
            </p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Warnings</p>
            <p
              className={`text-2xl font-bold ${warningCount > 0 ? "text-amber-400" : "text-emerald-400"}`}
            >
              {warningCount}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Approve All Button */}
          <button
            onClick={handleApproveAll}
            disabled={isApproving || approvalStatus === "success"}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${approvalStatus === "success"
              ? "bg-emerald-500/20 text-emerald-400 cursor-default"
              : "bg-emerald-600 hover:bg-emerald-500 text-white disabled:bg-slate-600"
              }`}
          >
            {isApproving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : approvalStatus === "success" ? (
              <CheckCircle className="w-4 h-4" />
            ) : (
              <Check className="w-4 h-4" />
            )}
            {approvalStatus === "success" ? "All Approved" : "Approve All"}
          </button>

          <button
            onClick={exportToCSV}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors font-medium"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
          <button
            onClick={exportToJSON}
            className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
          >
            <Download className="w-4 h-4" />
            Export JSON
          </button>
        </div>
      </div>

      {/* Edit Instructions */}
      <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-lg p-3 flex items-center gap-3">
        <Edit3 className="w-5 h-5 text-indigo-400" />
        <p className="text-sm text-indigo-200">
          <strong>Click any cell</strong> to edit. Changes are saved automatically.
          <span className="ml-2 inline-flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-indigo-500" />
            indicates manually edited cells.
          </span>
        </p>
      </div>

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

