import { useState, useMemo } from "react";
import {
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
  type ColumnDef,
  type Table,
} from "@tanstack/react-table";
import {
  Eye,
  FileText,
  CheckCircle,
} from "lucide-react";
import type { SchemaDefinition } from "../../../types";
import type { EditableExtractionResult } from "../../EditableResultsTable";
import { SmartCell } from "../../SmartCell";

interface UseTableConfigProps {
  results: EditableExtractionResult[];
  schema: SchemaDefinition;
  onRowClick?: (result: EditableExtractionResult) => void;
  onCellUpdate: (extractionId: string, fieldName: string, value: unknown) => Promise<void>;
  calculatingDocumentIds: Set<string>;
}

export function useTableConfig({
  results,
  schema,
  onRowClick,
  onCellUpdate,
  calculatingDocumentIds,
}: UseTableConfigProps): {
  table: Table<EditableExtractionResult>;
  sorting: SortingState;
  setSorting: (sorting: SortingState) => void;
} {
  const [sorting, setSorting] = useState<SortingState>([]);

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
        cell: (info) => {
          const confidence = info.getValue() as number;
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
        },
      },
      {
        accessorKey: "warnings",
        header: "Status",
        cell: (info) => {
          const warnings = info.getValue() as string[];
          return warnings.length > 0 ? (
            <div className="relative inline-block">
              <button className="p-1 text-amber-400 hover:bg-amber-400/20 rounded transition-colors">
                <span className="text-xs">⚠ {warnings.length}</span>
              </button>
            </div>
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

          const hasOverride = row.manual_overrides?.[field.name] !== undefined;
          const overrideData = row.manual_overrides?.[field.name];
          const isAutoCalculated = Boolean(
            overrideData &&
            typeof overrideData === "object" &&
            overrideData !== null &&
            "auto_calculated" in overrideData
          );
          const isCalculating = calculatingDocumentIds.has(row.document_id);
          // Use per-field confidence if available, else fall back to global
          const fieldConfidence = row.field_confidences?.[field.name] ?? row.confidence;

          // For arrays, use SmartCell with onSave for editing
          if (Array.isArray(value)) {
            return (
              <div className="relative">
                <div className={isAutoCalculated ? "bg-blue-500/10 border border-blue-500/30 rounded px-1" : ""}>
                  {isCalculating && (
                    <div className="absolute inset-0 bg-blue-500/20 flex items-center justify-center z-10 rounded">
                      <span className="text-xs text-blue-400">Calculating...</span>
                    </div>
                  )}
                  <SmartCell
                    value={value}
                    confidence={fieldConfidence}
                    fieldName={field.name}
                    onSave={async (newValue) => {
                      await onCellUpdate(row.id, field.name, newValue);
                    }}
                    editable={true}
                  />
                </div>
                {hasOverride && !isAutoCalculated && (
                  <div
                    className="absolute -left-1 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-indigo-500 z-10"
                    title="Manually edited"
                  />
                )}
                {isAutoCalculated && (
                  <div
                    className="absolute -left-1 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-blue-500 z-10"
                    title="Smart-calculated by AI"
                  />
                )}
              </div>
            );
          }

          // Use SmartCell for scalar values too (with editing support)
          return (
            <div className="relative">
              <div className={isAutoCalculated ? "bg-blue-500/10 border border-blue-500/30 rounded px-1" : ""}>
                {isCalculating && (
                  <div className="absolute inset-0 bg-blue-500/20 flex items-center justify-center z-10 rounded">
                    <span className="text-xs text-blue-400">Calculating...</span>
                  </div>
                )}
                <SmartCell
                  value={value}
                  confidence={fieldConfidence}
                  fieldName={field.name}
                  onSave={async (newValue) => {
                    await onCellUpdate(row.id, field.name, newValue);
                  }}
                  editable={true}
                />
              </div>
              {hasOverride && !isAutoCalculated && (
                <div
                  className="absolute -left-1 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-indigo-500 z-10"
                  title="Manually edited"
                />
              )}
              {isAutoCalculated && (
                <div
                  className="absolute -left-1 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-blue-500 z-10"
                  title="Smart-calculated by AI"
                />
              )}
            </div>
          );
        },
      });
    });

    return cols;
  }, [schema.fields, onCellUpdate, onRowClick, calculatingDocumentIds]);

  const table = useReactTable({
    data: results,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return {
    table,
    sorting,
    setSorting,
  };
}

