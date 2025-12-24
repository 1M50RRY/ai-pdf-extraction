import { useState, useMemo } from "react";
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
  ChevronDown,
  ChevronUp,
  Download,
  Eye,
  FileText,
  Info,
} from "lucide-react";
import type { ExtractionResult, ExtractBatchResponse, SchemaDefinition } from "../types";

interface ResultsTableProps {
  results: ExtractBatchResponse[];
  schema: SchemaDefinition;
  onRowClick?: (result: ExtractionResult) => void;
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

export function ResultsTable({ results, schema, onRowClick }: ResultsTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);

  // Flatten results for table display
  const flatResults = useMemo(() => {
    return results.flatMap((batch) => batch.results);
  }, [results]);

  const columns = useMemo((): ColumnDef<ExtractionResult, unknown>[] => {
    const cols: ColumnDef<ExtractionResult, unknown>[] = [
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
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-slate-500" />
            <span className="text-sm font-mono text-slate-300">
              {info.getValue() as string}
            </span>
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

    // Add dynamic columns for each field in schema
    schema.fields.forEach((field) => {
      cols.push({
        id: field.name,
        accessorFn: (row) => row.extracted_data[field.name],
        header: field.name.replace(/_/g, " ").toUpperCase(),
        cell: (info) => {
          const value = info.getValue();
          const confidence = info.row.original.confidence;

          let cellClass = "text-sm text-slate-300";
          if (confidence < 0.5) {
            cellClass = "text-sm text-red-300 bg-red-500/10 px-2 py-1 rounded";
          } else if (confidence < 0.8) {
            cellClass =
              "text-sm text-amber-300 bg-amber-500/10 px-2 py-1 rounded";
          }

          return (
            <span className={cellClass}>
              {value !== null && value !== undefined ? String(value) : "—"}
            </span>
          );
        },
      });
    });

    return cols;
  }, [schema.fields]);

  const table = useReactTable({
    data: flatResults,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const exportToCSV = () => {
    const headers = ["source_file", "confidence", "warnings", ...schema.fields.map((f) => f.name)];
    const rows = flatResults.map((result) => [
      result.source_file,
      result.confidence,
      result.warnings.join("; "),
      ...schema.fields.map((f) => result.extracted_data[f.name] ?? ""),
    ]);

    const csv = [
      headers.join(","),
      ...rows.map((row) =>
        row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(",")
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
      exported_at: new Date().toISOString(),
      total_documents: flatResults.length,
      average_confidence: avgConfidence,
      results: flatResults,
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
    flatResults.reduce((sum, r) => sum + r.confidence, 0) / flatResults.length;
  const warningCount = flatResults.filter((r) => r.warnings.length > 0).length;

  return (
    <div className="space-y-6">
      {/* Stats Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div>
            <p className="text-sm text-slate-400">Total Extractions</p>
            <p className="text-2xl font-bold text-slate-100">
              {flatResults.length}
            </p>
          </div>
          <div>
            <p className="text-sm text-slate-400">Avg. Confidence</p>
            <p className="text-2xl font-bold text-slate-100">
              {Math.round(avgConfidence * 100)}%
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
                  className="hover:bg-slate-700/30 transition-colors cursor-pointer group"
                  onClick={() => onRowClick?.(row.original)}
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

