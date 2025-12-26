import {
  Download,
  Sparkles,
  Check,
  CheckCircle,
  Loader2,
} from "lucide-react";
import type { SchemaDefinition } from "../../types";
import type { EditableExtractionResult } from "../EditableResultsTable";

interface TableToolbarProps {
  results: EditableExtractionResult[];
  schema: SchemaDefinition;
  batchId: string;
  isAutoCalculating: boolean;
  isApproving: boolean;
  approvalStatus: "idle" | "success" | "error";
  reviewedCount: number;
  totalCount: number;
  onAutoCalculate: () => void;
  onApproveAll: () => void;
  onExportCSV: () => void;
  onExportJSON: () => void;
}

export function TableToolbar({
  results,
  schema,
  batchId,
  isAutoCalculating,
  isApproving,
  approvalStatus,
  reviewedCount,
  totalCount,
  onAutoCalculate,
  onApproveAll,
  onExportCSV,
  onExportJSON,
}: TableToolbarProps) {
  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
      {/* Stats */}
      <div className="flex items-center gap-6">
        <div>
          <p className="text-sm text-slate-400">Total Extractions</p>
          <p className="text-2xl font-bold text-slate-100">
            {results.length}
          </p>
        </div>
        <div>
          <p className="text-sm text-slate-400">Avg. Confidence</p>
          <p className="text-2xl font-bold text-slate-100">
            {Math.round(
              results.reduce((sum, r) => sum + r.confidence, 0) / results.length
            )}%
          </p>
        </div>
        <div>
          <p className="text-sm text-slate-400">Reviewed</p>
          <p
            className={`text-2xl font-bold ${reviewedCount === totalCount
              ? "text-emerald-400"
              : "text-amber-400"
              }`}
          >
            {reviewedCount}/{totalCount}
          </p>
        </div>
        <div>
          <p className="text-sm text-slate-400">Warnings</p>
          <p
            className={`text-2xl font-bold ${results.filter((r) => r.warnings.length > 0).length > 0 ? "text-amber-400" : "text-emerald-400"}`}
          >
            {results.filter((r) => r.warnings.length > 0).length}
          </p>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 flex-wrap">
        {/* Dynamic Calculation Engine Button */}
        <button
          onClick={onAutoCalculate}
          disabled={isAutoCalculating}
          className="flex items-center gap-1.5 px-3 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition-colors text-sm font-medium disabled:bg-slate-600 disabled:cursor-not-allowed whitespace-nowrap"
          title="Dynamic Calculation Engine: Calculates missing values, infers formulas from field names, computes totals/averages, cross-references data, and completes patterns"
        >
          {isAutoCalculating ? (
            <Loader2 className="w-4 h-4 animate-spin flex-shrink-0" />
          ) : (
            <Sparkles className="w-4 h-4 flex-shrink-0" />
          )}
          <span className="hidden sm:inline">🧮 Smart Calculate</span>
          <span className="sm:hidden">🧮 Calculate</span>
        </button>

        {/* Approve All Button - Only show if not all approved */}
        {reviewedCount < totalCount && (
          <button
            onClick={onApproveAll}
            disabled={isApproving || approvalStatus === "success"}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap ${approvalStatus === "success"
              ? "bg-emerald-500/20 text-emerald-400 cursor-default"
              : "bg-emerald-600 hover:bg-emerald-500 text-white disabled:bg-slate-600"
              }`}
          >
            {isApproving ? (
              <Loader2 className="w-4 h-4 animate-spin flex-shrink-0" />
            ) : approvalStatus === "success" ? (
              <CheckCircle className="w-4 h-4 flex-shrink-0" />
            ) : (
              <Check className="w-4 h-4 flex-shrink-0" />
            )}
            <span className="hidden sm:inline">{approvalStatus === "success" ? "All Approved" : "Approve All"}</span>
            <span className="sm:hidden">{approvalStatus === "success" ? "Approved" : "Approve"}</span>
          </button>
        )}

        <button
          onClick={onExportCSV}
          className="flex items-center gap-1.5 px-3 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors text-sm font-medium whitespace-nowrap"
        >
          <Download className="w-4 h-4 flex-shrink-0" />
          <span className="hidden sm:inline">Export CSV</span>
          <span className="sm:hidden">CSV</span>
        </button>
        <button
          onClick={onExportJSON}
          className="flex items-center gap-1.5 px-3 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg transition-colors text-sm font-medium whitespace-nowrap"
        >
          <Download className="w-4 h-4 flex-shrink-0" />
          <span className="hidden sm:inline">Export JSON</span>
          <span className="sm:hidden">JSON</span>
        </button>
      </div>
    </div>
  );
}

