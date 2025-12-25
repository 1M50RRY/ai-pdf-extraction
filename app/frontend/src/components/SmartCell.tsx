import { useState } from "react";
import { X, Table } from "lucide-react";

interface SmartCellProps {
  value: unknown;
  confidence?: number;
  fieldName: string;
}

interface ArrayItemModalProps {
  items: unknown[];
  fieldName: string;
  onClose: () => void;
}

function ArrayItemModal({ items, fieldName, onClose }: ArrayItemModalProps) {
  if (!Array.isArray(items) || items.length === 0) {
    return null;
  }

  // Determine columns from first item
  const firstItem = items[0];
  const columns =
    typeof firstItem === "object" && firstItem !== null
      ? Object.keys(firstItem)
      : ["value"];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-[90vw] max-w-4xl max-h-[80vh] bg-slate-900 rounded-xl shadow-2xl border border-slate-700 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700 bg-slate-800/50">
          <div className="flex items-center gap-3">
            <Table className="w-5 h-5 text-indigo-400" />
            <div>
              <h3 className="text-lg font-semibold text-slate-100">
                {fieldName.replace(/_/g, " ").toUpperCase()}
              </h3>
              <p className="text-sm text-slate-400">
                {items.length} item{items.length !== 1 ? "s" : ""}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Table Content */}
        <div className="flex-1 overflow-auto p-4">
          <table className="w-full border-collapse">
            <thead>
              <tr className="bg-slate-800/50 border-b border-slate-700">
                {columns.map((col) => (
                  <th
                    key={col}
                    className="px-4 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wide"
                  >
                    {col.replace(/_/g, " ")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {items.map((item, idx) => (
                <tr
                  key={idx}
                  className="hover:bg-slate-800/30 transition-colors"
                >
                  {columns.map((col) => {
                    const cellValue =
                      typeof item === "object" && item !== null
                        ? (item as Record<string, unknown>)[col]
                        : item;
                    return (
                      <td
                        key={col}
                        className="px-4 py-3 text-sm text-slate-300 font-mono"
                      >
                        {cellValue !== null && cellValue !== undefined
                          ? typeof cellValue === "object"
                            ? JSON.stringify(cellValue)
                            : String(cellValue)
                          : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export function SmartCell({
  value,
  confidence,
  fieldName,
}: SmartCellProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Determine background color based on confidence
  let bgClass = "";
  let textClass = "text-slate-300";
  if (confidence !== undefined) {
    if (confidence >= 0.9) {
      bgClass = "";
      textClass = "text-emerald-300";
    } else if (confidence < 0.5) {
      bgClass = "bg-red-500/10";
      textClass = "text-red-400";
    } else if (confidence < 0.7) {
      bgClass = "bg-amber-500/10";
      textClass = "text-amber-400";
    } else if (confidence < 0.8) {
      bgClass = "bg-amber-500/5";
      textClass = "text-amber-300";
    }
  }

  // Handle Array values
  if (Array.isArray(value)) {
    const itemCount = value.length;
    return (
      <>
        <div
          className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-600/50 cursor-pointer hover:border-indigo-500/50 hover:bg-slate-800/50 transition-colors ${bgClass} ${textClass}`}
          onClick={() => setIsModalOpen(true)}
        >
          <Table className="w-4 h-4" />
          <span className="text-sm font-medium">
            {itemCount} item{itemCount !== 1 ? "s" : ""}
          </span>
          {confidence !== undefined && (
            <span className="text-xs text-slate-500">
              {Math.round(confidence * 100)}%
            </span>
          )}
        </div>
        {isModalOpen && (
          <ArrayItemModal
            items={value}
            fieldName={fieldName}
            onClose={() => setIsModalOpen(false)}
          />
        )}
      </>
    );
  }

  // Handle scalar values
  const displayValue =
    value !== null && value !== undefined ? String(value) : "—";

  return (
    <div className={`inline-flex items-center gap-2 ${bgClass} px-2 py-1 rounded`}>
      <span className={`text-sm ${textClass}`}>{displayValue}</span>
      {confidence !== undefined && (
        <span className="text-xs text-slate-500">
          {Math.round(confidence * 100)}%
        </span>
      )}
    </div>
  );
}

