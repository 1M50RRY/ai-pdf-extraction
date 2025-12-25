import { useState, useEffect } from "react";
import { X, Table, Save, Loader2 } from "lucide-react";

interface SmartCellProps {
  value: unknown;
  confidence?: number;
  fieldName: string;
  onSave?: (newValue: unknown) => Promise<void>;
  editable?: boolean;
}

interface ArrayItemModalProps {
  items: unknown[];
  fieldName: string;
  onClose: () => void;
}

interface ArrayEditorModalProps {
  items: unknown[];
  fieldName: string;
  onClose: () => void;
  onSave: (newValue: unknown[]) => Promise<void>;
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

function ArrayEditorModal({
  items,
  fieldName,
  onClose,
  onSave,
}: ArrayEditorModalProps) {
  const [jsonText, setJsonText] = useState(
    JSON.stringify(items, null, 2)
  );
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSave = async () => {
    setError(null);
    try {
      const parsed = JSON.parse(jsonText);
      if (!Array.isArray(parsed)) {
        setError("Value must be a JSON array");
        return;
      }
      setIsSaving(true);
      await onSave(parsed);
      onClose();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Invalid JSON format"
      );
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal - Wider and taller */}
      <div className="relative w-[95vw] max-w-6xl h-[90vh] bg-slate-900 rounded-xl shadow-2xl border border-slate-700 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700 bg-slate-800/50">
          <div className="flex items-center gap-3">
            <Table className="w-5 h-5 text-indigo-400" />
            <div>
              <h3 className="text-lg font-semibold text-slate-100">
                Edit {fieldName.replace(/_/g, " ").toUpperCase()}
              </h3>
              <p className="text-sm text-slate-400">
                Edit JSON array structure
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={isSaving}
            className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded-lg transition-colors disabled:opacity-50"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {error && (
            <div className="px-6 py-3 bg-red-500/10 border-b border-red-500/30">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}
          <div className="flex-1 p-6 overflow-auto">
            <textarea
              value={jsonText}
              onChange={(e) => {
                setJsonText(e.target.value);
                setError(null);
              }}
              className="w-full h-[400px] font-mono text-sm p-4 border border-slate-700 rounded-md bg-slate-950 text-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-none"
              spellCheck={false}
            />
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-slate-700 bg-slate-800/50">
          <button
            onClick={onClose}
            disabled={isSaving}
            className="px-4 py-2 text-slate-400 hover:text-slate-200 transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSaving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Save className="w-4 h-4" />
            )}
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

export function SmartCell({
  value,
  confidence,
  fieldName,
  onSave,
  editable = false,
}: SmartCellProps) {
  const [isViewModalOpen, setIsViewModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(
    value !== null && value !== undefined ? String(value) : ""
  );
  const [isSaving, setIsSaving] = useState(false);

  // Update editValue when value prop changes (but not while editing)
  useEffect(() => {
    if (!isEditing) {
      setEditValue(value !== null && value !== undefined ? String(value) : "");
    }
  }, [value, isEditing]);

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
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <button
              className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-600/50 cursor-pointer hover:border-indigo-500/50 hover:bg-slate-800/50 transition-colors ${bgClass} ${textClass}`}
              onClick={() => setIsViewModalOpen(true)}
            >
              <Table className="w-4 h-4" />
              <span className="text-sm font-medium">
                {itemCount} item{itemCount !== 1 ? "s" : ""}
              </span>
            </button>
            {onSave && (
              <button
                onClick={() => setIsEditModalOpen(true)}
                className="px-2 py-1 text-xs text-indigo-400 hover:text-indigo-300 hover:bg-indigo-500/10 rounded transition-colors"
              >
                Edit
              </button>
            )}
          </div>
          {confidence !== undefined && (
            <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300">
              {Math.round(confidence * 100)}%
            </span>
          )}
        </div>
        {isViewModalOpen && (
          <ArrayItemModal
            items={value}
            fieldName={fieldName}
            onClose={() => setIsViewModalOpen(false)}
          />
        )}
        {isEditModalOpen && onSave && (
          <ArrayEditorModal
            items={value}
            fieldName={fieldName}
            onClose={() => setIsEditModalOpen(false)}
            onSave={async (newValue) => {
              await onSave(newValue);
              setIsEditModalOpen(false);
            }}
          />
        )}
      </>
    );
  }

  // Handle scalar values
  const displayValue =
    value !== null && value !== undefined ? String(value) : "—";

  // Confidence badge color
  const getConfidenceBadgeColor = () => {
    if (confidence === undefined) return "";
    if (confidence >= 0.9) {
      return "bg-emerald-500/20 text-emerald-300";
    } else if (confidence < 0.5) {
      return "bg-red-500/20 text-red-300";
    } else if (confidence < 0.7) {
      return "bg-amber-500/20 text-amber-300";
    } else {
      return "bg-amber-500/10 text-amber-200";
    }
  };

  const handleBlur = async () => {
    if (editValue !== displayValue && onSave) {
      setIsSaving(true);
      try {
        // Try to parse as number if it looks like a number
        let parsedValue: unknown = editValue;
        if (editValue.trim() !== "" && !isNaN(Number(editValue))) {
          parsedValue = Number(editValue);
        }
        await onSave(parsedValue);
      } catch (err) {
        console.error("Save failed:", err);
        setEditValue(displayValue); // Revert on error
      } finally {
        setIsSaving(false);
        setIsEditing(false);
      }
    } else {
      setIsEditing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleBlur();
    } else if (e.key === "Escape") {
      setEditValue(displayValue);
      setIsEditing(false);
    }
  };

  if (isEditing && editable && onSave) {
    return (
      <div className="flex items-center justify-between gap-2 h-full w-full px-2 py-1">
        <div className="relative flex-1">
          <input
            type="text"
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={handleBlur}
            onKeyDown={handleKeyDown}
            autoFocus
            disabled={isSaving}
            className="w-full bg-slate-700 border-2 border-indigo-500 rounded px-2 py-1 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          {isSaving && (
            <div className="absolute right-2 top-1/2 -translate-y-1/2">
              <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
            </div>
          )}
        </div>
        {confidence !== undefined && (
          <span
            className={`text-[10px] font-bold px-1.5 py-0.5 rounded flex-shrink-0 ${getConfidenceBadgeColor()}`}
          >
            {Math.round(confidence * 100)}%
          </span>
        )}
      </div>
    );
  }

  return (
    <div
      className={`flex items-center justify-between gap-2 h-full w-full px-2 py-1 ${bgClass} rounded ${
        editable && onSave ? "cursor-text hover:bg-slate-800/50 transition-colors" : ""
      }`}
      onClick={() => {
        if (editable && onSave) {
          setIsEditing(true);
        }
      }}
    >
      <span className={`text-sm ${textClass} truncate flex-1`}>
        {displayValue}
      </span>
      {confidence !== undefined && (
        <span
          className={`text-[10px] font-bold px-1.5 py-0.5 rounded flex-shrink-0 ${getConfidenceBadgeColor()}`}
        >
          {Math.round(confidence * 100)}%
        </span>
      )}
    </div>
  );
}

