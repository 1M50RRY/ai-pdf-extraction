import { useEffect, useState } from "react";
import {
  X,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  FileText,
  ChevronRight,
  History,
} from "lucide-react";
import { getBatchHistory, type BatchSummary } from "../api";

interface HistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectBatch: (batchId: string) => void;
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

export function HistoryModal({ isOpen, onClose, onSelectBatch }: HistoryModalProps) {
  const [batches, setBatches] = useState<BatchSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      fetchHistory();
    }
  }, [isOpen]);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getBatchHistory(50, 0);
      setBatches(response.batches);
    } catch (err) {
      console.error("Failed to fetch history:", err);
      setError("Failed to load batch history");
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="bg-slate-900 rounded-2xl border border-slate-700 shadow-2xl w-full max-w-3xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-500/20 flex items-center justify-center">
              <History className="w-5 h-5 text-indigo-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-slate-100">Batch History</h2>
              <p className="text-sm text-slate-400">
                View past extraction jobs
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

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
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
              <p className="text-slate-400 font-medium">No batch history yet</p>
              <p className="text-sm text-slate-500 mt-1">
                Your extraction jobs will appear here
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {batches.map((batch) => (
                <button
                  key={batch.id}
                  onClick={() => {
                    onSelectBatch(batch.id);
                    onClose();
                  }}
                  className="w-full flex items-center justify-between p-4 bg-slate-800/50 hover:bg-slate-800 border border-slate-700 hover:border-indigo-500/50 rounded-xl transition-all group"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-slate-700 flex items-center justify-center">
                      <FileText className="w-6 h-6 text-slate-400" />
                    </div>
                    <div className="text-left">
                      <div className="flex items-center gap-3">
                        <span className="font-medium text-slate-200">
                          {batch.schema_name || "Untitled Batch"}
                        </span>
                        <StatusBadge status={batch.status} />
                      </div>
                      <div className="flex items-center gap-4 mt-1 text-sm text-slate-400">
                        <span>{batch.total_documents} documents</span>
                        <span>•</span>
                        <span>
                          {batch.successful_documents} success
                          {batch.failed_documents > 0 && (
                            <span className="text-red-400">
                              , {batch.failed_documents} failed
                            </span>
                          )}
                        </span>
                        <span>•</span>
                        <span>{formatDate(batch.created_at)}</span>
                      </div>
                    </div>
                  </div>
                  <ChevronRight className="w-5 h-5 text-slate-500 group-hover:text-indigo-400 transition-colors" />
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-700 bg-slate-800/30">
          <button
            onClick={onClose}
            className="w-full py-2.5 text-slate-300 hover:text-slate-100 bg-slate-700 hover:bg-slate-600 rounded-lg font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

