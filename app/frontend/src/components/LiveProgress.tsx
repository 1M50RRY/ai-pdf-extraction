import { useEffect, useState } from "react";
import {
  CheckCircle,
  XCircle,
  Loader2,
  Clock,
  FileText,
  AlertTriangle,
} from "lucide-react";
import { getBatchStatus, type BatchStatusResponse, type DocumentStatus } from "../api";

interface LiveProgressProps {
  batchId: string;
  onComplete: (status: BatchStatusResponse) => void;
  onError: (error: string) => void;
  pollInterval?: number;
}

function DocumentStatusIcon({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return <CheckCircle className="w-5 h-5 text-emerald-400" />;
    case "failed":
      return <XCircle className="w-5 h-5 text-red-400" />;
    case "processing":
      return <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />;
    default:
      return <Clock className="w-5 h-5 text-slate-500" />;
  }
}

function DocumentRow({ doc }: { doc: DocumentStatus }) {
  return (
    <div
      className={`flex items-center justify-between p-3 rounded-lg transition-colors ${
        doc.status === "completed"
          ? "bg-emerald-500/10 border border-emerald-500/20"
          : doc.status === "failed"
            ? "bg-red-500/10 border border-red-500/20"
            : doc.status === "processing"
              ? "bg-indigo-500/10 border border-indigo-500/20"
              : "bg-slate-800/50 border border-slate-700/50"
      }`}
    >
      <div className="flex items-center gap-3">
        <DocumentStatusIcon status={doc.status} />
        <div>
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-slate-500" />
            <span className="text-sm font-mono text-slate-200 truncate max-w-xs">
              {doc.filename}
            </span>
          </div>
          {doc.error_message && (
            <p className="text-xs text-red-400 mt-1">{doc.error_message}</p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-3">
        {doc.warnings.length > 0 && (
          <div className="flex items-center gap-1 text-amber-400">
            <AlertTriangle className="w-4 h-4" />
            <span className="text-xs">{doc.warnings.length}</span>
          </div>
        )}
        {doc.confidence !== null && (
          <span
            className={`text-sm font-medium ${
              doc.confidence >= 0.8
                ? "text-emerald-400"
                : doc.confidence >= 0.5
                  ? "text-amber-400"
                  : "text-red-400"
            }`}
          >
            {Math.round(doc.confidence * 100)}%
          </span>
        )}
        <span
          className={`text-xs px-2 py-1 rounded-full ${
            doc.status === "completed"
              ? "bg-emerald-500/20 text-emerald-300"
              : doc.status === "failed"
                ? "bg-red-500/20 text-red-300"
                : doc.status === "processing"
                  ? "bg-indigo-500/20 text-indigo-300"
                  : "bg-slate-700 text-slate-400"
          }`}
        >
          {doc.status}
        </span>
      </div>
    </div>
  );
}

export function LiveProgress({
  batchId,
  onComplete,
  onError,
  pollInterval = 2000,
}: LiveProgressProps) {
  const [status, setStatus] = useState<BatchStatusResponse | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);

  // Poll for status
  useEffect(() => {
    let isMounted = true;
    let pollCount = 0;
    const maxPolls = 300; // 10 minutes max

    const poll = async () => {
      if (!isMounted) return;

      try {
        const batchStatus = await getBatchStatus(batchId);
        if (!isMounted) return;

        setStatus(batchStatus);

        if (batchStatus.status === "completed" || batchStatus.progress_percent >= 100) {
          onComplete(batchStatus);
          return;
        }

        pollCount++;
        if (pollCount >= maxPolls) {
          onError("Processing timed out");
          return;
        }

        // Continue polling
        setTimeout(poll, pollInterval);
      } catch (err) {
        if (!isMounted) return;
        console.error("Polling error:", err);
        onError(err instanceof Error ? err.message : "Failed to get status");
      }
    };

    poll();

    return () => {
      isMounted = false;
    };
  }, [batchId, pollInterval, onComplete, onError]);

  // Track elapsed time
  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime((t) => t + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  if (!status) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Loader2 className="w-12 h-12 text-indigo-400 animate-spin mb-4" />
        <p className="text-slate-300">Starting batch processing...</p>
      </div>
    );
  }

  const progressPercent = Math.round(status.progress_percent);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-slate-100">
          Processing Documents
        </h2>
        <p className="text-slate-400 mt-1">
          {status.schema_name && (
            <>Using schema: <strong>{status.schema_name}</strong> • </>
          )}
          Elapsed: {formatTime(elapsedTime)}
        </p>
      </div>

      {/* Progress Bar */}
      <div className="bg-slate-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm text-slate-400">
            {status.completed_documents} of {status.total_documents} documents
          </span>
          <span className="text-2xl font-bold text-indigo-400">
            {progressPercent}%
          </span>
        </div>
        <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500 ease-out"
            style={{ width: `${progressPercent}%` }}
          />
        </div>

        {/* Stats */}
        <div className="flex items-center justify-center gap-8 mt-4">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-emerald-400" />
            <span className="text-sm text-slate-300">
              {status.completed_documents - status.failed_documents} completed
            </span>
          </div>
          {status.failed_documents > 0 && (
            <div className="flex items-center gap-2">
              <XCircle className="w-4 h-4 text-red-400" />
              <span className="text-sm text-slate-300">
                {status.failed_documents} failed
              </span>
            </div>
          )}
          <div className="flex items-center gap-2">
            <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
            <span className="text-sm text-slate-300">
              {status.total_documents - status.completed_documents} remaining
            </span>
          </div>
        </div>
      </div>

      {/* Document List */}
      <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
        <h3 className="text-sm font-medium text-slate-400 mb-3">
          Document Status
        </h3>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {status.documents.map((doc) => (
            <DocumentRow key={doc.id} doc={doc} />
          ))}
        </div>
      </div>
    </div>
  );
}

