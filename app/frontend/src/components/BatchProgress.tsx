import { Loader2, CheckCircle, XCircle, FileText } from "lucide-react";
import type { BatchProgress as BatchProgressType } from "../types";

interface BatchProgressProps {
  progress: BatchProgressType;
  files: File[];
}

export function BatchProgress({ progress, files }: BatchProgressProps) {
  const percentage =
    progress.total > 0 ? (progress.current / progress.total) * 100 : 0;

  return (
    <div className="space-y-6">
      {/* Progress Header */}
      <div className="text-center">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-slate-800 mb-4">
          {progress.status === "processing" && (
            <Loader2 className="w-10 h-10 text-indigo-400 animate-spin" />
          )}
          {progress.status === "complete" && (
            <CheckCircle className="w-10 h-10 text-emerald-400" />
          )}
          {progress.status === "error" && (
            <XCircle className="w-10 h-10 text-red-400" />
          )}
          {progress.status === "idle" && (
            <FileText className="w-10 h-10 text-slate-400" />
          )}
        </div>

        <h2 className="text-2xl font-bold text-slate-100">
          {progress.status === "processing" && "Processing Documents..."}
          {progress.status === "complete" && "Extraction Complete!"}
          {progress.status === "error" && "Processing Error"}
          {progress.status === "idle" && "Ready to Process"}
        </h2>

        <p className="text-slate-400 mt-2">
          {progress.status === "processing" &&
            `Processing file ${progress.current} of ${progress.total}`}
          {progress.status === "complete" &&
            `Successfully processed ${progress.total} files`}
          {progress.status === "error" && progress.message}
          {progress.status === "idle" && `${files.length} files ready`}
        </p>
      </div>

      {/* Progress Bar */}
      <div className="max-w-lg mx-auto">
        <div className="flex justify-between text-sm text-slate-400 mb-2">
          <span>Progress</span>
          <span>{Math.round(percentage)}%</span>
        </div>
        <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ease-out ${
              progress.status === "complete"
                ? "bg-emerald-500"
                : progress.status === "error"
                  ? "bg-red-500"
                  : "bg-indigo-500"
            }`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>

      {/* File List */}
      <div className="max-w-lg mx-auto bg-slate-800/50 rounded-xl p-4 max-h-48 overflow-y-auto">
        <div className="space-y-2">
          {files.map((file, index) => {
            const isProcessed = index < progress.current;
            const isProcessing =
              index === progress.current - 1 &&
              progress.status === "processing";

            return (
              <div
                key={`${file.name}-${index}`}
                className={`flex items-center justify-between rounded-lg px-4 py-2 transition-colors ${
                  isProcessed
                    ? "bg-emerald-500/10"
                    : isProcessing
                      ? "bg-indigo-500/10"
                      : "bg-slate-700/30"
                }`}
              >
                <div className="flex items-center gap-3">
                  <FileText
                    className={`w-4 h-4 ${
                      isProcessed
                        ? "text-emerald-400"
                        : isProcessing
                          ? "text-indigo-400"
                          : "text-slate-500"
                    }`}
                  />
                  <span
                    className={`text-sm truncate max-w-xs ${
                      isProcessed ? "text-emerald-300" : "text-slate-300"
                    }`}
                  >
                    {file.name}
                  </span>
                </div>
                {isProcessed && (
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                )}
                {isProcessing && (
                  <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

