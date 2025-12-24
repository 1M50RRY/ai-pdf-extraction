import { useState, useEffect } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import {
  X,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  AlertTriangle,
  CheckCircle,
  FileText,
  Code,
} from "lucide-react";
import type { ExtractionResult, SchemaDefinition } from "../types";

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface ValidationModalProps {
  result: ExtractionResult;
  schema: SchemaDefinition;
  pdfFile?: File;
  onClose: () => void;
}

function ConfidenceIndicator({ confidence }: { confidence: number }) {
  const percentage = Math.round(confidence * 100);

  let colorClass = "text-emerald-400 bg-emerald-500/20";
  let icon = <CheckCircle className="w-5 h-5" />;

  if (confidence < 0.5) {
    colorClass = "text-red-400 bg-red-500/20";
    icon = <AlertTriangle className="w-5 h-5" />;
  } else if (confidence < 0.8) {
    colorClass = "text-amber-400 bg-amber-500/20";
    icon = <AlertTriangle className="w-5 h-5" />;
  }

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg ${colorClass}`}>
      {icon}
      <span className="font-medium">{percentage}% Confidence</span>
    </div>
  );
}

export function ValidationModal({
  result,
  schema,
  pdfFile,
  onClose,
}: ValidationModalProps) {
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [scale, setScale] = useState(1.0);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"formatted" | "json">("formatted");

  // Create object URL for PDF
  useEffect(() => {
    if (pdfFile) {
      const url = URL.createObjectURL(pdfFile);
      setPdfUrl(url);
      return () => URL.revokeObjectURL(url);
    }
  }, [pdfFile]);

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
  };

  const goToPrevPage = () => setCurrentPage((p) => Math.max(1, p - 1));
  const goToNextPage = () => setCurrentPage((p) => Math.min(numPages, p + 1));
  const zoomIn = () => setScale((s) => Math.min(2, s + 0.2));
  const zoomOut = () => setScale((s) => Math.max(0.5, s - 0.2));

  // Handle escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-[95vw] h-[90vh] bg-slate-900 rounded-2xl shadow-2xl border border-slate-700 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700 bg-slate-800/50">
          <div className="flex items-center gap-4">
            <FileText className="w-6 h-6 text-indigo-400" />
            <div>
              <h2 className="text-lg font-semibold text-slate-100">
                {result.source_file}
              </h2>
              <p className="text-sm text-slate-400">
                {schema.name} • {schema.fields.length} fields
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <ConfidenceIndicator confidence={result.confidence} />
            <button
              onClick={onClose}
              className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content - Split View */}
        <div className="flex-1 flex overflow-hidden">
          {/* Left Side - PDF Viewer */}
          <div className="w-1/2 border-r border-slate-700 flex flex-col bg-slate-950">
            {/* PDF Controls */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-900">
              <div className="flex items-center gap-2">
                <button
                  onClick={goToPrevPage}
                  disabled={currentPage <= 1}
                  className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <span className="text-sm text-slate-300">
                  Page {currentPage} of {numPages || "..."}
                </span>
                <button
                  onClick={goToNextPage}
                  disabled={currentPage >= numPages}
                  className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={zoomOut}
                  disabled={scale <= 0.5}
                  className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded disabled:opacity-50 transition-colors"
                >
                  <ZoomOut className="w-4 h-4" />
                </button>
                <span className="text-sm text-slate-400 w-12 text-center">
                  {Math.round(scale * 100)}%
                </span>
                <button
                  onClick={zoomIn}
                  disabled={scale >= 2}
                  className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded disabled:opacity-50 transition-colors"
                >
                  <ZoomIn className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* PDF Display */}
            <div className="flex-1 overflow-auto flex items-start justify-center p-4">
              {pdfUrl ? (
                <Document
                  file={pdfUrl}
                  onLoadSuccess={onDocumentLoadSuccess}
                  loading={
                    <div className="flex items-center justify-center h-full">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500" />
                    </div>
                  }
                  error={
                    <div className="text-center text-slate-400 p-8">
                      <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p>Unable to load PDF preview</p>
                    </div>
                  }
                >
                  <Page
                    pageNumber={currentPage}
                    scale={scale}
                    renderTextLayer={false}
                    renderAnnotationLayer={false}
                    className="shadow-2xl"
                  />
                </Document>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-slate-400">
                  <FileText className="w-16 h-16 mb-4 opacity-50" />
                  <p className="text-lg">No PDF preview available</p>
                  <p className="text-sm mt-1">
                    Upload the original file to enable preview
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Right Side - Extracted Data */}
          <div className="w-1/2 flex flex-col bg-slate-900">
            {/* Tabs */}
            <div className="flex border-b border-slate-700">
              <button
                onClick={() => setActiveTab("formatted")}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                  activeTab === "formatted"
                    ? "text-indigo-400 border-b-2 border-indigo-400"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <FileText className="w-4 h-4" />
                Formatted View
              </button>
              <button
                onClick={() => setActiveTab("json")}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                  activeTab === "json"
                    ? "text-indigo-400 border-b-2 border-indigo-400"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Code className="w-4 h-4" />
                Raw JSON
              </button>
            </div>

            {/* Data Content */}
            <div className="flex-1 overflow-auto p-4">
              {activeTab === "formatted" ? (
                <div className="space-y-3">
                  {schema.fields.map((field) => {
                    const value = result.extracted_data[field.name];
                    const hasValue = value !== null && value !== undefined && value !== "";

                    return (
                      <div
                        key={field.name}
                        className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div>
                            <label className="text-xs font-medium text-slate-400 uppercase tracking-wide">
                              {field.name.replace(/_/g, " ")}
                            </label>
                            {field.required && (
                              <span className="ml-2 text-xs text-amber-400">Required</span>
                            )}
                          </div>
                          <span
                            className={`text-xs px-2 py-0.5 rounded ${
                              field.type === "currency"
                                ? "bg-emerald-500/20 text-emerald-300"
                                : field.type === "date"
                                  ? "bg-purple-500/20 text-purple-300"
                                  : "bg-slate-600/50 text-slate-400"
                            }`}
                          >
                            {field.type}
                          </span>
                        </div>
                        <div
                          className={`text-lg font-mono ${
                            hasValue ? "text-slate-100" : "text-slate-500 italic"
                          }`}
                        >
                          {hasValue ? String(value) : "—"}
                        </div>
                        {field.description && (
                          <p className="text-xs text-slate-500 mt-2">
                            {field.description}
                          </p>
                        )}
                      </div>
                    );
                  })}

                  {/* Warnings Section */}
                  {result.warnings.length > 0 && (
                    <div className="mt-6 bg-amber-500/10 border border-amber-500/30 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-3">
                        <AlertTriangle className="w-5 h-5 text-amber-400" />
                        <span className="font-medium text-amber-300">
                          Warnings ({result.warnings.length})
                        </span>
                      </div>
                      <ul className="space-y-2">
                        {result.warnings.map((warning, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-amber-200/80">
                            <span className="text-amber-500">•</span>
                            <span>{warning}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ) : (
                <pre className="text-sm text-slate-300 font-mono whitespace-pre-wrap bg-slate-950 rounded-lg p-4 border border-slate-800 overflow-auto">
                  {JSON.stringify(
                    {
                      source_file: result.source_file,
                      confidence: result.confidence,
                      extracted_data: result.extracted_data,
                      warnings: result.warnings,
                    },
                    null,
                    2
                  )}
                </pre>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

