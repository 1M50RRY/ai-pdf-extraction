import { useState, useEffect, useCallback } from "react";
import {
  FileSearch,
  Settings,
  Play,
  Table2,
  ArrowLeft,
  Zap,
  AlertCircle,
} from "lucide-react";
import {
  UploadZone,
  MultiUploadZone,
  SchemaEditor,
  BatchProgress,
  ResultsTable,
  ValidationModal,
} from "./components";
import { uploadSample, extractBatch, healthCheck } from "./api";
import type {
  AppStep,
  SchemaDefinition,
  ExtractBatchResponse,
  ExtractionResult,
  BatchProgress as BatchProgressType,
} from "./types";

const STEPS: { id: AppStep; label: string; icon: React.ReactNode }[] = [
  { id: "upload", label: "Sample Upload", icon: <FileSearch className="w-5 h-5" /> },
  { id: "schema", label: "Schema Editor", icon: <Settings className="w-5 h-5" /> },
  { id: "batch", label: "Batch Process", icon: <Play className="w-5 h-5" /> },
  { id: "results", label: "Results", icon: <Table2 className="w-5 h-5" /> },
];

function StepIndicator({
  currentStep,
  onStepClick,
}: {
  currentStep: AppStep;
  onStepClick: (step: AppStep) => void;
}) {
  const currentIndex = STEPS.findIndex((s) => s.id === currentStep);

  return (
    <div className="flex items-center justify-center gap-2">
      {STEPS.map((step, index) => {
        const isActive = step.id === currentStep;
        const isPast = index < currentIndex;
        const isClickable = isPast;

        return (
          <button
            key={step.id}
            onClick={() => isClickable && onStepClick(step.id)}
            disabled={!isClickable}
            className={`
              flex items-center gap-2 px-4 py-2 rounded-lg transition-all
              ${isActive ? "bg-indigo-600 text-white" : ""}
              ${isPast ? "bg-slate-700 text-slate-300 hover:bg-slate-600 cursor-pointer" : ""}
              ${!isActive && !isPast ? "bg-slate-800 text-slate-500 cursor-not-allowed" : ""}
            `}
          >
            {step.icon}
            <span className="hidden sm:inline text-sm font-medium">
              {step.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}

export default function App() {
  const [currentStep, setCurrentStep] = useState<AppStep>("upload");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);

  // Data state
  const [schema, setSchema] = useState<SchemaDefinition | null>(null);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchProgress, setBatchProgress] = useState<BatchProgressType>({
    current: 0,
    total: 0,
    status: "idle",
  });
  const [results, setResults] = useState<ExtractBatchResponse[]>([]);

  // Validation modal state
  const [selectedResult, setSelectedResult] = useState<ExtractionResult | null>(null);
  const [selectedPdfFile, setSelectedPdfFile] = useState<File | undefined>(undefined);

  // Check backend health on mount
  useEffect(() => {
    healthCheck().then(setIsBackendHealthy);
  }, []);

  const handleSampleUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await uploadSample(file);
      setSchema(response.suggested_schema);
      setCurrentStep("schema");
    } catch (err) {
      console.error("Upload failed:", err);
      setError(
        err instanceof Error
          ? err.message
          : "Failed to upload sample. Is the backend running?"
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleSchemaConfirm = () => {
    setCurrentStep("batch");
  };

  const handleBatchStart = async () => {
    if (!schema || batchFiles.length === 0) return;

    setIsLoading(true);
    setError(null);
    setBatchProgress({
      current: 0,
      total: batchFiles.length,
      status: "processing",
    });

    try {
      const batchResults = await extractBatch(batchFiles, schema, (current, total) => {
        setBatchProgress({
          current,
          total,
          status: "processing",
        });
      });

      setResults(batchResults);
      setBatchProgress({
        current: batchFiles.length,
        total: batchFiles.length,
        status: "complete",
      });

      // Auto-advance to results after a short delay
      setTimeout(() => setCurrentStep("results"), 1000);
    } catch (err) {
      console.error("Batch processing failed:", err);
      setBatchProgress({
        current: batchProgress.current,
        total: batchFiles.length,
        status: "error",
        message:
          err instanceof Error ? err.message : "Batch processing failed",
      });
      setError(
        err instanceof Error ? err.message : "Batch processing failed"
      );
    } finally {
      setIsLoading(false);
    }
  };

  const resetToStart = () => {
    setCurrentStep("upload");
    setSchema(null);
    setBatchFiles([]);
    setBatchProgress({ current: 0, total: 0, status: "idle" });
    setResults([]);
    setError(null);
    setSelectedResult(null);
    setSelectedPdfFile(undefined);
  };

  // Handle row click to open validation modal
  const handleRowClick = useCallback(
    (result: ExtractionResult) => {
      setSelectedResult(result);
      // Find the matching PDF file from batch files
      const matchingFile = batchFiles.find(
        (file) => file.name === result.source_file
      );
      setSelectedPdfFile(matchingFile);
    },
    [batchFiles]
  );

  const handleCloseModal = () => {
    setSelectedResult(null);
    setSelectedPdfFile(undefined);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-900 to-indigo-950">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-100">
                  PDF Extractor
                </h1>
                <p className="text-xs text-slate-400">AI-Powered Document Processing</p>
              </div>
            </div>

            {/* Backend Status */}
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  isBackendHealthy === true
                    ? "bg-emerald-400"
                    : isBackendHealthy === false
                      ? "bg-red-400"
                      : "bg-amber-400 animate-pulse"
                }`}
              />
              <span className="text-xs text-slate-400">
                {isBackendHealthy === true
                  ? "Backend Connected"
                  : isBackendHealthy === false
                    ? "Backend Offline"
                    : "Checking..."}
              </span>
            </div>
          </div>

          {/* Step Indicator */}
          <div className="mt-4">
            <StepIndicator currentStep={currentStep} onStepClick={setCurrentStep} />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Error Alert */}
        {error && (
          <div className="mb-6 bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-red-300 font-medium">Error</p>
              <p className="text-red-200/80 text-sm mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Step Content */}
        <div className="bg-slate-800/30 rounded-2xl border border-slate-700/50 p-8">
          {/* Step 1: Upload Sample */}
          {currentStep === "upload" && (
            <div className="max-w-2xl mx-auto">
              <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-slate-100">
                  Upload a Sample PDF
                </h2>
                <p className="text-slate-400 mt-2">
                  We'll analyze the document structure and suggest an extraction
                  schema
                </p>
              </div>
              <UploadZone
                onFileSelect={handleSampleUpload}
                isLoading={isLoading}
              />
            </div>
          )}

          {/* Step 2: Schema Editor */}
          {currentStep === "schema" && schema && (
      <div>
              <button
                onClick={() => setCurrentStep("upload")}
                className="flex items-center gap-2 text-slate-400 hover:text-slate-200 mb-6 transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                Back to Upload
              </button>
              <SchemaEditor
                schema={schema}
                onSchemaChange={setSchema}
                onConfirm={handleSchemaConfirm}
                isLoading={isLoading}
              />
      </div>
          )}

          {/* Step 3: Batch Processing */}
          {currentStep === "batch" && schema && (
            <div>
              <button
                onClick={() => setCurrentStep("schema")}
                className="flex items-center gap-2 text-slate-400 hover:text-slate-200 mb-6 transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                Back to Schema
        </button>

              {batchProgress.status === "idle" ? (
                <div className="max-w-2xl mx-auto space-y-6">
                  <div className="text-center">
                    <h2 className="text-2xl font-bold text-slate-100">
                      Upload PDFs for Batch Extraction
                    </h2>
                    <p className="text-slate-400 mt-2">
                      Using schema: <strong>{schema.name}</strong> (
                      {schema.fields.length} fields)
        </p>
      </div>

                  <MultiUploadZone
                    onFilesSelect={setBatchFiles}
                    selectedFiles={batchFiles}
                    isLoading={isLoading}
                  />

                  <div className="flex justify-center">
                    <button
                      onClick={handleBatchStart}
                      disabled={batchFiles.length === 0 || isLoading}
                      className="flex items-center gap-2 px-8 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-600 disabled:cursor-not-allowed rounded-xl font-medium transition-colors"
                    >
                      <Play className="w-5 h-5" />
                      Start Extraction ({batchFiles.length} files)
                    </button>
                  </div>
                </div>
              ) : (
                <BatchProgress progress={batchProgress} files={batchFiles} />
              )}
            </div>
          )}

          {/* Step 4: Results */}
          {currentStep === "results" && schema && results.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-6">
                <button
                  onClick={resetToStart}
                  className="flex items-center gap-2 text-slate-400 hover:text-slate-200 transition-colors"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Start New Extraction
                </button>
              </div>
              <ResultsTable
                results={results as ExtractBatchResponse[]}
                schema={schema}
                onRowClick={handleRowClick}
              />
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 mt-16 py-6">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-slate-500">
          AI PDF Extraction • Powered by GPT-4o
        </div>
      </footer>

      {/* Validation Modal */}
      {selectedResult && schema && (
        <ValidationModal
          result={selectedResult}
          schema={schema}
          pdfFile={selectedPdfFile}
          onClose={handleCloseModal}
        />
      )}
    </div>
  );
}
