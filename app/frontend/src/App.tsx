import { useState, useEffect, useCallback } from "react";
import {
  FileSearch,
  Settings,
  Play,
  Table2,
  ArrowLeft,
  Zap,
  AlertCircle,
  History,
} from "lucide-react";
import {
  UploadZone,
  MultiUploadZone,
  SchemaEditor,
  LiveProgress,
  EditableResultsTable,
  ValidationModal,
  HistoryModal,
  type EditableExtractionResult,
} from "./components";
import {
  uploadSample,
  healthCheck,
  startBatchExtraction,
  getBatchStatus,
  getSchema,
  type BatchStatusResponse,
} from "./api";
import type {
  AppStep,
  SchemaDefinition,
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
  const [schemaId, setSchemaId] = useState<string | null>(null);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<EditableExtractionResult[]>([]);

  // Validation modal state
  const [selectedResult, setSelectedResult] = useState<EditableExtractionResult | null>(null);
  const [selectedPdfFile, setSelectedPdfFile] = useState<File | undefined>(undefined);

  // History modal state
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);

  // Check backend health on mount
  useEffect(() => {
    healthCheck().then(setIsBackendHealthy);
  }, []);

  // Handle sample upload (for schema discovery)
  const handleSampleUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await uploadSample(file);
      setSchema(response.suggested_schema);
      setSchemaId(null); // Clear any previously selected template
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

  // Handle template selection (skip schema discovery)
  const handleTemplateSelect = (selectedSchema: SchemaDefinition, selectedSchemaId: string) => {
    setSchema(selectedSchema);
    setSchemaId(selectedSchemaId);
    setCurrentStep("batch"); // Skip schema editing, go straight to batch
  };

  // Handle schema confirmation
  const handleSchemaConfirm = () => {
    setCurrentStep("batch");
  };

  // Handle schema saved as template
  const handleSchemaSaved = (newSchemaId: string) => {
    setSchemaId(newSchemaId);
  };

  // Handle batch start
  const handleBatchStart = async () => {
    if (!schema || batchFiles.length === 0) return;

    setIsLoading(true);
    setError(null);
    setIsProcessing(true);

    try {
      const response = await startBatchExtraction(
        batchFiles,
        schema,
        schemaId || undefined
      );
      setBatchId(response.batch_id);
    } catch (err) {
      console.error("Batch start failed:", err);
      setError(
        err instanceof Error ? err.message : "Failed to start batch processing"
      );
      setIsProcessing(false);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle batch completion
  const handleBatchComplete = useCallback((status: BatchStatusResponse) => {
    // Convert batch status documents to editable results format
    const editableResults: EditableExtractionResult[] = status.documents
      .filter((doc) => doc.status === "completed" && doc.extracted_data)
      .map((doc) => ({
        id: doc.extraction_id || doc.id, // Use extraction_id for editing
        document_id: doc.id,
        source_file: doc.filename,
        page_number: 1,
        extracted_data: doc.extracted_data || {},
        confidence: doc.confidence || 0,
        field_confidences: doc.field_confidences, // Per-field confidence for cell highlighting
        warnings: doc.warnings,
        is_reviewed: false,
        manual_overrides: null,
      }));

    setResults(editableResults);
    setIsProcessing(false);
    setCurrentStep("results");
  }, []);

  // Handle batch error
  const handleBatchError = useCallback((errorMsg: string) => {
    setError(errorMsg);
    setIsProcessing(false);
  }, []);

  // Reset to start
  const resetToStart = () => {
    setCurrentStep("upload");
    setSchema(null);
    setSchemaId(null);
    setBatchFiles([]);
    setBatchId(null);
    setIsProcessing(false);
    setResults([]);
    setError(null);
    setSelectedResult(null);
    setSelectedPdfFile(undefined);
  };

  // Handle row click to open validation modal
  const handleRowClick = useCallback(
    (result: EditableExtractionResult) => {
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

  // Handle selecting a batch from history
  const handleSelectHistoryBatch = useCallback(async (selectedBatchId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const batchStatus = await getBatchStatus(selectedBatchId);
      
      // Fetch schema if available (needed to display results)
      if (batchStatus.schema_id) {
        try {
          const savedSchema = await getSchema(batchStatus.schema_id);
          setSchema(savedSchema.structure);
          setSchemaId(savedSchema.id);
        } catch (schemaErr) {
          console.warn("Could not load schema for batch:", schemaErr);
          // Continue without schema - results will still be shown
        }
      }
      
      // If batch is completed, show results
      if (batchStatus.status === "completed" || batchStatus.progress_percent >= 100) {
        setBatchId(selectedBatchId);
        handleBatchComplete(batchStatus);
      } else {
        // If still processing, show progress view
        setBatchId(selectedBatchId);
        setIsProcessing(true);
        setCurrentStep("batch");
      }
    } catch (err) {
      console.error("Failed to load batch:", err);
      setError(err instanceof Error ? err.message : "Failed to load batch");
    } finally {
      setIsLoading(false);
    }
  }, [handleBatchComplete]);

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
                <p className="text-xs text-slate-400">Enterprise Edition</p>
              </div>
            </div>

            {/* History Button & Backend Status */}
            <div className="flex items-center gap-4">
              <button
                onClick={() => setIsHistoryOpen(true)}
                className="flex items-center gap-2 px-3 py-2 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded-lg transition-colors"
              >
                <History className="w-4 h-4" />
                <span className="text-sm">History</span>
              </button>
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
                  Start Extraction
                </h2>
                <p className="text-slate-400 mt-2">
                  Select a saved template or upload a sample PDF to discover the schema
                </p>
              </div>
              <UploadZone
                onFileSelect={handleSampleUpload}
                onTemplateSelect={handleTemplateSelect}
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
                onSchemaSaved={handleSchemaSaved}
                isLoading={isLoading}
              />
      </div>
          )}

          {/* Step 3: Batch Processing */}
          {currentStep === "batch" && schema && (
            <div>
              <button
                onClick={() => setCurrentStep(schemaId ? "upload" : "schema")}
                className="flex items-center gap-2 text-slate-400 hover:text-slate-200 mb-6 transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                {schemaId ? "Back to Start" : "Back to Schema"}
        </button>

              {isProcessing && batchId ? (
                <LiveProgress
                  batchId={batchId}
                  onComplete={handleBatchComplete}
                  onError={handleBatchError}
                />
              ) : (
                <div className="max-w-2xl mx-auto space-y-6">
                  <div className="text-center">
                    <h2 className="text-2xl font-bold text-slate-100">
                      Upload PDFs for Batch Extraction
                    </h2>
                    <p className="text-slate-400 mt-2">
                      Using schema: <strong>{schema.name}</strong> (
                      {schema.fields.length} fields)
                      {schemaId && (
                        <span className="ml-2 text-emerald-400">• Saved Template</span>
                      )}
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
              )}
            </div>
          )}

          {/* Step 4: Results */}
          {currentStep === "results" && schema && results.length > 0 && batchId && (
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
              <EditableResultsTable
                results={results}
                schema={schema}
                batchId={batchId}
                onRowClick={handleRowClick}
              />
            </div>
          )}

          {/* Empty results fallback */}
          {currentStep === "results" && results.length === 0 && (
            <div className="text-center py-12">
              <p className="text-slate-400">No results to display.</p>
              <button
                onClick={resetToStart}
                className="mt-4 text-indigo-400 hover:text-indigo-300"
              >
                Start a new extraction
              </button>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 mt-16 py-6">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-slate-500">
          AI PDF Extraction • Enterprise Edition • Powered by GPT-4o
        </div>
      </footer>

      {/* Validation Modal */}
      {selectedResult && schema && (
        <ValidationModal
          result={{
            source_file: selectedResult.source_file,
            detected_schema: schema,
            extracted_data: selectedResult.extracted_data,
            confidence: selectedResult.confidence,
            warnings: selectedResult.warnings,
          }}
          schema={schema}
          pdfFile={selectedPdfFile}
          onClose={handleCloseModal}
        />
      )}

      {/* History Modal */}
      <HistoryModal
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        onSelectBatch={handleSelectHistoryBatch}
      />
    </div>
  );
}
