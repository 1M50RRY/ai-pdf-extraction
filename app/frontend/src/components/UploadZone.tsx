import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import {
  Upload,
  FileText,
  Loader2,
  ChevronDown,
  Bookmark,
  Sparkles,
} from "lucide-react";
import { listSchemas, type SavedSchema } from "../api";
import type { SchemaDefinition } from "../types";

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  onTemplateSelect?: (schema: SchemaDefinition, schemaId: string) => void;
  isLoading?: boolean;
  accept?: Record<string, string[]>;
  title?: string;
  subtitle?: string;
}

export function UploadZone({
  onFileSelect,
  onTemplateSelect,
  isLoading = false,
  accept = { "application/pdf": [".pdf"] },
  title = "Drop your PDF here",
  subtitle = "or click to browse",
}: UploadZoneProps) {
  const [templates, setTemplates] = useState<SavedSchema[]>([]);
  const [loadingTemplates, setLoadingTemplates] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<SavedSchema | null>(null);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  // Fetch templates on mount
  useEffect(() => {
    const fetchTemplates = async () => {
      setLoadingTemplates(true);
      try {
        const response = await listSchemas();
        setTemplates(response.schemas);
      } catch (err) {
        console.error("Failed to fetch templates:", err);
      } finally {
        setLoadingTemplates(false);
      }
    };
    fetchTemplates();
  }, []);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onFileSelect(acceptedFiles[0]);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive, acceptedFiles } =
    useDropzone({
      onDrop,
      accept,
      multiple: false,
      disabled: isLoading,
    });

  const handleTemplateSelect = (template: SavedSchema) => {
    setSelectedTemplate(template);
    setIsDropdownOpen(false);
    onTemplateSelect?.(template.structure, template.id);
  };

  const clearTemplate = () => {
    setSelectedTemplate(null);
    setIsDropdownOpen(false);
  };

  return (
    <div className="space-y-6">
      {/* Template Selector */}
      {templates.length > 0 && (
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center gap-3 mb-3">
            <Bookmark className="w-5 h-5 text-indigo-400" />
            <span className="font-medium text-slate-200">
              Use a Saved Template (Optional)
            </span>
          </div>

          <div className="relative">
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              disabled={loadingTemplates}
              className="w-full flex items-center justify-between px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-left hover:border-indigo-500 transition-colors"
            >
              <span className={selectedTemplate ? "text-slate-200" : "text-slate-400"}>
                {loadingTemplates
                  ? "Loading templates..."
                  : selectedTemplate
                    ? selectedTemplate.name
                    : "Select a template..."}
              </span>
              <ChevronDown
                className={`w-5 h-5 text-slate-400 transition-transform ${
                  isDropdownOpen ? "rotate-180" : ""
                }`}
              />
            </button>

            {isDropdownOpen && (
              <div className="absolute z-10 w-full mt-2 bg-slate-800 border border-slate-600 rounded-lg shadow-xl overflow-hidden">
                <button
                  onClick={clearTemplate}
                  className="w-full px-4 py-3 text-left text-slate-400 hover:bg-slate-700 transition-colors flex items-center gap-2"
                >
                  <Sparkles className="w-4 h-4" />
                  Discover schema from PDF (AI)
                </button>
                <div className="border-t border-slate-700" />
                {templates.map((template) => (
                  <button
                    key={template.id}
                    onClick={() => handleTemplateSelect(template)}
                    className="w-full px-4 py-3 text-left hover:bg-slate-700 transition-colors"
                  >
                    <div className="text-slate-200">{template.name}</div>
                    <div className="text-xs text-slate-500 mt-0.5">
                      {template.structure.fields.length} fields • v{template.version}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {selectedTemplate && (
            <p className="text-sm text-emerald-400 mt-3">
              ✓ Template selected. Upload PDFs to start batch extraction.
            </p>
          )}
        </div>
      )}

      {/* Upload Zone */}
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer
          transition-all duration-300 ease-out
          ${
            isDragActive
              ? "border-indigo-500 bg-indigo-500/10 scale-[1.02]"
              : "border-slate-600 hover:border-indigo-400 hover:bg-slate-800/50"
          }
          ${isLoading ? "pointer-events-none opacity-60" : ""}
        `}
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center gap-4">
          {isLoading ? (
            <Loader2 className="w-16 h-16 text-indigo-400 animate-spin" />
          ) : acceptedFiles.length > 0 ? (
            <FileText className="w-16 h-16 text-emerald-400" />
          ) : (
            <Upload
              className={`w-16 h-16 transition-colors ${
                isDragActive ? "text-indigo-400" : "text-slate-400"
              }`}
            />
          )}

          <div>
            <p className="text-xl font-semibold text-slate-200">
              {isLoading
                ? selectedTemplate
                  ? "Ready to process..."
                  : "Analyzing document..."
                : acceptedFiles.length > 0
                  ? acceptedFiles[0].name
                  : title}
            </p>
            <p className="text-sm text-slate-400 mt-1">
              {isLoading
                ? selectedTemplate
                  ? "Drop files to start batch extraction"
                  : "This may take a few seconds"
                : acceptedFiles.length > 0
                  ? `${(acceptedFiles[0].size / 1024).toFixed(1)} KB`
                  : selectedTemplate
                    ? "Upload PDF to start batch extraction"
                    : subtitle}
            </p>
          </div>
        </div>

        {isDragActive && (
          <div className="absolute inset-0 rounded-2xl bg-indigo-500/5 pointer-events-none" />
        )}
      </div>
    </div>
  );
}

interface MultiUploadZoneProps {
  onFilesSelect: (files: File[]) => void;
  selectedFiles: File[];
  isLoading?: boolean;
}

export function MultiUploadZone({
  onFilesSelect,
  selectedFiles,
  isLoading = false,
}: MultiUploadZoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      onFilesSelect([...selectedFiles, ...acceptedFiles]);
    },
    [onFilesSelect, selectedFiles]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    multiple: true,
    disabled: isLoading,
  });

  const removeFile = (index: number) => {
    const newFiles = selectedFiles.filter((_, i) => i !== index);
    onFilesSelect(newFiles);
  };

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-xl p-8 text-center cursor-pointer
          transition-all duration-200
          ${
            isDragActive
              ? "border-indigo-500 bg-indigo-500/10"
              : "border-slate-600 hover:border-indigo-400"
          }
          ${isLoading ? "pointer-events-none opacity-60" : ""}
        `}
      >
        <input {...getInputProps()} />
        <Upload className="w-10 h-10 text-slate-400 mx-auto mb-3" />
        <p className="text-slate-300">
          Drop multiple PDFs here or click to browse
        </p>
        <p className="text-sm text-slate-500 mt-1">
          {selectedFiles.length} file(s) selected
        </p>
      </div>

      {selectedFiles.length > 0 && (
        <div className="bg-slate-800/50 rounded-xl p-4 max-h-48 overflow-y-auto">
          <div className="space-y-2">
            {selectedFiles.map((file, index) => (
              <div
                key={`${file.name}-${index}`}
                className="flex items-center justify-between bg-slate-700/50 rounded-lg px-4 py-2"
              >
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5 text-indigo-400" />
                  <span className="text-sm text-slate-200 truncate max-w-xs">
                    {file.name}
                  </span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile(index);
                  }}
                  className="text-slate-400 hover:text-red-400 transition-colors"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
