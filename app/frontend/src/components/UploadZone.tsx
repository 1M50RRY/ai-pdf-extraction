import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileText, Loader2 } from "lucide-react";

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  isLoading?: boolean;
  accept?: Record<string, string[]>;
  title?: string;
  subtitle?: string;
}

export function UploadZone({
  onFileSelect,
  isLoading = false,
  accept = { "application/pdf": [".pdf"] },
  title = "Drop your PDF here",
  subtitle = "or click to browse",
}: UploadZoneProps) {
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

  return (
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
              ? "Analyzing document..."
              : acceptedFiles.length > 0
                ? acceptedFiles[0].name
                : title}
          </p>
          <p className="text-sm text-slate-400 mt-1">
            {isLoading
              ? "This may take a few seconds"
              : acceptedFiles.length > 0
                ? `${(acceptedFiles[0].size / 1024).toFixed(1)} KB`
                : subtitle}
          </p>
        </div>
      </div>

      {isDragActive && (
        <div className="absolute inset-0 rounded-2xl bg-indigo-500/5 pointer-events-none" />
      )}
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

