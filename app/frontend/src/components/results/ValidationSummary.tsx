import { Edit3 } from "lucide-react";

export function ValidationSummary() {
  return (
    <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-lg p-3 flex items-center gap-3">
      <Edit3 className="w-5 h-5 text-indigo-400" />
      <p className="text-sm text-indigo-200">
        <strong>Click any cell</strong> to edit. Changes are saved automatically.
        <span className="ml-2 inline-flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-indigo-500" />
          indicates manually edited cells.
        </span>
      </p>
    </div>
  );
}

