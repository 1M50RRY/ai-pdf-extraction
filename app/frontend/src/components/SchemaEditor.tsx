import { useState } from "react";
import {
  Edit3,
  Trash2,
  Plus,
  Check,
  X,
  ChevronDown,
  AlertCircle,
  Save,
  Bookmark,
  Loader2,
} from "lucide-react";
import { saveSchema } from "../api";
import type { FieldDefinition, FieldType, SchemaDefinition } from "../types";

const FIELD_TYPES: FieldType[] = [
  "string",
  "currency",
  "date",
  "number",
  "boolean",
  "email",
  "phone",
  "address",
  "percentage",
];

const TYPE_COLORS: Record<FieldType, string> = {
  string: "bg-blue-500/20 text-blue-300 border-blue-500/30",
  currency: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
  date: "bg-purple-500/20 text-purple-300 border-purple-500/30",
  number: "bg-amber-500/20 text-amber-300 border-amber-500/30",
  boolean: "bg-pink-500/20 text-pink-300 border-pink-500/30",
  email: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
  phone: "bg-orange-500/20 text-orange-300 border-orange-500/30",
  address: "bg-teal-500/20 text-teal-300 border-teal-500/30",
  percentage: "bg-rose-500/20 text-rose-300 border-rose-500/30",
};

interface SaveTemplateModalProps {
  schema: SchemaDefinition;
  onClose: () => void;
  onSaved: (schemaId: string) => void;
}

function SaveTemplateModal({ schema, onClose, onSaved }: SaveTemplateModalProps) {
  const [name, setName] = useState(schema.name || "");
  const [description, setDescription] = useState(schema.description || "");
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSave = async () => {
    if (!name.trim()) {
      setError("Template name is required");
      return;
    }

    setIsSaving(true);
    setError(null);

    try {
      const schemaToSave = {
        ...schema,
        name: name.trim(),
        description: description.trim(),
      };
      const saved = await saveSchema(schemaToSave);
      onSaved(saved.id);
      onClose();
    } catch (err) {
      console.error("Failed to save template:", err);
      setError(
        err instanceof Error ? err.message : "Failed to save template"
      );
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative bg-slate-800 rounded-2xl shadow-2xl border border-slate-700 w-full max-w-md p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-indigo-500/20 flex items-center justify-center">
            <Bookmark className="w-5 h-5 text-indigo-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-100">
              Save as Template
            </h3>
            <p className="text-sm text-slate-400">
              Reuse this schema for future extractions
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Template Name *
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Vendor X Invoice"
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description..."
              rows={3}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 resize-none"
            />
          </div>

          <div className="bg-slate-700/50 rounded-lg p-3">
            <p className="text-sm text-slate-400">
              <strong className="text-slate-300">{schema.fields.length}</strong> fields
              {schema.validation_rules.length > 0 && (
                <> • <strong className="text-slate-300">{schema.validation_rules.length}</strong> validation rules</>
              )}
            </p>
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 text-slate-400 hover:text-slate-200 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving || !name.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-600 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
          >
            {isSaving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Save className="w-4 h-4" />
            )}
            Save Template
          </button>
        </div>
      </div>
    </div>
  );
}

interface SchemaEditorProps {
  schema: SchemaDefinition;
  onSchemaChange: (schema: SchemaDefinition) => void;
  onConfirm: () => void;
  isLoading?: boolean;
  onSchemaSaved?: (schemaId: string) => void;
}

export function SchemaEditor({
  schema,
  onSchemaChange,
  onConfirm,
  isLoading = false,
  onSchemaSaved,
}: SchemaEditorProps) {
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editField, setEditField] = useState<FieldDefinition | null>(null);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [savedTemplateId, setSavedTemplateId] = useState<string | null>(null);

  const startEditing = (index: number) => {
    setEditingIndex(index);
    setEditField({ ...schema.fields[index] });
  };

  const saveEdit = () => {
    if (editingIndex !== null && editField) {
      const newFields = [...schema.fields];
      newFields[editingIndex] = editField;
      onSchemaChange({ ...schema, fields: newFields });
      setEditingIndex(null);
      setEditField(null);
    }
  };

  const cancelEdit = () => {
    setEditingIndex(null);
    setEditField(null);
  };

  const deleteField = (index: number) => {
    const newFields = schema.fields.filter((_, i) => i !== index);
    onSchemaChange({ ...schema, fields: newFields });
  };

  const addField = () => {
    const newField: FieldDefinition = {
      name: `new_field_${schema.fields.length + 1}`,
      type: "string",
      description: "New field description",
      required: false,
    };
    onSchemaChange({ ...schema, fields: [...schema.fields, newField] });
  };

  const toggleRequired = (index: number) => {
    const newFields = [...schema.fields];
    newFields[index] = {
      ...newFields[index],
      required: !newFields[index].required,
    };
    onSchemaChange({ ...schema, fields: newFields });
  };

  const handleSchemaSaved = (schemaId: string) => {
    setSavedTemplateId(schemaId);
    onSchemaSaved?.(schemaId);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-100">{schema.name}</h2>
          <p className="text-slate-400 mt-1 text-sm">
            {schema.fields.length} fields detected •{" "}
            {schema.validation_rules.length} validation rules
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowSaveModal(true)}
            disabled={savedTemplateId !== null}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
              savedTemplateId
                ? "bg-emerald-500/20 text-emerald-400 cursor-default"
                : "bg-slate-700 hover:bg-slate-600 text-slate-200"
            }`}
          >
            {savedTemplateId ? (
              <>
                <Check className="w-4 h-4" />
                Template Saved
              </>
            ) : (
              <>
                <Bookmark className="w-4 h-4" />
                Save as Template
              </>
            )}
          </button>
          <button
            onClick={addField}
            className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" />
            Add Field
          </button>
        </div>
      </div>

      {/* Validation Rules */}
      {schema.validation_rules.length > 0 && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-4 h-4 text-amber-400" />
            <span className="text-sm font-medium text-amber-300">
              Validation Rules
            </span>
          </div>
          <div className="space-y-1">
            {schema.validation_rules.map((rule, i) => (
              <code
                key={i}
                className="block text-xs text-amber-200/80 font-mono"
              >
                {rule}
              </code>
            ))}
          </div>
        </div>
      )}

      {/* Fields Table */}
      <div className="bg-slate-800/50 rounded-xl overflow-hidden border border-slate-700">
        <table className="w-full">
          <thead>
            <tr className="bg-slate-800">
              <th className="text-left px-4 py-3 text-sm font-medium text-slate-400">
                Field Name
              </th>
              <th className="text-left px-4 py-3 text-sm font-medium text-slate-400">
                Type
              </th>
              <th className="text-left px-4 py-3 text-sm font-medium text-slate-400">
                Description
              </th>
              <th className="text-center px-4 py-3 text-sm font-medium text-slate-400">
                Required
              </th>
              <th className="text-right px-4 py-3 text-sm font-medium text-slate-400">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700/50">
            {schema.fields.map((field, index) => (
              <tr
                key={index}
                className="hover:bg-slate-700/30 transition-colors"
              >
                {editingIndex === index && editField ? (
                  <>
                    <td className="px-4 py-3">
                      <input
                        type="text"
                        value={editField.name}
                        onChange={(e) =>
                          setEditField({ ...editField, name: e.target.value })
                        }
                        className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1 text-sm focus:outline-none focus:border-indigo-500"
                      />
                    </td>
                    <td className="px-4 py-3">
                      <div className="relative">
                        <select
                          value={editField.type}
                          onChange={(e) =>
                            setEditField({
                              ...editField,
                              type: e.target.value as FieldType,
                            })
                          }
                          className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1 text-sm appearance-none focus:outline-none focus:border-indigo-500"
                        >
                          {FIELD_TYPES.map((type) => (
                            <option key={type} value={type}>
                              {type}
                            </option>
                          ))}
                        </select>
                        <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <input
                        type="text"
                        value={editField.description}
                        onChange={(e) =>
                          setEditField({
                            ...editField,
                            description: e.target.value,
                          })
                        }
                        className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1 text-sm focus:outline-none focus:border-indigo-500"
                      />
                    </td>
                    <td className="px-4 py-3 text-center">
                      <input
                        type="checkbox"
                        checked={editField.required}
                        onChange={(e) =>
                          setEditField({
                            ...editField,
                            required: e.target.checked,
                          })
                        }
                        className="w-4 h-4 rounded border-slate-600 bg-slate-700 text-indigo-500 focus:ring-indigo-500"
                      />
                    </td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={saveEdit}
                          className="p-1 text-emerald-400 hover:bg-emerald-400/20 rounded transition-colors"
                        >
                          <Check className="w-4 h-4" />
                        </button>
                        <button
                          onClick={cancelEdit}
                          className="p-1 text-slate-400 hover:bg-slate-600 rounded transition-colors"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </>
                ) : (
                  <>
                    <td className="px-4 py-3">
                      <code className="text-sm font-mono text-slate-200">
                        {field.name}
                      </code>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-block px-2 py-0.5 rounded text-xs font-medium border ${TYPE_COLORS[field.type]}`}
                      >
                        {field.type}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-slate-400 max-w-xs truncate">
                      {field.description}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <button
                        onClick={() => toggleRequired(index)}
                        className={`w-5 h-5 rounded flex items-center justify-center transition-colors ${
                          field.required
                            ? "bg-indigo-500 text-white"
                            : "bg-slate-700 text-slate-500"
                        }`}
                      >
                        {field.required && <Check className="w-3 h-3" />}
                      </button>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => startEditing(index)}
                          className="p-1 text-slate-400 hover:text-indigo-400 hover:bg-indigo-400/20 rounded transition-colors"
                        >
                          <Edit3 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => deleteField(index)}
                          className="p-1 text-slate-400 hover:text-red-400 hover:bg-red-400/20 rounded transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Confirm Button */}
      <div className="flex justify-end">
        <button
          onClick={onConfirm}
          disabled={isLoading || schema.fields.length === 0}
          className="flex items-center gap-2 px-6 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-600 disabled:cursor-not-allowed rounded-xl font-medium transition-colors"
        >
          <Check className="w-5 h-5" />
          Confirm Schema & Start Batch
        </button>
      </div>

      {/* Save Template Modal */}
      {showSaveModal && (
        <SaveTemplateModal
          schema={schema}
          onClose={() => setShowSaveModal(false)}
          onSaved={handleSchemaSaved}
        />
      )}
    </div>
  );
}
