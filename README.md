# AI PDF Extraction Application

A production-grade PDF data extraction service using OpenAI GPT-4o with structured outputs.

## Features

- **Smart Schema Discovery**: Upload a sample PDF and get AI-suggested extraction schemas with Chain-of-Thought reasoning
- **Dynamic Validation Rules**: AI-generated math validation rules (e.g., `total == subtotal + tax`)
- **Batch Extraction**: Extract structured data from multiple PDFs with progress tracking
- **Confidence Scoring**: Logprob-based confidence calculations for each extraction
- **Side-by-Side Validation**: View PDF alongside extracted data to verify low-confidence fields
- **Export Functionality**: Export results to CSV or JSON with schema-matched headers
- **Type-Safe**: Full Pydantic validation for all inputs and outputs
- **Universal Currency Parsing**: Handles international formats ($1,000.00, €1.000,00, etc.)

## Tech Stack

### Backend
- **Framework**: Python FastAPI + Pydantic + Uvicorn
- **AI**: OpenAI GPT-4o with vision capabilities and structured outputs
- **PDF Processing**: pdf2image (poppler) for PDF to image conversion
- **Validation**: simpleeval for safe dynamic math expression evaluation
- **Currency Parsing**: price-parser for universal currency format support

### Frontend
- **Framework**: React 19 + TypeScript + Vite
- **Styling**: Tailwind CSS 4
- **PDF Viewing**: react-pdf for in-browser PDF rendering
- **Data Tables**: @tanstack/react-table for sortable, exportable tables
- **Icons**: lucide-react

## Prerequisites

### System Dependencies

**macOS:**

```bash
brew install poppler
```

**Ubuntu/Debian:**

```bash
sudo apt-get install poppler-utils
```

**Windows:**
Download poppler from: https://github.com/oschwartz10612/poppler-windows/releases

### Node.js

Node.js 18+ recommended for the frontend.

### Python Environment

Python 3.11+ recommended.

## Installation

### Backend Setup

1. **Clone and navigate:**

```bash
cd ai-pdf-extraction
```

2. **Create virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Frontend Setup

1. **Navigate to frontend:**

```bash
cd app/frontend
```

2. **Install dependencies:**

```bash
npm install
```

3. **Start development server:**

```bash
npm run dev
```

The frontend will be available at http://localhost:5173

## Running the Application

### Backend (Development Server)

```bash
uvicorn app.backend.main:app --reload --port 8000
```

Or run directly:

```bash
python -m app.backend.main
```

### Frontend (Development Server)

```bash
cd app/frontend
npm run dev
```

### Production Build

```bash
cd app/frontend
npm run build
```

### API Documentation

Once the backend is running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## UI Workflow

### 1. Sample Upload
Upload a sample PDF to analyze. The AI will detect the document type and suggest an extraction schema.

### 2. Schema Editor
Review and customize the detected schema:
- Rename fields
- Change field types
- Mark fields as required/optional
- View AI-generated validation rules

### 3. Batch Processing
Upload multiple PDFs (5+) for batch extraction. Progress is displayed in real-time.

### 4. Results View
View extraction results in a sortable table:
- **Confidence indicators**: Green (≥80%), Yellow (50-79%), Red (<50%)
- **Warning badges**: Hover to see validation warnings
- **Click any row** to open the side-by-side validation view

### 5. Side-by-Side Validation
When you click a row:
- **Left panel**: PDF viewer with zoom and page navigation
- **Right panel**: Extracted data in formatted or JSON view
- Visually verify low-confidence extractions

### 6. Export
Export results using the buttons above the table:
- **Export CSV**: Spreadsheet-compatible format with schema-matched headers
- **Export JSON**: Full extraction data including schema and metadata

## API Endpoints

### Health Check

```sh
GET /health
```

### Upload Sample PDF

```sh
POST /upload-sample
Content-Type: multipart/form-data

file: <PDF file>
```

Returns a suggested extraction schema based on AI analysis.

### Extract Batch

```sh
POST /extract-batch
Content-Type: multipart/form-data

file: <PDF file>
confirmed_schema: <JSON SchemaDefinition>
```

Returns extracted data for all pages.

## Project Structure

```
ai-pdf-extraction/
├── app/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── models.py            # Pydantic models
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── pdf_service.py   # PDF processing
│   │       └── ai_service.py    # OpenAI integration
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── components/
│   │   │   │   ├── UploadZone.tsx      # File upload with drag-and-drop
│   │   │   │   ├── SchemaEditor.tsx    # Schema customization
│   │   │   │   ├── BatchProgress.tsx   # Progress indicator
│   │   │   │   ├── ResultsTable.tsx    # Data table with export
│   │   │   │   └── ValidationModal.tsx # Side-by-side PDF/data view
│   │   │   ├── api.ts           # Backend API client
│   │   │   ├── types.ts         # TypeScript types
│   │   │   └── App.tsx          # Main application
│   │   ├── package.json
│   │   └── vite.config.ts
│   └── test-pdfs/               # Sample PDFs for testing
├── tests/                       # Test suite
├── requirements.txt
└── README.md
```

## Schema Definition Example

```json
{
  "name": "Invoice Schema",
  "description": "Standard invoice extraction",
  "version": "1.0",
  "fields": [
    {
      "name": "invoice_number",
      "type": "string",
      "description": "The unique invoice identifier",
      "required": true
    },
    {
      "name": "total_amount",
      "type": "currency",
      "description": "Total amount due",
      "required": true
    },
    {
      "name": "invoice_date",
      "type": "date",
      "description": "Date of the invoice",
      "required": true
    }
  ],
  "validation_rules": [
    "total_amount == subtotal + tax"
  ]
}
```

## Supported Field Types

- `string` - Text fields (default catch-all)
- `currency` - Money amounts (auto-parsed from any format)
- `date` - Dates (normalized to YYYY-MM-DD)
- `number` - Numeric values
- `boolean` - Yes/No values
- `email` - Email addresses
- `phone` - Phone numbers
- `address` - Physical addresses
- `percentage` - Percentage values

## Validation Rules

The AI can detect and output validation rules for numerical relationships:

```python
# Supported syntax
"total == subtotal + tax"
"net_income == gross_income - expenses"
"discount_amount == round(subtotal * discount_rate, 2)"

# Available functions
sum(), round(x, n), abs(), min(), max(), sqrt(), log(), len()
```

## Testing

```bash
pytest tests/ -v --cov=app
```

## Data Sources

- https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr
- https://universe.roboflow.com/jakob-awn1e/receipt-or-invoice
- https://data.nsw.gov.au/data/dataset/nsw-education-government-school-student-attendance-bulletin
- https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000789019&type=8&dateb=&owner=include&count=40&search_text=

## License

MIT
