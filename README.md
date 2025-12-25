# AI PDF Extraction Application

A production-grade PDF data extraction service using OpenAI GPT-4.1 with structured outputs.

## Features

- **Smart Schema Discovery**: Upload a sample PDF and get AI-suggested extraction schemas with Chain-of-Thought reasoning
- **Dynamic Validation Rules**: AI-generated math validation rules (e.g., `total == subtotal + tax`)
- **Batch Extraction**: Extract structured data from multiple PDFs with progress tracking
- **Confidence Scoring**: Logprob-based confidence calculations for each extraction
- **Side-by-Side Validation**: View PDF alongside extracted data to verify low-confidence fields
- **Export Functionality**: Export results to CSV or JSON with schema-matched headers
- **Type-Safe**: Full Pydantic validation for all inputs and outputs
- **Universal Currency Parsing**: Handles international formats ($1,000.00, €1.000,00, etc.)
- **PostgreSQL Persistence**: Schemas, documents, and extractions stored in a relational database
- **Docker Ready**: Full Docker Compose setup for production deployment

## Tech Stack

### Backend

- **Framework**: Python FastAPI + Pydantic + Uvicorn
- **Database**: PostgreSQL 15 + SQLAlchemy 2.0 + Alembic
- **AI**: OpenAI GPT-4.1 with vision capabilities and structured outputs
- **PDF Processing**: pdf2image (poppler) for PDF to image conversion
- **Validation**: simpleeval for safe dynamic math expression evaluation
- **Currency Parsing**: price-parser for universal currency format support

### Frontend

- **Framework**: React 19 + TypeScript + Vite
- **Styling**: Tailwind CSS 4
- **PDF Viewing**: react-pdf for in-browser PDF rendering
- **Data Tables**: @tanstack/react-table for sortable, exportable tables
- **Icons**: lucide-react

### Infrastructure

- **Containerization**: Docker + Docker Compose
- **Database**: PostgreSQL 15 Alpine
- **Migrations**: Alembic with autogenerate support

## Quick Start with Docker

The fastest way to run the full application:

```bash
# Clone the repository
git clone <repository-url>
cd ai-pdf-extraction

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Start all services
docker compose up --build
```

Services will be available at:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432

### Docker Commands

```bash
# Start in background
docker compose up -d

# View logs
docker compose logs -f backend

# Stop all services
docker compose down

# Reset database (removes all data)
docker compose down -v
```

## Prerequisites (Local Development)

### System Dependencies

**macOS:**

```bash
brew install poppler postgresql
```

**Ubuntu/Debian:**

```bash
sudo apt-get install poppler-utils postgresql-client libpq-dev
```

**Windows:**
Download poppler from: https://github.com/oschwartz10612/poppler-windows/releases

### Node.js

Node.js 18+ recommended for the frontend.

### Python Environment

Python 3.10+ recommended.

## Local Development Setup

### 1. Database Setup

Start PostgreSQL (or use Docker):

```bash
# Using Docker for just the database
docker compose up db -d

# Or create a local database
createdb extraction_db
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key-here"
export DATABASE_URL="postgresql://user:password@localhost:5432/extraction_db"

# Run database migrations
cd app/backend
alembic upgrade head

# Start the backend
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup

```bash
cd app/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at http://localhost:5173

## Database Schema

### Entity Relationship

```
SavedSchema (1) ──────< DocumentBatch (1) ──────< Document (1) ──────< Extraction
     │                       │                        │                    │
     │                       │                        │                    │
  • id (UUID)             • id (UUID)              • id (UUID)          • id (UUID)
  • name                  • schema_id (FK)         • batch_id (FK)      • document_id (FK)
  • description           • name                   • filename           • page_number
  • version               • created_at             • status             • data (JSON)
  • structure (JSON)      • completed_at           • file_hash          • confidence
  • created_at            • total_documents        • upload_date        • warnings (JSON)
  • is_active             • successful_documents   • processed_at       • is_reviewed
                          • failed_documents       • error_message      • manual_overrides
```

### Document Status Flow

```
PENDING → PROCESSING → COMPLETED
                    ↘ FAILED
```

## Database Migrations

```bash
cd app/backend

# Create a new migration (after modifying models_db.py)
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# View migration history
alembic history
```

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
├── docker-compose.yml          # Docker Compose configuration
├── app/
│   ├── backend/
│   │   ├── Dockerfile          # Backend container definition
│   │   ├── main.py             # FastAPI application
│   │   ├── models.py           # Pydantic API models
│   │   ├── models_db.py        # SQLAlchemy ORM models
│   │   ├── database.py         # Database connection setup
│   │   ├── alembic.ini         # Alembic configuration
│   │   ├── alembic/
│   │   │   ├── env.py          # Migration environment
│   │   │   └── versions/       # Migration files
│   │   └── services/
│   │       ├── pdf_service.py  # PDF processing
│   │       └── ai_service.py   # OpenAI integration
│   ├── frontend/
│   │   ├── Dockerfile          # Frontend container definition
│   │   ├── src/
│   │   │   ├── components/
│   │   │   │   ├── UploadZone.tsx
│   │   │   │   ├── SchemaEditor.tsx
│   │   │   │   ├── BatchProgress.tsx
│   │   │   │   ├── ResultsTable.tsx
│   │   │   │   └── ValidationModal.tsx
│   │   │   ├── api.ts
│   │   │   ├── types.ts
│   │   │   └── App.tsx
│   │   └── package.json
│   └── test-pdfs/
├── tests/
├── requirements.txt
└── README.md
```

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
  "validation_rules": ["total_amount == subtotal + tax"]
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

## Environment Variables

| Variable         | Description                  | Default                                                   |
| ---------------- | ---------------------------- | --------------------------------------------------------- |
| `OPENAI_API_KEY` | OpenAI API key (required)    | -                                                         |
| `DATABASE_URL`   | PostgreSQL connection string | `postgresql://user:password@localhost:5432/extraction_db` |
| `SQL_DEBUG`      | Enable SQL query logging     | `false`                                                   |

## Testing

```bash
# Run all tests
pytest tests/ -v --cov=app

# Run specific test file
pytest tests/test_ai_service.py -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=html
```

## Data Sources

- https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr
- https://universe.roboflow.com/jakob-awn1e/receipt-or-invoice
- https://data.nsw.gov.au/data/dataset/nsw-education-government-school-student-attendance-bulletin
- https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000789019&type=8&dateb=&owner=include&count=40&search_text=

## License

MIT
