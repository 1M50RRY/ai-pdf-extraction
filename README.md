# AI Document Extraction Service

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

A production-grade AI-powered document extraction service that automatically discovers schemas, extracts structured data from PDFs, and provides intelligent data repair capabilities. Built with FastAPI, React, and OpenAI GPT-4.1.

## 🚀 Key Features

### 🔍 Adaptive Schema Discovery
- **Multi-page sampling**: Intelligently analyzes first, middle, and last pages to detect tables and data structures hidden deep in documents
- **Zero-configuration**: Upload a sample PDF and get AI-suggested extraction schemas with semantic field names
- **Array detection**: Automatically identifies tables and lists as array fields (e.g., invoice line items, transaction history)
- **Chain-of-Thought reasoning**: AI explains document classification and field selection rationale

### ⚡ Batch Processing
- **Parallel extraction**: Process multiple PDFs concurrently with configurable concurrency limits (default: 5)
- **Real-time progress**: Live status updates with per-file processing indicators
- **Queue system**: Asynchronous background processing with PostgreSQL-backed job tracking
- **Error handling**: Graceful failure recovery with detailed error messages

### 🧠 Smart Repair (AI Agent)
- **Forensic Accountant persona**: Secondary LLM call that acts as a data analyst to complete missing values
- **Formula inference**: Automatically calculates missing fields based on schema semantics (e.g., `total_cost` → sum of `line_items`)
- **Cross-reference logic**: Finds values in headers/footers that were missed in main body extraction
- **Mathematical consistency**: Enforces logical relationships (e.g., `tax = total - subtotal`)
- **On-demand repair**: Trigger smart repair for individual documents or entire batches

### 🎨 Interactive UI
- **Confidence scoring**: Per-field confidence indicators with color-coded cells (Green ≥80%, Yellow 50-79%, Red <50%)
- **Visual warnings**: Tooltips and badges for validation errors and missing fields
- **Full JSON editing**: Click any cell to edit extracted values with automatic persistence
- **Side-by-side validation**: View PDF alongside extracted data for visual verification
- **Smart cells**: Special handling for arrays with nested table views and inline editing
- **Export functionality**: CSV and JSON export with schema-matched headers and confidence scores

### 📊 History & Analytics
- **Persistent storage**: All extractions saved to PostgreSQL with full audit trail
- **Batch history**: View past extractions with filtering and search
- **PDF preview**: Access original PDFs from history via API endpoints
- **Review workflow**: Human-in-the-loop approval system with manual override tracking
- **Schema templates**: Save and reuse extraction schemas for recurring document types

## 🛠️ Tech Stack

### Backend
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) 0.109+ with Pydantic 2.5+ for type-safe APIs
- **Database**: [PostgreSQL](https://www.postgresql.org/) 15 with [SQLAlchemy](https://www.sqlalchemy.org/) 2.0 ORM
- **Migrations**: [Alembic](https://alembic.sqlalchemy.org/) for database versioning
- **AI Engine**: [OpenAI GPT-4.1](https://openai.com/) with vision capabilities and structured outputs
- **PDF Processing**: [pdf2image](https://github.com/Belval/pdf2image) (Poppler) for PDF to image conversion
- **Validation**: [simpleeval](https://github.com/danthedeckie/simpleeval) for safe dynamic math expression evaluation
- **Currency Parsing**: [price-parser](https://github.com/scrapinghub/price-parser) for universal currency format support

### Frontend
- **Framework**: [React](https://react.dev/) 19 with [TypeScript](https://www.typescriptlang.org/) 5.9+
- **Build Tool**: [Vite](https://vitejs.dev/) 7.2+ for fast development and optimized builds
- **Styling**: [Tailwind CSS](https://tailwindcss.com/) 4.1+ for utility-first styling
- **Routing**: [React Router](https://reactrouter.com/) 7.11+ for client-side navigation
- **Data Tables**: [TanStack Table](https://tanstack.com/table) 8.21+ for sortable, filterable tables
- **PDF Viewing**: [react-pdf](https://react-pdf.org/) 10.2+ for in-browser PDF rendering
- **Icons**: [Lucide React](https://lucide.dev/) for consistent iconography
- **HTTP Client**: [Axios](https://axios-http.com/) for API communication

### Infrastructure
- **Containerization**: Docker + Docker Compose for reproducible deployments
- **Database**: PostgreSQL 15 Alpine for lightweight container
- **Reverse Proxy**: Nginx (in production) for serving frontend and proxying API

## 📁 Project Structure

```
ai-pdf-extraction/
├── docker-compose.yml              # Docker Compose configuration
├── app/
│   ├── backend/
│   │   ├── Dockerfile              # Backend container definition
│   │   ├── main.py                 # FastAPI app initialization & routing
│   │   ├── models.py               # Pydantic API request/response models
│   │   ├── models_db.py            # SQLAlchemy ORM models
│   │   ├── database.py             # Database connection & session management
│   │   ├── config.py               # Configuration & environment variables
│   │   ├── requirements.txt        # Python dependencies
│   │   ├── alembic.ini             # Alembic configuration
│   │   ├── alembic/                 # Database migrations
│   │   │   ├── env.py
│   │   │   └── versions/
│   │   ├── routers/                 # API route modules
│   │   │   ├── batches.py          # Batch processing endpoints
│   │   │   ├── documents.py        # Document content & smart repair
│   │   │   ├── history.py          # Batch history & extraction details
│   │   │   ├── schemas.py          # Schema template management
│   │   │   └── upload.py           # Sample upload & schema discovery
│   │   └── services/
│   │       ├── pdf_service.py       # PDF to image conversion
│   │       ├── ai_service.py       # Backward-compatible wrapper
│   │       └── ai/                 # AI service modules
│   │           ├── discovery.py    # Schema discovery logic
│   │           ├── extraction.py   # Data extraction with map-reduce
│   │           ├── repair.py       # Smart repair (Forensic Accountant)
│   │           ├── validation.py  # Data validation & normalization
│   │           └── exceptions.py   # Custom exceptions
│   ├── frontend/
│   │   ├── Dockerfile              # Frontend container definition
│   │   ├── nginx.conf              # Nginx configuration (production)
│   │   ├── package.json            # Node.js dependencies
│   │   └── src/
│   │       ├── pages/               # Page components
│   │       │   ├── DashboardPage.tsx    # Main extraction workflow
│   │       │   └── HistoryPage.tsx      # Batch history view
│   │       ├── components/         # React components
│   │       │   ├── UploadZone.tsx        # File upload with drag-drop
│   │       │   ├── SchemaEditor.tsx     # Schema customization UI
│   │       │   ├── LiveProgress.tsx     # Real-time batch progress
│   │       │   ├── EditableResultsTable.tsx  # Main results table
│   │       │   ├── SmartCell.tsx         # Editable cell with confidence
│   │       │   ├── ValidationModal.tsx   # Side-by-side PDF + JSON view
│   │       │   ├── results/              # Results table sub-components
│   │       │   │   ├── TableToolbar.tsx  # Stats & action buttons
│   │       │   │   ├── ValidationSummary.tsx  # Edit instructions
│   │       │   │   └── hooks/
│   │       │   │       └── useTableConfig.ts  # Table configuration hook
│   │       │   └── ...
│   │       ├── api.ts              # API client functions
│   │       ├── types.ts            # TypeScript type definitions
│   │       ├── App.tsx              # Root component with routing
│   │       └── main.tsx            # Application entry point
│   └── test-pdfs/                  # Sample PDFs for testing
├── tests/                          # Backend test suite
│   ├── test_ai_service.py         # AI service tests
│   ├── test_api.py                # API endpoint tests
│   └── ...
├── prompt-documentation.md         # Prompt engineering documentation
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- **Docker** and **Docker Compose** (recommended)
- **Python** 3.10+ (for local development)
- **Node.js** 18+ (for local frontend development)
- **PostgreSQL** 15+ (or use Docker)
- **Poppler** (for PDF processing): `brew install poppler` (macOS) or `apt-get install poppler-utils` (Linux)
- **OpenAI API Key**: Get one from [OpenAI Platform](https://platform.openai.com/api-keys)

### Option 1: Docker Compose (Recommended)

The fastest way to run the entire application:

```bash
# Clone the repository
git clone <repository-url>
cd ai-pdf-extraction

# Create environment file for backend
cat > app/backend/.env << EOF
OPENAI_API_KEY=your-api-key-here
DATABASE_URL=postgresql://user:password@db:5432/extraction_db
EOF

# Start all services
docker compose up --build
```

Services will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432

### Option 2: Local Development

#### 1. Database Setup

```bash
# Using Docker for just the database
docker compose up db -d

# Or create a local PostgreSQL database
createdb extraction_db
```

#### 2. Backend Setup

```bash
cd app/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key-here"
export DATABASE_URL="postgresql://user:password@localhost:5432/extraction_db"

# Run database migrations
alembic upgrade head

# Start the backend server
uvicorn main:app --reload --port 8000
```

The backend API will be available at http://localhost:8000

#### 3. Frontend Setup

```bash
cd app/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at http://localhost:5173

## ⚙️ Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4.1 | ✅ Yes | - |
| `DATABASE_URL` | PostgreSQL connection string | ✅ Yes | `postgresql://user:password@localhost:5432/extraction_db` |
| `SQL_DEBUG` | Enable SQL query logging | No | `false` |

### Example `.env` file (for backend)

```bash
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:password@localhost:5432/extraction_db
SQL_DEBUG=false
```

## 🔄 Workflow

### 1. **Schema Discovery**
   - Upload a sample PDF (invoice, report, form, etc.)
   - AI analyzes the document structure and suggests extraction fields
   - Review and customize the detected schema

### 2. **Batch Processing**
   - Upload multiple PDFs (5+ files)
   - System processes files in parallel with real-time progress
   - Each extraction includes confidence scores and validation warnings

### 3. **Review & Edit**
   - View results in an interactive table with color-coded confidence indicators
   - Click any cell to edit extracted values
   - Use side-by-side view to verify extractions against the original PDF

### 4. **Smart Repair** (Optional)
   - Click "🧮 Smart Calculate" to trigger the AI repair agent
   - System infers missing values and fixes logical inconsistencies
   - Calculated fields are marked with a blue indicator

### 5. **Export & Approve**
   - Export results to CSV or JSON
   - Approve all extractions to mark them as reviewed
   - Access past extractions from the History page

## 📊 Database Schema

### Core Entities

- **SavedSchema**: Reusable extraction templates with field definitions
- **DocumentBatch**: Groups of documents processed together
- **Document**: Individual PDF files with processing status
- **Extraction**: Extracted data with confidence scores and validation warnings

### Status Flow

```
Document: PENDING → PROCESSING → COMPLETED
                        ↓
                     FAILED
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test file
pytest tests/test_ai_service.py -v
```

## 📚 Documentation

- **[Prompt Engineering Documentation](./prompt-documentation.md)**: Detailed explanation of AI prompts, iteration history, and design decisions
- **API Documentation**: Interactive docs available at http://localhost:8000/docs when backend is running

## 🎯 Key Design Decisions

### Adaptive Sampling
Instead of analyzing all pages (expensive) or just the first page (misses mid-document tables), we sample 6 representative pages (first 2, last 2, middle 2) to balance cost and coverage.

### Map-Reduce Extraction
For large documents (>10 pages), we split into chunks of 5 pages, process in parallel, then merge results. This reduces latency from 5 minutes to ~1.5 minutes for 50-page documents.

### Post-Merge Validation
Validation runs **once** on the final merged data, not per-chunk, preventing false warnings from partial extractions.

### Smart Repair vs. Hardcoded Rules
Instead of hardcoded heuristics (`if "tax" is None: tax = total - subtotal`), we use a secondary LLM call that infers formulas from schema semantics, handling edge cases like multi-currency and custom tax rates.

## 📄 License

MIT

---

**Built with ❤️ using FastAPI, React, and OpenAI GPT-4.1**
