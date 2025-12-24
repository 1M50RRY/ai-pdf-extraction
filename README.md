# AI PDF Extraction Application

A production-grade PDF data extraction service using OpenAI GPT-4o with structured outputs.

## Features

- **Schema Detection**: Upload a sample PDF and get AI-suggested extraction schemas
- **Batch Extraction**: Extract structured data from multi-page PDFs
- **Type-Safe**: Full Pydantic validation for all inputs and outputs
- **Extensible**: Support for multiple field types (currency, date, email, etc.)

## Tech Stack

- **Backend**: Python FastAPI + Pydantic + Uvicorn
- **AI**: OpenAI GPT-4o with vision capabilities
- **PDF Processing**: pdf2image (poppler) for PDF to image conversion
- **Frontend**: React + Tailwind (coming soon)

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

### Python Environment

Python 3.11+ recommended.

## Installation

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

## Running the Application

### Development Server

```bash
uvicorn app.backend.main:app --reload --port 8000
```

Or run directly:

```bash
python -m app.backend.main
```

### API Documentation

Once running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

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

```ini
ai-pdf-extraction/
├── app/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI application
│   │   ├── models.py         # Pydantic models
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── pdf_service.py   # PDF processing
│   │       └── ai_service.py    # OpenAI integration
│   ├── frontend/             # React frontend (coming soon)
│   └── test-pdfs/            # Sample PDFs for testing
├── tests/                    # Test suite
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
  ]
}
```

## Supported Field Types

- `string` - Text fields
- `currency` - Money amounts
- `date` - Dates in any format
- `number` - Numeric values
- `boolean` - Yes/No values
- `email` - Email addresses
- `phone` - Phone numbers
- `address` - Physical addresses
- `percentage` - Percentage values

## Testing

```bash
pytest tests/ -v --cov=app
```

# Data Sources

- https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr
- https://universe.roboflow.com/jakob-awn1e/receipt-or-invoice
- https://data.nsw.gov.au/data/dataset/nsw-education-government-school-student-attendance-bulletin
- https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000789019&type=8&dateb=&owner=include&count=40&search_text=

## License

MIT

