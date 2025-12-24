# Python 3.10 slim base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - poppler-utils: For PDF to image conversion
# - gcc: For compiling Python packages
# - libpq-dev: For PostgreSQL client library
# - curl: For healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run migrations and start server
CMD ["/bin/sh", "-c", "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000"]

