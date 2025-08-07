# Stage 1: Builder
FROM python:3.12.9-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY services/data_ingestion/requirements.txt data_ingestion-requirements.txt
COPY core/requirements.txt core-requirements.txt

# Combine both requirements into one
RUN cat data_ingestion-requirements.txt core-requirements.txt > requirements.txt

# Install build tools and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir asyncpg

# Stage 2: Runtime
FROM python:3.12.9-slim

# Set working directory
WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    netcat-traditional \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy data ingestion service code
COPY services/data_ingestion /app/services/data_ingestion

# Copy core shared code
COPY /core /app/core

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run the application directly - the service will handle DB initialization
CMD ["uvicorn", "services.data_ingestion.main:app", "--host", "0.0.0.0", "--port", "8000"]
