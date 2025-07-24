# Stage 1: Builder
FROM python:3.12.9-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install build tools and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # Install torch first to ensure availability for xformers
    pip install --no-cache-dir torch>=2.0.1 && \
    # Install remaining dependencies
    pip install --no-cache-dir -r requirements.txt


RUN python -m textblob.download_corpora

# Stage 2: Runtime
FROM python:3.12.9-slim

# Set working directory
WORKDIR /app  

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    HF_HUB_DOWNLOAD_TIMEOUT=120 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_OFFLINE=0 \
    HF_HUB_DISABLE_TELEMETRY=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
