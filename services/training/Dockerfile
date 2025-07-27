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

# Copy core and training requirements
COPY services/training/requirements.txt training-requirements.txt
COPY core/requirements.txt core-requirements.txt

# Combine both requirements into one
RUN cat training-requirements.txt core-requirements.txt > requirements.txt

# Install build tools and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # Install remaining dependencies
    pip install --no-cache-dir -r requirements.txt


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
COPY services/training /app/services/training
COPY core /app/core

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    #Ensures no GPU usage, aligns with CPU-only setup
    TF_CPP_MIN_LOG_LEVEL=1 \ 
    GIT_PYTHON_REFRESH="quiet"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "services.training.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]