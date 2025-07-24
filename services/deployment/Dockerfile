# Stage 1: Builder
FROM python:3.12.9-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

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
COPY . /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    #Ensures no GPU usage, aligns with CPU-only setup
    TF_CPP_MIN_LOG_LEVEL=2  \
    #Suppresses TensorFlow logs, though not strictly needed now
    #Sets PyTorch as the Keras backend
    KERAS_BACKEND="torch" 

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]