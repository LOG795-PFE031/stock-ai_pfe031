# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install build tools and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install torch first to ensure availability for xformers
RUN pip install --no-cache-dir torch>=2.0.1

# Install ta package separately to ensure proper installation
RUN pip install --no-cache-dir ta>=0.10.0

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY core/ ./core/
COPY services/ ./services/
COPY training/ ./training/
COPY main.py .
COPY monitoring/ ./monitoring/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=-1  
#Ensures no GPU usage, aligns with CPU-only setup
ENV TF_CPP_MIN_LOG_LEVEL=2  
#Suppresses TensorFlow logs, though not strictly needed now
ENV HF_HUB_DOWNLOAD_TIMEOUT=120
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_OFFLINE=0
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV KERAS_BACKEND="torch" 
#Sets PyTorch as the Keras backend

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]