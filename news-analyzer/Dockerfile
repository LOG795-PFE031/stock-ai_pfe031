# Use PyTorch as the base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    gnupg2 \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements files first to leverage caching
COPY requirements-base.txt ./
COPY requirements.txt ./

# Install dependencies as root
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements-base.txt && \
    python -m pip install --no-cache-dir -r requirements.txt

# Create a non-root user for running the application
RUN useradd -m -U app_user && \
    chown -R app_user:app_user /app

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R app_user:app_user /app/models /app/data /app/logs && \
    chmod -R 755 /app/models /app/data /app/logs

# Copy application code
COPY . .
RUN chown -R app_user:app_user /app

# Set environment variables
ENV PYTHONPATH=/app:/app/src \
    PATH=/opt/conda/bin:$PATH \
    # RabbitMQ configuration
    RABBITMQ_HOST=rabbitmq \
    RABBITMQ_PORT=5672 \
    RABBITMQ_USER=guest \
    RABBITMQ_PASS=guest \
    # API configuration
    API_HOST=0.0.0.0 \
    API_PORT=8092 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    REDIS_HOST=localhost \
    REDIS_PORT=6379 \
    YF_RATE_LIMIT=2 \
    YF_RATE_PERIOD=60

# Make port 8092 available
EXPOSE 8092

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8092/api/health || exit 1

# Start Redis and the application
CMD service redis-server start && su - app_user -c "cd /app && export PATH=/opt/conda/bin:$PATH && PYTHONPATH=/app:/app/src python -m src.main" 