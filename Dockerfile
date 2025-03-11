# Use an official Python runtime as the base image
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /build

# Copy requirements files
COPY requirements-base.txt requirements.txt ./

# Download all requirements including dependencies
RUN pip download --no-cache-dir -r requirements.txt -d /build/wheels

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy wheels from builder
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements*.txt ./

# Install dependencies
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy only the application code and essential configuration files
COPY inference /app/inference

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app \
    MODELS_DIR=/app/models \
    GENERAL_MODEL_PATH=/app/models/general/general_model.keras \
    SYMBOL_ENCODER_PATH=/app/models/general/symbol_encoder.gz \
    SECTOR_ENCODER_PATH=/app/models/general/sector_encoder.gz \
    # RabbitMQ configuration
    RABBITMQ_HOST=rabbitmq \
    RABBITMQ_PORT=5672 \
    RABBITMQ_USER=guest \
    RABBITMQ_PASS=guest \
    # API configuration
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Make port 8000 available
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "inference.api_server"]