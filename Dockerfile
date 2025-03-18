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

# Copy the entire stock-prediction directory
COPY stock-prediction /app/stock-prediction

# Set environment variables
ENV PYTHONPATH=/app \
    MODELS_DIR=/app/stock-prediction/models \
    GENERAL_MODEL_PATH=/app/stock-prediction/models/general/general_model.keras \
    SYMBOL_ENCODER_PATH=/app/stock-prediction/models/general/symbol_encoder.gz \
    SECTOR_ENCODER_PATH=/app/stock-prediction/models/general/sector_encoder.gz \
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
CMD ["python", "-m", "stock-prediction.inference.api_server"]