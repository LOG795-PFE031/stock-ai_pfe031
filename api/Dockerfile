# Stage 1: Builder
FROM python:3.12.9-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY ../requirements.txt .


# Copy core and api requirements
COPY api/requirements.txt api-requirements.txt
COPY core/requirements.txt core-requirements.txt

# Combine all requirements into one file
RUN cat api-requirements.txt core-requirements.txt > requirements.txt

# Install build tools and wheel
# Install build tools and wheel
RUN pip install --no-cache-dir -r requirements.txt


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

# Copy gateway code and its dependencies
COPY api/        ./api/
COPY core/             ./core/
COPY services/monitoring ./services/monitoring

# Set environment variables
ENV API_HOST=0.0.0.0 \
  API_PORT=8000 \
  PREFECT_API_URL=http://prefect-server:4200/api \
  MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
