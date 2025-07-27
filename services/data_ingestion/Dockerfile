# Use Python 3.12 slim image
FROM python:3.12.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY services/data_ingestion/requirements.txt data_ingestion-requirements.txt
COPY core/requirements.txt core-requirements.txt

# Combine both requirements into one
RUN cat data_ingestion-requirements.txt core-requirements.txt > requirements.txt


# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project for core modules
COPY . /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with uvicorn
CMD ["uvicorn", "services.data_ingestion.main:app", "--host", "0.0.0.0", "--port", "8000"]
