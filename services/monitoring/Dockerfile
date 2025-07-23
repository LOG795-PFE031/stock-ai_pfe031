FROM python:3.10-slim
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY services/monitoring/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/monitoring/main.py .
COPY services/monitoring/monitoring_service.py .
COPY services/    ./services/  

# Copy shared instrumentation (Prometheus metrics & utils)
COPY core/             ./core/
COPY monitoring/ monitoring/

# Expose and healthcheck
EXPOSE 8008
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8008/health || exit 1

# Run the monitoring service
CMD ["python", "main.py"]
