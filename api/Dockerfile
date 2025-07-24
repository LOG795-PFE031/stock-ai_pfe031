FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl git procps \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY ../requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

# Copy gateway code and its dependencies
COPY api/        ./api/
COPY core/             ./core/
COPY db/               ./db/
COPY services/         ./services/
COPY prometheus/       ./prometheus/
COPY grafana/          ./grafana/

ENV API_HOST=0.0.0.0 \
  API_PORT=8000 \
  PREFECT_API_URL=http://prefect-server:4200/api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]