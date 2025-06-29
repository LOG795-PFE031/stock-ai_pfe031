FROM python:3.10-slim

WORKDIR /app

# # Installer MLflow
# COPY ../requirements.txt requirements
# RUN pip install requirements

COPY requirements.txt .                 
RUN pip install --no-cache-dir -r requirements.txt 

COPY deployment/deployment.py .
# COPY deployment.py .

# # Par défaut, servir le modèle 'model_a' (ou celui que tu veux)
# ENV MODEL_NAME=specific
COPY deployment/script.sh .


EXPOSE 5000
# Exécuter le script
# CMD ["python", "deployment.py"]

# CMD ["mlflow ui --backend-store-uri /mlruns --host 0.0.0.0 --port 5000"]

RUN chmod +x ./script.sh

CMD ["./script.sh"]