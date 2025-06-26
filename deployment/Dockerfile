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

EXPOSE 5000
# Exécuter le script
CMD ["python", "deployment.py"]

# ENTRYPOINT ["sh", "-c", "mlflow models serve -m /opt/ml/models/$MODEL_NAME -h 0.0.0.0 -p 5000"]