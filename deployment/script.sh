#!/bin/bash

# Lancer ton script Python
python ./deployment.py

# Ensuite lancer le UI de MLflow
mlflow ui --backend-store-uri app/mlruns --host 0.0.0.0 --port 5000

