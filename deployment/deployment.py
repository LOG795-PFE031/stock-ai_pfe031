from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os


app = FastAPI()
MODELS_DIR = "/app/models/specific"
loaded_models = {}

# Charger tous les modèles au démarrage
print(os.listvolumes())
for model_name in os.listdir(MODELS_DIR):
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.isdir(model_path):
        try:
            loaded_models[model_name] = mlflow.pyfunc.load_model(model_path)
        except Exception as e:
            print(f"Erreur chargement {model_name} : {e}")

# class PredictionRequest(BaseModel):
#     columns: list[str]
#     data: list[list]

# @app.post("/predict/{model_name}")
# def predict(model_name: str, request: PredictionRequest):
#     if model_name not in loaded_models:
#         raise HTTPException(status_code=404, detail="Modèle non trouvé")
#     df = pd.DataFrame(data=request.data, columns=request.columns)
#     prediction = loaded_models[model_name].predict(df)
#     return {"prediction": prediction.tolist()}



# while (True) :
#     print('impresseion')