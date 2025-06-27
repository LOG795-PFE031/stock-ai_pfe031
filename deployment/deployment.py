from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# import mlflow.pyfunc
import mlflow
import keras
import pandas as pd
import os


app = FastAPI()
MODELS_DIR = "/app/models/specific"
MLFLOW_DIR = "/app/mlflowModels"
loaded_models = {}

# Charger tous les modèles au démarrage
# print(os.listvolumes())
# print(os.listdir(MODELS_DIR))
# with mlflow.start_run():
#     for model_name in os.listdir(MODELS_DIR):
#         model_path = os.path.join(MODELS_DIR, model_name)
#         print('the model path is : '+ model_path)
#         if os.path.isdir(model_path):
#             choosenModel = os.path.join(model_path,model_name+'_model.keras')

#             print('the final  model path is : '+ choosenModel)
#             try:
#                 # loaded_models[model_name] = mlflow.pyfunc.load_model(model_path)
#                 # loaded_models[model_name] = mlflow.keras.load_model(model_path)
#                 model = keras.models.load_model(choosenModel)
#                 mlflow.keras.log_model(model, "model", registered_model_name=model_name)
#                 # with mlflow.start_run():
#             except Exception as e:
#                 print(f"Erreur chargement {model_name} : {e}")

with mlflow.start_run():
    for model_name in os.listdir(MODELS_DIR):
        if(model_name == 'AAPL'):
            model_path = os.path.join(MODELS_DIR, model_name)
            print('the model path is : '+ model_path)
            if os.path.isdir(model_path):
                chosenModel = os.path.join(model_path,model_name+'_model.keras')

                print('the final  model path is : '+ chosenModel)
                try:
                    # loaded_models[model_name] = mlflow.pyfunc.load_model(model_path)
                    # loaded_models[model_name] = mlflow.keras.load_model(model_path)
                    model = keras.models.load_model(filepath=chosenModel)
                    mlflow.keras.save_model(model, path=MLFLOW_DIR+"/mlflow_model")
                    model = mlflow.keras.load_model(MLFLOW_DIR+"/mlflow_model")
                    # with mlflow.start_run():
                except Exception as e:
                    print(f"Erreur chargement {model_name} : {e}")

class PredictionRequest(BaseModel):
    columns: list[str]
    data: list[list]

@app.post("/predict/{model_name}")
def predict(model_name: str, request: PredictionRequest):
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    df = pd.DataFrame(data=request.data, columns=request.columns)
    prediction = loaded_models[model_name].predict(df)
    return {"prediction": prediction.tolist()}



# while (True) :
#     print('impresseion')