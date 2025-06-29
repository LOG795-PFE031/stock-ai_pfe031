from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# import mlflow.pyfunc
import mlflow
import keras
import pandas as pd
import os
import shutil


app = FastAPI()
MODELS_DIR = "/app/models/specific"
MLFLOW_DIR = "/app/mlflowModels"
loaded_models = {}


### pour voir si tout marchait bien 
for filename in os.listdir(MLFLOW_DIR):
        file_path = os.path.join(MLFLOW_DIR, filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Supprime les fichiers
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Supprime les sous-dossiers
        except Exception as e:
            print(f"Erreur en supprimant {file_path}: {e}")







for model_name in os.listdir(MODELS_DIR):
        if model_name == "AAPL" :    
            model_path = os.path.join(MODELS_DIR, model_name)
            if os.path.isdir(model_path):
                chosenModel = os.path.join(model_path,model_name+'_model.keras')

                try:
                    model = keras.models.load_model(filepath=chosenModel)

                    save_path = os.path.join(MLFLOW_DIR, model_name)
                    mlflow.keras.save_model(model, path=save_path)              

                    print('Les modèles sauvegardés sont : ')
                    print(os.listdir(MLFLOW_DIR))    

                    mlflow.set_tracking_uri("app/mlruns")  # adapte à ton chemin

                    experiment_name = "AAPL_experiment"

                    # Récupérer ou créer l'expérience
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment is None:
                        experiment_id = mlflow.create_experiment(experiment_name)
                    else:
                        experiment_id = experiment.experiment_id           

                    # Relancer une run et logger le modèle
                    with mlflow.start_run(experiment_id=experiment_id):

                        mlflow.keras.log_model(keras_model=model, artifact_path="model", registered_model_name=model_name)
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

