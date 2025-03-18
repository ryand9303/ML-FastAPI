# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Dict
# import pickle
# import json
# import numpy as np
# import uvicorn

# app = FastAPI()

# # Load models and performance metrics
# MODEL_PATHS = {
#     "model_1": "tuned_multi_output_model.pkl",
#     "model_2": "tuned_random_forest_model.pkl"
# }

# METRICS_PATHS = {
#     "model_1": "performance_metrics.json",
#     #"model_2": "path_to_metrics_2.json"
# }

# # Load models and metrics
# models = {}
# metrics = {}

# for key, path in MODEL_PATHS.items():
#     try:
#         with open(path, "rb") as model_file:
#             models[key] = pickle.load(model_file)
#     except Exception as e:
#         print(f"Error loading {key}: {e}")

# for key, path in METRICS_PATHS.items():
#     try:
#         with open(path, "r") as metrics_file:
#             metrics[key] = json.load(metrics_file)
#     except Exception as e:
#         print(f"Error loading metrics for {key}: {e}")

# FEATURES = [
#     'PM1_P', 'PM2_P', 'Mods1_Q', 'T1', 'T2', 'T3', 'T4', 'T4b',
#     'Mods1_Bridge_Position', 'T5', 'T6', 'T7', 'R8', 'PM1_Alpha', 'PM2_Alpha',
#     'Mods1_Inner_Alpha', 'k_Mods1_Alpha', 'PM1_EM_Theta0', 'PM2_EM_Theta0'
# ]

# TARGETS = ['Rotor1_Torque.Torque', 'Rotor2_Torque.Torque', 'Mods1_Torque.Torque']

# class PredictionInput(BaseModel):
#     model_id: str
#     features: Dict[str, float]  # Expecting a dictionary of feature names and values

# @app.get("/")
# def home():
#     return {"message": "Hello, World!"}

# @app.post("/predictGearBox")
# def predict_gearbox(input_data: PredictionInput):
#     if input_data.model_id not in models:
#         raise HTTPException(status_code=400, detail="Invalid model_id")

#     # Extract feature values in the correct order
#     try:
#         input_values = [input_data.features[feat] for feat in FEATURES]
#     except KeyError as e:
#         raise HTTPException(status_code=400, detail=f"Missing feature: {e}")

#     # Convert to NumPy array and reshape for the model
#     input_array = np.array(input_values).reshape(1, -1)

#     # Make predictions
#     model = models[input_data.model_id]
#     predictions = model.predict(input_array)

#     # Format output
#     result = {target: pred for target, pred in zip(TARGETS, predictions[0])}

#     return {
#         "model_id": input_data.model_id,
#         "predictions": result
#     }

# @app.get("/getPerformanceMetrics/{model_id}")
# def get_performance_metrics(model_id: str):
#     if model_id not in metrics:
#         raise HTTPException(status_code=400, detail="Invalid model_id")

#     return metrics[model_id]

# if __name__ == "__main__":
#   uvicorn.run(app, host="127.0.0.1", port=5555)

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

from typing import Dict, List

import pickle

import numpy as np
import os

import json

import uvicorn

import plotly.express as px

import plotly.io as pio
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://revmagneticgearml.com"],  # For production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define your six model types with their names, types, and versions

MODELS = [

    {"model_type": "Random Forest 1", "version": "1.0"},

    {"model_type": "Random Forest 2", "version": "2.0"},

    {"model_type": "Gradient Boosting 1", "version": "1.0"},

    {"model_type": "Gradient Boosting 2", "version": "2.0"},

    {"model_type": "Gauss Process Regression 1", "version": "1.0"},

    {"model_type": "Gauss Process Regression 2", "version": "2.0"},

]


# Base directory for the models

BASE_MODEL_DIR = "/content/drive/Shared drives/Capstone REV Group/404 MODEL Files/CAPstone"  # Parent directory containing model folders


# Load all models from the directory

models = {}


def load_models():

    """Load all models from the specified directory."""

    for model in MODELS:
        model_type = model["model_type"]

        model_dir = os.path.join(BASE_MODEL_DIR, model_type)


        # Ensure the model directory exists

        if not os.path.exists(model_dir):

            print(f"Warning: The directory '{model_dir}' does not exist. Skipping model.")
            continue
        

        try:

            # Load the model and output model

            model_file = os.path.join(model_dir, f"{model_type}.pkl")

            output_model_file = os.path.join(model_dir, f"{model_type}_output.pkl")

            with open(model_file, "rb") as f:

                models[model_type] = {

                    "model": pickle.load(f),

                    "output_model": pickle.load(open(output_model_file, "rb")),

                    "features": {},

                    "performance_metrics": {}

                }


            # Load features from JSON

            features_file = os.path.join(model_dir, f"{model_type}_features.json")

            if os.path.exists(features_file):

                with open(features_file, "r") as f:

                    models[model_type]["features"] = json.load(f)


            # Load performance metrics from JSON

            metrics_file = os.path.join(model_dir, f"{model_type}_performance.json")

            if os.path.exists(metrics_file):

                with open(metrics_file, "r") as f:

                    models[model_type]["performance_metrics"] = json.load(f)


        except Exception as e:

            print(f"Error loading model {model_type}: {e}")


load_models()  # Load the models at startup

@app.get("/test")
def test_connection():
    return {"message": "FastAPI is connected and working!"}


@app.get("/")

def home():

    return {"message": "Machine Learning API is running!", "Available Models": get_available_models()}
    #return{"Available Models": get_available_models()}


@app.get("/getAvailableModels")

def get_available_models():

    """Returns a list of available models."""

    return MODELS


@app.get("/getModelFeatures/{model_id}")

def get_model_features(model_id: str):

    """Returns the features for a specific model."""

    if model_id not in models:

        raise HTTPException(status_code=404, detail="Model not found")
    

    return models[model_id]["features"]


@app.get("/getModelPerformanceMetrics/{model_id}")

def get_model_performance_metrics(model_id: str):

    """Returns the performance metrics for a specific model."""

    if model_id not in models:

        raise HTTPException(status_code=404, detail="Model not found")
    

    return models[model_id]["performance_metrics"]


class PredictionInput(BaseModel):

    model_id: str

    features: Dict[str, float]


@app.post("/predict")

def predict(input_data: PredictionInput):
    """

    Takes a model_id and feature values, runs prediction, and returns target values.
    """
    model_id = input_data.model_id
    features = input_data.features


    if model_id not in models:

        raise HTTPException(status_code=404, detail="Model not found")


    # Ensure all required features are provided

    # Ensure all required features are provided

    missing_features = [f for f in models[model_id]["features"].get("features", []) if f not in features]

    if missing_features:

        raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")


    # Convert input features to NumPy array

    feature_array = np.array([features[f] for f in models[model_id]["features"].get("features", [])]).reshape(1, -1)


    # Get the model and predict

    model = models[model_id]["model"]

    prediction = model.predict(feature_array)


    # Ensure output format matches the expected targets (assumes targets are the same across all models)

    targets = models[model_id]["features"].get("targets", [])

    if len(prediction) != len(targets):

        raise HTTPException(status_code=500, detail="Model output does not match expected targets.")


    # Return predictions as JSON

    return {targets[i]: prediction[i] for i in range(len(targets))}


@app.get("/getPlot")

def get_plot(plot_type: str, variable: str):

    # Ensure dataset.csv is available

    df = pd.read_csv("dataset.csv")  
    

    if variable not in df.columns:

        raise HTTPException(status_code=400, detail="Invalid variable")
    

    if plot_type == "histogram":

        fig = px.histogram(df, x=variable)

    elif plot_type == "correlation":

        fig = px.imshow(df.corr(), text_auto=True)

    elif plot_type == "violin":

        fig = px.violin(df, y=variable, box=True, points="all")

    else:

        raise HTTPException(status_code=400, detail="Invalid plot type")
    

    plot_path = f"plots/{plot_type}_{variable}.html"

    os.makedirs('plots', exist_ok=True)  # Ensure the plots directory exists

    pio.write_html(fig, file=plot_path)
    

    return {"plot_url": plot_path}


if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=5555)
