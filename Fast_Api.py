from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import pickle
import numpy as np
import os
import json
import uvicorn
import requests  # To download files from GitHub
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# FastAPI Setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://revmagneticgearml.com"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model structure (List format)
MODELS = [
    {"model_type": "Random Forest", "version": "1.0"},
    {"model_type": "Random Forest", "version": "2.0"},
    {"model_type": "Gradient Boosting", "version": "1.0"},
    {"model_type": "Gradient Boosting", "version": "2.0"},
    {"model_type": "Gauss Process Regression", "version": "1.0"},
    {"model_type": "Gauss Process Regression", "version": "2.0"}
]

# GitHub repository details (CHANGE URL TO MATCH YOUR REPO)
GITHUB_REPO_URL = "https://raw.githubusercontent.com/ryand9303/ML-FastAPI/main"

# Store models in memory
models = {}
model_availability = {}  # Dictionary to store model availability status

def download_file_from_github(model_folder, filename):
    """Download a file from GitHub and check if it exists."""
    url = f"{GITHUB_REPO_URL}/{model_folder}/{filename}"
    response = requests.get(url)

    if response.status_code == 200:
        return response.content
    else:
        print(f"⚠️ Warning: Could not download {filename} for {model_folder}")
        return None

def load_models():
    """Download and load models from GitHub dynamically, checking availability."""
    for model in MODELS:
        model_folder = model["model_type"]
        model_version = model["version"]

        try:
            # Dynamically determine filenames based on model type
            feature_file = f"features{model_version.replace('.', '')}.json"
            metrics_file = f"performance_metrics{model_version.replace('.', '')}.json"
            model_file = f"tuned_multi_output_model{model_version.replace('.', '')}.pkl"

            # Check availability of all required files
            model_content = download_file_from_github(model_folder, model_file)
            features_content = download_file_from_github(model_folder, feature_file)
            metrics_content = download_file_from_github(model_folder, metrics_file)

            if not all([model_content, features_content, metrics_content]):
                print(f"⚠️ Model {model_folder} is missing files and will be marked as unavailable.")
                model_availability[model_folder] = False
                continue  # Skip loading this model

            # Deserialize files if available
            model_obj = pickle.loads(model_content)
            features = json.loads(features_content)
            performance_metrics = json.loads(metrics_content)

            # Store model in dictionary
            models[model_folder] = {
                "model": model_obj,
                "features": features,
                "performance_metrics": performance_metrics,
            }

            model_availability[model_folder] = True  # Mark model as available
            print(f"✅ Loaded {model_folder} successfully!")

        except Exception as e:
            print(f"❌ Error loading {model_folder}: {e}")
            model_availability[model_folder] = False  # Mark model as unavailable

# Load models at startup
load_models()

@app.get("/")
def home():
    return {"message": "Machine Learning API is running!", "Available Models": get_available_models()}

@app.get("/getAvailableModels")
def get_available_models():
    """Returns a list of available models with their versions and availability status."""
    return [{"model": model["model_type"], "version": model["version"], "available": model_availability.get(model["model_type"], False)} for model in MODELS]

@app.get("/getModelFeatures/{model_id}")
def get_model_features(model_id: str):
    """Returns the features for a specific model."""
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found or unavailable")
    return models[model_id]["features"]

@app.get("/getModelPerformanceMetrics/{model_id}")
def get_model_performance_metrics(model_id: str):
    """Returns the performance metrics for a specific model."""
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found or unavailable")
    return models[model_id]["performance_metrics"]

# Define prediction input format
class PredictionInput(BaseModel):
    model_id: str
    features: Dict[str, float]

@app.post("/predict")
def predict(input_data: PredictionInput):
    """Takes a model_id and feature values, runs prediction, and returns target values."""
    model_id = input_data.model_id
    features = input_data.features

    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found or unavailable")

    # Ensure all required features are provided
    required_features = models[model_id]["features"].get("features", [])
    missing_features = [f for f in required_features if f not in features]

    if missing_features:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")

    # Convert input features to NumPy array
    feature_array = np.array([features[f] for f in required_features]).reshape(1, -1)

    # Get model and predict
    model = models[model_id]["model"]
    prediction = model.predict(feature_array)

    # Ensure output format matches expected targets
    targets = models[model_id]["features"].get("targets", [])
    if len(prediction[0]) != len(targets):
        raise HTTPException(status_code=500, detail="Model output does not match expected targets.")

    # Return predictions as JSON
    return {targets[i]: prediction[0][i] for i in range(len(targets))}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5555)
