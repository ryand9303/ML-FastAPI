from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import pickle
import numpy as np
import os
import json
import uvicorn
import requests  # New: To download files from GitHub
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

# Define model structure based on GitHub repository
MODELS = {
    "Random Forest 1": {"version": "1.0", "feature_file": "features95.json", "metrics_file": "performance_metrics95.json", "model_file": "tuned_multi_output_model95.pkl"},
    "Random Forest 2": {"version": "2.0", "feature_file": "features99.json", "metrics_file": "performance_metrics99.json", "model_file": "tuned_multi_output_model99.pkl"},
    "Gradient Boosting 1": {"version": "1.0", "feature_file": "features95.json", "metrics_file": "performance_metrics95.json", "model_file": "tuned_multi_output_model95.pkl"},
    "Gradient Boosting 2": {"version": "2.0", "feature_file": "features99.json", "metrics_file": "performance_metrics99.json", "model_file": "tuned_multi_output_model99.pkl"}
}

# GitHub repository details (CHANGE URL TO MATCH YOUR REPO)
GITHUB_REPO_URL = "https://raw.githubusercontent.com/ryand9303/ML-FastAPI/main"

# Store models in memory
models = {}

def download_file_from_github(model_name, filename):
    """Download a file from GitHub."""
    url = f"{GITHUB_REPO_URL}/{model_name}/{filename}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.content
    else:
        print(f"Warning: Could not download {filename} for {model_name}")
        return None

def load_models():
    """Download and load models from GitHub."""
    for model_name, model_data in MODELS.items():
        try:
            # Load Model (.pkl)
            model_file_content = download_file_from_github(model_name, model_data["model_file"])
            if model_file_content:
                model = pickle.loads(model_file_content)
            else:
                continue  # Skip model if it couldn't be downloaded

            # Load Features JSON
            features_content = download_file_from_github(model_name, model_data["feature_file"])
            features = json.loads(features_content) if features_content else {}

            # Load Performance Metrics JSON
            metrics_content = download_file_from_github(model_name, model_data["metrics_file"])
            performance_metrics = json.loads(metrics_content) if metrics_content else {}

            # Store model in dictionary
            models[model_name] = {
                "model": model,
                "features": features,
                "performance_metrics": performance_metrics,
            }

            print(f"✅ Loaded {model_name} successfully!")

        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")

# Load models at startup
load_models()

@app.get("/")
def home():
    return {"message": "Machine Learning API is running!", "Available Models": get_available_models()}

@app.get("/getAvailableModels")
def get_available_models():
    """Returns a list of available models with their versions."""
    return [{"model": name, "version": data["version"]} for name, data in MODELS.items()]

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
        raise HTTPException(status_code=404, detail="Model not found")

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
