
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Dict
# import pickle
# import numpy as np
# import os
# import json
# import uvicorn
# import requests  # To download files from GitHub
# import pandas as pd
# import plotly.express as px
# import plotly.io as pio
# from fastapi.middleware.cors import CORSMiddleware

# # FastAPI Setup
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["https://revmagneticgearml.com"],  
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Define model structure (List format)
# MODELS = [
#     {"model_type": "Random Forest", "version": "1.0"},
#     {"model_type": "Random Forest", "version": "2.0"},
#     {"model_type": "Gradient Boosting", "version": "1.0"},
#     {"model_type": "Gradient Boosting", "version": "2.0"},
#     {"model_type": "Gauss Process Regression", "version": "1.0"},
#     {"model_type": "Gauss Process Regression", "version": "2.0"}
# ]

# # GitHub repository details (CHANGE URL TO MATCH YOUR REPO)
# GITHUB_REPO_URL = "https://raw.githubusercontent.com/ryand9303/ML-FastAPI/main"

# # Store models in memory
# models = {}
# model_availability = {}  # Dictionary to store model availability status
# data_summary = {}  # Store feature and target type information

# def download_file_from_github(model_folder, filename):
#     """Download a file from GitHub and check if it exists."""
#     url = f"{GITHUB_REPO_URL}/{model_folder}/{filename}"
#     response = requests.get(url)

#     if response.status_code == 200:
#         return response.content
#     else:
#         print(f"⚠️ Warning: Could not download {filename} for {model_folder}")
#         return None

# def load_models():
#     """Download and load models from GitHub dynamically, checking availability."""
#     for model in MODELS:
#         model_type = model["model_type"]
#         model_version = model["version"]
#         model_key = f"{model_type} {model_version}"  # Unique identifier: Type + Version

#         try:
#             # Dynamically determine filenames based on model type and version
#             feature_file = f"features{model_version.replace('.', '')}.json"
#             metrics_file = f"performance_metrics{model_version.replace('.', '')}.json"
#             model_file = f"tuned_multi_output_model{model_version.replace('.', '')}.pkl"

#             # Check availability of all required files
#             model_content = download_file_from_github(model_key, model_file)
#             features_content = download_file_from_github(model_key, feature_file)
#             metrics_content = download_file_from_github(model_key, metrics_file)

#             if not all([model_content, features_content, metrics_content]):
#                 print(f"⚠️ Model {model_key} is missing files and will be marked as unavailable.")
#                 model_availability[model_key] = False
#                 continue  # Skip loading this model

#             # Deserialize files if available
#             model_obj = pickle.loads(model_content)
#             features = json.loads(features_content)
#             performance_metrics = json.loads(metrics_content)

#             # Store model in dictionary with unique identifier
#             models[model_key] = {
#                 "model": model_obj,
#                 "features": features,
#                 "performance_metrics": performance_metrics,
#             }

#             # Store feature and target details for the `getDataSummary` function
#             data_summary[model_key] = {
#                 "features": features.get("features", []),
#                 "targets": features.get("targets", [])
#             }

#             model_availability[model_key] = True  # Mark model as available
#             print(f"✅ Loaded {model_key} successfully!")

#         except Exception as e:
#             print(f"❌ Error loading {model_key}: {e}")
#             model_availability[model_key] = False  # Mark model as unavailable

# # Load models at startup
# load_models()

# @app.get("/getPlot")
# def get_plot(plot_type: str, variable: str):
#     """
#     Generates a plot based on the requested type and variable.
#     """
#     # Load the dataset (Ensure `dataset.csv` is in the correct location)
#     try:
#         df = pd.read_csv("dataset.csv")
#     except FileNotFoundError:
#         raise HTTPException(status_code=500, detail="Dataset file not found.")

#     # Check if the variable exists in the dataset
#     if variable not in df.columns:
#         raise HTTPException(status_code=400, detail="Invalid variable.")

#     # Generate the requested plot
#     if plot_type == "histogram":
#         fig = px.histogram(df, x=variable, title=f"Histogram of {variable}")

#     elif plot_type == "correlation":
#         fig = px.imshow(df.corr(), text_auto=True, title="Correlation Matrix")

#     elif plot_type == "violin":
#         fig = px.violin(df, y=variable, box=True, points="all", title=f"Violin Plot of {variable}")

#     else:
#         raise HTTPException(status_code=400, detail="Invalid plot type.")

#     # Save plot as an HTML file
#     plot_path = f"plots/{plot_type}_{variable}.html"
#     os.makedirs('plots', exist_ok=True)  # Ensure the directory exists
#     pio.write_html(fig, file=plot_path)

#     return {"plot_url": plot_path}

# @app.get("/getDataSummary")
# def get_data_summary():
#     """Returns a summary of all features and targets available across models."""
#     return data_summary

# # Define prediction input format
# class PredictionInput(BaseModel):
#     model_type: str
#     version: str
#     features: Dict[str, float]

# @app.post("/predict")
# def predict(input_data: PredictionInput):
#     """Takes a model_type, version, and feature values, runs prediction, and returns target values."""
#     model_key = f"{input_data.model_type} {input_data.version}"
#     features = input_data.features

#     if model_key not in models:
#         raise HTTPException(status_code=404, detail="Model not found or unavailable")

#     # Ensure all required features are provided
#     required_features = models[model_key]["features"].get("features", [])
#     missing_features = [f for f in required_features if f not in features]

#     if missing_features:
#         raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")

#     # Convert input features to NumPy array
#     feature_array = np.array([features[f] for f in required_features]).reshape(1, -1)

#     # Get model and predict
#     model = models[model_key]["model"]
#     prediction = model.predict(feature_array)

#     # Ensure output format matches expected targets
#     targets = models[model_key]["features"].get("targets", [])
#     if len(prediction[0]) != len(targets):
#         raise HTTPException(status_code=500, detail="Model output does not match expected targets.")

#     # Return predictions as JSON
#     return {targets[i]: prediction[0][i] for i in range(len(targets))}

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=5555)


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

# Define model structure (List format) - Uniquely identifying models with type + version
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
data_summary = {}  # Store feature and target type information

def download_file_from_github(model_folder, filename):
    """Download a file from GitHub and check if it exists."""
    model_folder = model_folder.replace(" ", "_")  # Handle spaces in model names
    url = f"{GITHUB_REPO_URL}/{model_folder}/{filename}"
    
    print(f"🔗 Attempting to download: {url}")  # Debugging
    response = requests.get(url)

    if response.status_code == 200:
        print(f"✅ Successfully downloaded {filename} for {model_folder}")
        return response.content
    else:
        print(f"⚠️ Warning: Could not download {filename} for {model_folder}")
        return None

def load_models():
    """Download and load models from GitHub dynamically, checking availability."""
    for model in MODELS:
        model_type = model["model_type"]
        model_version = model["version"]
        model_key = f"{model_type} {model_version}"  # Unique identifier: Type + Version

        print(f"🔄 Checking model {model_key} ...")  # Debugging

        try:
            # Dynamically determine filenames based on model type and version
            feature_file = f"features{model_version.replace('.', '')}.json"
            metrics_file = f"performance_metrics{model_version.replace('.', '')}.json"
            model_file = f"tuned_multi_output_model{model_version.replace('.', '')}.pkl"

            # Check availability of all required files
            model_content = download_file_from_github(model_key, model_file)
            features_content = download_file_from_github(model_key, feature_file)
            metrics_content = download_file_from_github(model_key, metrics_file)

            # Check if files exist before proceeding
            if not all([model_content, features_content, metrics_content]):
                print(f"⚠️ Model {model_key} is missing files and will be marked as unavailable.")
                model_availability[model_key] = False
                continue  # Skip loading this model

            # Deserialize files safely
            try:
                model_obj = pickle.loads(model_content)
            except pickle.UnpicklingError as e:
                print(f"❌ Error unpickling {model_key}: {e}")
                continue  # Skip this model

            features = json.loads(features_content)
            performance_metrics = json.loads(metrics_content)

            # Store model in dictionary with unique identifier
            models[model_key] = {
                "model": model_obj,
                "features": features,
                "performance_metrics": performance_metrics,
            }

            # Store feature and target details for the `getDataSummary` function
            data_summary[model_key] = {
                "features": features.get("features", []),
                "targets": features.get("targets", [])
            }

            model_availability[model_key] = True  # Mark model as available
            print(f"✅ Loaded {model_key} successfully!")

        except Exception as e:
            print(f"❌ Error loading {model_key}: {e}")
            model_availability[model_key] = False  # Mark model as unavailable

# Load models at startup
load_models()

@app.get("/")
def home():
    return {"message": "Machine Learning API is running!", "Available Models": get_available_models()}

@app.get("/getAvailableModels")
def get_available_models():
    """Returns a list of available models with their versions and availability status."""
    return [{"model_type": model["model_type"], "version": model["version"], "available": model_availability.get(f"{model['model_type']} {model['version']}", False)} for model in MODELS]

@app.get("/getModelFeatures/{model_type}/{version}")
def get_model_features(model_type: str, version: str):
    """Returns the features for a specific model (requires type and version)."""
    model_key = f"{model_type} {version}"
    
    if model_key not in models:
        raise HTTPException(status_code=404, detail="Model not found or unavailable")
    
    return models[model_key]["features"]

@app.get("/getModelPerformanceMetrics/{model_type}/{version}")
def get_model_performance_metrics(model_type: str, version: str):
    """Returns the performance metrics for a specific model (requires type and version)."""
    model_key = f"{model_type} {version}"
    
    if model_key not in models:
        raise HTTPException(status_code=404, detail="Model not found or unavailable")
    
    return models[model_key]["performance_metrics"]

@app.get("/getDataSummary")
def get_data_summary():
    """Returns a summary of all features and targets available across models."""
    return data_summary

# Define prediction input format
class PredictionInput(BaseModel):
    model_type: str
    version: str
    features: Dict[str, float]

@app.post("/predict")
def predict(input_data: PredictionInput):
    """Takes a model_type, version, and feature values, runs prediction, and returns target values."""
    model_key = f"{input_data.model_type} {input_data.version}"
    features = input_data.features

    if model_key not in models:
        raise HTTPException(status_code=404, detail="Model not found or unavailable")

    # Ensure all required features are provided
    required_features = models[model_key]["features"].get("features", [])
    missing_features = [f for f in required_features if f not in features]

    if missing_features:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")

    # Convert input features to NumPy array
    feature_array = np.array([features[f] for f in required_features]).reshape(1, -1)

    # Get model and predict
    model = models[model_key]["model"]
    prediction = model.predict(feature_array)

    print(f"🔎 Model Prediction: {prediction}")  # Debugging

    # Ensure output format matches expected targets
    targets = models[model_key]["features"].get("targets", [])
    
    if not isinstance(prediction, (list, np.ndarray)):
        prediction = [prediction]  # Convert scalar to list
    
    if len(prediction) != len(targets):
        raise HTTPException(status_code=500, detail=f"Model output does not match expected targets. Output: {prediction}")

    # Return predictions as JSON
    return {targets[i]: prediction[i] for i in range(len(targets))}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5555)



