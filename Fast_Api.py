pip install python-multipart
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, RootModel
from typing import Dict, List, Union, Optional
import joblib
import numpy as np
import json
import requests
import pandas as pd
import random
from io import BytesIO, StringIO
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os
import plotly.express as px
import plotly.io as pio
from sklearn.decomposition import PCA


# FastAPI Setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://revmagneticgearml.com"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model structure - Uniquely identifying models with type + version
MODELS = [
    {"model_type": "Random Forest", "version": "1.0"},
    {"model_type": "Random Forest", "version": "2.0"},
    {"model_type": "Gradient Boosting", "version": "1.0"},
    {"model_type": "Gradient Boosting", "version": "2.0"}
]

# GitHub repository details (CHANGE URL TO MATCH YOUR REPO)
GITHUB_REPO_URL = "https://raw.githubusercontent.com/ryand9303/ML-FastAPI/main/Models"

# Store models in memory
models = {}
model_availability = {}  # Dictionary to store model availability status
data_summary = {}  # Store feature and target type information

def download_file_from_github(model_type, model_version, filename):
    """Download a file from GitHub and check if it exists."""
    
    # Ensure model type names are correctly formatted for URLs
    model_type_encoded = model_type.replace(" ", "%20")  
    url = f"{GITHUB_REPO_URL}/{model_type_encoded}/{model_version}/{filename}"
    
    print(f"🔗 Attempting to download: {url}")  # Debugging
    response = requests.get(url, stream=True)

    if response.status_code == 200 and "text/html" not in response.headers.get("Content-Type", ""):
        print(f"✅ Successfully downloaded {filename} for {model_type} {model_version}")  
        return response.content  # Ensures binary content is returned
    else:
        print(f"⚠️ Warning: Could not download {filename} for {model_type} {model_version} (Status: {response.status_code})")  
        return None

def load_models():
    """Download and load models from GitHub dynamically, checking availability."""
    for model in MODELS:
        model_type = model["model_type"]
        model_version = model["version"]
        model_key = f"{model_type} {model_version}"  # Unique identifier

        print(f"🔄 Checking model {model_key} ...")  # Debugging

        try:
            # Dynamically determine filenames
            feature_file = f"features{model_version}.json"
            metrics_file = f"performance_metrics{model_version}.json"
            model_file = f"tuned_multi_output_model{model_version}.pkl"

            # ✅ Download files
            model_content = download_file_from_github(model_type, model_version, model_file)
            features_content = download_file_from_github(model_type, model_version, feature_file)
            metrics_content = download_file_from_github(model_type, model_version, metrics_file)

            # Check if files exist
            if not all([model_content, features_content, metrics_content]):
                print(f"⚠️ Model {model_key} is missing files. Marking as unavailable.")
                model_availability[model_key] = False
                continue

            # Save model file temporarily before loading
            temp_model_path = f"temp_model_{model_version}.pkl"
            with open(temp_model_path, "wb") as f:
                f.write(model_content)

            # Verify that the saved file is a valid pickle file before loading
            try:
                print(f"📦 Validating and unpacking model {model_key}")

                # First, check if it's a valid joblib file
                with open(temp_model_path, "rb") as f:
                    first_byte = f.read(1)  # Read the first byte

                if first_byte == b'\x80':  # Joblib format starts with 0x80
                    model_obj = joblib.load(temp_model_path)  # ✅ Use joblib
                else:
                    print(f"⚠️ {model_key} might be a non-joblib pickle. Trying with pickle.")
                    with open(temp_model_path, "rb") as f:
                        model_obj = pickle.load(f)

            except Exception as e:
                print(f"❌ Unpickling failed for {model_key}: {e}")
                model_availability[model_key] = False
                continue

            # Load JSON data properly
            features = json.loads(features_content.decode("utf-8"))
            performance_metrics = json.loads(metrics_content.decode("utf-8"))

            # Store in memory
            models[model_key] = {
                "model": model_obj,
                "features": features,
                "performance_metrics": performance_metrics,
            }

            # Store for getDataSummary
            data_summary[model_key] = {
                "features": features.get("features", []),
                "targets": features.get("targets", [])
            }

            model_availability[model_key] = True  # Mark as available
            print(f"✅ Loaded {model_key} successfully!")

        except Exception as e:
            print(f"❌ Error loading {model_key}: {e}")
            model_availability[model_key] = False  # Mark as unavailable

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
    """Returns a summary of all features and their type (feature or target) for each model."""
    
    # Fixed target names
    targets = ['Rotor1_Torque.Torque', 'Rotor2_Torque.Torque', 'Mods1_Torque.Torque']

    summary = {}

    for model in MODELS:
        model_type = model["model_type"]
        model_version = model["version"]
        model_key = f"{model_type} {model_version}"

        # Get model features using the existing function
        if model_key in models:
            features = models[model_key]["features"]
        else:
            features = []

        summary[model_key] = {
            "features": features,  # Feature list from the model
            "targets": targets  # Fixed list of target values
        }

    return summary


# Define the labeled features for version 1.0 and version 2.0
class Features1(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float

class Features2(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float
    feature_10: float
    feature_11: float
    feature_12: float
    feature_13: float
    feature_14: float
    feature_15: float
    feature_16: float
    feature_17: float
    feature_18: float
    feature_19: float
    feature_20: float
    feature_21: float
    feature_22: float
    feature_23: float

class PredictionInput(BaseModel):
    model_type: str
    version: str
    use_random: Optional[bool] = Field(default=False, description="Set to true to use random feature values.")
    features_1: Optional[Features1] = None
    features_2: Optional[Features2] = None
@app.get("/predict")
async def predict_csv(
    model_type: str,
    version: str,
    file: UploadFile = File(...),
):
    """Handles model selection, CSV file processing, PCA, and runs prediction."""
    
    model_key = f"{model_type} {version}"

    # Validate model existence
    if model_key not in models:
        raise HTTPException(status_code=404, detail="Model not found or unavailable")

    # Read CSV content
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    
    # Check if the first column contains the features
    features = df.columns.tolist()
    if features[0] != 'features':
        raise HTTPException(status_code=400, detail="CSV must have 'features' in the first column")
    
    # Get the feature columns and values columns
    feature_names = features[1:]  # All columns except the first one (features)
    feature_values = df[feature_names].values  # All values of the features
    
    # Prepare PCA based on the version
    pca = PCA(n_components=9 if version == "1.0" else 23)
    
    # Perform PCA on the feature values to get the desired number of components
    pca_result = pca.fit_transform(feature_values)

    predictions = []
    for column_idx in range(pca_result.shape[1]):
        # Select a column of PCA results to use for prediction
        pca_features = pca_result[:, column_idx].reshape(1, -1)

        # Load the model for prediction
        model_file_path = f"Models/{model_type}/{version}/tuned_multi_output_model{version}.pkl"
        try:
            model = joblib.load(model_file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

        # Predict with the model
        prediction = model.predict(pca_features)

        # Add result to the list
        predictions.append({
            "model_type": model_type,
            "version": version,
            "input_features": pca_features.tolist(),
            "prediction": prediction.tolist()
        })
    
    return {"predictions": predictions}


GITHUB_PLOTS_URL = "https://raw.githubusercontent.com/ryand9303/ML-FastAPI/main/Plots"

# Define dataset location (Change if needed)
DATASET_PATH = "data.json"

# Ensure local directory for plots exists
PLOTS_DIR = "Plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

@app.get("/getPlot/{plot_name}")
def get_plot(plot_name: str):
    """
    Fetches the requested plot HTML file from the GitHub repository's 'Plots' folder.
    """

    # Construct the GitHub URL for the plot file
    plot_url = f"{GITHUB_PLOTS_URL}/{plot_name}.html"

    # Attempt to download the plot file
    response = requests.get(plot_url)

    if response.status_code == 200:
        return {"message": "Plot retrieved successfully", "plot_url": plot_url}
    else:
        raise HTTPException(status_code=404, detail="Plot not found in GitHub repository")


@app.get("/generatePlot")
def generate_plot(plot_type: str, variables: List[str]):
    """
    Generates a plot dynamically based on user input.
    
    Parameters:
    - plot_type: "histogram", "correlation_map", or "violin_plot"
    - variables: List of feature names to plot
    """

    # Load dataset from JSON
    try:
        with open(DATASET_PATH, "r") as f:
            data = pd.DataFrame(json.load(f)["values"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

    # Check if variables exist in the dataset
    missing_vars = [var for var in variables if var not in data.columns]
    if missing_vars:
        raise HTTPException(status_code=400, detail=f"Missing variables in dataset: {missing_vars}")

    # Create and save the requested plot
    plot_path = f"{PLOTS_DIR}/{plot_type}_{'_'.join(variables)}.html"

    if plot_type.lower() == "histogram":
        fig = px.histogram(data, x=variables[0], title=f"Histogram of {variables[0]}")
    elif plot_type.lower() == "correlation_map":
        corr_matrix = data[variables].corr()
        fig = px.imshow(corr_matrix, title="Correlation Map", labels=dict(color="Correlation"))
    elif plot_type.lower() == "violin_plot":
        if len(variables) < 2:
            raise HTTPException(status_code=400, detail="Violin plot requires at least two variables (x and y).")
        fig = px.violin(data, x=variables[0], y=variables[1], box=True, title=f"Violin Plot: {variables[1]} vs {variables[0]}")
    else:
        raise HTTPException(status_code=400, detail="Invalid plot type. Choose from 'histogram', 'correlation_map', or 'violin_plot'.")

    # Save the plot as an HTML file
    pio.write_html(fig, plot_path)

    return {"message": "Plot generated successfully", "plot_url": f"/{plot_path}"}

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5555)

