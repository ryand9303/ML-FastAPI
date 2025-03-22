from fastapi import FastAPI, HTTPException #, UploadFile, File
from pydantic import BaseModel, Field, RootModel
from typing import Dict, List, Union, Optional
import joblib
import numpy as np
import json
import requests
import pandas as pd
import random
from io import BytesIO #, StringIO
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os
import plotly.express as px
import plotly.io as pio
#from sklearn.decomposition import PCA



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
@app.post("/predict")     # change into a get file and get inputs externally from the UI and not internally created
def predict(input_data: PredictionInput):
    """Handles model selection, input validation, and runs prediction."""
    
    model_key = f"{input_data.model_type} {input_data.version}"

    # Validate model existence
    if model_key not in models:
        raise HTTPException(status_code=404, detail="Model not found or unavailable")

    features = None
    expected_length = 0

    # Choose which features to work with based on version
    if input_data.version == "1.0":
        expected_length = 9
        if input_data.use_random:
            features = Features1(
                feature_1=random.uniform(-10, 10),
                feature_2=random.uniform(-10, 10),
                feature_3=random.uniform(-10, 10),
                feature_4=random.uniform(-10, 10),
                feature_5=random.uniform(-10, 10),
                feature_6=random.uniform(-10, 10),
                feature_7=random.uniform(-10, 10),
                feature_8=random.uniform(-10, 10),
                feature_9=random.uniform(-10, 10)
            )
        elif input_data.features_1 is None:
            raise HTTPException(status_code=400, detail="Features for version 1.0 must be provided.")
        else:
            features = input_data.features_1            
    else:
        expected_length = 23
        if input_data.use_random:
            features = Features2(
                feature_1=random.uniform(-10, 10),
                feature_2=random.uniform(-10, 10),
                feature_3=random.uniform(-10, 10),
                feature_4=random.uniform(-10, 10),
                feature_5=random.uniform(-10, 10),
                feature_6=random.uniform(-10, 10),
                feature_7=random.uniform(-10, 10),
                feature_8=random.uniform(-10, 10),
                feature_9=random.uniform(-10, 10),
                feature_10=random.uniform(-10, 10),
                feature_11=random.uniform(-10, 10),
                feature_12=random.uniform(-10, 10),
                feature_13=random.uniform(-10, 10),
                feature_14=random.uniform(-10, 10),
                feature_15=random.uniform(-10, 10),
                feature_16=random.uniform(-10, 10),
                feature_17=random.uniform(-10, 10),
                feature_18=random.uniform(-10, 10),
                feature_19=random.uniform(-10, 10),
                feature_20=random.uniform(-10, 10),
                feature_21=random.uniform(-10, 10),
                feature_22=random.uniform(-10, 10),
                feature_23=random.uniform(-10, 10)
            )
        elif input_data.features_2 is None:
            raise HTTPException(status_code=400, detail="Features for version 2.0 must be provided.")
        else:
            features = input_data.features_2

    # Convert features into a list for prediction
    feature_values = [getattr(features, f"feature_{i + 1}") for i in range(expected_length)]

    # Save to data.json as before
    input_json = {"values": feature_values}
    
    # Save to data.json
    json_filename = 'data.json'
    with open(json_filename, "w") as f:
        json.dump(input_json, f)

    # Load the corresponding model
    model_file_path = f"Models/{input_data.model_type}/{input_data.version}/tuned_multi_output_model{input_data.version}.pkl"
    try:
        model = joblib.load(model_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    # Convert input features into NumPy array and predict
    feature_array = np.array(feature_values).reshape(1, -1)
    prediction = model.predict(feature_array)

    return {
        "model_type": input_data.model_type,
        "version": input_data.version,
        "input_features": feature_values,
        "prediction": prediction.tolist()
    }





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

