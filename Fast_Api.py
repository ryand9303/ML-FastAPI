from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field, RootModel
from typing import Dict, List, Union, Optional
import joblib
import numpy as np
import json
import requests
import pandas as pd
import random
#from io import BytesIO #, StringIO
import io
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os
import plotly.express as px
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import base64
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import os
import zipfile  # Make sure to import this
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles







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



# Hardcoded dataset summary (obtained from Google Colab)
num_rows = 751406  # found in google colab code
num_cols = 212  # found in google colab code
columns = ['Sim_Num', 'Case_Num', 'PM1_P', 'PM2_P', 'Mods1_Q', 'T1', 'T2', 'T3', 'T4', 'T4b', 'Mods1_Bridge_Position', 'T5', 'T6', 'T7', 'R8', 'PM1_Alpha', 'PM2_Alpha', 
           'Mods1_Inner_Alpha', 'k_Mods1_Alpha', 'PM1_Radial_Portion', 'PM2_Radial_Portion', 'PM1_EM_Theta0', 'PM2_EM_Theta0', 'pm1rMatIndx', 'pm1tMatIndx', 'pm2rMatIndx', 
           'pm2tMatIndx', 'Temperature', 'pm1rBknee', 'pm1tBknee', 'pm2rBknee', 'pm2tBknee', 'Gen_Num', 'Ind_Num', 'symmetryFactor', 'Rotor1_Torque.Torque', 
           'Rotor1_Force.Force_mag', 'Rotor2_Torque.Torque', 'Rotor2_Force.Force_mag', 'Mods1_Torque.Torque', 'Mods1_Force.Force_mag', 'PM1_Torque.Torque', 
           'PM1_Force.Force_mag', 'PM2_Torque.Torque', 'PM2_Force.Force_mag', 'Rotor1_Demag_Percentage', 'Rotor2_Demag_Percentage', 'PM1_Outs_Demag_Percentage', 
           'PM2_Outs_Demag_Percentage', 'PM1_Ins_Demag_Percentage', 'PM2_Ins_Demag_Percentage', 'PM1_Radial_Demag_Percentage', 'PM2_Radial_Demag_Percentage', 
           'AG1 Middle Circle max Mag_B', 'AG1 Middle Circle rms Mag_B', 'AG1 Middle Circle max Br', 'AG1 Middle Circle rms Br', 'AG2 Middle Circle max Mag_B', 
           'AG2 Middle Circle rms Mag_B', 'AG2 Middle Circle max Br', 'AG2 Middle Circle rms Br', 'BI1 Middle Circle max Mag_B', 'BI1 Middle Circle rms Mag_B', 
           'BI1 Middle Circle max Br', 'BI1 Middle Circle rms Br', 'BI2 Middle Circle max Mag_B', 'BI2 Middle Circle rms Mag_B', 'BI2 Middle Circle max Br', 
           'BI2 Middle Circle rms Br', 'Bore 1mm Circle max Mag_B', 'Bore 1mm Circle rms Mag_B', 'Bore 1mm Circle max Br', 'Bore 1mm Circle rms Br', 
           'Bore 2mm Circle max Mag_B', 'Bore 2mm Circle rms Mag_B', 'Bore 2mm Circle max Br', 'Bore 2mm Circle rms Br', 'Bore 3mm Circle max Mag_B', 
           'Bore 3mm Circle rms Mag_B', 'Bore 3mm Circle max Br', 'Bore 3mm Circle rms Br', 'Bore 5mm Circle max Mag_B', 'Bore 5mm Circle rms Mag_B', 'Bore 5mm Circle max Br', 
           'Bore 5mm Circle rms Br', 'Bore 7mm Circle max Mag_B', 'Bore 7mm Circle rms Mag_B', 'Bore 7mm Circle max Br', 'Bore 7mm Circle rms Br', 'Bore 10mm Circle max Mag_B', 
           'Bore 10mm Circle rms Mag_B', 'Bore 10mm Circle max Br', 'Bore 10mm Circle rms Br', 'Bore 15mm Circle max Mag_B', 'Bore 15mm Circle rms Mag_B', 
           'Bore 15mm Circle max Br', 'Bore 15mm Circle rms Br', 'Bore 20mm Circle max Mag_B', 'Bore 20mm Circle rms Mag_B', 'Bore 20mm Circle max Br', 
           'Bore 20mm Circle rms Br', 'Bore 25mm Circle max Mag_B', 'Bore 25mm Circle rms Mag_B', 'Bore 25mm Circle max Br', 'Bore 25mm Circle rms Br', 
           'Bore 30mm Circle max Mag_B', 'Bore 30mm Circle rms Mag_B', 'Bore 30mm Circle max Br', 'Bore 30mm Circle rms Br', 'Exterior 1mm Circle max Mag_B', 
           'Exterior 1mm Circle rms Mag_B', 'Exterior 1mm Circle max Br', 'Exterior 1mm Circle rms Br', 'Exterior 2mm Circle max Mag_B', 'Exterior 2mm Circle rms Mag_B', 
           'Exterior 2mm Circle max Br', 'Exterior 2mm Circle rms Br', 'Exterior 3mm Circle max Mag_B', 'Exterior 3mm Circle rms Mag_B', 'Exterior 3mm Circle max Br', 
           'Exterior 3mm Circle rms Br', 'Exterior 10mm Circle max Mag_B', 'Exterior 10mm Circle rms Mag_B', 'Exterior 10mm Circle max Br', 'Exterior 10mm Circle rms Br', 
           'Exterior 15mm Circle max Mag_B', 'Exterior 15mm Circle rms Mag_B', 'Exterior 15mm Circle max Br', 'Exterior 15mm Circle rms Br', 'Exterior 20mm Circle max Mag_B', 
           'Exterior 20mm Circle rms Mag_B', 'Exterior 20mm Circle max Br', 'Exterior 20mm Circle rms Br', 'Exterior 25mm Circle max Mag_B', 'Exterior 25mm Circle rms Mag_B', 
           'Exterior 25mm Circle max Br', 'Exterior 25mm Circle rms Br', 'Exterior 5mm Circle max Mag_B', 'Exterior 5mm Circle rms Mag_B', 'Exterior 5mm Circle max Br', 
           'Exterior 5mm Circle rms Br', 'Exterior 7mm Circle max Mag_B', 'Exterior 7mm Circle rms Mag_B', 'Exterior 7mm Circle max Br', 'Exterior 7mm Circle rms Br', 
           'Exterior 30mm Circle max Mag_B', 'Exterior 30mm Circle rms Mag_B', 'Exterior 30mm Circle max Br', 'Exterior 30mm Circle rms Br', 'Exterior 35mm Circle max Mag_B', 
           'Exterior 35mm Circle rms Mag_B', 'Exterior 35mm Circle max Br', 'Exterior 35mm Circle rms Br', 'Exterior 40mm Circle max Mag_B', 'Exterior 40mm Circle rms Mag_B', 
           'Exterior 40mm Circle max Br', 'Exterior 40mm Circle rms Br', 'Exterior 45mm Circle max Mag_B', 'Exterior 45mm Circle rms Mag_B', 'Exterior 45mm Circle max Br', 
           'Exterior 45mm Circle rms Br', 'Exterior 50mm Circle max Mag_B', 'Exterior 50mm Circle rms Mag_B', 'Exterior 50mm Circle max Br', 'Exterior 50mm Circle rms Br', 
           'Mods1 Middle Circle max Mag_B', 'Mods1 Middle Circle rms Mag_B', 'Mods1 Middle Circle max Br', 'Mods1 Middle Circle rms Br', 'PM1 Middle Circle max Mag_B', 
           'PM1 Middle Circle rms Mag_B', 'PM1 Middle Circle max Br', 'PM1 Middle Circle rms Br', 'PM2 Middle Circle max Mag_B', 'PM2 Middle Circle rms Mag_B', 
           'PM2 Middle Circle max Br', 'PM2 Middle Circle rms Br', 'R1 Circle max Mag_B', 'R1 Circle rms Mag_B', 'R1 Circle max Br', 'R1 Circle rms Br', 'R2 Circle max Mag_B', 
           'R2 Circle rms Mag_B', 'R2 Circle max Br', 'R2 Circle rms Br', 'R3 Circle max Mag_B', 'R3 Circle rms Mag_B', 'R3 Circle max Br', 'R3 Circle rms Br', 
           'R4 Circle max Mag_B', 'R4 Circle rms Mag_B', 'R4 Circle max Br', 'R4 Circle rms Br', 'R5 Circle max Mag_B', 'R5 Circle rms Mag_B', 'R5 Circle max Br', 
           'R5 Circle rms Br', 'R6 Circle max Mag_B', 'R6 Circle rms Mag_B', 'R6 Circle max Br', 'R6 Circle rms Br', 'R7 Circle max Mag_B', 'R7 Circle rms Mag_B', 
           'R7 Circle max Br', 'R7 Circle rms Br', 'R8 Circle max Mag_B', 'R8 Circle rms Mag_B', 'R8 Circle max Br', 'R8 Circle rms Br', 'Sim_Time', 'Memory_Utilized', 
           'Cluster_Used']

# Hardcoded target variables
targets = ['Rotor1_Torque.Torque', 'Rotor2_Torque.Torque', 'Mods1_Torque.Torque']

# Endpoint to get data summary (hardcoded)
@app.get("/getDataSummary")
def get_data_summary():
    try:
        # Prepare the response data
        summary = {
            "num_rows": num_rows,
            "num_columns": num_cols,
            "columns": columns,
            "targets": targets  # Add target variables to the summary
        }

        return JSONResponse(content=summary)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)














# Define the structure for the prediction data 
class PredictionData(BaseModel):
    features: Dict[str, float]  # Features are given as key-value pairs

@app.post("/predict")
def predict(
    model_type: str, 
    version: str, 
    file: UploadFile = File(...)  # Accepts the file upload
):
    """Handles model selection, input validation, and runs prediction."""
    
    # Read the uploaded file content and parse it as JSON
    try:
        file_content = file.file.read()  # Read file content synchronously
        input_data = json.loads(file_content)  # Parse JSON content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading or parsing the JSON file: {str(e)}")
    
    predictions = []

    # Wrap the incoming JSON data inside "features" to match the structure
    if isinstance(input_data, dict):  # Single data point
        input_data = [{"features": input_data}]
    
    for data in input_data:
        if "features" not in data:
            raise HTTPException(status_code=400, detail="JSON must contain 'features' with corresponding values.")

        # Ensure features are not empty
        if not data["features"]:
            raise HTTPException(status_code=400, detail="JSON must contain 'features' with corresponding values.")

        # Load pre-trained scaler
        try:
            print("Loading scaler...")
            scaler = joblib.load("scaler.pkl")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading scaler: {str(e)}")
        
        # Load the correct PCA version based on the 'version' parameter
        try:
            if version == "1.0":
                pca = joblib.load("pca_1.0.pkl")
            elif version == "2.0":
                pca = joblib.load("pca_2.0.pkl")
            else:
                raise HTTPException(status_code=400, detail="Invalid version specified. Only '1.0' and '2.0' are supported.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading PCA: {str(e)}")

        # Convert feature dictionary to list
        features = list(data["features"].values())
        feature_values = np.array(features).reshape(1, -1)  # Reshape if necessary

        try:
            print(f"Feature values: {feature_values}")
            scaled_values = scaler.transform(feature_values)
            pca_result = pca.transform(scaled_values)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error during scaling or PCA transformation: {str(e)}")

        # Ensure correct PCA output size (9 features for version 1.0, 23 for version 2.0)
        expected_length = 9 if version == "1.0" else 23
        if pca_result.shape[1] != expected_length:
            raise HTTPException(status_code=400, detail=f"PCA output should have {expected_length} features, but it has {pca_result.shape[1]}.")

        print(f"PCA Result Shape: {pca_result.shape}")

        # Load model
        model_file_path = f"Models/{model_type}/{version}/tuned_multi_output_model{version}.pkl"
        try:
            model = joblib.load(model_file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

        # Make prediction
        try:
            prediction = model.predict(pca_result)
            predictions.append({
                "model_type": model_type,
                "version": version,
                "input_features": pca_result.tolist(),
                "prediction": prediction.tolist()
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    return {"predictions": predictions}







from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/getPlot/{plot_type}")
def get_plot(plot_type: str, feature: str):
    """Simply return the URL for the plot file."""
    valid_plot_types = ["histograms", "violins"]
    if plot_type not in valid_plot_types:
        raise HTTPException(status_code=400, detail="Invalid plot type.")

    plot_file = f"{feature}_{plot_type}.html"
    plot_url = f"/static/{plot_file}"

    # Return the URL of the plot
    return JSONResponse(content={"file_url": plot_url})















    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5555)

