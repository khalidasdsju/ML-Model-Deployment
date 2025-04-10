import os
import sys
import joblib
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from fastapi import FastAPI, HTTPException, Query, Depends, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from datetime import datetime
import io
import csv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HF.logger import logging
from HF.utils.s3_utils import S3Utils

# Initialize FastAPI app
app = FastAPI(
    title="Heart Failure Prediction API",
    description="API for predicting heart failure risk using machine learning",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="templates/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Global variables for model and parameters
MODEL = None
MODEL_FEATURES = None
PREDICTION_THRESHOLD = 0.001
IMPORTANT_FEATURES = None
AWS_ACCESS_KEY = None
AWS_SECRET_KEY = None
S3_BUCKET_NAME = "hf-predication-model2025"
MODEL_VERSION = None

# Pydantic models for request and response
class PatientData(BaseModel):
    """Patient data for heart failure prediction"""
    # Original 16 features (required)
    FS: float = Field(..., description="Fractional Shortening (%)", example=25)
    DT: float = Field(..., description="Deceleration Time (ms)", example=160)
    NYHA: int = Field(..., description="New York Heart Association classification (1-4)", example=3)
    HR: float = Field(..., description="Heart Rate (bpm)", example=95)
    BNP: float = Field(..., description="B-type Natriuretic Peptide (pg/mL)", example=800)
    LVIDs: float = Field(..., description="Left Ventricular Internal Dimension in systole (cm)", example=4.8)
    BMI: float = Field(..., description="Body Mass Index (kg/m²)", example=28.5)
    LAV: float = Field(..., description="Left Atrial Volume (mL)", example=45)
    Wall_Subendocardial: int = Field(..., description="Subendocardial Wall (0=No, 1=Yes)", example=1)
    LDLc: float = Field(..., description="Low-Density Lipoprotein cholesterol (mg/dL)", example=140)
    Age: float = Field(..., description="Age (years)", example=65)
    ECG_T_inversion: int = Field(..., description="ECG T-wave inversion (0=No, 1=Yes)", example=1)
    ICT: float = Field(..., description="Isovolumic Contraction Time (ms)", example=110)
    RBS: float = Field(..., description="Random Blood Sugar (mg/dL)", example=180)
    EA: float = Field(..., description="E/A ratio", example=0.8)
    Chest_pain: int = Field(..., description="Chest pain (0=No, 1=Yes)", example=1)

    # Additional 9 features (optional)
    LVEF: Optional[float] = Field(None, description="Left Ventricular Ejection Fraction (%)", example=45)
    Sex: Optional[int] = Field(None, description="Sex (0=Female, 1=Male)", example=1)
    HTN: Optional[int] = Field(None, description="Hypertension (0=No, 1=Yes)", example=1)
    DM: Optional[int] = Field(None, description="Diabetes Mellitus (0=No, 1=Yes)", example=1)
    Smoker: Optional[int] = Field(None, description="Smoker (0=No, 1=Yes)", example=1)
    DL: Optional[int] = Field(None, description="Dyslipidemia (0=No, 1=Yes)", example=1)
    TropI: Optional[float] = Field(None, description="Troponin I (ng/mL)", example=0.5)
    RWMA: Optional[int] = Field(None, description="Regional Wall Motion Abnormality (0=No, 1=Yes)", example=1)
    MR: Optional[int] = Field(None, description="Mitral Regurgitation (0=None, 1=Mild, 2=Moderate)", example=1)

class PredictionResponse(BaseModel):
    """Response model for heart failure prediction"""
    probability: float = Field(..., description="Probability of heart failure")
    prediction: int = Field(..., description="Prediction (0=No HF, 1=HF)")
    threshold: float = Field(..., description="Threshold used for prediction")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: str = Field(..., description="Timestamp of prediction")
    features_used: List[str] = Field(..., description="Features used for prediction")
    patient_data: Dict[str, Union[float, int]] = Field(..., description="Patient data used for prediction")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    patients: List[PatientData] = Field(..., description="List of patient data for batch prediction")

class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    summary: Dict[str, Union[int, float]] = Field(..., description="Summary of predictions")

def load_model_from_local():
    """Load the model from local files"""
    global MODEL, MODEL_FEATURES, PREDICTION_THRESHOLD, IMPORTANT_FEATURES

    try:
        # Load the model
        model_path = "saved_models/latest/xgboost_model.pkl"
        if not os.path.exists(model_path):
            model_path = "best_xgboost_model.pkl"

        MODEL = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")

        # Get model features
        if hasattr(MODEL, 'feature_names_in_'):
            MODEL_FEATURES = MODEL.feature_names_in_
        elif hasattr(MODEL, 'feature_names'):
            MODEL_FEATURES = MODEL.feature_names
        else:
            MODEL_FEATURES = None

        # Load parameters
        params_path = "config/prediction_params.yaml"
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = yaml.safe_load(f)

            # Get prediction threshold
            if 'prediction_threshold' in params:
                PREDICTION_THRESHOLD = params['prediction_threshold']

            # Get important features
            if 'important_features' in params:
                IMPORTANT_FEATURES = params['important_features']

        return True
    except Exception as e:
        logging.error(f"Error loading model from local: {e}")
        return False

def load_model_from_s3(aws_access_key, aws_secret_key, timestamp=None):
    """Load the model from S3"""
    global MODEL, MODEL_FEATURES, PREDICTION_THRESHOLD, IMPORTANT_FEATURES, MODEL_VERSION

    try:
        # Initialize S3 client
        s3_utils = S3Utils(
            bucket_name=S3_BUCKET_NAME,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )

        # If timestamp is not provided, find the latest version
        if timestamp is None:
            # List all manifests
            manifests = s3_utils.list_files("manifests/")

            if not manifests:
                logging.error("No manifests found in S3")
                return False

            # Sort manifests by timestamp
            manifests.sort(reverse=True)

            # Extract timestamp from the latest manifest
            latest_manifest = manifests[0]
            timestamp = latest_manifest.split("/")[1].split("_")[0]

        MODEL_VERSION = timestamp

        # Create temporary directory for downloads
        temp_dir = f"temp_model_{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)

        # Download the model
        model_path = None
        s3_key = f"models/{timestamp}/best_xgboost_model.pkl"
        try:
            model_path = s3_utils.download_file(s3_key, f"{temp_dir}/model.pkl")
            logging.info(f"Model downloaded to {model_path}")
        except Exception as e:
            logging.warning(f"Could not download best model, trying saved model: {e}")

            # Try downloading the saved model
            s3_key = f"models/{timestamp}/xgboost_model.pkl"
            try:
                model_path = s3_utils.download_file(s3_key, f"{temp_dir}/model.pkl")
                logging.info(f"Model downloaded to {model_path}")
            except Exception as e:
                logging.error(f"Could not download any model: {e}")
                return False

        # Load the model
        MODEL = joblib.load(model_path)

        # Get model features
        if hasattr(MODEL, 'feature_names_in_'):
            MODEL_FEATURES = MODEL.feature_names_in_
        elif hasattr(MODEL, 'feature_names'):
            MODEL_FEATURES = MODEL.feature_names
        else:
            MODEL_FEATURES = None

        # Download model parameters
        params_path = None
        s3_key = f"config/{timestamp}/prediction_params.yaml"
        try:
            params_path = s3_utils.download_file(s3_key, f"{temp_dir}/params.yaml")
            logging.info(f"Parameters downloaded to {params_path}")

            # Load parameters
            with open(params_path, 'r') as f:
                params = yaml.safe_load(f)

            # Get prediction threshold
            if 'prediction_threshold' in params:
                PREDICTION_THRESHOLD = params['prediction_threshold']

            # Get important features
            if 'important_features' in params:
                IMPORTANT_FEATURES = params['important_features']
        except Exception as e:
            logging.warning(f"Could not download parameters: {e}")

        return True
    except Exception as e:
        logging.error(f"Error loading model from S3: {e}")
        return False

def get_model():
    """Get the model, loading it if necessary"""
    global MODEL, AWS_ACCESS_KEY, AWS_SECRET_KEY

    if MODEL is None:
        # Try loading from S3 if credentials are available
        if AWS_ACCESS_KEY and AWS_SECRET_KEY:
            if not load_model_from_s3(AWS_ACCESS_KEY, AWS_SECRET_KEY):
                # Fall back to local model if S3 fails
                if not load_model_from_local():
                    raise HTTPException(status_code=500, detail="Failed to load model")
        else:
            # Load from local if no S3 credentials
            if not load_model_from_local():
                raise HTTPException(status_code=500, detail="Failed to load model")

    return MODEL

def make_prediction(patient_data: Dict):
    """Make a prediction for a single patient"""
    model = get_model()

    # Convert patient data to DataFrame
    df = pd.DataFrame([patient_data])

    # Filter features if model expects specific features
    if MODEL_FEATURES is not None:
        # Check which features are available in the input
        available_features = [f for f in MODEL_FEATURES if f in df.columns]

        # If any features are missing, raise an error
        missing_features = [f for f in MODEL_FEATURES if f not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )

        # Filter to only include features the model expects
        df_filtered = df[available_features]
    else:
        df_filtered = df

    # Make prediction
    probability = model.predict_proba(df_filtered)[0, 1]
    prediction = 1 if probability >= PREDICTION_THRESHOLD else 0

    # Create response
    response = {
        "probability": float(probability),
        "prediction": int(prediction),
        "threshold": float(PREDICTION_THRESHOLD),
        "model_version": MODEL_VERSION or "local",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features_used": MODEL_FEATURES.tolist() if MODEL_FEATURES is not None else [],
        "patient_data": patient_data
    }

    return response

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api")
async def api_root():
    """Root endpoint for API"""
    return {
        "message": "Heart Failure Prediction API",
        "docs": "/docs",
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION or "local",
        "threshold": PREDICTION_THRESHOLD,
        "important_features": IMPORTANT_FEATURES
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """Predict heart failure for a single patient"""
    # Convert Pydantic model to dict
    patient_dict = patient.dict()

    # Remove None values
    patient_dict = {k: v for k, v in patient_dict.items() if v is not None}

    # Make prediction
    result = make_prediction(patient_dict)

    return result

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Predict heart failure for multiple patients"""
    predictions = []

    for patient in request.patients:
        # Convert Pydantic model to dict
        patient_dict = patient.dict()

        # Remove None values
        patient_dict = {k: v for k, v in patient_dict.items() if v is not None}

        # Make prediction
        try:
            result = make_prediction(patient_dict)
            predictions.append(result)
        except Exception as e:
            # Skip failed predictions in batch mode
            logging.error(f"Error predicting for patient: {e}")

    # Calculate summary statistics
    total = len(predictions)
    positive = sum(1 for p in predictions if p["prediction"] == 1)
    negative = total - positive

    summary = {
        "total_patients": total,
        "positive_predictions": positive,
        "negative_predictions": negative,
        "positive_percentage": (positive / total) * 100 if total > 0 else 0,
        "negative_percentage": (negative / total) * 100 if total > 0 else 0
    }

    return {
        "predictions": predictions,
        "summary": summary
    }

@app.post("/batch-predict-file")
async def batch_predict_file(file: UploadFile = File(...)):
    """Predict heart failure for multiple patients from a CSV file"""
    # Read CSV file
    content = await file.read()
    csv_data = content.decode('utf-8').splitlines()
    
    # Parse CSV
    reader = csv.DictReader(csv_data)
    
    # Convert to list of patient data
    patients = []
    for row in reader:
        # Convert string values to appropriate types
        patient_data = {}
        for key, value in row.items():
            if value.strip() == '':
                continue
                
            # Try to convert to number if possible
            try:
                if '.' in value:
                    patient_data[key] = float(value)
                else:
                    patient_data[key] = int(value)
            except ValueError:
                patient_data[key] = value
                
        patients.append(patient_data)
    
    # Make predictions
    predictions = []
    for patient_data in patients:
        try:
            result = make_prediction(patient_data)
            predictions.append(result)
        except Exception as e:
            logging.error(f"Error predicting for patient: {e}")
    
    # Calculate summary statistics
    total = len(predictions)
    positive = sum(1 for p in predictions if p["prediction"] == 1)
    negative = total - positive
    
    summary = {
        "total_patients": total,
        "positive_predictions": positive,
        "negative_predictions": negative,
        "positive_percentage": (positive / total) * 100 if total > 0 else 0,
        "negative_percentage": (negative / total) * 100 if total > 0 else 0
    }
    
    return {
        "predictions": predictions,
        "summary": summary
    }

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    model = get_model()

    return {
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION or "local",
        "model_features": MODEL_FEATURES.tolist() if MODEL_FEATURES is not None else [],
        "threshold": PREDICTION_THRESHOLD,
        "important_features": IMPORTANT_FEATURES
    }

@app.post("/load-model")
async def load_model(
    aws_access_key: str = Query(..., description="AWS access key ID"),
    aws_secret_key: str = Query(..., description="AWS secret access key"),
    timestamp: Optional[str] = Query(None, description="Model version timestamp (if None, will use latest)")
):
    """Load a specific model version from S3"""
    global AWS_ACCESS_KEY, AWS_SECRET_KEY

    # Store credentials for future use
    AWS_ACCESS_KEY = aws_access_key
    AWS_SECRET_KEY = aws_secret_key

    # Load the model
    if load_model_from_s3(aws_access_key, aws_secret_key, timestamp):
        return {
            "message": "Model loaded successfully",
            "model_version": MODEL_VERSION,
            "threshold": PREDICTION_THRESHOLD,
            "important_features": IMPORTANT_FEATURES
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to load model from S3")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Load the model on startup
@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    try:
        load_model_from_local()
    except Exception as e:
        logging.error(f"Error loading model on startup: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
