# API Documentation

## Endpoints

### GET /

Home page with UI for making predictions.

### GET /api

Root endpoint for API information.

### POST /predict

Predict heart failure for a single patient.

#### Request Body

```json
{
  "FS": 25,
  "DT": 160,
  "NYHA": 3,
  "HR": 95,
  "BNP": 800,
  "LVIDs": 4.8,
  "BMI": 28.5,
  "LAV": 45,
  "Wall_Subendocardial": 1,
  "LDLc": 140,
  "Age": 65,
  "ECG_T_inversion": 1,
  "ICT": 110,
  "RBS": 180,
  "EA": 0.8,
  "Chest_pain": 1,
  "LVEF": 45,
  "Sex": 1,
  "HTN": 1,
  "DM": 1,
  "Smoker": 1,
  "DL": 1,
  "TropI": 0.5,
  "RWMA": 1,
  "MR": 1
}
```

#### Response

```json
{
  "probability": 0.85,
  "prediction": 1,
  "threshold": 0.5,
  "model_version": "20250409",
  "timestamp": "2025-04-09 15:30:00",
  "features_used": ["FS", "DT", "NYHA", ...],
  "patient_data": {
    "FS": 25,
    "DT": 160,
    ...
  }
}
```

### POST /batch-predict

Predict heart failure for multiple patients.

### POST /batch-predict-file

Predict heart failure from a CSV file.

### GET /model-info

Get information about the loaded model.

### POST /load-model

Load a specific model version from S3.

### GET /health

Health check endpoint.
