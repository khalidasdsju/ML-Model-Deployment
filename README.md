# Heart Failure Prediction API

This project provides a FastAPI-based REST API for heart failure prediction using machine learning models.

## Features

- **Heart Failure Prediction**: Predict the risk of heart failure based on patient data
- **Batch Prediction**: Make predictions for multiple patients at once
- **Model Management**: Load models from local files or AWS S3
- **API Documentation**: Interactive API documentation with Swagger UI
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- Docker (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ML-Model-Deployment
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the API:
   ```bash
   uvicorn app:app --reload
   ```

4. Access the API documentation:
   - Open your browser and go to [http://localhost:8000/docs](http://localhost:8000/docs)

### Using Docker

1. Build and run the Docker container:
   ```bash
   docker-compose up --build
   ```

2. Access the API documentation:
   - Open your browser and go to [http://localhost:8000/docs](http://localhost:8000/docs)

## API Endpoints

### Root Endpoint

- **GET /** - Get basic information about the API

### Health Check

- **GET /health** - Check if the API is running

### Model Information

- **GET /model-info** - Get information about the loaded model

### Prediction

- **POST /predict** - Predict heart failure for a single patient
  - Request body: Patient data (see example below)
  - Response: Prediction result with probability and threshold

### Batch Prediction

- **POST /batch-predict** - Predict heart failure for multiple patients
  - Request body: List of patient data
  - Response: List of prediction results and summary statistics

### Load Model from S3

- **POST /load-model** - Load a specific model version from S3
  - Query parameters:
    - `aws_access_key`: AWS access key ID
    - `aws_secret_key`: AWS secret access key
    - `timestamp`: Model version timestamp (optional)

## Example Usage

### Single Prediction

```python
import requests

# Patient data
patient_data = {
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
    "Chest_pain": 1
}

# Make prediction
response = requests.post("http://localhost:8000/predict", json=patient_data)
result = response.json()

print(f"Probability: {result['probability']}")
print(f"Prediction: {result['prediction']}")  # 0 = No HF, 1 = HF
print(f"Threshold: {result['threshold']}")
```

### Batch Prediction

```python
import requests

# List of patients
batch_data = {
    "patients": [
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
            "Chest_pain": 1
        },
        {
            "FS": 40,
            "DT": 220,
            "NYHA": 1,
            "HR": 75,
            "BNP": 50,
            "LVIDs": 3.0,
            "BMI": 22.1,
            "LAV": 30,
            "Wall_Subendocardial": 0,
            "LDLc": 100,
            "Age": 45,
            "ECG_T_inversion": 0,
            "ICT": 80,
            "RBS": 110,
            "EA": 1.5,
            "Chest_pain": 0
        }
    ]
}

# Make batch prediction
response = requests.post("http://localhost:8000/batch-predict", json=batch_data)
result = response.json()

print(f"Total patients: {result['summary']['total_patients']}")
print(f"Positive predictions: {result['summary']['positive_predictions']}")
print(f"Negative predictions: {result['summary']['negative_predictions']}")
```

### Loading a Model from S3

```python
import requests

# Load model from S3
response = requests.post(
    "http://localhost:8000/load-model",
    params={
        "aws_access_key": "YOUR_AWS_ACCESS_KEY",
        "aws_secret_key": "YOUR_AWS_SECRET_KEY",
        "timestamp": "20250409_153349"  # Optional
    }
)

print(response.json())
```

## Testing the API

You can use the included test script to test the API:

```bash
python test_api.py
```

To test with a CSV file containing patient data:

```bash
python test_api.py --csv path/to/patients.csv
```

## Model Features

The model uses the following features for prediction:

### Required Features (16)
- FS: Fractional Shortening (%)
- DT: Deceleration Time (ms)
- NYHA: New York Heart Association classification (1-4)
- HR: Heart Rate (bpm)
- BNP: B-type Natriuretic Peptide (pg/mL)
- LVIDs: Left Ventricular Internal Dimension in systole (cm)
- BMI: Body Mass Index (kg/m²)
- LAV: Left Atrial Volume (mL)
- Wall_Subendocardial: Subendocardial Wall (0=No, 1=Yes)
- LDLc: Low-Density Lipoprotein cholesterol (mg/dL)
- Age: Age (years)
- ECG_T_inversion: ECG T-wave inversion (0=No, 1=Yes)
- ICT: Isovolumic Contraction Time (ms)
- RBS: Random Blood Sugar (mg/dL)
- EA: E/A ratio
- Chest_pain: Chest pain (0=No, 1=Yes)

### Optional Features (9)
- LVEF: Left Ventricular Ejection Fraction (%)
- Sex: Sex (0=Female, 1=Male)
- HTN: Hypertension (0=No, 1=Yes)
- DM: Diabetes Mellitus (0=No, 1=Yes)
- Smoker: Smoker (0=No, 1=Yes)
- DL: Dyslipidemia (0=No, 1=Yes)
- TropI: Troponin I (ng/mL)
- RWMA: Regional Wall Motion Abnormality (0=No, 1=Yes)
- MR: Mitral Regurgitation (0=None, 1=Mild, 2=Moderate)

## AWS S3 Integration

The API can load models from AWS S3. To use this feature:

1. Set up an AWS S3 bucket named "hf-predication-model2025"
2. Upload your models to the bucket using the `upload_to_s3.py` script
3. Use the `/load-model` endpoint to load a specific model version

## Original Project Workflow

### Workflow: Create
#### 1.template.py
![alt text](<folder structure.png>)
#### 2. Update setup.py to install requirements.txt pakages
#### 3. Stoe data in Mongodb
#### 4. Add logger files code
#### 5. Write Custom exception code
#### 6. Add main Utils Files code
#### 7. Create jypeter files for EDA and Model Training
#### 8. Start Update HF/Component:
#####   "Work flow every components:"
######          "1.Constant"
######          "2.entity"
######          "3.components"
######          "4.Pipeline"
######          "5.Main Files"
####    1.Data Ingestion
####     ![alt text](<Data Ingestion.png>)
####    2. Data Drift with Evidently(MLOps Tool) & Data Validation
####      ![alt text](<Data Validation.png>)
####    3. Data Transformation & Model training
####      ![alt text](<Data Transformation-1.png>)
####