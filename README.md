# Heart Failure Prediction System

![Heart Failure Prediction](https://img.shields.io/badge/ML-Heart%20Failure%20Prediction-red)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-FastAPI%2FFlask-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A sophisticated machine learning system for predicting heart failure risk using clinical parameters. This application provides healthcare professionals with an intuitive interface to assess patient risk and make informed clinical decisions.

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Information](#model-information)
- [AWS S3 Integration](#aws-s3-integration)
- [Project Workflow](#project-workflow)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Interactive Dashboard**: Real-time visualization of prediction statistics and model performance
- **Single Patient Prediction**: Comprehensive form for individual patient assessment
- **Batch Prediction**: Process multiple patients simultaneously via CSV upload
- **Model Management**: Load models from local files or AWS S3
- **Prediction History**: Track and review previous predictions
- **Responsive Design**: Optimized for both desktop and mobile devices
- **API Documentation**: Interactive documentation with Swagger UI
- **Docker Support**: Easy deployment with Docker and Docker Compose

## 🏗️ System Architecture

The system consists of three main components:

1. **Machine Learning Backend**:
   - Ensemble of gradient boosting models (XGBoost, LightGBM, CatBoost)
   - Feature preprocessing and normalization pipeline
   - Model validation and calibration system

2. **API Layer**:
   - RESTful API built with FastAPI
   - Alternative Flask implementation for compatibility
   - Endpoint validation and error handling

3. **Web Interface**:
   - Responsive Bootstrap-based UI
   - Interactive charts using Chart.js
   - Client-side data validation

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ML-Model-Deployment
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Application

You can run the application using either the FastAPI or Flask backend:

#### FastAPI Version (Original)
```bash
python run_app_1010.py
# Or use uvicorn directly
uvicorn app:app --reload --host 0.0.0.0 --port 1010
```
Access the application at: http://localhost:1010

#### Flask Alternative
```bash
python run_flask.py
```
Access the application at: http://localhost:1020

### Using Docker

1. Build and run the Docker container:
   ```bash
   docker-compose up --build
   ```

2. Access the application:
   - Open your browser and go to [http://localhost:8000](http://localhost:8000)

### Making Predictions

1. **Single Patient**:
   - Navigate to the "Single Prediction" tab
   - Fill in the required patient parameters
   - Click "Predict" to see the results

2. **Batch Prediction**:
   - Navigate to the "Batch Prediction" tab
   - Download the CSV template if needed
   - Upload your CSV file with patient data
   - Click "Run Batch Prediction"

## 📚 API Documentation

The API provides the following endpoints:

### Core Endpoints

- **GET /** - Get basic information about the API and access the web interface
- **GET /health** - Health check endpoint to verify API status
- **GET /api/model-info** - Retrieve information about the loaded model

### Prediction Endpoints

- **POST /api/predict** - Predict heart failure for a single patient
  - Request body: Patient data (see example below)
  - Response: Prediction result with probability and threshold

- **POST /api/batch-predict** - Predict heart failure for multiple patients
  - Request body: List of patient data
  - Response: List of prediction results and summary statistics

### Model Management

- **POST /load-model** - Load a specific model version from S3
  - Query parameters:
    - `aws_access_key`: AWS access key ID
    - `aws_secret_key`: AWS secret access key
    - `timestamp`: Model version timestamp (optional)

For detailed API documentation, visit `/docs` when the server is running.

### Code Examples

#### Single Prediction API Call

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
response = requests.post("http://localhost:1010/api/predict", json=patient_data)
result = response.json()

print(f"Probability: {result['probability']}")
print(f"Prediction: {result['prediction']}")  # 0 = No HF, 1 = HF
print(f"Threshold: {result['threshold']}")
```

#### Batch Prediction API Call

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
response = requests.post("http://localhost:1010/api/batch-predict", json=batch_data)
result = response.json()

print(f"Total patients: {result['summary']['total_patients']}")
print(f"Positive predictions: {result['summary']['positive_predictions']}")
print(f"Negative predictions: {result['summary']['negative_predictions']}")
```

### Testing

You can use the included test scripts to verify the system:

```bash
# Test the API endpoints
python test_api.py

# Test with a CSV file containing patient data
python test_api.py --csv path/to/patients.csv

# Test model predictions
python test_model_prediction.py
```

## 🧠 Model Information

### Models Used

- **Primary Model**: XGBoost (Gradient Boosting)
- **Alternative Models**: LightGBM, CatBoost

### Performance Metrics

| Metric      | Value |
|-------------|-------|
| Accuracy    | 85%   |
| Sensitivity | 82%   |
| Specificity | 88%   |
| AUC-ROC     | 0.89  |

### Required Features (16)

The model requires the following clinical parameters:

- **FS**: Fractional Shortening (%)
- **DT**: Deceleration Time (ms)
- **NYHA**: New York Heart Association classification (1-4)
- **HR**: Heart Rate (bpm)
- **BNP**: B-type Natriuretic Peptide (pg/mL)
- **LVIDs**: Left Ventricular Internal Dimension in systole (cm)
- **BMI**: Body Mass Index (kg/m²)
- **LAV**: Left Atrial Volume (mL)
- **Wall_Subendocardial**: Subendocardial Wall (0=No, 1=Yes)
- **LDLc**: Low-Density Lipoprotein cholesterol (mg/dL)
- **Age**: Age (years)
- **ECG_T_inversion**: ECG T-wave inversion (0=No, 1=Yes)
- **ICT**: Isovolumic Contraction Time (ms)
- **RBS**: Random Blood Sugar (mg/dL)
- **EA**: E/A ratio
- **Chest_pain**: Chest pain (0=No, 1=Yes)

### Optional Features (9)

The following parameters can enhance prediction accuracy:

- **LVEF**: Left Ventricular Ejection Fraction (%)
- **Sex**: Sex (0=Female, 1=Male)
- **HTN**: Hypertension (0=No, 1=Yes)
- **DM**: Diabetes Mellitus (0=No, 1=Yes)
- **Smoker**: Smoker (0=No, 1=Yes)
- **DL**: Dyslipidemia (0=No, 1=Yes)
- **TropI**: Troponin I (ng/mL)
- **RWMA**: Regional Wall Motion Abnormality (0=No, 1=Yes)
- **MR**: Mitral Regurgitation (0=None, 1=Mild, 2=Moderate)

## 💾 AWS S3 Integration

The system can load models from AWS S3 for versioning and deployment:

1. **Setup**: Create an AWS S3 bucket named "hf-predication-model2025"
2. **Upload**: Use the `upload_to_s3.py` script to upload models to the bucket
   ```bash
   python upload_to_s3.py --aws-access-key YOUR_ACCESS_KEY --aws-secret-key YOUR_SECRET_KEY
   ```
3. **Load**: Use the `/load-model` endpoint to load a specific model version

## 📝 Project Workflow

This project follows a structured MLOps workflow:

### Development Process

1. **Project Setup**
   - Template creation and folder structure setup
   - Configuration of dependencies and environment
   - Logger and exception handling implementation

2. **Data Pipeline**
   - Data ingestion from various sources
   - Data validation and drift detection using Evidently
   - Feature engineering and preprocessing

3. **Model Development**
   - Exploratory data analysis
   - Model training and hyperparameter optimization
   - Model evaluation and selection

4. **Deployment**
   - API development with FastAPI
   - Web interface creation
   - Docker containerization

### Component Architecture

Each component follows a consistent structure:
1. Constants definition
2. Entity creation
3. Component implementation
4. Pipeline integration
5. Main execution files

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Disclaimer**: This tool is intended to assist healthcare professionals and should not replace clinical judgment. Always consult with a qualified healthcare provider for diagnosis and treatment decisions.

© 2025 Heart Failure Prediction System
