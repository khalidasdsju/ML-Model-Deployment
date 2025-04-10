import requests
import json
import argparse
import pandas as pd

def test_api(url="http://localhost:8000"):
    """Test the FastAPI endpoints"""
    print(f"Testing API at {url}")
    
    # Test root endpoint
    print("\n1. Testing root endpoint...")
    response = requests.get(f"{url}/")
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test health endpoint
    print("\n2. Testing health endpoint...")
    response = requests.get(f"{url}/health")
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test model info endpoint
    print("\n3. Testing model info endpoint...")
    response = requests.get(f"{url}/model-info")
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test prediction endpoint with sample data
    print("\n4. Testing prediction endpoint...")
    sample_data = {
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
    
    response = requests.post(f"{url}/predict", json=sample_data)
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test batch prediction endpoint with sample data
    print("\n5. Testing batch prediction endpoint...")
    batch_data = {
        "patients": [
            sample_data,
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
                "Chest_pain": 0,
                "LVEF": 60,
                "Sex": 0,
                "HTN": 0,
                "DM": 0,
                "Smoker": 0,
                "DL": 0,
                "TropI": 0.1,
                "RWMA": 0,
                "MR": 0
            }
        ]
    }
    
    response = requests.post(f"{url}/batch-predict", json=batch_data)
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_with_csv(csv_file, url="http://localhost:8000"):
    """Test the API with data from a CSV file"""
    print(f"Testing API with data from {csv_file}")
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert DataFrame to list of dictionaries
    patients = df.to_dict(orient="records")
    
    # Create batch request
    batch_data = {
        "patients": patients
    }
    
    # Send batch prediction request
    response = requests.post(f"{url}/batch-predict", json=batch_data)
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Summary: {json.dumps(result['summary'], indent=2)}")
        
        # Save predictions to CSV
        predictions_df = pd.DataFrame([
            {
                "patient_id": i + 1,
                "probability": pred["probability"],
                "prediction": pred["prediction"],
                "threshold": pred["threshold"]
            }
            for i, pred in enumerate(result["predictions"])
        ])
        
        output_file = "api_predictions.csv"
        predictions_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Heart Failure Prediction API")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--csv", help="CSV file with patient data")
    
    args = parser.parse_args()
    
    if args.csv:
        test_with_csv(args.csv, args.url)
    else:
        test_api(args.url)
