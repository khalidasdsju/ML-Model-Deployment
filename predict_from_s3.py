import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import yaml
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HF.utils.s3_utils import S3Utils
from HF.logger import logging

def download_model(aws_access_key_id, aws_secret_access_key, timestamp=None, bucket_name="hf-predication-model2025"):
    """
    Download the model from S3

    Parameters:
    -----------
    aws_access_key_id : str
        AWS access key ID
    aws_secret_access_key : str
        AWS secret access key
    timestamp : str, optional
        Timestamp of the version to download. If None, will use the latest version
    bucket_name : str, optional
        S3 bucket name

    Returns:
    --------
    tuple
        (model_path, params_path) - Paths to the downloaded model and parameters
    """
    try:
        # Initialize S3 client
        s3_utils = S3Utils(
            bucket_name=bucket_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        # If timestamp is not provided, find the latest version
        if timestamp is None:
            # List all manifests
            manifests = s3_utils.list_files("manifests/")

            if not manifests:
                print("No manifests found in S3")
                return None, None

            # Sort manifests by timestamp (assuming format: manifests/TIMESTAMP_manifest.txt)
            manifests.sort(reverse=True)

            # Extract timestamp from the latest manifest
            latest_manifest = manifests[0]
            timestamp = latest_manifest.split("/")[1].split("_")[0]

            print(f"Using latest version: {timestamp}")

        # Create temporary directory for downloads
        temp_dir = f"temp_model_{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)

        # Download the model
        model_path = None
        s3_key = f"models/{timestamp}/best_xgboost_model.pkl"
        try:
            model_path = s3_utils.download_file(s3_key, f"{temp_dir}/model.pkl")
            print(f"Model downloaded to {model_path}")
        except Exception as e:
            logging.warning(f"Could not download best model, trying saved model: {e}")

            # Try downloading the saved model
            s3_key = f"models/{timestamp}/xgboost_model.pkl"
            try:
                model_path = s3_utils.download_file(s3_key, f"{temp_dir}/model.pkl")
                print(f"Model downloaded to {model_path}")
            except Exception as e:
                logging.error(f"Could not download any model: {e}")
                print(f"Error: Could not download any model")
                return None, None

        # Download model parameters
        params_path = None
        s3_key = f"config/{timestamp}/prediction_params.yaml"
        try:
            params_path = s3_utils.download_file(s3_key, f"{temp_dir}/params.yaml")
            print(f"Parameters downloaded to {params_path}")
        except Exception as e:
            logging.warning(f"Could not download parameters: {e}")
            print(f"Warning: Could not download parameters, will use default values")

        return model_path, params_path
    except Exception as e:
        logging.error(f"Error downloading model from S3: {e}")
        print(f"Error downloading model from S3: {e}")
        return None, None

def make_prediction(input_data, model_path, params_path=None, threshold=0.001):
    """
    Make predictions using the downloaded model

    Parameters:
    -----------
    input_data : str or pandas.DataFrame
        Path to CSV file or DataFrame with input data
    model_path : str
        Path to the model file
    params_path : str, optional
        Path to the parameters file
    threshold : float, optional
        Prediction threshold (default: 0.001)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with predictions
    """
    try:
        # Load the model
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

        # Load parameters if available
        if params_path and os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = yaml.safe_load(f)

            # Get important features
            important_features = None
            if 'important_features' in params:
                important_features = params['important_features']
                print(f"Important features: {important_features}")

        # Use the provided threshold
        print(f"Using prediction threshold: {threshold}")

        # Load input data
        if isinstance(input_data, str):
            # Load from CSV file
            data = pd.read_csv(input_data)
            print(f"Input data loaded from {input_data}, shape: {data.shape}")
        else:
            # Use provided DataFrame
            data = input_data
            print(f"Using provided DataFrame, shape: {data.shape}")

        # Make predictions
        try:
            # Check if model expects different features than what we have
            model_features = None
            if hasattr(model, 'feature_names_in_'):
                model_features = model.feature_names_in_
            elif hasattr(model, 'feature_names'):
                model_features = model.feature_names

            if model_features is not None:
                # Filter data to only include features the model expects
                print(f"\nFiltering data to include only the {len(model_features)} features the model was trained on")
                data_filtered = data[model_features]
                print(f"Filtered data shape: {data_filtered.shape}")
            else:
                data_filtered = data

            # Make predictions using the filtered data
            probabilities = model.predict_proba(data_filtered)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

            # Create results DataFrame
            results = pd.DataFrame({
                'Probability': probabilities,
                'Prediction': predictions
            })

            # Add original data (all 25 features)
            for col in data.columns:
                results[col] = data[col].values

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = f"predictions_{timestamp}.csv"
            results.to_csv(results_path, index=False)

            print(f"\nPredictions saved to {results_path}")
            print(f"\nPrediction Summary:")
            print(f"  Total samples: {len(results)}")
            print(f"  Positive predictions (HF): {results['Prediction'].sum()} ({results['Prediction'].mean()*100:.1f}%)")
            print(f"  Negative predictions (No HF): {len(results) - results['Prediction'].sum()} ({(1-results['Prediction'].mean())*100:.1f}%)")

            return results
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            print(f"Error making predictions: {e}")

            # Try to identify the issue
            model_features = None
            try:
                if hasattr(model, 'feature_names_in_'):
                    model_features = model.feature_names_in_
                elif hasattr(model, 'feature_names'):
                    model_features = model.feature_names

                if model_features is not None:
                    print(f"\nModel expects these features: {model_features}")
                    print(f"Input data has these features: {data.columns.tolist()}")

                    # Find missing features
                    missing_features = set(model_features) - set(data.columns)
                    if missing_features:
                        print(f"\nMissing features in input data: {missing_features}")

                    # Find extra features
                    extra_features = set(data.columns) - set(model_features)
                    if extra_features:
                        print(f"\nExtra features in input data: {extra_features}")
            except Exception:
                pass

            return None
    except Exception as e:
        logging.error(f"Error in prediction process: {e}")
        print(f"Error in prediction process: {e}")
        return None

def create_sample_data():
    """
    Create a sample dataset for testing

    Returns:
    --------
    pandas.DataFrame
        Sample DataFrame
    """
    # Create sample data with the required features
    sample_data = {
        # Original 16 features
        'FS': [25, 40, 20, 35],
        'DT': [160, 220, 150, 200],
        'NYHA': [3, 1, 4, 2],
        'HR': [95, 75, 110, 80],
        'BNP': [800, 50, 1200, 100],
        'LVIDs': [4.8, 3.0, 5.2, 3.5],
        'BMI': [28.5, 22.1, 30.2, 24.5],
        'LAV': [45, 30, 50, 35],
        'Wall_Subendocardial': [1, 0, 1, 0],
        'LDLc': [140, 100, 150, 110],
        'Age': [65, 45, 70, 55],
        'ECG_T_inversion': [1, 0, 1, 0],
        'ICT': [110, 80, 120, 90],
        'RBS': [180, 110, 200, 120],
        'EA': [0.8, 1.5, 0.7, 1.2],
        'Chest_pain': [1, 0, 1, 0],

        # Additional 9 features to reach 25
        'LVEF': [45, 60, 40, 55],
        'Sex': [1, 0, 1, 0],  # 1 for male, 0 for female
        'HTN': [1, 0, 1, 0],  # Hypertension
        'DM': [1, 0, 1, 0],   # Diabetes Mellitus
        'Smoker': [1, 0, 1, 0],
        'DL': [1, 0, 1, 0],    # Dyslipidemia
        'TropI': [0.5, 0.1, 0.8, 0.2],  # Troponin I
        'RWMA': [1, 0, 1, 0],  # Regional Wall Motion Abnormality
        'MR': [1, 0, 2, 0]     # Mitral Regurgitation (0=None, 1=Mild, 2=Moderate)
    }

    # Create DataFrame
    df = pd.DataFrame(sample_data)

    # Save sample data to CSV
    sample_file_path = "sample_data_for_prediction.csv"
    df.to_csv(sample_file_path, index=False)
    print(f"Sample data saved to {sample_file_path}")

    return df, sample_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using heart failure model from S3")
    parser.add_argument("--access-key", required=True,
                        help="AWS access key ID")
    parser.add_argument("--secret-key", required=True,
                        help="AWS secret access key")
    parser.add_argument("--bucket", default="hf-predication-model2025",
                        help="S3 bucket name")
    parser.add_argument("--timestamp",
                        help="Timestamp of the version to use (if None, will use the latest version)")
    parser.add_argument("--input",
                        help="Path to input CSV file (if None, will use sample data)")
    parser.add_argument("--threshold", type=float, default=0.001,
                        help="Prediction threshold (default: 0.001)")

    args = parser.parse_args()

    # Download model from S3
    model_path, params_path = download_model(args.access_key, args.secret_key, args.timestamp, args.bucket)

    if model_path:
        # Prepare input data
        if args.input:
            # Use provided input file
            make_prediction(args.input, model_path, params_path, args.threshold)
        else:
            # Create and use sample data
            _, sample_file_path = create_sample_data()
            make_prediction(sample_file_path, model_path, params_path, args.threshold)
