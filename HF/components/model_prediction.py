# HF/components/model_prediction.py
import os
import sys
import yaml
import numpy as np
import pandas as pd
import joblib
from HF.logger import logging
from HF.exception import HFException
from HF.entity.config_entity import ModelPredictionConfig
from HF.entity.artifact_entity import ModelPredictionArtifact

class ModelPrediction:
    def __init__(self, model_prediction_config: ModelPredictionConfig):
        try:
            self.model_prediction_config = model_prediction_config
        except Exception as e:
            raise HFException(f"Error in ModelPrediction initialization: {e}", sys)
    
    def load_model(self):
        try:
            logging.info(f"Loading model from {self.model_prediction_config.model_file_path}")
            model = joblib.load(self.model_prediction_config.model_file_path)
            return model
        except Exception as e:
            raise HFException(f"Error loading model: {e}", sys)
    
    def load_prediction_params(self):
        try:
            logging.info(f"Loading prediction parameters from {self.model_prediction_config.params_file_path}")
            with open(self.model_prediction_config.params_file_path, 'r') as f:
                params = yaml.safe_load(f)
            return params
        except Exception as e:
            raise HFException(f"Error loading prediction parameters: {e}", sys)
    
    def preprocess_data(self, data):
        try:
            logging.info("Preprocessing input data for prediction")
            
            # Load prediction parameters
            params = self.load_prediction_params()
            
            # Make a copy of the data
            processed_data = data.copy()
            
            # Drop unnecessary columns
            if 'drop_columns' in params['preprocessing']:
                columns_to_drop = [col for col in params['preprocessing']['drop_columns'] if col in processed_data.columns]
                if columns_to_drop:
                    processed_data = processed_data.drop(columns=columns_to_drop)
                    logging.info(f"Dropped columns: {columns_to_drop}")
            
            # Ensure all required columns are present
            required_columns = (
                params['preprocessing'].get('numerical_columns', []) + 
                params['preprocessing'].get('categorical_columns', [])
            )
            
            missing_columns = [col for col in required_columns if col not in processed_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert categorical columns to numeric if needed
            for col in params['preprocessing'].get('categorical_columns', []):
                if col in processed_data.columns and processed_data[col].dtype == 'object':
                    processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            
            # Fill missing values
            for col in processed_data.columns:
                if processed_data[col].isna().any():
                    if col in params['preprocessing'].get('numerical_columns', []):
                        # Fill numerical columns with median
                        processed_data[col] = processed_data[col].fillna(processed_data[col].median())
                    else:
                        # Fill categorical columns with mode
                        processed_data[col] = processed_data[col].fillna(processed_data[col].mode()[0])
            
            logging.info(f"Preprocessed data shape: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            raise HFException(f"Error preprocessing data: {e}", sys)
    
    def make_prediction(self, data):
        try:
            logging.info("Making predictions")
            
            # Load model
            model = self.load_model()
            
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            # Make predictions
            predictions_prob = model.predict_proba(processed_data)[:, 1]
            
            # Load prediction parameters
            params = self.load_prediction_params()
            threshold = params.get('prediction_threshold', 0.5)
            
            # Convert probabilities to binary predictions
            predictions = (predictions_prob >= threshold).astype(int)
            
            # Create a DataFrame with predictions
            result_df = pd.DataFrame({
                'Probability': predictions_prob,
                'Prediction': predictions
            })
            
            # Add original data
            for col in data.columns:
                result_df[col] = data[col].values
            
            # Save predictions
            os.makedirs(os.path.dirname(self.model_prediction_config.prediction_file_path), exist_ok=True)
            result_df.to_csv(self.model_prediction_config.prediction_file_path, index=False)
            
            logging.info(f"Predictions saved to {self.model_prediction_config.prediction_file_path}")
            
            # Calculate accuracy if target column is available
            accuracy = None
            if 'HF' in data.columns:
                accuracy = (predictions == data['HF']).mean()
                logging.info(f"Prediction accuracy: {accuracy}")
            
            # Create prediction artifact
            prediction_artifact = ModelPredictionArtifact(
                prediction_file_path=self.model_prediction_config.prediction_file_path,
                model_accuracy=accuracy if accuracy is not None else 0.0
            )
            
            return prediction_artifact
            
        except Exception as e:
            raise HFException(f"Error making predictions: {e}", sys)
    
    def initiate_model_prediction(self, data=None) -> ModelPredictionArtifact:
        try:
            logging.info("Initiating model prediction")
            
            # If data is not provided, load from a default location
            if data is None:
                # You can modify this to load data from a specific location
                logging.warning("No data provided for prediction. Using sample data.")
                # Create a sample DataFrame with required columns
                params = self.load_prediction_params()
                required_columns = (
                    params['preprocessing'].get('numerical_columns', []) + 
                    params['preprocessing'].get('categorical_columns', [])
                )
                data = pd.DataFrame(columns=required_columns)
                # Add some sample data
                data.loc[0] = 0  # Initialize with zeros
            
            # Make predictions
            prediction_artifact = self.make_prediction(data)
            
            logging.info(f"Model prediction completed: {prediction_artifact}")
            return prediction_artifact
            
        except Exception as e:
            raise HFException(f"Error in model prediction: {e}", sys)
