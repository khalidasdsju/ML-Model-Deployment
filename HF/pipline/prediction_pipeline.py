# HF/pipline/prediction_pipeline.py
import sys
import pandas as pd
from HF.exception import HFException
from HF.logger import logging
from HF.entity.config_entity import ModelPredictionConfig
from HF.entity.artifact_entity import ModelPredictionArtifact
from HF.components.model_prediction import ModelPrediction

class PredictionPipeline:
    def __init__(self):
        self.model_prediction_config = ModelPredictionConfig()

    def predict(self, input_data=None):
        try:
            logging.info("Starting prediction pipeline")

            # Initialize ModelPrediction
            model_prediction = ModelPrediction(
                model_prediction_config=self.model_prediction_config
            )

            # Make predictions
            prediction_artifact = model_prediction.initiate_model_prediction(input_data)

            logging.info("Prediction completed successfully")
            return prediction_artifact

        except Exception as e:
            raise HFException(f"Error in prediction pipeline: {e}", sys)

    def run_batch_prediction(self, file_path):
        try:
            logging.info(f"Starting batch prediction on file: {file_path}")

            # Load data from file
            data = pd.read_csv(file_path)
            logging.info(f"Loaded data with shape: {data.shape}")

            # Make predictions
            prediction_artifact = self.predict(data)

            logging.info(f"Batch prediction completed. Results saved to: {prediction_artifact.prediction_file_path}")
            return prediction_artifact

        except Exception as e:
            raise HFException(f"Error in batch prediction: {e}", sys)