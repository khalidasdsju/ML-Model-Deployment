import os
import pandas as pd
from HF.pipline.training_pipeline import TrainPipeline
from HF.pipline.prediction_pipeline import PredictionPipeline
from HF.entity.config_entity import TrainingPipelineConfig
from HF.logger import logging

def train_model():
    """Train the model using the training pipeline"""
    try:
        logging.info("Starting model training")
        
        # Initialize training pipeline
        training_pipeline_config = TrainingPipelineConfig()
        train_pipeline = TrainPipeline(training_pipeline_config)
        
        # Run the pipeline
        model_pusher_artifact = train_pipeline.run()
        
        logging.info(f"Model training completed. Model saved at: {model_pusher_artifact.saved_model_path}")
        return model_pusher_artifact
        
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise e

def predict_sample():
    """Make predictions on a sample dataset"""
    try:
        logging.info("Starting sample prediction")
        
        # Create a sample dataset
        sample_data = {
            'Age': [65, 45, 70, 55],
            'Sex': [1, 0, 1, 0],
            'BMI': [28.5, 22.1, 30.2, 24.5],
            'NYHA': [3, 1, 4, 2],
            'HR': [95, 75, 110, 80],
            'HTN': [1, 0, 1, 1],
            'DM': [1, 0, 1, 0],
            'Smoker': [1, 0, 0, 1],
            'DL': [1, 0, 1, 0],
            'CXR': [1, 0, 1, 0],
            'ECG': [2, 0, 2, 1],
            'LVIDd': [6.2, 4.5, 6.5, 5.0],
            'FS': [25, 40, 20, 35],
            'LVIDs': [4.8, 3.0, 5.2, 3.5],
            'LVEF': [35, 60, 30, 55],
            'LAV': [45, 30, 50, 35],
            'ICT': [110, 80, 120, 90],
            'IRT': [95, 70, 100, 75],
            'EA': [0.8, 1.5, 0.7, 1.2],
            'DT': [160, 220, 150, 200],
            'RR': [22, 16, 24, 18],
            'TC': [220, 180, 240, 190],
            'LDLc': [140, 100, 150, 110],
            'BNP': [800, 50, 1200, 100],
            'Chest_pain': [1, 0, 1, 0],
            'RWMA': [1, 0, 1, 0],
            'MI': [1, 0, 1, 0],
            'ACS': [1, 0, 2, 0],
            'Wall': [1, 0, 2, 0],
            'Thrombolysis': [1, 0, 0, 0],
            'MR': [1, 0, 1, 0]
        }
        
        # Create DataFrame
        df = pd.DataFrame(sample_data)
        
        # Save sample data to CSV
        sample_file_path = "sample_data.csv"
        df.to_csv(sample_file_path, index=False)
        logging.info(f"Sample data saved to {sample_file_path}")
        
        # Initialize prediction pipeline
        prediction_pipeline = PredictionPipeline()
        
        # Make predictions
        prediction_artifact = prediction_pipeline.run_batch_prediction(sample_file_path)
        
        # Load and display predictions
        predictions = pd.read_csv(prediction_artifact.prediction_file_path)
        logging.info(f"Predictions:\n{predictions[['Probability', 'Prediction']].head()}")
        
        return prediction_artifact
        
    except Exception as e:
        logging.error(f"Error in sample prediction: {e}")
        raise e

def main():
    """Main function to demonstrate model training and prediction"""
    try:
        # Train the model
        train_model()
        
        # Make predictions
        predict_sample()
        
        logging.info("Model demo completed successfully")
        
    except Exception as e:
        logging.error(f"Error in model demo: {e}")
        raise e

if __name__ == "__main__":
    main()
