# demo.py
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

        # Define the configuration for the training pipeline
        training_pipeline_config = TrainingPipelineConfig()

        # Create the TrainPipeline object
        train_pipeline = TrainPipeline(training_pipeline_config=training_pipeline_config)

        # Run the pipeline
        model_pusher_artifact = train_pipeline.run()

        logging.info(f"Model training completed. Model saved at: {model_pusher_artifact.saved_model_path}")
        return model_pusher_artifact

    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise e

def create_sample_data():
    """Create a sample dataset for testing"""
    sample_data = {
        # Required features based on the model
        'FS': [25, 40, 20, 35],
        'DT': [160, 220, 150, 200],
        'NYHA': [3, 1, 4, 2],
        'HR': [95, 75, 110, 80],
        'BNP': [800, 50, 1200, 100],
        'LVIDs': [4.8, 3.0, 5.2, 3.5],
        'BMI': [28.5, 22.1, 30.2, 24.5],
        'LAV': [45, 30, 50, 35],
        'Wall_Subendocardial': [1, 0, 1, 0],  # Required by the model
        'LDLc': [140, 100, 150, 110],
        'Age': [65, 45, 70, 55],
        'ECG_T_inversion': [1, 0, 1, 0],  # Required by the model
        'ICT': [110, 80, 120, 90],
        'RBS': [180, 110, 200, 120],
        'EA': [0.8, 1.5, 0.7, 1.2],
        'Chest_pain': [1, 0, 1, 0],

        # Additional features (not required by the model but included for completeness)
        'Sex': [1, 0, 1, 0],
        'Creatinine': [1.2, 0.8, 1.5, 0.9],
        'LVIDd': [6.2, 4.5, 6.5, 5.0],
        'LVEF': [35, 60, 30, 55],
        'IRT': [95, 70, 100, 75],
        'RR': [22, 16, 24, 18],
        'TC': [220, 180, 240, 190],
        'HTN': [1, 0, 1, 1],
        'DM': [1, 0, 1, 0],
        'Smoker': [1, 0, 0, 1],
        'DL': [1, 0, 1, 0],
        'CXR': [1, 0, 1, 0],
        'RWMA': [1, 0, 1, 0],
        'MI': [1, 0, 1, 0],
        'Thrombolysis': [1, 0, 0, 0],
        'MR': [1, 0, 1, 0],

        # Expected target (HF = heart failure, No HF = no heart failure)
        'HF': ['HF', 'No HF', 'HF', 'No HF']
    }

    # Create DataFrame
    df = pd.DataFrame(sample_data)

    # Save sample data to CSV
    sample_file_path = "sample_data.csv"
    df.to_csv(sample_file_path, index=False)
    logging.info(f"Sample data saved to {sample_file_path}")

    return df, sample_file_path

def predict_sample():
    """Make predictions on a sample dataset"""
    try:
        logging.info("Starting sample prediction")

        # Create a sample dataset
        _, sample_file_path = create_sample_data()

        # Initialize prediction pipeline
        prediction_pipeline = PredictionPipeline()

        # Make predictions
        prediction_artifact = prediction_pipeline.run_batch_prediction(sample_file_path)

        # Load and display predictions
        try:
            predictions = pd.read_csv(prediction_artifact.prediction_file_path)
            # Check if the expected columns are present
            if 'Probability' in predictions.columns and 'Prediction' in predictions.columns:
                logging.info(f"Predictions:\n{predictions[['Probability', 'Prediction']].head()}")
            else:
                # Display whatever columns are available
                logging.info(f"Predictions (columns available: {predictions.columns.tolist()}):\n{predictions.head()}")
        except Exception as e:
            # If it's not a CSV file, just read the content as text
            with open(prediction_artifact.prediction_file_path, 'r') as f:
                content = f.read()
            logging.info(f"Prediction output:\n{content}")

        return prediction_artifact

    except Exception as e:
        logging.error(f"Error in sample prediction: {e}")
        raise e

if __name__ == "__main__":
    try:
        # Train the model
        train_model()

        # Make predictions
        predict_sample()

        logging.info("Demo completed successfully")

    except Exception as e:
        logging.error(f"Error in demo: {e}")
        raise e
