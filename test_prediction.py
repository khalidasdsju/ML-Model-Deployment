import os
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def create_sample_data():
    """Create a sample dataset for testing"""
    # Create sample data with the exact features the model was trained on
    sample_data = {
        'FS': [25, 40, 20, 35],
        'DT': [160, 220, 150, 200],
        'NYHA': [3, 1, 4, 2],
        'HR': [95, 75, 110, 80],
        'BNP': [800, 50, 1200, 100],
        'LVIDs': [4.8, 3.0, 5.2, 3.5],
        'BMI': [28.5, 22.1, 30.2, 24.5],
        'LAV': [45, 30, 50, 35],
        'Wall_Subendocardial': [1, 0, 1, 0],  # Assuming this is a binary feature
        'LDLc': [140, 100, 150, 110],
        'Age': [65, 45, 70, 55],
        'ECG_T_inversion': [1, 0, 1, 0],  # Assuming this is a binary feature
        'ICT': [110, 80, 120, 90],
        'RBS': [180, 110, 200, 120],  # Random Blood Sugar
        'EA': [0.8, 1.5, 0.7, 1.2],
        'Chest_pain': [1, 0, 1, 0],
        # Expected target (1 = heart failure, 0 = no heart failure)
        'HF': [1, 0, 1, 0]
    }

    # Create DataFrame
    df = pd.DataFrame(sample_data)

    # Save sample data to CSV
    sample_file_path = "sample_data.csv"
    df.to_csv(sample_file_path, index=False)
    logging.info(f"Sample data saved to {sample_file_path}")

    return df, sample_file_path

def load_model():
    """Load the pre-trained XGBoost model"""
    try:
        # Try to load the model from the best_xgboost_model.pkl file
        model_path = "best_xgboost_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logging.info(f"Loaded model from {model_path}")
            return model
        else:
            # If the model doesn't exist, create a new XGBoost model with default parameters
            from xgboost import XGBClassifier
            logging.warning(f"Model file {model_path} not found. Creating a new XGBoost model.")
            model = XGBClassifier(
                learning_rate=0.03,
                n_estimators=250,
                max_depth=12,
                min_child_weight=4,
                gamma=0.1,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise e

def make_predictions(model, data):
    """Make predictions using the model"""
    try:
        # Separate features and target
        X = data.drop('HF', axis=1) if 'HF' in data.columns else data
        y = data['HF'] if 'HF' in data.columns else None

        # Make predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # Create a DataFrame with predictions
        predictions = pd.DataFrame({
            'Probability': y_prob,
            'Prediction': y_pred
        })

        # Add original data
        for col in X.columns:
            predictions[col] = X[col].values

        # Add actual target if available
        if y is not None:
            predictions['Actual'] = y.values

        # Save predictions
        predictions.to_csv("predictions.csv", index=False)
        logging.info("Predictions saved to predictions.csv")

        # Evaluate if actual values are available
        if y is not None:
            accuracy = accuracy_score(y, y_pred)
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Classification Report:\n{classification_report(y, y_pred)}")

        return predictions
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        raise e

def main():
    """Main function to test the model"""
    try:
        # Create sample data
        data, _ = create_sample_data()

        # Load model
        model = load_model()

        # Train the model if it's a new model
        if not hasattr(model, 'feature_importances_'):
            X = data.drop('HF', axis=1)
            y = data['HF']
            model.fit(X, y)
            logging.info("Model trained on sample data")

        # Make predictions
        predictions = make_predictions(model, data)

        # Display predictions
        logging.info(f"Predictions:\n{predictions[['Probability', 'Prediction', 'Actual']].head()}")

        logging.info("Model testing completed successfully")

    except Exception as e:
        logging.error(f"Error in model testing: {e}")
        raise e

if __name__ == "__main__":
    main()
