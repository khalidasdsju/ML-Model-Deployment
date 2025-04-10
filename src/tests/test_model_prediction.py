import os
import yaml
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_prediction_params():
    """Load prediction parameters from YAML file"""
    try:
        params_file = os.path.join("config", "prediction_params.yaml")
        with open(params_file, 'r') as f:
            params = yaml.safe_load(f)
        logging.info(f"Loaded prediction parameters from {params_file}")
        return params
    except Exception as e:
        logging.error(f"Error loading prediction parameters: {e}")
        raise e

def create_sample_data(params):
    """Create a sample dataset for testing based on prediction parameters"""
    # Get the feature lists from parameters
    categorical_features = params['preprocessing']['categorical_columns']
    numerical_features = params['preprocessing']['numerical_columns']
    
    # Create sample data with appropriate features
    sample_data = {}
    
    # Add categorical features (using binary values for simplicity)
    for feature in categorical_features:
        sample_data[feature] = [1, 0, 1, 0]
    
    # Add numerical features with appropriate ranges
    sample_data['Age'] = [65, 45, 70, 55]
    sample_data['BMI'] = [28.5, 22.1, 30.2, 24.5]
    sample_data['HR'] = [95, 75, 110, 80]
    sample_data['RBS'] = [180, 110, 200, 120]
    sample_data['FS'] = [25, 40, 20, 35]
    sample_data['LVIDs'] = [4.8, 3.0, 5.2, 3.5]
    sample_data['LAV'] = [45, 30, 50, 35]
    sample_data['ICT'] = [110, 80, 120, 90]
    sample_data['EA'] = [0.8, 1.5, 0.7, 1.2]
    sample_data['DT'] = [160, 220, 150, 200]
    sample_data['LDLc'] = [140, 100, 150, 110]
    sample_data['BNP'] = [800, 50, 1200, 100]
    
    # Add target variable (for evaluation)
    sample_data['HF'] = [1, 0, 1, 0]
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Save sample data to CSV
    sample_file_path = "sample_data_for_prediction.csv"
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
            logging.error(f"Model file {model_path} not found")
            return None
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise e

def make_predictions(model, data, params):
    """Make predictions using the model and parameters"""
    try:
        # Separate features and target
        X = data.drop('HF', axis=1) if 'HF' in data.columns else data
        y = data['HF'] if 'HF' in data.columns else None
        
        # Make predictions
        y_prob = model.predict_proba(X)[:, 1]
        
        # Apply threshold from parameters
        threshold = params.get('prediction_threshold', 0.5)
        y_pred = (y_prob >= threshold).astype(int)
        
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
        predictions.to_csv("model_predictions.csv", index=False)
        logging.info("Predictions saved to model_predictions.csv")
        
        # Evaluate if actual values are available
        if y is not None:
            accuracy = accuracy_score(y, y_pred)
            try:
                roc_auc = roc_auc_score(y, y_prob)
            except:
                roc_auc = 0.0
                logging.warning("Could not calculate ROC AUC score")
                
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"ROC AUC: {roc_auc:.4f}")
            logging.info(f"Classification Report:\n{classification_report(y, y_pred)}")
        
        return predictions
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        raise e

def analyze_feature_importance(model, feature_names):
    """Analyze and display feature importance"""
    try:
        # Get feature importance
        importance = model.feature_importances_
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Save to CSV
        feature_importance.to_csv("feature_importance.csv", index=False)
        
        # Display top features
        logging.info("Top 5 most important features:")
        for i, row in feature_importance.head(5).iterrows():
            logging.info(f"{row['Feature']}: {row['Importance']:.4f}")
            
        return feature_importance
    except Exception as e:
        logging.error(f"Error analyzing feature importance: {e}")
        raise e

def main():
    """Main function to test the model prediction"""
    try:
        # Load prediction parameters
        params = load_prediction_params()
        
        # Create sample data
        data, _ = create_sample_data(params)
        
        # Load model
        model = load_model()
        if model is None:
            return
        
        # Make predictions
        predictions = make_predictions(model, data, params)
        
        # Analyze feature importance
        feature_importance = analyze_feature_importance(model, data.drop('HF', axis=1).columns)
        
        # Display predictions
        logging.info(f"Predictions:\n{predictions[['Probability', 'Prediction', 'Actual']].head()}")
        
        logging.info("Model prediction testing completed successfully")
        
    except Exception as e:
        logging.error(f"Error in model prediction testing: {e}")
        raise e

if __name__ == "__main__":
    main()
