# HF/components/model_trainer.py
import os
import sys
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from HF.logger import logging
from HF.exception import HFException
from HF.entity.config_entity import ModelTrainerConfig
from HF.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise HFException(f"Error in ModelTrainer initialization: {e}", sys)

    def train_model(self, X_train, y_train):
        try:
            logging.info("Training XGBoost model with optimized parameters")

            # Create XGBoost model with optimized parameters
            xgb_model = XGBClassifier(
                **self.model_trainer_config.model_params,
                use_label_encoder=False,
                eval_metric='logloss'
            )

            # Train the model
            xgb_model.fit(X_train, y_train)

            return xgb_model

        except Exception as e:
            raise HFException(f"Error in model training: {e}", sys)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logging.info("Evaluating model performance")

            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            try:
                roc_auc = roc_auc_score(y_test, y_prob)
            except:
                roc_auc = 0.0
                logging.warning("Could not calculate ROC AUC score")

            f1 = f1_score(y_test, y_pred)

            # Log classification report
            logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

            # Check if model meets expected accuracy
            if accuracy < self.model_trainer_config.expected_accuracy:
                logging.warning(f"Model accuracy {accuracy} is below expected accuracy {self.model_trainer_config.expected_accuracy}")

            return accuracy, roc_auc, f1

        except Exception as e:
            raise HFException(f"Error in model evaluation: {e}", sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating model trainer")

            # Load transformed training and testing data
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info(f"Loading transformed training data from {train_file_path}")
            train_arr = np.load(train_file_path, allow_pickle=True)

            logging.info(f"Loading transformed testing data from {test_file_path}")
            test_arr = np.load(test_file_path, allow_pickle=True)

            # Split into features and target
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Convert target values to numeric if they are strings
            if isinstance(y_train[0], str):
                logging.info("Converting string target values to numeric")
                # Map 'HF' to 1 and 'No HF' to 0
                y_train = np.array([1 if y == 'HF' else 0 for y in y_train])
                y_test = np.array([1 if y == 'HF' else 0 for y in y_test])

            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            # Train the model
            model = self.train_model(X_train, y_train)

            # Evaluate the model
            accuracy, roc_auc, f1 = self.evaluate_model(model, X_test, y_test)

            logging.info(f"Model performance - Accuracy: {accuracy}, ROC AUC: {roc_auc}, F1 Score: {f1}")

            # Create directory for model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

            # Save the model
            joblib.dump(model, self.model_trainer_config.trained_model_file_path)
            logging.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")

            # Create model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                model_accuracy=accuracy
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise HFException(f"Error in model trainer: {e}", sys)