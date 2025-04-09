# HF/components/data_transformation.py
import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.combine import SMOTEENN
from HF.exception import HFException
from HF.logger import logging
from HF.entity.config_entity import DataTransformationConfig  # Ensure this import exists
from HF.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from HF.utils.main_utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_transformation_config: DataTransformationConfig):
        try:
            logging.info("Data Transformation Initialization")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise HFException(e, sys)

    def get_data_transformer_object(self):
        try:
            logging.info("Creating data transformer object")

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, self.data_transformation_config.num_features),
                ("cat_pipeline", cat_pipeline, self.data_transformation_config.or_columns),
                ("oh_pipeline", cat_pipeline, self.data_transformation_config.oh_columns),
            ])

            return preprocessor

        except Exception as e:
            raise HFException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")

            # Create directories for transformed data
            transformed_train_dir = os.path.dirname(self.data_transformation_config.transformed_train_file_path)
            transformed_test_dir = os.path.dirname(self.data_transformation_config.transformed_test_file_path)
            transformed_object_dir = os.path.dirname(self.data_transformation_config.transformed_object_file_path)

            os.makedirs(transformed_train_dir, exist_ok=True)
            os.makedirs(transformed_test_dir, exist_ok=True)
            os.makedirs(transformed_object_dir, exist_ok=True)

            # Load the train and test datasets
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Strip spaces and standardize column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            # Drop 'StudyID' column
            train_df = train_df.drop(columns=self.data_transformation_config.drop_columns, axis=1)
            test_df = test_df.drop(columns=self.data_transformation_config.drop_columns, axis=1)

            # Rename the target column if necessary
            train_df.rename(columns={"HF": "HF"}, inplace=True)
            test_df.rename(columns={"HF": "HF"}, inplace=True)

            # Print available columns for debugging
            logging.info(f"Available columns in train_df: {train_df.columns.tolist()}")

            # Define target column name
            target_column = "HF"

            # Check if target column exists in both DataFrames
            if target_column not in train_df.columns or target_column not in test_df.columns:
                raise ValueError(f"Target column '{target_column}' is missing! Available columns: {train_df.columns.tolist()}")

            target_feature_train_df = train_df[target_column]
            target_feature_test_df = test_df[target_column]

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)

            # Fit and transform the features using the preprocessor
            preprocessor = self.get_data_transformer_object()
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Handle imbalanced data using SMOTEENN
            logging.info("Handling imbalanced data using SMOTEENN")
            smote_enn = SMOTEENN()
            input_feature_train_arr, target_feature_train_df = smote_enn.fit_resample(input_feature_train_arr, target_feature_train_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the transformed data and preprocessor object
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
            )

            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise HFException(e, sys)
