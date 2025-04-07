# HF/components/data_validation.py
import json
import pandas as pd
from HF.exception import HFException
from HF.logger import logging
from HF.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from HF.entity.config_entity import DataValidationConfig
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise HFException(f"Error in DataValidation initialization: {e}", sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Validation status of columns: {status}")
            return status
        except Exception as e:
            raise HFException(f"Error validating columns: {e}", sys)

    def detect_data_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_df, current_data=current_df)
            drift_status = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
            return drift_status
        except Exception as e:
            raise HFException(f"Error detecting data drift: {e}", sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            validation_status = self.validate_number_of_columns(train_df) and self.validate_number_of_columns(test_df)

            drift_status = self.detect_data_drift(train_df, test_df)
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            return DataValidationArtifact(
                validation_status=validation_status,
                message="Validation successful",
                drift_report_file_path=drift_report_file_path
            )
        except Exception as e:
            raise HFException(f"Error during data validation: {e}", sys)
