# HF/pipline/training_pipeline.py
import os
import sys
from HF.exception import HFException
from HF.logger import logging
from HF.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from HF.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from HF.components.data_ingestion import DataIngestion
from HF.components.data_validation import DataValidation
from HF.components.data_transformation import DataTransformation

class TrainPipeline:
    def __init__(self, training_pipeline_config):
        self.training_pipeline_config = training_pipeline_config
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")

            # Initialize DataIngestion
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Data ingestion completed")
            return data_ingestion_artifact

        except Exception as e:
            raise HFException(e, sys)

    def start_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")

            # Initialize DataValidation
            data_validation = DataValidation(self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Data validation completed")
            return data_validation_artifact

        except Exception as e:
            raise HFException(e, sys)

    def start_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")

            # Fetch the previous artifacts
            data_ingestion_artifact = self.start_data_ingestion()

            # Initialize DataTransformation
            data_transformation = DataTransformation(data_ingestion_artifact, self.data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            logging.info("Data transformation completed")
            return data_transformation_artifact

        except Exception as e:
            raise HFException(e, sys)

    def run(self):
        try:
            # Start the pipeline
            logging.info("Pipeline started")
            
            # Start the data ingestion step
            self.start_data_ingestion()

            # Start the data validation step
            self.start_data_validation()

            # Start the data transformation step
            self.start_data_transformation()

            logging.info("Pipeline completed successfully")

        except Exception as e:
            raise HFException(e, sys)
