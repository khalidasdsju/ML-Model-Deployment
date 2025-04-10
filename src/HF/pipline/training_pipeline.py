# HF/pipline/training_pipeline.py
import os
import sys
from HF.exception import HFException
from HF.logger import logging
from HF.entity.config_entity import (
    DataIngestionConfig, DataValidationConfig, DataTransformationConfig,
    ModelTrainerConfig, ModelPusherConfig
)
from HF.entity.artifact_entity import (
    DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact,
    ModelTrainerArtifact, ModelPusherArtifact
)
from HF.components.data_ingestion import DataIngestion
from HF.components.data_validation import DataValidation
from HF.components.data_transformation import DataTransformation
from HF.components.model_trainer import ModelTrainer
from HF.components.model_pusher import ModelPusher

class TrainPipeline:
    def __init__(self, training_pipeline_config):
        self.training_pipeline_config = training_pipeline_config
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_pusher_config = ModelPusherConfig()

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

            # Get data ingestion artifact first
            data_ingestion_artifact = self.start_data_ingestion()

            # Initialize DataValidation with both required arguments
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )
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

    def start_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model trainer")

            # Fetch the previous artifacts
            data_transformation_artifact = self.start_data_transformation()

            # Initialize ModelTrainer
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            logging.info("Model training completed")
            return model_trainer_artifact

        except Exception as e:
            raise HFException(e, sys)

    def start_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Starting model pusher")

            # Fetch the previous artifacts
            model_trainer_artifact = self.start_model_trainer()

            # Initialize ModelPusher
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_trainer_artifact=model_trainer_artifact
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()

            logging.info("Model pushing completed")
            return model_pusher_artifact

        except Exception as e:
            raise HFException(e, sys)

    def run(self):
        try:
            # Start the pipeline
            logging.info("Pipeline started")

            # Start the model pusher (which will trigger the entire pipeline)
            model_pusher_artifact = self.start_model_pusher()

            logging.info("Pipeline completed successfully")
            return model_pusher_artifact

        except Exception as e:
            raise HFException(e, sys)
