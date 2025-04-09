# HF/components/model_pusher.py
import os
import sys
import shutil
from HF.logger import logging
from HF.exception import HFException
from HF.entity.config_entity import ModelPusherConfig
from HF.entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise HFException(f"Error in ModelPusher initialization: {e}", sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Initiating model pusher")

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.model_pusher_config.model_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.model_pusher_config.saved_model_path), exist_ok=True)

            # Copy the trained model to the model pusher directory
            shutil.copy(
                src=self.model_trainer_artifact.trained_model_file_path,
                dst=self.model_pusher_config.model_file_path
            )
            logging.info(f"Copied model to: {self.model_pusher_config.model_file_path}")

            # Copy the model to the saved models directory for deployment
            shutil.copy(
                src=self.model_trainer_artifact.trained_model_file_path,
                dst=self.model_pusher_config.saved_model_path
            )
            logging.info(f"Saved model for deployment at: {self.model_pusher_config.saved_model_path}")

            # Create a symbolic link to the latest model
            latest_dir = os.path.join(self.model_pusher_config.model_pusher_dir, "latest")
            os.makedirs(latest_dir, exist_ok=True)

            latest_model_path = os.path.join(
                latest_dir,
                os.path.basename(self.model_pusher_config.saved_model_path)
            )

            # If a latest model already exists, remove it
            if os.path.exists(latest_model_path):
                os.remove(latest_model_path)

            # Create a copy of the latest model
            shutil.copy(
                src=self.model_pusher_config.saved_model_path,
                dst=latest_model_path
            )
            logging.info(f"Created latest model at: {latest_model_path}")

            # Create model pusher artifact
            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=self.model_pusher_config.saved_model_path,
                model_file_path=self.model_pusher_config.model_file_path
            )

            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise HFException(f"Error in model pusher: {e}", sys)