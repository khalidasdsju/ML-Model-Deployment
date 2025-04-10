# HF/components/model_pusher.py
import os
import sys
import shutil
from datetime import datetime
from HF.logger import logging
from HF.exception import HFException
from HF.entity.config_entity import ModelPusherConfig
from HF.entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact
from HF.utils.s3_utils import S3Utils

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_trainer_artifact: ModelTrainerArtifact,
                 aws_access_key_id=None, aws_secret_access_key=None, s3_bucket_name=None):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_trainer_artifact = model_trainer_artifact
            self.aws_access_key_id = aws_access_key_id
            self.aws_secret_access_key = aws_secret_access_key
            self.s3_bucket_name = s3_bucket_name
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

            # Upload model to S3 if credentials are provided
            s3_uri = None
            if self.aws_access_key_id and self.aws_secret_access_key and self.s3_bucket_name:
                try:
                    logging.info(f"Uploading model to S3 bucket: {self.s3_bucket_name}")

                    # Initialize S3 client
                    s3_utils = S3Utils(
                        bucket_name=self.s3_bucket_name,
                        aws_access_key_id=self.aws_access_key_id,
                        aws_secret_access_key=self.aws_secret_access_key
                    )

                    # Create a timestamp for versioning
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Upload the model
                    s3_key = f"models/{timestamp}/model.pkl"
                    s3_uri = s3_utils.upload_file(latest_model_path, s3_key)
                    logging.info(f"Model uploaded to S3: {s3_uri}")

                    # Upload model configuration if available
                    config_path = os.path.join("config", "prediction_params.yaml")
                    if os.path.exists(config_path):
                        s3_key = f"config/{timestamp}/prediction_params.yaml"
                        config_uri = s3_utils.upload_file(config_path, s3_key)
                        logging.info(f"Model configuration uploaded to S3: {config_uri}")
                except Exception as s3_error:
                    logging.warning(f"Error uploading model to S3: {s3_error}")
                    logging.warning("Continuing with local model deployment")

            # Create model pusher artifact
            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=self.model_pusher_config.saved_model_path,
                model_file_path=self.model_pusher_config.model_file_path
            )

            logging.info(f"Model pusher artifact: {model_pusher_artifact}")

            # Log S3 URI if available
            if s3_uri:
                logging.info(f"Model S3 URI: {s3_uri}")
                print(f"\nModel deployed to S3: {s3_uri}")

            return model_pusher_artifact

        except Exception as e:
            raise HFException(f"Error in model pusher: {e}", sys)