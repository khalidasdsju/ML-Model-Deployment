import os
import sys
import pandas as pd
from HF.exception import HFException
from HF.logger import logging
from HF.entity.config_entity import DataIngestionConfig
from HF.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split
from HF.data_access import StudyData

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise HFException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        try:
            logging.info("Exporting data from MongoDB")
            # Assuming StudyData handles the data fetching from MongoDB
            study_data = StudyData()
            dataframe = study_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            return dataframe
        except Exception as e:
            raise HFException(f"Error exporting data: {e}", sys)

    def create_directories(self):
        """Ensure all required directories exist"""
        directory_path = self.data_ingestion_config.feature_store_path
        if not os.path.exists(directory_path):
            logging.info(f"Creating directory: {directory_path}")
            os.makedirs(directory_path)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            logging.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False)
            logging.info(f"Train and test data saved at: {self.data_ingestion_config.training_file_path}, {self.data_ingestion_config.testing_file_path}")
        except Exception as e:
            raise HFException(f"Error splitting data: {e}", sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Initiating data ingestion")
            
            # Check if the raw data path exists
            if not os.path.exists(self.data_ingestion_config.raw_data_path):
                raise FileNotFoundError(f"Raw data file not found: {self.data_ingestion_config.raw_data_path}")

            # Ensure directories exist
            self.create_directories()

            df = pd.read_csv(self.data_ingestion_config.raw_data_path)
            self.split_data_as_train_test(df)
            
            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
        except Exception as e:
            raise HFException(f"Error in data ingestion: {e}", sys)
