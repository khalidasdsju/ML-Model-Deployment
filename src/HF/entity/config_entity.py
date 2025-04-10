import os
from datetime import datetime
from dataclasses import dataclass, field
from HF.constants import *
from HF.exception import HFException

# Base path for artifact directory
BASE_ARTIFACT_DIR = "/Users/khalid/Desktop/ML-Model-Deployment/artifact"  # Modify this as per your requirement

# Creating a dynamic timestamp-based folder name
def get_timestamp():
    return datetime.now().strftime('%d_%m_%Y_%H_%M_%S')

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(BASE_ARTIFACT_DIR, f"{PIPELINE_NAME}_{get_timestamp()}")
    # artifact directory where we store all the pipeline results

# Initialize training pipeline config
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME
    ingested_data_dir: str = os.path.join(data_ingestion_dir, 'data_ingestion')  # Directory to store the ingested data
    # We'll use MongoDB instead of a raw data file

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    drift_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR, DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
    # Directory and path for storing drift report files

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TRAIN_FILE_NAME.replace("csv", "npy"))
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TEST_FILE_NAME.replace("csv", "npy"))
    transformed_object_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR)
    # Paths for storing transformed data and transformation objects
    drop_columns: list = field(default_factory=lambda: ["StudyID"])  # Columns to drop during transformation
    num_features: list = field(default_factory=lambda: ["Age", "BMI", "HR", "RBS", "HbA1C", "Creatinine", "Na", "K", "Cl", "Hb", "TropI", "LVIDd", "FS", "LVIDs", "LVEF", "LAV", "ICT", "IRT", "EA", "DT", "MPI", "RR", "TC", "LDLc", "HDLc", "TG", "BNP"])
    or_columns: list = field(default_factory=lambda: ["Sex", "NYHA", "HTN", "DM", "Smoker", "DL", "BA", "CXR", "RWMA", "MI", "Thrombolysis", "MR", "Chest_pain"])
    oh_columns: list = field(default_factory=lambda: ["ECG", "ACS", "Wall"])

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME)
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
    # XGBoost model parameters
    model_params: dict = field(default_factory=lambda: {
        "learning_rate": 0.03,
        "n_estimators": 250,
        "max_depth": 12,
        "min_child_weight": 4,
        "gamma": 0.1,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42
    })

@dataclass
class ModelPusherConfig:
    model_pusher_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_PUSHER_DIR_NAME)
    model_file_path: str = os.path.join(model_pusher_dir, MODEL_TRAINER_TRAINED_MODEL_NAME)
    saved_model_path: str = os.path.join(MODEL_PUSHER_SAVED_MODEL_DIR, f"{get_timestamp()}", MODEL_TRAINER_TRAINED_MODEL_NAME)
    # Path to save the model for deployment

@dataclass
class ModelPredictionConfig:
    model_prediction_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_PREDICTION_DIR_NAME)
    prediction_file_path: str = os.path.join(model_prediction_dir, "predictions.csv")
    model_file_path: str = os.path.join(MODEL_PUSHER_SAVED_MODEL_DIR, "latest", MODEL_TRAINER_TRAINED_MODEL_NAME)
    params_file_path: str = os.path.join("config", MODEL_PREDICTION_PARAMETERS_FILE_NAME)
    # Parameters for model prediction
