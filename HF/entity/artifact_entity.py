# HF/entity/artifact_entity.py
class DataIngestionArtifact:
    def __init__(self, trained_file_path: str, test_file_path: str):
        self.trained_file_path = trained_file_path
        self.test_file_path = test_file_path

class DataValidationArtifact:
    def __init__(self, drift_report_file_path: str, validation_status: bool = True, message: str = ""):
        self.drift_report_file_path = drift_report_file_path
        self.validation_status = validation_status
        self.message = message

class DataTransformationArtifact:
    def __init__(self, transformed_train_file_path: str, transformed_test_file_path: str, transformed_object_file_path: str):
        self.transformed_train_file_path = transformed_train_file_path
        self.transformed_test_file_path = transformed_test_file_path
        self.transformed_object_file_path = transformed_object_file_path

class ModelTrainerArtifact:
    def __init__(self, trained_model_file_path: str, model_accuracy: float):
        self.trained_model_file_path = trained_model_file_path
        self.model_accuracy = model_accuracy

class ModelPusherArtifact:
    def __init__(self, saved_model_path: str, model_file_path: str):
        self.saved_model_path = saved_model_path
        self.model_file_path = model_file_path

class ModelPredictionArtifact:
    def __init__(self, prediction_file_path: str, model_accuracy: float):
        self.prediction_file_path = prediction_file_path
        self.model_accuracy = model_accuracy
