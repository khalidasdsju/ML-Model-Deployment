# HF/entity/artifact_entity.py
class DataIngestionArtifact:
    def __init__(self, trained_file_path: str, test_file_path: str):
        self.trained_file_path = trained_file_path
        self.test_file_path = test_file_path

class DataValidationArtifact:
    def __init__(self, drift_report_file_path: str):
        self.drift_report_file_path = drift_report_file_path

class DataTransformationArtifact:
    def __init__(self, transformed_train_file_path: str, transformed_test_file_path: str, transformed_object_file_path: str):
        self.transformed_train_file_path = transformed_train_file_path
        self.transformed_test_file_path = transformed_test_file_path
        self.transformed_object_file_path = transformed_object_file_path
