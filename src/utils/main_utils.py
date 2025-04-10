# HF/utils/main_utils.py
import yaml
import numpy as np
import joblib
from HF.exception import HFException
def read_yaml_file(file_path: str) -> dict:
    """
    Reads the YAML file and returns its content as a dictionary
    """
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise HFException(f"Error reading YAML file at {file_path}: {e}", sys)

def write_yaml_file(file_path: str, content: dict) -> None:
    """
    Writes the content into the specified YAML file
    """
    try:
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise HFException(f"Error writing to YAML file at {file_path}: {e}", sys)

def save_numpy_array_data(file_path: str, data: np.ndarray) -> None:
    """
    Saves numpy array data to the specified path
    """
    try:
        np.save(file_path, data)
    except Exception as e:
        raise HFException(f"Error saving numpy array to {file_path}: {e}", sys)

def save_object(file_path: str, obj: object) -> None:
    """
    Saves the given object as a pickle file
    """
    try:
        joblib.dump(obj, file_path)
    except Exception as e:
        raise HFException(f"Error saving object to {file_path}: {e}", sys)

def create_directories(directories: list):
    """
    Create directories if they do not exist.
    :param directories: list of directory paths
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory {directory} created or already exists.")
