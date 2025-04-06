import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from HF.exception import HFException
from HF.logger import logging



class TargetValueMapping:
    def __init__(self):
        # Define the custom label-to-integer mapping
        self.mapping = {
            " No HF": 0,
            "HF": 1
        }

    def _asdict(self):
        """
        Returns the mapping dictionary.
        """
        return self.mapping

    def reverse_mapping(self):
        """
        Returns the inverse of the mapping dictionary, i.e., from int to label.
        """
        return {v: k for k, v in self.mapping.items()}