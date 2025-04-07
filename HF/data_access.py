# HF/data_access.py

class StudyData:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        # Implement loading logic (e.g., pandas read_csv)
        import pandas as pd
        return pd.read_csv(self.data_path)
