import pandas as pd

def validate_columns_exist(df: pd.DataFrame, columns: list) -> None:
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following columns are missing from the DataFrame: {missing_columns}")