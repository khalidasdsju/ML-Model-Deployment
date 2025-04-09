import os
import pandas as pd
from HF.logger import logging

def get_data_path(filename):
    """
    Get the full path to a file in the data directory.
    
    Parameters:
    -----------
    filename : str
        The filename (without path)
    
    Returns:
    --------
    str
        The full path to the file in the data directory
    """
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, filename)

def save_dataframe(df, filename, index=False):
    """
    Save a pandas DataFrame to the data directory.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to save
    filename : str
        The filename (without path)
    index : bool, default=False
        Whether to include the index in the CSV file
    
    Returns:
    --------
    str
        The full path to the saved file
    """
    try:
        # Create data directory if it doesn't exist
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)
        
        # Construct the full path
        filepath = os.path.join(data_dir, filename)
        
        # Save the DataFrame
        df.to_csv(filepath, index=index)
        logging.info(f"DataFrame saved to {filepath}")
        
        return filepath
    except Exception as e:
        logging.error(f"Error saving DataFrame: {e}")
        # Save to current directory as fallback
        filepath = filename
        df.to_csv(filepath, index=index)
        logging.warning(f"DataFrame saved to current directory: {filepath}")
        return filepath

def load_dataframe(filename):
    """
    Load a pandas DataFrame from the data directory.
    
    Parameters:
    -----------
    filename : str
        The filename (without path)
    
    Returns:
    --------
    pandas.DataFrame
        The loaded DataFrame
    """
    try:
        # Construct the full path
        filepath = get_data_path(filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            logging.warning(f"File {filepath} not found. Trying current directory.")
            filepath = filename
        
        # Load the DataFrame
        df = pd.read_csv(filepath)
        logging.info(f"DataFrame loaded from {filepath}")
        
        return df
    except Exception as e:
        logging.error(f"Error loading DataFrame: {e}")
        raise e
