import os
from HF.logger import logging

def get_report_path(filename):
    """
    Get the full path to a file in the reports directory.
    
    Parameters:
    -----------
    filename : str
        The filename (without path)
    
    Returns:
    --------
    str
        The full path to the file in the reports directory
    """
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    return os.path.join(reports_dir, filename)

def save_report(content, filename):
    """
    Save content to a text file in the reports directory.
    
    Parameters:
    -----------
    content : str
        The content to save
    filename : str
        The filename (without path)
    
    Returns:
    --------
    str
        The full path to the saved file
    """
    try:
        # Create reports directory if it doesn't exist
        reports_dir = 'reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        # Construct the full path
        filepath = os.path.join(reports_dir, filename)
        
        # Save the content
        with open(filepath, 'w') as f:
            f.write(content)
        
        logging.info(f"Report saved to {filepath}")
        
        return filepath
    except Exception as e:
        logging.error(f"Error saving report: {e}")
        # Save to current directory as fallback
        filepath = filename
        with open(filepath, 'w') as f:
            f.write(content)
        logging.warning(f"Report saved to current directory: {filepath}")
        return filepath

def read_report(filename):
    """
    Read content from a text file in the reports directory.
    
    Parameters:
    -----------
    filename : str
        The filename (without path)
    
    Returns:
    --------
    str
        The content of the file
    """
    try:
        # Construct the full path
        filepath = get_report_path(filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            logging.warning(f"File {filepath} not found. Trying current directory.")
            filepath = filename
        
        # Read the content
        with open(filepath, 'r') as f:
            content = f.read()
        
        logging.info(f"Report read from {filepath}")
        
        return content
    except Exception as e:
        logging.error(f"Error reading report: {e}")
        raise e

def append_to_report(content, filename):
    """
    Append content to a text file in the reports directory.
    
    Parameters:
    -----------
    content : str
        The content to append
    filename : str
        The filename (without path)
    
    Returns:
    --------
    str
        The full path to the file
    """
    try:
        # Construct the full path
        filepath = get_report_path(filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            logging.warning(f"File {filepath} not found. Creating new file.")
            return save_report(content, filename)
        
        # Append the content
        with open(filepath, 'a') as f:
            f.write(content)
        
        logging.info(f"Content appended to {filepath}")
        
        return filepath
    except Exception as e:
        logging.error(f"Error appending to report: {e}")
        raise e
