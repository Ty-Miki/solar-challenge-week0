import pandas as pd
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_csv(filepath):
    """
    Load a CSV file and return a DataFrame.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame or None
    """
    if not os.path.exists(filepath):
        logging.error(f"File does not exist: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath, parse_dates=True)
        logging.info(f"Successfully loaded data from {filepath} with shape {df.shape}")
        return df
    except Exception as e:
        logging.exception(f"Failed to load data from {filepath}")
        return None
