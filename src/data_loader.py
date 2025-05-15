import pandas as pd
import logging
import os

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Format logs
formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(message)s')
console_handler.setFormatter(formatter)

# Add handler only once to prevent duplicates in Jupyter
if not logger.hasHandlers():
    logger.addHandler(console_handler)

def load_csv(filepath):
    """
    Load a CSV file and return a DataFrame.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame or None
    """
    if not os.path.exists(filepath):
        logger.error(f"File does not exist: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath, parse_dates=True)
        logger.info(f"✅ Successfully loaded data from {filepath} with shape {df.shape}")
        return df
    except Exception as e:
        logger.exception(f"❌ Failed to load data from {filepath}")
        return None
