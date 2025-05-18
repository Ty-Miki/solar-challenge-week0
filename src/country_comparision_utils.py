import logging
from typing import Dict
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ComparisionUtils:
    """
    A utility class to be used for country comparison EDA.
    """

    def __init__(self):
        logging.info("Comparsion Utility Class Initialized.")

    def combine_country_data(self, country_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple country data into a single DataFrame, adding a 'country' column for each row.
        Args:
            country_data (dict): A dictionary with country names as keys and DataFrames as values.
        Returns:
            pd.DataFrame: Combined DataFrame with a 'country' column.
        """
        try:
            dfs = []
            for country, df in country_data.items():
                if not isinstance(country, str):
                    logging.error("Country name must be a string.")
                    raise ValueError("Country name must be a string.")
                if not isinstance(df, pd.DataFrame):
                    logging.error("Value must be a pandas DataFrame.")
                    raise ValueError("Value must be a pandas DataFrame.")
                if df.empty:
                    logging.error(f"DataFrame for {country} is empty.")
                    raise ValueError(f"DataFrame for {country} is empty.")
                df_copy = df.copy()
                df_copy["country"] = country
                dfs.append(df_copy)
            logging.info("All data is valid.")

            combined_data = pd.concat(dfs, ignore_index=True)
            logging.info("Data combined successfully.")
            return combined_data
        except Exception as e:
            logging.error(f"Error combining country data: {e}")
            raise