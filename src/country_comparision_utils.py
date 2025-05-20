import logging
from typing import Dict, List, Optional
import pandas as pd
from scipy.stats import f_oneway

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
    
    def generate_summary_stats(self, df: pd.DataFrame, 
                               metrics: List[str] = ["GHI", "DNI", "DHI"], 
                               stats: List[str] = ["mean", "median", "std"],
                               country_col: str = "country") -> pd.DataFrame:
        """
        Generates a summary table comparing statistical measures across countries.
        Logs errors gracefully and validates inputs.

        Args:
            df: Combined DataFrame (from `combine_country_data`).
            metrics: Columns to analyze (default: ["GHI", "DNI", "DHI"]).
            stats: Aggregation functions (default: ["mean", "median", "std"]).
            country_col: Name of the country column (default: "country").

        Returns:
            pd.DataFrame: Summary table with countries as rows and metrics/stats as columns.

        Raises:
            ValueError: If required columns are missing or inputs are invalid.
        """
        try:
            # Validate inputs
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            if df.empty:
                raise ValueError("DataFrame is empty")
            if country_col not in df.columns:
                raise ValueError(f"Country column '{country_col}' not found in DataFrame")
            
            # Check if specified metrics exist
            missing_metrics = [m for m in metrics if m not in df.columns]
            if missing_metrics:
                raise ValueError(f"Metrics not found in DataFrame: {missing_metrics}")

            # Log start of operation
            logging.info(
                f"Generating summary stats for metrics: {metrics} "
                f"using aggregations: {stats}"
            )

            # Compute stats
            summary = (
                df.groupby(country_col)[metrics]
                .agg(stats)
                .round(2)
            )
            
            # Flatten multi-index columns
            summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
            summary = summary.reset_index()

            logging.info("Successfully generated summary stats")
            return summary

        except ValueError as e:
            logging.error(f"Input validation error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error generating summary stats: {e}")

    def perform_anova(self, df: pd.DataFrame, group_col: str, value_col: str, interpret: bool = False):
        """
        Performs one-way ANOVA to test if means of 'value_col' differ significantly between groups in 'group_col'.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            group_col (str): The name of the column with categorical groups (e.g., 'Country').
            value_col (str): The name of the numeric column to compare (e.g., 'GHI').

        Returns:
            tuple: F-statistic and p-value of the ANOVA test.
        """
        try:
            groups = [group[value_col].dropna() for _, group in df.groupby(group_col)]
            f_stat, p_value = f_oneway(*groups)
            logging.info(f"ANOVA test performed on '{value_col}' grouped by '{group_col}'.")

            if interpret:
                if p_value < 0.05:
                    print(f"Significant differences detected in '{value_col}' between groups in '{group_col}'.")
                else:
                    print(f"No significant differences found in '{value_col}' between groups in '{group_col}'.")
                    
            return f_stat, p_value
        except Exception as e:
            logging.error(f"ANOVA test failed: {e}")
            return None, None