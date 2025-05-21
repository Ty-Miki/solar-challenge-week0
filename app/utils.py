import logging
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.data_loader import load_csv
from src.country_comparision_utils import ComparisionUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_combine_each_country_data(country_filename_pairs: dict[str, str]) -> pd.DataFrame:
    """
    Loads data for each country from the provided file paths, combines them into a single DataFrame, and handles errors.
    Args:
        country_filename_pairs (dict[str, str]): A dictionary mapping country names to their respective data file paths.
    Returns:
        pd.DataFrame or None: The combined DataFrame containing data from all valid countries, or None if no valid data is found.
    Logs:
        - The number of successfully loaded dataframes.
        - Errors if no valid data files are found or if the combined data is empty.
    Displays:
        - Streamlit error messages if no valid data files are found or if the combined data is empty.
    """

    dataframes = [df for df in map(load_csv, country_filename_pairs.values()) if df is not None]
    logging.info(f"Loaded {len(dataframes)} dataframes.")

    if not dataframes:
        logging.error("No valid data files found.")
        st.error("No valid data files found.")
        return None
    
    comparator = ComparisionUtils()
    combined_data = comparator.combine_country_data({country_name: df for country_name, df in zip(country_filename_pairs.keys(), dataframes)})
    if combined_data is None or combined_data.empty:
        logging.error("No valid data to display.")
        st.error("No valid data to display.")
        return None
    
    logging.info("Data combined successfully.")
    return combined_data

def boxplot_by_country(df: pd.DataFrame, group_col: str = "country") -> None:
    """
    Displays interactive country-wise boxplots for selected numeric columns in a pandas DataFrame using Streamlit.

    This function provides a Streamlit interface to:
    - Select numeric columns for plotting boxplots grouped by the 'Country' column.
    - Optionally customize the color palette of the plots.
    - Optionally display summary statistics (describe) grouped by country.
    - Handle empty DataFrames or missing 'Country' columns gracefully.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing at least a 'Country' column and one or more numeric columns.
        group_col (str): The column name to group by for the boxplots. Default is 'country'.

    Returns:
        None. The function renders plots and controls directly in the Streamlit app.
    """
    if df.empty:
        st.warning("DataFrame is empty.")
        return

    if group_col not in df.columns:
        st.error("'Country' column not found in the dataset.")
        return

    # Attempt to convert all columns (except 'Country') to numeric if they look like numbers
    for col in df.columns:
        if col != group_col and col != 'Timestamp' and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    with st.sidebar:
        st.subheader("🔧 Controls")
        selected_cols = st.multiselect("Select numeric columns to plot", numeric_cols, default=numeric_cols[:1])

        use_color = st.checkbox("Customize color palette", value=False)
        palette_option = None
        if use_color:
            palette_option = st.selectbox(
                "Choose color palette",
                options=["husl", "Set2", "pastel", "muted", "dark", "colorblind", None],
                index=0
            )

        if st.checkbox("Show summary statistics"):
            st.write(df[selected_cols + [group_col]].groupby(group_col).describe())

    if not selected_cols:
        st.warning("Please select at least one numeric column to plot.")
        return

    fig, axes = plt.subplots(len(selected_cols), 1, figsize=(8, 5 * len(selected_cols)))
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    for ax, col in zip(axes, selected_cols):
        try:
            sns.boxplot(
                x=group_col, 
                y=col,
                data=df,
                hue=group_col,
                palette=palette_option if use_color else None,
                dodge=False,
                ax=ax,
            )
            ax.set_title(f"{col} by {group_col}")
            ax.set_xlabel("Country")
            ax.set_ylabel(col)
        except Exception as e:
            st.error(f"Error plotting {col}: {e}")
            logging.exception(f"Failed to generate boxplot for {col}: {e}")


    st.pyplot(fig)

def write_summary_table(df: pd.DataFrame, group_col: str = "country") -> None:
    """
    Displays a summary table of the DataFrame grouped by the specified column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        group_col (str): The column name to group by for the summary table. Default is 'country'.

    Returns:
        None. The function renders the summary table directly in the Streamlit app.
    """
    if df.empty:
        st.warning("DataFrame is empty.")
        return

    if group_col not in df.columns:
        st.error("'Country' column not found in the dataset.")
        return

    summary = df.groupby(group_col)[['GHI', "DNI", 'DHI']].describe()
    st.write(summary)

def per_country_ghi_trends(filename: str, column: str = 'GHI', time_column: str = 'Timestamp'):

    df = load_csv(filename)
    df[time_column] = pd.to_datetime(df[time_column])

    fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharex=True)
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    for ax in axes:
        try:
            sns.lineplot(x=df[time_column], y=df[column], ax=ax)
            ax.set_title(f'Time Series of {column}')
            ax.set_ylabel(column)
            logging.info(f"Time Series for {column} created successfully.")
        except Exception as e:
            logging.error(f"Error generating time series plot for {column}: {e}")
    
    st.pyplot(fig)

