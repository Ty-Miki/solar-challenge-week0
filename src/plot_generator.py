import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import logging
import os
from typing import Union, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PlotGenerator:
    """
    A utility class for generating common data visualizations using pandas DataFrames and seaborn/matplotlib.
    Methods
    -------
    
    methods:
        plot_box(df: pd.DataFrame, columns: Union[str, List[str]]):
        plot_histogram(df: pd.DataFrame, columns: Union[str, List[str]], bins=30):
        plot_time_series(df: pd.DataFrame, columns: Union[str, List[str]], time_column: str):
    """
   
    def __init__(self):
        logging.info("PlotGenerator initialized successfully.")

    def plot_box(self, df: pd.DataFrame, columns: Union[str, List[str]]):
        """
        Generates boxplots for the specified columns in the given DataFrame.
        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data.
            columns (Union[str, List[str]]): Column name or list of column names to plot.
        """
        columns = [columns] if isinstance(columns, str) else columns
        n = len(columns)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        for ax, col in zip(axes, columns):
            try:
                sns.boxplot(y=df[col], ax=ax)
                ax.set_title(f'Boxplot of {col}')
                ax.set_ylabel(col)
                logging.info(f"Boxplot for {col} created successfully.")
            except Exception as e:
                logging.error(f"Error generating boxplot for {col}: {e}")
        
        plt.tight_layout()
        plt.show()

    def plot_histogram(self, df: pd.DataFrame, columns: Union[str, List[str]], bins=30):
        """
        Generates histograms (with KDE) for the specified columns in the given DataFrame.
        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data.
            columns (Union[str, List[str]]): Column name or list of column names to plot.
            bins (int, optional): Number of bins for the histogram. Default is 30.
        """
        columns = [columns] if isinstance(columns, str) else columns
        n = len(columns)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        for ax, col in zip(axes, columns):
            try:
                sns.histplot(df[col], bins=bins, kde=True, ax=ax)
                ax.set_title(f'Histogram of {col}')
                ax.set_xlabel(col)
                logging.info(f"Histogram for {col} created successfully.")
            except Exception as e:
                logging.error(f"Error generating histogram for {col}: {e}")
        
        plt.tight_layout()
        plt.show()

    def plot_time_series(self, df: pd.DataFrame, columns: Union[str, List[str]], time_column: str):
        """
        Plots time series line plots for the specified columns against a time column.
        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data.
            columns (Union[str, List[str]]): Column name or list of column names to plot.
            time_column (str): Name of the column representing time or sequence.
        """
        columns = [columns] if isinstance(columns, str) else columns
        n = len(columns)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        for ax, col in zip(axes, columns):
            try:
                sns.lineplot(x=df[time_column], y=df[col], ax=ax)
                ax.set_title(f'Time Series of {col}')
                ax.set_ylabel(col)
                logging.info(f"Time Series for {col} created successfully.")
            except Exception as e:
                logging.error(f"Error generating time series plot for {col}: {e}")
        
        plt.xlabel(time_column)
        plt.tight_layout()
        plt.show()

    def plot_box_grouped(self, df: pd.DataFrame, columns: Union[str, List[str]], group_column: str):
        """
        Generates grouped boxplots for the specified columns by a categorical group.
        Parameters:
            df (pd.DataFrame): Input data.
            columns (str or list): Columns to plot.
            group_column (str): Categorical column to group by (e.g., 'Cleaning').
        """
        columns = [columns] if isinstance(columns, str) else columns
        n = len(columns)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        for ax, col in zip(axes, columns):
            try:
                sns.boxplot(x=df[group_column], y=df[col], ax=ax)
                ax.set_title(f'{col} by {group_column}')
                logging.info(f"Grouped boxplot for {col} created successfully.")
            except Exception as e:
                logging.error(f"Error generating grouped boxplot for {col}: {e}")
        
        plt.tight_layout()
        plt.show()

    def plot_time_series_grouped(self, df: pd.DataFrame, columns: Union[str, List[str]], time_column: str, group_column: str):
        """
        Plots time series lines for each group in group_column for the specified columns.
        Parameters:
            df (pd.DataFrame): Input data.
            columns (str or list): Columns to plot.
            time_column (str): Time column (e.g., 'Timestamp').
            group_column (str): Grouping column (e.g., 'Cleaning').
        """
        columns = [columns] if isinstance(columns, str) else columns
        n = len(columns)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=True)
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        for ax, col in zip(axes, columns):
            try:
                sns.lineplot(data=df, x=time_column, y=col, hue=group_column, ax=ax)
                ax.set_title(f'{col} over Time by {group_column}')
                logging.info(f"Grouped time series plot for {col} created successfully.")
            except Exception as e:
                logging.error(f"Error generating grouped time series plot for {col}: {e}")
        
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, df: pd.DataFrame, columns: List[str], annot=True, cmap="coolwarm"):
        """
        Plots a heatmap showing pairwise correlation coefficients between the specified columns.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data.
            columns (List[str]): List of column names to include in the correlation matrix.
            annot (bool): Whether to annotate the heatmap with correlation values. Default is True.
            cmap (str): The color map style to use. Default is "coolwarm".
        """
        try:
            corr_matrix = df[columns].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=".2f", square=True, linewidths=0.5)
            plt.title("Correlation Heatmap")
            logging.info("Correlation heatmap created successfully.")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error generating correlation heatmap: {e}")


