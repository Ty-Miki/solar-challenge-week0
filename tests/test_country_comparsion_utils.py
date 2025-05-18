import pytest
import pandas as pd
import logging
from src.country_comparision_utils import ComparisionUtils

@pytest.fixture
def comparison_utils():
    return ComparisionUtils()

@pytest.fixture
def sample_country_data():
    return {
        "USA": pd.DataFrame({"GDP": [20, 21], "Population": [330, 331]}),
        "Canada": pd.DataFrame({"GDP": [1.6, 1.7], "Population": [38, 39]}),
    }

def test_combine_country_data_success(comparison_utils, sample_country_data):
    """Test successful combination of valid country DataFrames."""
    combined_df = comparison_utils.combine_country_data(sample_country_data)
    
    # Check if output is a DataFrame
    assert isinstance(combined_df, pd.DataFrame)
    
    # Check if 'country' column was added
    assert "country" in combined_df.columns
    
    # Check all countries are included
    assert set(combined_df["country"].unique()) == {"USA", "Canada"}
    
    # Check row count (2 rows per country)
    assert len(combined_df) == 4

def test_combine_single_country(comparison_utils):
    """Test with a single country."""
    data = {"Germany": pd.DataFrame({"GDP": [3.8]})}
    combined_df = comparison_utils.combine_country_data(data)
    
    assert len(combined_df) == 1
    assert combined_df["country"].iloc[0] == "Germany"

def test_non_string_country_key(comparison_utils):
    """Test error when country key is not a string."""
    with pytest.raises(ValueError, match="Country name must be a string"):
        comparison_utils.combine_country_data({123: pd.DataFrame()})

def test_non_dataframe_value(comparison_utils):
    """Test error when value is not a DataFrame."""
    with pytest.raises(ValueError, match="Value must be a pandas DataFrame"):
        comparison_utils.combine_country_data({"France": "not_a_df"})

def test_empty_dataframe(comparison_utils):
    """Test error when a DataFrame is empty."""
    with pytest.raises(ValueError, match="DataFrame for Italy is empty"):
        comparison_utils.combine_country_data({"Italy": pd.DataFrame()})

def test_mixed_column_structure(comparison_utils):
    """Test combining DataFrames with different columns."""
    data = {
        "Japan": pd.DataFrame({"GDP": [5.0]}),
        "Brazil": pd.DataFrame({"GDP": [1.5], "Area": [8.5]}),  # Extra column
    }
    combined_df = comparison_utils.combine_country_data(data)
    
    # Check missing column (Area) is filled with NaN for Japan
    assert "Area" in combined_df.columns
    assert combined_df["Area"].isna().sum() == 1  # Japan's row has NaN