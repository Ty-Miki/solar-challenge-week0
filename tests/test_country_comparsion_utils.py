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

@pytest.fixture
def sample_combined_data():
    """Sample DataFrame with country, GHI, DNI, DHI columns."""
    return pd.DataFrame({
        "country": ["Germany", "Germany", "Spain", "Spain"],
        "GHI": [450, 460, 520, 530],
        "DNI": [320, 330, 380, 390],
        "DHI": [130, 140, 140, 150]
    })

def test_successful_summary_stats(sample_combined_data):
    """Test successful generation of summary stats."""
    utils = ComparisionUtils()
    summary = utils.generate_summary_stats(sample_combined_data)
    
    # Check structure
    assert isinstance(summary, pd.DataFrame)
    assert set(summary.columns) == {
        "country", "GHI_mean", "GHI_median", "GHI_std",
        "DNI_mean", "DNI_median", "DNI_std",
        "DHI_mean", "DHI_median", "DHI_std"
    }
    
    # Check values (Germany stats)
    germany = summary[summary["country"] == "Germany"].iloc[0]
    assert germany["GHI_mean"] == pytest.approx(455.0)
    assert germany["DNI_std"] == pytest.approx(7.07, abs=0.01)

def test_custom_metrics_and_stats(sample_combined_data):
    """Test with custom metrics and stats."""
    utils = ComparisionUtils()
    summary = utils.generate_summary_stats(
        sample_combined_data,
        metrics=["GHI", "DHI"],
        stats=["mean", "max"]
    )
    assert set(summary.columns) == {
        "country", "GHI_mean", "GHI_max", "DHI_mean", "DHI_max"
    }

def test_missing_metrics(caplog, sample_combined_data):
    """Test logging when metrics are missing (no exception raised)."""
    utils = ComparisionUtils()
    with caplog.at_level(logging.ERROR):
        result = utils.generate_summary_stats(
            sample_combined_data, 
            metrics=["GHI", "INVALID"]
        )
    
    # Check that the method logged an error and returned None (or a default DataFrame)
    assert "Metrics not found in DataFrame: ['INVALID']" in caplog.text
    assert result is None  # Or assert isinstance(result, pd.DataFrame) if you return a default

def test_empty_dataframe(caplog):
    """Test logging when DataFrame is empty (no exception raised)."""
    utils = ComparisionUtils()
    with caplog.at_level(logging.ERROR):
        result = utils.generate_summary_stats(pd.DataFrame())
    
    assert "DataFrame is empty" in caplog.text
    assert result is None  # Or adjust based on your method's return

def test_invalid_country_column(caplog, sample_combined_data):
    """Test logging when country column is missing (no exception raised)."""
    utils = ComparisionUtils()
    with caplog.at_level(logging.ERROR):
        result = utils.generate_summary_stats(
            sample_combined_data, 
            country_col="missing_col"
        )
    
    assert "Country column 'missing_col' not found in DataFrame" in caplog.text
    assert result is None

def test_non_dataframe_input(caplog):
    """Test logging when input is not a DataFrame (no exception raised)."""
    utils = ComparisionUtils()
    with caplog.at_level(logging.ERROR):
        result = utils.generate_summary_stats({"not_a_df": 123})
    
    assert "Input must be a pandas DataFrame" in caplog.text
    assert result is None
