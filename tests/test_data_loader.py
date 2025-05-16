import os
import pandas as pd
import pytest
from src.data_loader import load_csv

# Sample test CSV content
TEST_CSV = "test_sample.csv"
DATA = """Timestamp,GHI,DNI,DHI
2025-01-01 00:00,0,0,0
2025-01-01 01:00,10,15,5
"""

@pytest.fixture(scope="module")
def setup_test_csv():
    # Create test CSV file
    with open(TEST_CSV, "w") as f:
        f.write(DATA)
    yield TEST_CSV
    # Cleanup after tests
    if os.path.exists(TEST_CSV):
        os.remove(TEST_CSV)

def test_load_csv_success(setup_test_csv):
    df = load_csv(setup_test_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 4)
    assert "GHI" in df.columns

def test_load_csv_file_not_found():
    df = load_csv("non_existent_file.csv")
    assert df is None