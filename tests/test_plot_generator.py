import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.plot_generator import PlotGenerator

@pytest.fixture
def sample_df():
    np.random.seed(0)
    return pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.rand(100) * 100,
        'time': pd.date_range('2023-01-01', periods=100)
    })

@pytest.fixture
def plot_gen():
    return PlotGenerator()

def test_plot_box_single_column(plot_gen, sample_df, monkeypatch):
    # Patch plt.show to avoid GUI during test
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_gen.plot_box(sample_df, 'A')

def test_plot_box_multiple_columns(plot_gen, sample_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_gen.plot_box(sample_df, ['A', 'B'])

def test_plot_histogram_single_column(plot_gen, sample_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_gen.plot_histogram(sample_df, 'B', bins=20)

def test_plot_histogram_multiple_columns(plot_gen, sample_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_gen.plot_histogram(sample_df, ['A', 'B'], bins=10)

def test_plot_time_series_single_column(plot_gen, sample_df, monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_gen.plot_time_series(sample_df, 'A', time_column='time')

def test_plot_time_series_multiple_columns(plot_gen, sample_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_gen.plot_time_series(sample_df, ['A', 'B'], time_column='time')

def test_plot_box_invalid_column(plot_gen, sample_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    # Should not raise, but log error
    plot_gen.plot_box(sample_df, 'invalid_column')

def test_plot_box_grouped(plot_gen, sample_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    # Add a group column
    df = sample_df.copy()
    df['group'] = np.random.choice([0, 1], size=len(df))
    plot_gen.plot_box_grouped(df, ['A', 'B'], group_column='group')

def test_plot_time_series_grouped(plot_gen, sample_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    # Add a group column
    df = sample_df.copy()
    df['group'] = np.random.choice([0, 1], size=len(df))
    plot_gen.plot_time_series_grouped(df, ['A', 'B'], time_column='time', group_column='group')

def test_plot_correlation_heatmap(plot_gen, sample_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_gen.plot_correlation_heatmap(sample_df, ['A', 'B'])

def test_plot_scatter(plot_gen, sample_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_gen.plot_scatter(sample_df, x_col='A', y_col='B')