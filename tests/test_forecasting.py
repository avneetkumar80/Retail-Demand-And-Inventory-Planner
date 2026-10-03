import pytest
import pandas as pd
import numpy as np
from src.forecasting.baselines import (
    forecast_same_day_last_week,
    forecast_moving_average,
    forecast_exponential_smoothing,
    forecast_croston
)
from src.forecasting.models import MLForecaster

@pytest.fixture
def sample_series_df():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    sales = [10, 12, 15, 8, 9, 20, 25] * 4 + [14, 18]
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "product_id": "PRD-001",
        "units_sold": sales,
        "is_promotion": 0,
        "day_of_week": [d.weekday() for d in dates],
        "is_holiday": 0
    })

def test_baselines_outputs(sample_series_df):
    f_naive = forecast_same_day_last_week(sample_series_df, horizon_days=7)
    assert len(f_naive) == 7
    assert np.all(f_naive >= 0)

    f_ma = forecast_moving_average(sample_series_df, horizon_days=7, window=7)
    assert len(f_ma) == 7
    assert np.all(f_ma >= 0)

    f_es = forecast_exponential_smoothing(sample_series_df, horizon_days=7)
    assert len(f_es) == 7
    assert np.all(f_es >= 0)

    f_croston = forecast_croston(sample_series_df, horizon_days=7)
    assert len(f_croston) == 7
    assert np.all(f_croston >= 0)

def test_ml_forecaster_fit_predict(sample_series_df):
    forecaster = MLForecaster(model_type="ridge")
    forecaster.fit(sample_series_df)
    
    preds, importances = forecaster.predict_product_horizon(sample_series_df, horizon_days=14)
    assert len(preds) == 14
    assert np.all(preds >= 0)
    assert isinstance(importances, dict)

def test_ml_forecaster_short_series():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    short_df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "product_id": "PRD-SHORT",
        "units_sold": [5, 6, 4, 7, 5, 6, 8, 4, 5, 6],
        "is_promotion": 0,
        "day_of_week": [d.weekday() for d in dates],
        "is_holiday": 0
    })
    forecaster = MLForecaster(model_type="ridge")
    forecaster.fit(short_df)
    preds, importances = forecaster.predict_product_horizon(short_df, horizon_days=7)
    assert len(preds) == 7
    assert np.all(preds >= 0)
