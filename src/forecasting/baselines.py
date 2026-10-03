import numpy as np
import pandas as pd
from typing import List, Dict, Union

def forecast_same_day_last_week(history_df: pd.DataFrame, horizon_days: int) -> np.ndarray:
    """
    Naive Baseline: Predicts demand equal to the same day of the previous week (t-7).
    history_df must be sorted by date ascending.
    """
    sales = history_df["units_sold"].values
    if len(sales) < 7:
        mean_val = np.mean(sales) if len(sales) > 0 else 0
        return np.full(horizon_days, max(0.0, float(mean_val)))
    
    preds = []
    recent_7 = list(sales[-7:])
    for step in range(horizon_days):
        val = recent_7[step % 7]
        preds.append(max(0.0, float(val)))
    return np.array(preds)

def forecast_moving_average(history_df: pd.DataFrame, horizon_days: int, window: int = 7) -> np.ndarray:
    """
    Moving Average Baseline: Predicts average daily sales over recent window (e.g. 7 or 14 days).
    """
    sales = history_df["units_sold"].values
    if len(sales) == 0:
        return np.zeros(horizon_days)
    
    actual_window = min(len(sales), window)
    mean_val = float(np.mean(sales[-actual_window:]))
    return np.full(horizon_days, max(0.0, round(mean_val, 2)))

def forecast_exponential_smoothing(history_df: pd.DataFrame, horizon_days: int, alpha: float = 0.3) -> np.ndarray:
    """
    Simple Exponential Smoothing (SES) Baseline.
    """
    sales = history_df["units_sold"].values
    if len(sales) == 0:
        return np.zeros(horizon_days)
    
    s = float(sales[0])
    for val in sales[1:]:
        s = alpha * val + (1 - alpha) * s
        
    return np.full(horizon_days, max(0.0, round(s, 2)))

def forecast_croston(history_df: pd.DataFrame, horizon_days: int, alpha: float = 0.1) -> np.ndarray:
    """
    Croston's Method for intermittent / slow-moving product demand.
    Separately models demand size when non-zero and interval between non-zero demand events.
    """
    sales = history_df["units_sold"].values
    nz_indices = np.where(sales > 0)[0]
    
    if len(nz_indices) == 0:
        return np.zeros(horizon_days)
    if len(nz_indices) == 1:
        avg_val = sales[nz_indices[0]] / max(1, len(sales))
        return np.full(horizon_days, max(0.0, float(avg_val)))

    # Compute intervals and demand sizes
    intervals = np.diff(np.concatenate(([-1], nz_indices)))
    demand_sizes = sales[nz_indices]

    a = float(demand_sizes[0])
    p = float(intervals[0])

    for i in range(1, len(nz_indices)):
        a = alpha * demand_sizes[i] + (1 - alpha) * a
        p = alpha * intervals[i] + (1 - alpha) * p

    croston_forecast = a / max(0.1, p)
    return np.full(horizon_days, max(0.0, round(croston_forecast, 2)))
