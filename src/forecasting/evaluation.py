import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple

from src.config import DB_PATH, EVALUATION_RESULTS_PATH, TEST_SPLIT_DAYS
from src.data.pipeline import load_sales, load_products
from src.forecasting.baselines import (
    forecast_same_day_last_week,
    forecast_moving_average,
    forecast_exponential_smoothing,
    forecast_croston
)
from src.forecasting.models import MLForecaster

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes MAE, WAPE, and RMSE metrics.
    WAPE = (Sum |y_true - y_pred| / Sum y_true) * 100
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    sum_true = float(np.sum(y_true))
    if sum_true > 0:
        wape = float((np.sum(np.abs(y_true - y_pred)) / sum_true) * 100)
    else:
        wape = 0.0 if np.sum(np.abs(y_pred)) == 0 else 100.0
        
    return {
        "MAE": round(mae, 2),
        "WAPE_percent": round(wape, 2),
        "RMSE": round(rmse, 2)
    }

class BacktestEvaluator:
    """
    Evaluates forecasting models over out-of-sample time-based validation splits.
    Strictly prevents data leakage by restricting train set to data prior to cutoff date.
    """
    
    def __init__(self, db_path: Path = DB_PATH, test_days: int = TEST_SPLIT_DAYS, model_type: str = "lightgbm"):
        self.db_path = db_path
        self.test_days = test_days
        self.model_type = model_type
        self.df_sales = load_sales(db_path)
        self.df_products = load_products(db_path)

    def run_backtest(self) -> Dict[str, Any]:
        """
        Runs out-of-sample backtest comparing Baselines vs ML Model.
        Returns comprehensive evaluation dictionary and saves to evaluation_results.json.
        """
        # Reload latest data from DB
        self.df_sales = load_sales(self.db_path)
        self.df_products = load_products(self.db_path)

        # Define cutoff date with dynamic fallback for short datasets
        min_date = self.df_sales["date"].min()
        max_date = self.df_sales["date"].max()
        total_days = (max_date - min_date).days + 1
        
        effective_test_days = self.test_days
        if total_days <= self.test_days + 7:
            effective_test_days = max(3, int(total_days * 0.25))

        cutoff_date = max_date - pd.Timedelta(days=effective_test_days)
        
        df_train = self.df_sales[self.df_sales["date"] <= cutoff_date].copy()
        df_test = self.df_sales[self.df_sales["date"] > cutoff_date].copy()
        
        # Fit ML model strictly on train set
        ml_model = MLForecaster(model_type=self.model_type).fit(df_train)
        
        model_names = ["Naive_LastWeek", "MovingAvg_7D", "ExpSmoothing", "Croston_Intermittent", "ML_Model"]
        
        product_results = {}
        all_actuals = {m: [] for m in model_names}
        all_preds = {m: [] for m in model_names}

        for _, prod in self.df_products.iterrows():
            pid = prod["product_id"]
            p_name = prod["product_name"]
            
            p_train = df_train[df_train["product_id"] == pid].sort_values(by="date")
            p_test = df_test[df_test["product_id"] == pid].sort_values(by="date")
            
            y_actual = p_test["units_sold"].values
            horizon = len(y_actual)
            
            if horizon == 0:
                continue
                
            # Forecasts from each model
            preds_naive = forecast_same_day_last_week(p_train, horizon)
            preds_ma = forecast_moving_average(p_train, horizon, window=7)
            preds_es = forecast_exponential_smoothing(p_train, horizon, alpha=0.3)
            preds_croston = forecast_croston(p_train, horizon, alpha=0.1)
            preds_ml, _ = ml_model.predict_product_horizon(p_train, horizon)

            model_preds = {
                "Naive_LastWeek": preds_naive,
                "MovingAvg_7D": preds_ma,
                "ExpSmoothing": preds_es,
                "Croston_Intermittent": preds_croston,
                "ML_Model": preds_ml
            }

            p_metrics = {}
            for m_name, preds in model_preds.items():
                p_metrics[m_name] = calculate_metrics(y_actual, preds)
                all_actuals[m_name].extend(y_actual)
                all_preds[m_name].extend(preds)

            # Determine best model for this specific product
            best_m = min(p_metrics.keys(), key=lambda m: p_metrics[m]["WAPE_percent"])

            product_results[pid] = {
                "product_name": p_name,
                "category": prod["category"],
                "best_model": best_m,
                "metrics": p_metrics
            }

        # Compute Global Aggregate Metrics across all products
        global_metrics = {}
        for m_name in model_names:
            global_metrics[m_name] = calculate_metrics(
                np.array(all_actuals[m_name]),
                np.array(all_preds[m_name])
            )

        overall_best = min(global_metrics.keys(), key=lambda m: global_metrics[m]["WAPE_percent"])

        evaluation_output = {
            "test_period_days": self.test_days,
            "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
            "max_date": max_date.strftime("%Y-%m-%d"),
            "overall_best_model": overall_best,
            "global_metrics": global_metrics,
            "product_metrics": product_results
        }

        # Save outputs
        EVALUATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EVALUATION_RESULTS_PATH, "w") as f:
            json.dump(evaluation_output, f, indent=4)

        return evaluation_output

if __name__ == "__main__":
    evaluator = BacktestEvaluator()
    res = evaluator.run_backtest()
    print("Backtest Evaluation Completed!")
    print(f"Overall Best Model: {res['overall_best_model']}")
    print(json.dumps(res["global_metrics"], indent=2))
