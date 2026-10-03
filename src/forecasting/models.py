import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any, Optional
from sklearn.linear_model import Ridge
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

def create_time_features(df_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Creates temporal and rolling lag features for ML model without data leakage.
    Expects df_sales with columns: ['date', 'product_id', 'units_sold', 'is_promotion', 'day_of_week', 'is_holiday']
    """
    df = df_sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["product_id", "date"]).reset_index(drop=True)
    
    # Temporal indicators
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Lag and Rolling features calculated PER PRODUCT
    dfs = []
    for pid, group in df.groupby("product_id"):
        group = group.copy()
        
        # Lags
        group["lag_7"] = group["units_sold"].shift(7)
        group["lag_14"] = group["units_sold"].shift(14)
        group["lag_21"] = group["units_sold"].shift(21)
        
        # Rolling stats on shifted series to avoid leakage
        shifted = group["units_sold"].shift(1)
        group["rolling_mean_7"] = shifted.rolling(window=7, min_periods=1).mean()
        group["rolling_std_7"] = shifted.rolling(window=7, min_periods=1).std().fillna(0)
        group["rolling_mean_14"] = shifted.rolling(window=14, min_periods=1).mean()
        
        dfs.append(group)
        
    df_featured = pd.concat(dfs, ignore_index=True)
    df_featured = df_featured.sort_values(by=["date", "product_id"]).reset_index(drop=True)
    return df_featured

class MLForecaster:
    """
    Supervised Machine Learning Forecaster (LightGBM / Ridge) with recursive auto-regressive multi-step forecasting.
    """
    FEATURE_COLS = [
        "day_of_week", "day_of_month", "month", "is_weekend",
        "is_promotion", "is_holiday",
        "lag_7", "lag_14", "lag_21",
        "rolling_mean_7", "rolling_std_7", "rolling_mean_14"
    ]

    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type if (model_type == "lightgbm" and HAS_LIGHTGBM) else "ridge"
        self.model = None
        self.feature_importances_ = {}

    def fit(self, df_train: pd.DataFrame) -> "MLForecaster":
        """
        Trains model on feature-engineered training subset.
        """
        df_feat = create_time_features(df_train)
        # Drop rows with NaN in features caused by lag operations
        df_clean = df_feat.dropna(subset=self.FEATURE_COLS).copy()
        
        X = df_clean[self.FEATURE_COLS]
        y = df_clean["units_sold"]
        
        if self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                verbose=-1
            )
            self.model.fit(X, y)
            importances = self.model.feature_importances_
            self.feature_importances_ = dict(zip(self.FEATURE_COLS, [float(x) for x in importances]))
        else:
            self.model = Ridge(alpha=1.0, random_state=42)
            self.model.fit(X, y)
            importances = np.abs(self.model.coef_)
            self.feature_importances_ = dict(zip(self.FEATURE_COLS, [float(x) for x in importances]))

        return self

    def predict_product_horizon(
        self,
        product_history_df: pd.DataFrame,
        horizon_days: int
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generates recursive multi-step forecasts for a single product.
        Returns (predictions_array, top_feature_importances)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        df_history = product_history_df.copy().sort_values(by="date").reset_index(drop=True)
        last_date = pd.to_datetime(df_history["date"].iloc[-1])
        pid = df_history["product_id"].iloc[0]

        predictions = []
        simulated_history = df_history.copy()

        for step in range(1, horizon_days + 1):
            next_date = last_date + pd.Timedelta(days=step)
            dow = next_date.weekday()
            
            # Feature extraction on updated history
            df_feat = create_time_features(simulated_history)
            latest_row = df_feat.iloc[-1].copy()
            
            # Construct row for next_date
            next_row = {
                "date": next_date.strftime("%Y-%m-%d"),
                "product_id": pid,
                "units_sold": 0,
                "is_promotion": 0,
                "day_of_week": dow,
                "is_holiday": 0,
                "day_of_month": next_date.day,
                "month": next_date.month,
                "is_weekend": 1 if dow >= 5 else 0,
                "lag_7": latest_row["units_sold"] if step == 1 else (predictions[step-8] if step >= 8 else latest_row["lag_7"]),
                "lag_14": latest_row["lag_7"] if step == 1 else latest_row["lag_14"],
                "lag_21": latest_row["lag_14"] if step == 1 else latest_row["lag_21"],
                "rolling_mean_7": latest_row["rolling_mean_7"],
                "rolling_std_7": latest_row["rolling_std_7"],
                "rolling_mean_14": latest_row["rolling_mean_14"]
            }
            
            X_pred = pd.DataFrame([next_row])[self.FEATURE_COLS].fillna(0)
            pred_val = float(self.model.predict(X_pred)[0])
            pred_val = max(0.0, round(pred_val, 2))
            
            predictions.append(pred_val)
            
            # Append predicted step to simulated history for autoregression
            next_df_row = pd.DataFrame([{
                "date": next_date.strftime("%Y-%m-%d"),
                "product_id": pid,
                "units_sold": pred_val,
                "is_promotion": 0,
                "day_of_week": dow,
                "is_holiday": 0
            }])
            simulated_history = pd.concat([simulated_history, next_df_row], ignore_index=True)

        return np.array(predictions), self.feature_importances_
