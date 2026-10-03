import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.config import DB_PATH, DEFAULT_LEAD_TIME_DAYS, DEFAULT_SAFETY_FACTOR
from src.data.pipeline import load_products, load_sales, load_inventory
from src.forecasting.baselines import forecast_moving_average, forecast_croston
from src.forecasting.models import MLForecaster

class ReorderEngine:
    """
    Inventory Planning Engine calculating Reorder Points (ROP), Safety Stock (SS),
    and Recommended Reorder Quantities (Q) based on forecast demand and supply parameters.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def calculate_recommendation(
        self,
        product_row: pd.Series,
        inventory_row: pd.Series,
        daily_forecast: np.ndarray,
        historical_sales: pd.Series,
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
        custom_lead_time: Optional[int] = None,
        demand_multiplier: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculates reorder recommendation and returns full calculation transparency object.
        """
        pid = product_row["product_id"]
        pname = product_row["product_name"]
        category = product_row["category"]
        unit_cost = float(product_row["unit_cost"])
        unit_price = float(product_row["unit_price"])
        moq = int(product_row["min_order_qty"])
        
        lead_time = int(custom_lead_time) if custom_lead_time is not None else int(product_row["lead_time_days"])
        lead_time = max(1, lead_time)  # Edge case: minimum 1 day
        
        current_stock = int(inventory_row["current_stock"]) if inventory_row is not None else 0
        
        # Apply demand multiplier (for What-If scenarios)
        forecast_adjusted = np.array(daily_forecast) * float(demand_multiplier)
        
        # 1. Lead Time Demand (D_L)
        if len(forecast_adjusted) >= lead_time:
            lead_time_demand = float(np.sum(forecast_adjusted[:lead_time]))
            avg_daily_demand = float(np.mean(forecast_adjusted[:lead_time]))
        else:
            avg_daily_demand = float(np.mean(forecast_adjusted)) if len(forecast_adjusted) > 0 else 0.0
            lead_time_demand = avg_daily_demand * lead_time
            
        # 2. Historical Demand Volatility (Std Dev)
        if len(historical_sales) > 1:
            std_daily_demand = float(np.std(historical_sales.values, ddof=1))
        else:
            std_daily_demand = avg_daily_demand * 0.3  # Fallback estimate
            
        # 3. Safety Stock (SS) = Z * std_d * sqrt(Lead_Time)
        safety_stock = float(safety_factor * std_daily_demand * math.sqrt(lead_time))
        safety_stock = round(max(1.0, safety_stock), 1)
        
        # 4. Reorder Point (ROP) = D_L + SS
        reorder_point = round(lead_time_demand + safety_stock, 1)
        
        # 5. Raw Deficit = ROP - Current Stock
        raw_deficit = reorder_point - current_stock
        
        if raw_deficit > 0:
            # Must reorder enough to reach ROP, rounded up to MOQ
            raw_qty = math.ceil(raw_deficit)
            suggested_reorder_qty = max(raw_qty, moq)
        else:
            suggested_reorder_qty = 0
            
        total_reorder_cost = round(suggested_reorder_qty * unit_cost, 2)

        # 6. Action Status Classification
        if current_stock == 0:
            status = "CRITICAL_OUT_OF_STOCK"
            status_label = "🔴 Out of Stock"
        elif current_stock <= safety_stock:
            status = "URGENT_REORDER"
            status_label = "🟠 Urgent Reorder Needed"
        elif current_stock <= reorder_point:
            status = "REORDER_SOON"
            status_label = "🟡 Reorder Soon"
        elif current_stock > reorder_point + (14 * avg_daily_demand):
            status = "OVERSTOCKED"
            status_label = "🔵 Overstocked"
        else:
            status = "ADEQUATE_STOCK"
            status_label = "🟢 Stock Optimal"

        return {
            "product_id": pid,
            "product_name": pname,
            "category": category,
            "current_stock": current_stock,
            "lead_time_days": lead_time,
            "min_order_qty": moq,
            "unit_cost": unit_cost,
            "unit_price": unit_price,
            "safety_factor": safety_factor,
            "demand_multiplier": demand_multiplier,
            "avg_daily_demand": round(avg_daily_demand, 1),
            "lead_time_demand": round(lead_time_demand, 1),
            "std_daily_demand": round(std_daily_demand, 1),
            "safety_stock": math.ceil(safety_stock),
            "reorder_point": math.ceil(reorder_point),
            "suggested_reorder_qty": suggested_reorder_qty,
            "total_reorder_cost": total_reorder_cost,
            "status_code": status,
            "status_label": status_label
        }

    def generate_all_recommendations(
        self,
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
        custom_lead_times: Optional[Dict[str, int]] = None,
        demand_multipliers: Optional[Dict[str, float]] = None,
        horizon_days: int = 14
    ) -> List[Dict[str, Any]]:
        """
        Generates reorder recommendations across all products in database.
        """
        df_products = load_products(self.db_path)
        df_sales = load_sales(self.db_path)
        df_inventory = load_inventory(self.db_path)
        
        # Fit ML model on available historical data
        ml_model = MLForecaster(model_type="lightgbm").fit(df_sales)
        
        recommendations = []
        
        for _, p_row in df_products.iterrows():
            pid = p_row["product_id"]
            p_sales = df_sales[df_sales["product_id"] == pid].sort_values(by="date")
            p_inv_match = df_inventory[df_inventory["product_id"] == pid]
            inv_row = p_inv_match.iloc[0] if not p_inv_match.empty else None
            
            lt_override = custom_lead_times.get(pid) if custom_lead_times else None
            mult_override = demand_multipliers.get(pid, 1.0) if demand_multipliers else 1.0
            
            # Generate forecast using ML model (with fallback to Moving Average if needed)
            try:
                forecast, _ = ml_model.predict_product_horizon(p_sales, horizon_days)
            except Exception:
                forecast = forecast_moving_average(p_sales, horizon_days, window=7)
                
            rec = self.calculate_recommendation(
                product_row=p_row,
                inventory_row=inv_row,
                daily_forecast=forecast,
                historical_sales=p_sales["units_sold"],
                safety_factor=safety_factor,
                custom_lead_time=lt_override,
                demand_multiplier=mult_override
            )
            recommendations.append(rec)
            
        return recommendations
