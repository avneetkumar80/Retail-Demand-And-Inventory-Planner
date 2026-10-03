import pytest
import pandas as pd
import numpy as np
from src.inventory.reorder_engine import ReorderEngine

@pytest.fixture
def sample_product_inventory():
    p_row = pd.Series({
        "product_id": "PRD-TEST",
        "product_name": "Test Espresso",
        "category": "Coffee",
        "unit_cost": 10.0,
        "unit_price": 25.0,
        "lead_time_days": 3,
        "min_order_qty": 10
    })
    inv_row = pd.Series({
        "product_id": "PRD-TEST",
        "current_stock": 5
    })
    sales_series = pd.Series([10, 12, 11, 9, 10, 13, 15, 10, 11, 12])
    forecast = np.array([12, 12, 12, 12, 12, 12, 12])
    return p_row, inv_row, sales_series, forecast

def test_reorder_calculation(sample_product_inventory):
    p_row, inv_row, sales_series, forecast = sample_product_inventory
    engine = ReorderEngine()
    
    rec = engine.calculate_recommendation(
        product_row=p_row,
        inventory_row=inv_row,
        daily_forecast=forecast,
        historical_sales=sales_series,
        safety_factor=1.65
    )
    
    assert rec["product_id"] == "PRD-TEST"
    assert rec["lead_time_days"] == 3
    assert rec["lead_time_demand"] == 36.0  # 3 days * 12 units
    assert rec["reorder_point"] > 36.0      # ROP = D_L + Safety Stock
    assert rec["suggested_reorder_qty"] >= p_row["min_order_qty"]
    assert rec["status_code"] in ["URGENT_REORDER", "REORDER_SOON", "CRITICAL_OUT_OF_STOCK"]

def test_zero_stock_alert(sample_product_inventory):
    p_row, _, sales_series, forecast = sample_product_inventory
    inv_row = pd.Series({"product_id": "PRD-TEST", "current_stock": 0})
    engine = ReorderEngine()
    
    rec = engine.calculate_recommendation(
        product_row=p_row,
        inventory_row=inv_row,
        daily_forecast=forecast,
        historical_sales=sales_series
    )
    assert rec["status_code"] == "CRITICAL_OUT_OF_STOCK"
    assert rec["suggested_reorder_qty"] > 0
