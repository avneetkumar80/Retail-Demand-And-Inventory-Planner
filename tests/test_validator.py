import pytest
import pandas as pd
from src.data.validator import DataValidator

@pytest.fixture
def sample_valid_data():
    df_products = pd.DataFrame([{
        "product_id": "PRD-TEST-1",
        "product_name": "Test Beans",
        "category": "Coffee",
        "unit_cost": 10.0,
        "unit_price": 20.0,
        "lead_time_days": 3,
        "min_order_qty": 5
    }])
    df_sales = pd.DataFrame([{
        "date": "2026-01-01",
        "product_id": "PRD-TEST-1",
        "units_sold": 10,
        "is_promotion": 0,
        "day_of_week": 3,
        "is_holiday": 0
    }])
    df_inventory = pd.DataFrame([{
        "product_id": "PRD-TEST-1",
        "current_stock": 15
    }])
    return df_products, df_sales, df_inventory

def test_validator_pass(sample_valid_data):
    df_products, df_sales, df_inventory = sample_valid_data
    validator = DataValidator()
    is_valid, report = validator.validate_all(df_products, df_sales, df_inventory)
    assert is_valid is True
    assert report["status"] == "PASS"

def test_validator_negative_cost(sample_valid_data):
    df_products, df_sales, df_inventory = sample_valid_data
    df_products.loc[0, "unit_cost"] = -5.0
    validator = DataValidator()
    is_valid, report = validator.validate_all(df_products, df_sales, df_inventory)
    assert is_valid is False
    assert any("Negative unit cost" in e for e in report["errors"])

def test_validator_duplicate_product_id(sample_valid_data):
    df_products, df_sales, df_inventory = sample_valid_data
    dup_row = df_products.iloc[0].copy()
    df_products = pd.concat([df_products, pd.DataFrame([dup_row])], ignore_index=True)
    validator = DataValidator()
    is_valid, report = validator.validate_all(df_products, df_sales, df_inventory)
    assert is_valid is False
    assert any("Duplicate product IDs" in e for e in report["errors"])
