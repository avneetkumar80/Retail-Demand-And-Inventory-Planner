import pytest
import sqlite3
import pandas as pd
from pathlib import Path
from src.data.schema import init_db
from src.data.pipeline import run_pipeline, clean_and_fill_sales_gaps

def test_init_db(tmp_path):
    db_file = tmp_path / "test_stockwise.db"
    init_db(db_file)
    assert db_file.exists()
    
    with sqlite3.connect(db_file) as conn:
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
        assert "products" in tables
        assert "daily_sales" in tables
        assert "inventory" in tables

def test_clean_and_fill_sales_gaps():
    df_products = pd.DataFrame([{"product_id": "P1"}])
    df_sales = pd.DataFrame([
        {"date": "2026-01-01", "product_id": "P1", "units_sold": 5, "is_promotion": 0, "day_of_week": 3, "is_holiday": 0},
        {"date": "2026-01-03", "product_id": "P1", "units_sold": 8, "is_promotion": 0, "day_of_week": 5, "is_holiday": 0}
    ])
    
    df_clean = clean_and_fill_sales_gaps(df_sales, df_products)
    assert len(df_clean) == 3  # Gap on 2026-01-02 filled
    jan2_row = df_clean[df_clean["date"] == "2026-01-02"].iloc[0]
    assert jan2_row["units_sold"] == 0

def test_run_pipeline_idempotence(tmp_path):
    db_file = tmp_path / "test_pipeline.db"
    report1 = run_pipeline(db_path=db_file, generate_synthetic=True)
    report2 = run_pipeline(db_path=db_file, generate_synthetic=False)
    assert report1["status"] == "PASS"
    assert report2["status"] == "PASS"
