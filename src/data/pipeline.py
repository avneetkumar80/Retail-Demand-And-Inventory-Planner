import sqlite3
import argparse
import pandas as pd
from typing import Tuple, Dict, Any
from pathlib import Path

from src.config import DB_PATH, RAW_DATA_DIR
from src.data.schema import init_db
from src.data.generator import generate_synthetic_data
from src.data.validator import DataValidator, DataValidationError

def clean_and_fill_sales_gaps(df_sales: pd.DataFrame, df_products: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures complete, continuous daily time series for each product by filling
    missing calendar dates with 0 units_sold and correct day_of_week indicators.
    """
    df_sales = df_sales.copy()
    df_sales["date"] = pd.to_datetime(df_sales["date"])
    
    cleaned_records = []
    min_date = df_sales["date"].min()
    max_date = df_sales["date"].max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    for pid in df_products["product_id"].unique():
        p_sales = df_sales[df_sales["product_id"] == pid].set_index("date")
        
        # Reindex to full date range
        p_reindexed = p_sales.reindex(full_date_range)
        p_reindexed["product_id"] = pid
        p_reindexed["units_sold"] = p_reindexed["units_sold"].fillna(0).astype(int)
        p_reindexed["is_promotion"] = p_reindexed["is_promotion"].fillna(0).astype(int)
        p_reindexed["is_holiday"] = p_reindexed["is_holiday"].fillna(0).astype(int)
        
        p_reindexed["date"] = p_reindexed.index.strftime("%Y-%m-%d")
        p_reindexed["day_of_week"] = p_reindexed.index.dayofweek
        
        cleaned_records.append(p_reindexed.reset_index(drop=True))
        
    df_cleaned = pd.concat(cleaned_records, ignore_index=True)
    # Sort deterministically
    df_cleaned = df_cleaned.sort_values(by=["date", "product_id"]).reset_index(drop=True)
    return df_cleaned

def run_pipeline_from_dataframes(
    df_products: pd.DataFrame,
    df_sales: pd.DataFrame,
    df_inventory: pd.DataFrame,
    db_path: Path = DB_PATH
) -> Dict[str, Any]:
    """
    Ingests, validates, cleans, and loads user-provided custom DataFrames into SQLite database.
    """
    init_db(db_path)
    
    # Save raw custom inputs
    df_products.to_csv(RAW_DATA_DIR / "products.csv", index=False)
    df_sales.to_csv(RAW_DATA_DIR / "sales.csv", index=False)
    df_inventory.to_csv(RAW_DATA_DIR / "inventory.csv", index=False)

    # Clean & fill gaps
    df_sales_cleaned = clean_and_fill_sales_gaps(df_sales, df_products)

    # Validate Data
    validator = DataValidator()
    is_valid, report = validator.validate_all(df_products, df_sales_cleaned, df_inventory)
    
    if not is_valid:
        raise DataValidationError(f"Data Pipeline aborted due to validation errors: {report['errors']}")

    # Load to SQLite (Clear existing tables for fresh user upload)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM daily_sales")
        cursor.execute("DELETE FROM inventory")
        cursor.execute("DELETE FROM products")
        
        for _, row in df_products.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO products 
                (product_id, product_name, category, unit_cost, unit_price, lead_time_days, min_order_qty, shelf_life_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["product_id"], row["product_name"], row["category"],
                float(row["unit_cost"]), float(row["unit_price"]), int(row["lead_time_days"]),
                int(row["min_order_qty"]), row.get("shelf_life_days")
            ))
            
        for _, row in df_sales_cleaned.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO daily_sales
                (date, product_id, units_sold, is_promotion, day_of_week, is_holiday)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(row["date"]), str(row["product_id"]), int(row["units_sold"]),
                int(row.get("is_promotion", 0)), int(row.get("day_of_week", 0)), int(row.get("is_holiday", 0))
            ))

        for _, row in df_inventory.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO inventory
                (product_id, current_stock, last_restock_date)
                VALUES (?, ?, ?)
            """, (
                str(row["product_id"]), int(row["current_stock"]), str(row.get("last_restock_date", ""))
            ))
            
        conn.commit()

    report["records_loaded"] = {
        "products": len(df_products),
        "sales": len(df_sales_cleaned),
        "inventory": len(df_inventory)
    }
    return report

def run_pipeline(
    db_path: Path = DB_PATH,
    generate_synthetic: bool = True
) -> Dict[str, Any]:
    """
    Runs full ETL Pipeline: Init DB -> Ingest Raw -> Validate -> Transform -> Load to SQLite.
    Returns status and validation summary dictionary.
    """
    # 1. Init Database Schema
    init_db(db_path)
    
    if generate_synthetic:
        df_products, df_sales, df_inventory = generate_synthetic_data(num_days=180)
        # Save raw snapshots
        df_products.to_csv(RAW_DATA_DIR / "products.csv", index=False)
        df_sales.to_csv(RAW_DATA_DIR / "sales.csv", index=False)
        df_inventory.to_csv(RAW_DATA_DIR / "inventory.csv", index=False)
    else:
        df_products = pd.read_csv(RAW_DATA_DIR / "products.csv")
        df_sales = pd.read_csv(RAW_DATA_DIR / "sales.csv")
        df_inventory = pd.read_csv(RAW_DATA_DIR / "inventory.csv")

    return run_pipeline_from_dataframes(df_products, df_sales, df_inventory, db_path)

def load_products(db_path: Path = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM products", conn)

def load_sales(db_path: Path = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM daily_sales ORDER BY date ASC", conn)
        df["date"] = pd.to_datetime(df["date"])
        return df

def load_inventory(db_path: Path = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM inventory", conn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockWise ETL Pipeline Runner")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate fresh synthetic dataset")
    args = parser.parse_args()
    
    summary = run_pipeline(generate_synthetic=True)
    print("ETL Pipeline completed successfully!")
    print(summary)
