import sqlite3
from pathlib import Path
from typing import Union
from src.config import DB_PATH

CREATE_PRODUCTS_TABLE = """
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_cost REAL NOT NULL CHECK(unit_cost >= 0),
    unit_price REAL NOT NULL CHECK(unit_price >= 0),
    lead_time_days INTEGER NOT NULL CHECK(lead_time_days > 0),
    min_order_qty INTEGER NOT NULL CHECK(min_order_qty >= 1),
    shelf_life_days INTEGER
);
"""

CREATE_SALES_TABLE = """
CREATE TABLE IF NOT EXISTS daily_sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    product_id TEXT NOT NULL,
    units_sold INTEGER NOT NULL CHECK(units_sold >= 0),
    is_promotion INTEGER NOT NULL CHECK(is_promotion IN (0, 1)),
    day_of_week INTEGER NOT NULL CHECK(day_of_week BETWEEN 0 AND 6),
    is_holiday INTEGER NOT NULL CHECK(is_holiday IN (0, 1)),
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    UNIQUE(date, product_id)
);
"""

CREATE_INVENTORY_TABLE = """
CREATE TABLE IF NOT EXISTS inventory (
    product_id TEXT PRIMARY KEY,
    current_stock INTEGER NOT NULL CHECK(current_stock >= 0),
    last_restock_date TEXT,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_sales_date_product ON daily_sales(date, product_id);
CREATE INDEX IF NOT EXISTS idx_sales_product ON daily_sales(product_id);
"""

def init_db(db_path: Union[str, Path] = DB_PATH) -> None:
    """Initialize SQLite database with required relational schema and indexes."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(CREATE_PRODUCTS_TABLE)
        cursor.execute(CREATE_SALES_TABLE)
        cursor.execute(CREATE_INVENTORY_TABLE)
        cursor.executescript(CREATE_INDEXES)
        conn.commit()

if __name__ == "__main__":
    init_db()
    print("Database schema initialized successfully.")
