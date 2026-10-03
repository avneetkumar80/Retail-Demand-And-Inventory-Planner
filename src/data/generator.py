import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict

# Product catalog definition representing a realistic café/retail store
SAMPLE_PRODUCTS = [
    {
        "product_id": "PRD-001",
        "product_name": "Espresso Coffee Beans 1kg",
        "category": "Coffee & Tea",
        "unit_cost": 12.50,
        "unit_price": 28.00,
        "lead_time_days": 4,
        "min_order_qty": 10,
        "shelf_life_days": 180,
        "base_demand": 25,
        "weekend_mult": 1.4,
        "volatility": 4.0,
        "is_intermittent": False
    },
    {
        "product_id": "PRD-002",
        "product_name": "Oat Milk 1L (Case of 6)",
        "category": "Dairy & Alternatives",
        "unit_cost": 8.00,
        "unit_price": 16.50,
        "lead_time_days": 2,
        "min_order_qty": 15,
        "shelf_life_days": 60,
        "base_demand": 40,
        "weekend_mult": 1.2,
        "volatility": 5.5,
        "is_intermittent": False
    },
    {
        "product_id": "PRD-003",
        "product_name": "Fresh Butter Croissants (12-pack)",
        "category": "Bakery",
        "unit_cost": 6.20,
        "unit_price": 14.00,
        "lead_time_days": 1,
        "min_order_qty": 5,
        "shelf_life_days": 3,
        "base_demand": 30,
        "weekend_mult": 1.8,
        "volatility": 6.0,
        "is_intermittent": False
    },
    {
        "product_id": "PRD-004",
        "product_name": "Ceremonial Grade Matcha 100g",
        "category": "Coffee & Tea",
        "unit_cost": 18.00,
        "unit_price": 42.00,
        "lead_time_days": 7,
        "min_order_qty": 5,
        "shelf_life_days": 365,
        "base_demand": 4,
        "weekend_mult": 1.3,
        "volatility": 2.0,
        "is_intermittent": True  # Intermittent low volume sales
    },
    {
        "product_id": "PRD-005",
        "product_name": "Organic Almond Butter 500g",
        "category": "Pantry & Grab-and-Go",
        "unit_cost": 5.50,
        "unit_price": 11.90,
        "lead_time_days": 3,
        "min_order_qty": 12,
        "shelf_life_days": 120,
        "base_demand": 12,
        "weekend_mult": 1.1,
        "volatility": 3.0,
        "is_intermittent": False
    },
    {
        "product_id": "PRD-006",
        "product_name": "Gluten-Free Blueberry Muffins (6-pack)",
        "category": "Bakery",
        "unit_cost": 4.80,
        "unit_price": 10.50,
        "lead_time_days": 2,
        "min_order_qty": 8,
        "shelf_life_days": 5,
        "base_demand": 18,
        "weekend_mult": 1.6,
        "volatility": 4.5,
        "is_intermittent": False
    },
    {
        "product_id": "PRD-007",
        "product_name": "Sparkling Cold Brew Can 330ml",
        "category": "Beverages",
        "unit_cost": 1.20,
        "unit_price": 3.50,
        "lead_time_days": 3,
        "min_order_qty": 24,
        "shelf_life_days": 90,
        "base_demand": 35,
        "weekend_mult": 1.5,
        "volatility": 7.0,
        "is_intermittent": False
    }
]

def generate_synthetic_data(
    num_days: int = 180,
    end_date: str = None,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic synthetic sales, product, and inventory dataset for small retail/café.
    Returns (df_products, df_sales, df_inventory)
    """
    np.random.seed(seed)
    
    if end_date is None:
        end_dt = datetime.now()
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
    start_dt = end_dt - timedelta(days=num_days - 1)
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
    
    products_list = []
    inventory_list = []
    sales_list = []
    
    for p in SAMPLE_PRODUCTS:
        products_list.append({
            "product_id": p["product_id"],
            "product_name": p["product_name"],
            "category": p["category"],
            "unit_cost": p["unit_cost"],
            "unit_price": p["unit_price"],
            "lead_time_days": p["lead_time_days"],
            "min_order_qty": p["min_order_qty"],
            "shelf_life_days": p["shelf_life_days"]
        })
        
        # Simulate realistic current stock based on demand & lead time
        avg_daily = p["base_demand"]
        current_stock = int(avg_daily * (p["lead_time_days"] + np.random.uniform(0.5, 2.5)))
        # Introduce 1 or 2 products near stockout condition for testing alerts
        if p["product_id"] in ["PRD-001", "PRD-004"]:
            current_stock = int(avg_daily * 0.8)  # Danger low stock
            
        inventory_list.append({
            "product_id": p["product_id"],
            "current_stock": max(2, current_stock),
            "last_restock_date": (end_dt - timedelta(days=np.random.randint(1, 5))).strftime("%Y-%m-%d")
        })
        
        # Generate time series sales for this product
        trend = np.linspace(0.9, 1.15, len(date_range))  # Slight growth trend
        
        for i, dt in enumerate(date_range):
            dow = dt.weekday()
            is_weekend = 1 if dow >= 5 else 0
            
            # 5% chance of holiday or promotion
            is_promo = 1 if np.random.rand() < 0.08 else 0
            is_holiday = 1 if (dt.month == 12 and dt.day in [24, 25, 31]) or (dt.month == 7 and dt.day == 4) else 0
            
            day_mult = p["weekend_mult"] if is_weekend else 1.0
            promo_mult = 1.35 if is_promo else 1.0
            holiday_mult = 0.4 if is_holiday else 1.0  # Shop closes early or closed on holiday
            
            if p["is_intermittent"]:
                # Zero sales on ~60% of days
                has_sales = np.random.rand() > 0.6
                if has_sales:
                    units = int(np.random.poisson(lam=p["base_demand"] * promo_mult))
                else:
                    units = 0
            else:
                expected_demand = p["base_demand"] * trend[i] * day_mult * promo_mult * holiday_mult
                noise = np.random.normal(0, p["volatility"])
                units = max(0, int(round(expected_demand + noise)))
                
            sales_list.append({
                "date": dt.strftime("%Y-%m-%d"),
                "product_id": p["product_id"],
                "units_sold": units,
                "is_promotion": is_promo,
                "day_of_week": dow,
                "is_holiday": is_holiday
            })
            
    df_products = pd.DataFrame(products_list)
    df_sales = pd.DataFrame(sales_list)
    df_inventory = pd.DataFrame(inventory_list)
    
    return df_products, df_sales, df_inventory

if __name__ == "__main__":
    p, s, inv = generate_synthetic_data()
    print(f"Generated {len(p)} products, {len(s)} sales records, {len(inv)} inventory records.")
