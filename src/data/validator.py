import pandas as pd
from typing import Dict, List, Tuple, Any

class DataValidationError(Exception):
    """Custom exception raised when critical data validation fails."""
    pass

class DataValidator:
    """
    Validates products, sales, and inventory dataframes for data quality,
    temporal completeness, constraint adherence, and referential integrity.
    """
    
    def __init__(self):
        self.validation_summary = {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "metrics": {}
        }

    def validate_all(
        self,
        df_products: pd.DataFrame,
        df_sales: pd.DataFrame,
        df_inventory: pd.DataFrame
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Executes complete validation suite across all three relational tables.
        Returns (is_valid, validation_report_dict).
        """
        self.validation_summary = {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "metrics": {
                "total_products": len(df_products),
                "total_sales_records": len(df_sales),
                "total_inventory_records": len(df_inventory),
                "date_range": ""
            }
        }

        # 1. Product Table Checks
        self._validate_products(df_products)

        # 2. Sales Table Checks
        self._validate_sales(df_sales, df_products)

        # 3. Inventory Table Checks
        self._validate_inventory(df_inventory, df_products)

        # Final Status determination
        if len(self.validation_summary["errors"]) > 0:
            self.validation_summary["status"] = "FAIL"
        elif len(self.validation_summary["warnings"]) > 0:
            self.validation_summary["status"] = "WARNING"
            
        is_valid = self.validation_summary["status"] != "FAIL"
        return is_valid, self.validation_summary

    def _validate_products(self, df_products: pd.DataFrame) -> None:
        required_cols = {"product_id", "product_name", "category", "unit_cost", "unit_price", "lead_time_days", "min_order_qty"}
        missing_cols = required_cols - set(df_products.columns)
        if missing_cols:
            self.validation_summary["errors"].append(f"Products schema error: missing columns {missing_cols}")
            return

        # Duplicate product_id
        dups = df_products[df_products.duplicated(subset=["product_id"])]
        if not dups.empty:
            self.validation_summary["errors"].append(f"Duplicate product IDs found: {dups['product_id'].tolist()}")

        # Invalid prices or lead times
        invalid_cost = df_products[df_products["unit_cost"] < 0]
        if not invalid_cost.empty:
            self.validation_summary["errors"].append(f"Negative unit cost found for product IDs: {invalid_cost['product_id'].tolist()}")

        invalid_lead = df_products[df_products["lead_time_days"] <= 0]
        if not invalid_lead.empty:
            self.validation_summary["errors"].append(f"Invalid lead time <= 0 for product IDs: {invalid_lead['product_id'].tolist()}")

    def _validate_sales(self, df_sales: pd.DataFrame, df_products: pd.DataFrame) -> None:
        required_cols = {"date", "product_id", "units_sold", "is_promotion", "day_of_week", "is_holiday"}
        missing_cols = required_cols - set(df_sales.columns)
        if missing_cols:
            self.validation_summary["errors"].append(f"Sales schema error: missing columns {missing_cols}")
            return

        # Check negative sales
        neg_sales = df_sales[df_sales["units_sold"] < 0]
        if not neg_sales.empty:
            self.validation_summary["errors"].append(f"Negative sales quantities detected in {len(neg_sales)} records.")

        # Check duplicate (date, product_id)
        dups = df_sales[df_sales.duplicated(subset=["date", "product_id"])]
        if not dups.empty:
            self.validation_summary["errors"].append(f"Duplicate (date, product_id) combinations found: {len(dups)} duplicates.")

        # Foreign Key check: product_id in sales must exist in products
        valid_pids = set(df_products["product_id"])
        sales_pids = set(df_sales["product_id"])
        orphan_pids = sales_pids - valid_pids
        if orphan_pids:
            self.validation_summary["errors"].append(f"Orphan sales records with unlisted product IDs: {orphan_pids}")

        # Date completeness check per product
        if not df_sales.empty:
            df_sales["dt"] = pd.to_datetime(df_sales["date"])
            min_date = df_sales["dt"].min()
            max_date = df_sales["dt"].max()
            self.validation_summary["metrics"]["date_range"] = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            
            expected_days = (max_date - min_date).days + 1
            for pid in valid_pids:
                p_sales = df_sales[df_sales["product_id"] == pid]
                actual_days = len(p_sales["date"].unique())
                if actual_days < expected_days:
                    missing_count = expected_days - actual_days
                    self.validation_summary["warnings"].append(
                        f"Product {pid} has {missing_count} missing date records between {min_date.strftime('%Y-%m-%d')} and {max_date.strftime('%Y-%m-%d')}."
                    )

    def _validate_inventory(self, df_inventory: pd.DataFrame, df_products: pd.DataFrame) -> None:
        required_cols = {"product_id", "current_stock"}
        missing_cols = required_cols - set(df_inventory.columns)
        if missing_cols:
            self.validation_summary["errors"].append(f"Inventory schema error: missing columns {missing_cols}")
            return

        neg_stock = df_inventory[df_inventory["current_stock"] < 0]
        if not neg_stock.empty:
            self.validation_summary["errors"].append(f"Negative stock levels found for product IDs: {neg_stock['product_id'].tolist()}")

        valid_pids = set(df_products["product_id"])
        inv_pids = set(df_inventory["product_id"])
        missing_inv = valid_pids - inv_pids
        if missing_inv:
            self.validation_summary["warnings"].append(f"Products missing from inventory table: {missing_inv}")
