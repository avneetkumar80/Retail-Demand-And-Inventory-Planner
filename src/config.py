import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "stockwise.db"
EVALUATION_RESULTS_PATH = DATA_DIR / "evaluation_results.json"

# Ensure directory structure exists
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Forecasting Configurations
DEFAULT_HORIZONS = [7, 14]
HISTORICAL_TRAIN_DAYS = 180  # Default duration of synthetic dataset in days
TEST_SPLIT_DAYS = 28        # Last 28 days reserved for time-based evaluation

# Inventory Configurations
DEFAULT_SAFETY_FACTOR = 1.65  # ~95% service level standard normal z-score
DEFAULT_LEAD_TIME_DAYS = 3
DEFAULT_SERVICE_LEVEL = 0.95

# Streamlit App Custom Styling
APP_TITLE = "StockWise | Retail Demand Forecasting & Inventory Planner"
APP_ICON = "📦"
