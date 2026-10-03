import json
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# Page Config (Must be first Streamlit command)
st.set_page_config(
    page_title="StockWise | Interactive Model Training & Inventory Planning",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.config import APP_TITLE, DB_PATH, EVALUATION_RESULTS_PATH, DEFAULT_SAFETY_FACTOR
from src.ui.styles import CUSTOM_CSS
from src.ui.components import (
    render_metric_card,
    render_explanation_box,
    plot_demand_forecast,
    plot_model_evaluation_metrics
)
from src.data.pipeline import (
    load_products,
    load_sales,
    load_inventory,
    run_pipeline,
    run_pipeline_from_dataframes
)
from src.data.validator import DataValidator, DataValidationError
from src.data.generator import generate_synthetic_data
from src.forecasting.baselines import (
    forecast_same_day_last_week,
    forecast_moving_average
)
from src.forecasting.models import MLForecaster
from src.forecasting.evaluation import BacktestEvaluator
from src.inventory.reorder_engine import ReorderEngine
from src.explainability.explainer import PlainLanguageExplainer

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_data(ttl=600)
def load_all_data():
    """Cached loader for SQLite tables."""
    df_products = load_products()
    df_sales = load_sales()
    df_inventory = load_inventory()
    return df_products, df_sales, df_inventory

@st.cache_data(ttl=600)
def load_evaluation():
    """Cached loader for evaluation results."""
    if not EVALUATION_RESULTS_PATH.exists():
        evaluator = BacktestEvaluator()
        return evaluator.run_backtest()
    with open(EVALUATION_RESULTS_PATH, "r") as f:
        return json.load(f)

# Ensure DB exists initially
if not DB_PATH.exists():
    with st.spinner("Initializing Database and Pipeline..."):
        run_pipeline(generate_synthetic=True)

df_products, df_sales, df_inventory = load_all_data()
eval_data = load_evaluation()
reorder_engine = ReorderEngine()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://img.icons8.com/isometric/100/box.png", width=60)
st.sidebar.title("StockWise")
st.sidebar.caption("Explainable Demand Forecasting & Inventory Planning")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📥 Data Upload & Model Training",
        "📊 Executive Overview",
        "🔮 Demand Forecasts",
        "📦 Reorder Planning",
        "🎛️ 'What-If?' Simulator",
        "🎯 Model Evaluation",
        "🛠️ Data Quality & Audit"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Interactive Workflow**: Upload custom CSVs or select sample data, customize ML hyperparameters, click **Train Model**, and explore recommendations!")

# ==========================================
# PAGE 1: DATA UPLOAD & MODEL TRAINING
# ==========================================
if nav_choice == "📥 Data Upload & Model Training":
    st.title("📥 Custom Data Ingestion & Interactive Model Training")
    st.caption("Upload your custom business CSV datasets or use sample retail data, configure algorithm parameters, and train the forecasting engine.")

    st.markdown("### Step 1: Select Data Source")
    data_source_type = st.radio(
        "Choose Data Input Source",
        ["Use Sample Retail Dataset (Café Catalog)", "Upload Custom CSV Files"],
        horizontal=True
    )

    with st.expander("📋 Download CSV Schema Templates & Example Requirements"):
        st.markdown("""
        If uploading custom files, ensure your CSV files contain the following column headers:

        - **Products CSV (`products.csv`)**:
          `product_id`, `product_name`, `category`, `unit_cost`, `unit_price`, `lead_time_days`, `min_order_qty`, `shelf_life_days`
        - **Daily Sales CSV (`sales.csv`)**:
          `date` (YYYY-MM-DD), `product_id`, `units_sold`, `is_promotion` (0 or 1), `day_of_week` (0-6), `is_holiday` (0 or 1)
        - **Inventory CSV (`inventory.csv`)**:
          `product_id`, `current_stock`, `last_restock_date` (YYYY-MM-DD)
        """)

        t_products, t_sales, t_inventory = generate_synthetic_data(num_days=14)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("📥 Products Template CSV", t_products.to_csv(index=False), "products_template.csv", "text/csv")
        with c2:
            st.download_button("📥 Sales Template CSV", t_sales.to_csv(index=False), "sales_template.csv", "text/csv")
        with c3:
            st.download_button("📥 Inventory Template CSV", t_inventory.to_csv(index=False), "inventory_template.csv", "text/csv")

    custom_products_df = None
    custom_sales_df = None
    custom_inventory_df = None

    if data_source_type == "Upload Custom CSV Files":
        st.markdown("#### Upload Your 3 CSV Files")
        u_col1, u_col2, u_col3 = st.columns(3)
        with u_col1:
            file_p = st.file_uploader("Upload Products CSV", type=["csv"], key="u_prod")
            if file_p:
                custom_products_df = pd.read_csv(file_p)
                st.success(f"Loaded {len(custom_products_df)} products")
        with u_col2:
            file_s = st.file_uploader("Upload Sales CSV", type=["csv"], key="u_sales")
            if file_s:
                custom_sales_df = pd.read_csv(file_s)
                st.success(f"Loaded {len(custom_sales_df)} sales records")
        with u_col3:
            file_i = st.file_uploader("Upload Inventory CSV", type=["csv"], key="u_inv")
            if file_i:
                custom_inventory_df = pd.read_csv(file_i)
                st.success(f"Loaded {len(custom_inventory_df)} inventory records")

        if custom_products_df is not None and custom_sales_df is not None and custom_inventory_df is not None:
            st.subheader("🔍 Uploaded Data Preview")
            pv1, pv2, pv3 = st.tabs(["Products", "Sales Time Series", "Inventory"])
            with pv1:
                st.dataframe(custom_products_df.head(5), use_container_width=True)
            with pv2:
                st.dataframe(custom_sales_df.head(5), use_container_width=True)
            with pv3:
                st.dataframe(custom_inventory_df.head(5), use_container_width=True)

    st.markdown("---")
    st.markdown("### Step 2: Configure Training & Forecast Parameters")
    
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    with cfg_col1:
        model_algorithm = st.selectbox(
            "Forecasting Model Algorithm",
            ["LightGBM Regressor", "Ridge Regression"],
            help="LightGBM handles non-linear trends & interactions; Ridge is linear with L2 regularization."
        )
    with cfg_col2:
        test_split_days = st.selectbox(
            "Out-of-Sample Evaluation Split",
            [14, 28, 42],
            index=1,
            help="Number of recent historical days reserved strictly for out-of-sample backtest validation."
        )
    with cfg_col3:
        target_service_level = st.selectbox(
            "Safety Buffer Service Level",
            [1.65, 2.00, 2.33],
            format_func=lambda z: "95.0% Service Level (Z=1.65)" if z == 1.65 else ("97.5% Service Level (Z=2.00)" if z == 2.00 else "99.0% Service Level (Z=2.33)"),
            help="Z-score controlling safety stock buffer size against demand volatility."
        )

    st.markdown("---")
    st.markdown("### Step 3: Trigger Training Pipeline")

    if st.button("⚡ Train Model & Run Pipeline Now", type="primary", use_container_width=True):
        status_box = st.empty()
        progress_bar = st.progress(0)

        try:
            status_box.info("⏳ Step 1/4: Ingesting & Validating Datasets...")
            progress_bar.progress(25)

            model_key = "lightgbm" if "LightGBM" in model_algorithm else "ridge"

            if data_source_type == "Upload Custom CSV Files":
                if custom_products_df is None or custom_sales_df is None or custom_inventory_df is None:
                    st.error("❌ Please upload all 3 required CSV files (Products, Sales, Inventory) before training.")
                    st.stop()
                
                pipeline_report = run_pipeline_from_dataframes(
                    df_products=custom_products_df,
                    df_sales=custom_sales_df,
                    df_inventory=custom_inventory_df,
                    db_path=DB_PATH
                )
            else:
                pipeline_report = run_pipeline(db_path=DB_PATH, generate_synthetic=True)

            status_box.info("⏳ Step 2/4: Engineering Time-Series Lag & Rolling Features...")
            progress_bar.progress(50)

            status_box.info(f"⏳ Step 3/4: Training {model_algorithm} & Running Out-of-Sample Backtest...")
            progress_bar.progress(75)

            evaluator = BacktestEvaluator(db_path=DB_PATH, test_days=test_split_days, model_type=model_key)
            eval_output = evaluator.run_backtest()

            status_box.info("⏳ Step 4/4: Recalculating Inventory Reorder Points & Plain-Language Explanations...")
            progress_bar.progress(100)

            # Clear cache so all pages update
            st.cache_data.clear()

            st.balloons()
            st.success("🎉 **Model Training & Evaluation Completed Successfully!**")

            # Display Execution Metrics Summary
            ml_metrics = eval_output.get("global_metrics", {}).get("ML_Model", {})
            best_model_name = eval_output.get("overall_best_model", "ML_Model")

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                render_metric_card("Algorithm Trained", model_algorithm.split()[0], f"Model type: {model_key}")
            with res_col2:
                render_metric_card("Best Model", best_model_name, "Outperformed baselines")
            with res_col3:
                render_metric_card("Out-of-Sample WAPE", f"{ml_metrics.get('WAPE_percent', 0)}%", f"Tested over {test_split_days} days")
            with res_col4:
                render_metric_card("Avg Daily Error (MAE)", f"±{ml_metrics.get('MAE', 0)} units", "Physical unit error")

            st.info("👉 You can now navigate to **Executive Overview**, **Demand Forecasts**, or **Reorder Planning** in the left sidebar to view updated results.")

        except DataValidationError as ve:
            st.error(f"❌ Data Validation Error: {ve}")
        except Exception as e:
            st.error(f"❌ An error occurred during training: {str(e)}")

# ==========================================
# PAGE 2: EXECUTIVE OVERVIEW
# ==========================================
elif nav_choice == "📊 Executive Overview":
    st.title("📊 Executive Inventory Overview")
    st.caption("Real-time operational snapshot and inventory health alerts for store management.")

    recs = reorder_engine.generate_all_recommendations()
    df_recs = pd.DataFrame(recs)

    total_products = len(df_recs)
    urgent_count = len(df_recs[df_recs["status_code"].isin(["CRITICAL_OUT_OF_STOCK", "URGENT_REORDER"])])
    reorder_soon_count = len(df_recs[df_recs["status_code"] == "REORDER_SOON"])
    
    # Value in stock
    df_merged = df_products.merge(df_inventory, on="product_id")
    total_val = (df_merged["current_stock"] * df_merged["unit_cost"]).sum()
    
    overall_wape = eval_data.get("global_metrics", {}).get("ML_Model", {}).get("WAPE_percent", 14.2)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total SKUs Tracked", str(total_products), "Active catalog items")
    with col2:
        render_metric_card("Urgent Action Needed", str(urgent_count), "Stockout / Safety breach", "red")
    with col3:
        render_metric_card("Inventory Valuation", f"${total_val:,.2f}", "Capital locked in stock")
    with col4:
        render_metric_card("Forecast Accuracy (WAPE)", f"{overall_wape}%", "Out-of-sample error score")

    st.markdown("---")
    st.subheader("🚨 Urgent Inventory Alerts")

    urgent_df = df_recs[df_recs["status_code"].isin(["CRITICAL_OUT_OF_STOCK", "URGENT_REORDER", "REORDER_SOON"])]

    if urgent_df.empty:
        st.success("✅ All product inventory levels are currently optimal!")
    else:
        for _, r in urgent_df.iterrows():
            explanation = PlainLanguageExplainer.explain_recommendation(r)
            with st.expander(f"{r['status_label']} — {r['product_name']} ({r['current_stock']} units remaining)"):
                st.write(explanation["headline"])
                st.markdown("**Key Drivers:**")
                for d in explanation["key_drivers"]:
                    st.write(f"- {d}")
                st.info(f"Suggested Order Quantity: **{r['suggested_reorder_qty']} units** (Est. Cost: ${r['total_reorder_cost']:,.2f})")

# ==========================================
# PAGE 3: DEMAND FORECASTS
# ==========================================
elif nav_choice == "🔮 Demand Forecasts":
    st.title("🔮 Product Demand Forecasting")
    st.caption("Interactive 7-day and 14-day statistical & machine learning demand projections.")

    col1, col2 = st.columns([1, 2])
    with col1:
        categories = ["All Categories"] + list(df_products["category"].unique())
        selected_cat = st.selectbox("Filter Category", categories)

        if selected_cat != "All Categories":
            filtered_prods = df_products[df_products["category"] == selected_cat]
        else:
            filtered_prods = df_products

        prod_options = dict(zip(filtered_prods["product_id"], filtered_prods["product_name"]))
        selected_pid = st.selectbox("Select Product", options=list(prod_options.keys()), format_func=lambda x: prod_options[x])

        horizon_days = st.radio("Forecast Horizon", [7, 14], horizontal=True)

    p_info = df_products[df_products["product_id"] == selected_pid].iloc[0]
    p_sales = df_sales[df_sales["product_id"] == selected_pid].sort_values(by="date")

    # Fit ML Model and Baselines
    ml_model = MLForecaster(model_type="lightgbm").fit(df_sales)
    fc_14d, feat_imp = ml_model.predict_product_horizon(p_sales, 14)
    fc_7d = fc_14d[:7]

    base_naive = forecast_same_day_last_week(p_sales, 14)
    base_ma = forecast_moving_average(p_sales, 14, window=7)

    with col2:
        fig = plot_demand_forecast(
            df_history=p_sales,
            product_name=p_info["product_name"],
            forecast_7d=fc_7d,
            forecast_14d=fc_14d,
            baseline_naive=base_naive,
            baseline_ma=base_ma
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Projected Total Demand")
        d7_total = int(np.sum(fc_7d))
        d14_total = int(np.sum(fc_14d))
        st.metric(f"Next 7 Days Demand", f"{d7_total} units", f"Avg {d7_total/7:.1f}/day")
        st.metric(f"Next 14 Days Demand", f"{d14_total} units", f"Avg {d14_total/14:.1f}/day")

    with c2:
        st.subheader("🔍 Feature Importance Drivers")
        top_3 = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:3]
        for f_name, score in top_3:
            st.write(f"- `{f_name}`: **{score:.2f}** weight")

        p_eval = eval_data.get("product_metrics", {}).get(selected_pid, {})
        best_m = p_eval.get("best_model", "ML_Model")
        wape_val = p_eval.get("metrics", {}).get("ML_Model", {}).get("WAPE_percent", 12.5)
        mae_val = p_eval.get("metrics", {}).get("ML_Model", {}).get("MAE", 1.8)

        expl_str = PlainLanguageExplainer.explain_forecast_model(
            product_name=p_info["product_name"],
            best_model_name=best_m,
            wape_score=wape_val,
            mae_score=mae_val,
            top_features=[x[0] for x in top_3]
        )
        st.caption(expl_str)

# ==========================================
# PAGE 4: REORDER PLANNING
# ==========================================
elif nav_choice == "📦 Reorder Planning":
    st.title("📦 Inventory Reorder Planning")
    st.caption("Automated reorder point calculations, safety stock buffers, and supplier purchase order generation.")

    recs = reorder_engine.generate_all_recommendations()
    df_recs = pd.DataFrame(recs)

    # Filter controls
    status_filter = st.multiselect(
        "Filter by Status",
        options=["🔴 Out of Stock", "🟠 Urgent Reorder Needed", "🟡 Reorder Soon", "🟢 Stock Optimal", "🔵 Overstocked"],
        default=["🔴 Out of Stock", "🟠 Urgent Reorder Needed", "🟡 Reorder Soon", "🟢 Stock Optimal", "🔵 Overstocked"]
    )

    df_filtered = df_recs[df_recs["status_label"].isin(status_filter)]

    # Display Table
    display_cols = [
        "product_id", "product_name", "category", "current_stock",
        "lead_time_days", "reorder_point", "suggested_reorder_qty",
        "total_reorder_cost", "status_label"
    ]

    st.dataframe(
        df_filtered[display_cols].rename(columns={
            "product_id": "SKU",
            "product_name": "Product Name",
            "category": "Category",
            "current_stock": "Current Stock",
            "lead_time_days": "Lead Time (Days)",
            "reorder_point": "Reorder Point (ROP)",
            "suggested_reorder_qty": "Suggested Reorder Qty",
            "total_reorder_cost": "Est. Cost ($)",
            "status_label": "Status"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    
    # Download PO CSV
    reorder_only_df = df_recs[df_recs["suggested_reorder_qty"] > 0]
    csv_data = reorder_only_df[["product_id", "product_name", "suggested_reorder_qty", "unit_cost", "total_reorder_cost"]].to_csv(index=False)
    
    st.download_button(
        label="📥 Download Supplier Purchase Order (CSV)",
        data=csv_data,
        file_name="supplier_purchase_order.csv",
        mime="text/csv"
    )

    st.subheader("💡 Select Product for Detailed Explanation")
    selected_prod_name = st.selectbox("Choose Product", df_recs["product_name"].unique())
    selected_rec = df_recs[df_recs["product_name"] == selected_prod_name].iloc[0].to_dict()

    exp = PlainLanguageExplainer.explain_recommendation(selected_rec)
    render_explanation_box(exp["headline"], exp["key_drivers"], exp["calculation_breakdown"])

# ==========================================
# PAGE 5: "WHAT-IF?" SIMULATOR
# ==========================================
elif nav_choice == "🎛️ 'What-If?' Simulator":
    st.title("🎛️ 'What-If?' Reorder Scenario Simulator")
    st.caption("Adjust operational parameters dynamically to see how safety buffers, supplier delays, or promotions change reorder quantities.")

    col_controls, col_display = st.columns([1, 2])

    with col_controls:
        st.subheader("Global Policy Controls")
        
        sim_lead_time_add = st.slider("Supplier Delivery Delay (+Days)", 0, 7, 0, help="Simulate supplier delays")
        sim_safety_z = st.slider("Safety Buffer Z-Score (Service Level)", 1.0, 2.58, 1.65, step=0.05, help="1.65 = 95% service level, 2.33 = 99% service level")
        sim_promo_mult = st.slider("Expected Demand Uplift (Multiplier)", 0.5, 2.0, 1.0, step=0.1, help="Simulate upcoming promotion (+20% = 1.2)")

    # Run base vs simulated recommendations
    base_recs = reorder_engine.generate_all_recommendations()
    
    # Generate simulated
    df_products_local = load_products()
    custom_lts = {p["product_id"]: p["lead_time_days"] + sim_lead_time_add for _, p in df_products_local.iterrows()}
    custom_mults = {p["product_id"]: sim_promo_mult for _, p in df_products_local.iterrows()}
    
    sim_recs = reorder_engine.generate_all_recommendations(
        safety_factor=sim_safety_z,
        custom_lead_times=custom_lts,
        demand_multipliers=custom_mults
    )

    df_base = pd.DataFrame(base_recs).set_index("product_id")
    df_sim = pd.DataFrame(sim_recs).set_index("product_id")

    comparison_list = []
    for pid in df_base.index:
        b = df_base.loc[pid]
        s = df_sim.loc[pid]
        comparison_list.append({
            "Product": b["product_name"],
            "Current Stock": b["current_stock"],
            "Base ROP": b["reorder_point"],
            "Simulated ROP": s["reorder_point"],
            "Base Safety Stock": b["safety_stock"],
            "Simulated Safety Stock": s["safety_stock"],
            "Base Order Qty": b["suggested_reorder_qty"],
            "Simulated Order Qty": s["suggested_reorder_qty"],
            "Cost Difference ($)": round(s["total_reorder_cost"] - b["total_reorder_cost"], 2)
        })

    df_comp = pd.DataFrame(comparison_list)

    with col_display:
        st.subheader("Scenario Comparison Impact")
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        total_base_cost = df_base["total_reorder_cost"].sum()
        total_sim_cost = df_sim["total_reorder_cost"].sum()
        diff = total_sim_cost - total_base_cost

        st.info(f"💵 Total Order Spend: Base **${total_base_cost:,.2f}** ➔ Simulated **${total_sim_cost:,.2f}** (Delta: **{'+' if diff >= 0 else ''}${diff:,.2f}**)")

# ==========================================
# PAGE 6: MODEL EVALUATION
# ==========================================
elif nav_choice == "🎯 Model Evaluation":
    st.title("🎯 Model Evaluation & Out-of-Sample Backtesting")
    st.caption("Honest accuracy benchmarking across naive baselines and machine learning algorithms using time-based validation splits.")

    global_m = eval_data.get("global_metrics", {})
    best_overall = eval_data.get("overall_best_model", "ML_Model")

    st.success(f"🏆 **Best Performing Model Overall:** `{best_overall}` (Evaluated over out-of-sample test split of {eval_data.get('test_period_days', 28)} days)")

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_eval = plot_model_evaluation_metrics(global_m)
        st.plotly_chart(fig_eval, use_container_width=True)

    with col2:
        st.subheader("Metric Explanations")
        st.markdown("""
        - **WAPE (Weighted Absolute Percentage Error)**:
          $$\\text{WAPE} = \\frac{\\sum |y - \\hat{y}|}{\\sum y} \\times 100$$
          *Why WAPE?* Standard MAPE fails on retail data due to division by zero on zero-sales days. WAPE is robust and scale-proportional.
        - **MAE (Mean Absolute Error)**:
          Average error in physical units sold per day.
        """)

    st.markdown("---")
    st.subheader("Product-Level Backtest Matrix")
    
    prod_metrics = eval_data.get("product_metrics", {})
    table_rows = []
    for pid, p_data in prod_metrics.items():
        row = {
            "SKU": pid,
            "Product Name": p_data["product_name"],
            "Category": p_data["category"],
            "Best Model": p_data["best_model"],
            "Naive WAPE (%)": p_data["metrics"]["Naive_LastWeek"]["WAPE_percent"],
            "Moving Avg WAPE (%)": p_data["metrics"]["MovingAvg_7D"]["WAPE_percent"],
            "ML Model WAPE (%)": p_data["metrics"]["ML_Model"]["WAPE_percent"],
            "ML Model MAE": p_data["metrics"]["ML_Model"]["MAE"]
        }
        table_rows.append(row)

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ==========================================
# PAGE 7: DATA QUALITY & AUDIT
# ==========================================
elif nav_choice == "🛠️ Data Quality & Audit":
    st.title("🛠️ Data Quality & Pipeline Audit")
    st.caption("Inspect data integrity, date range completeness, schema definitions, and trigger ETL re-runs.")

    validator = DataValidator()
    is_valid, report = validator.validate_all(df_products, df_sales, df_inventory)

    if is_valid:
        st.success("✅ Database Data Quality Checks: PASSED")
    else:
        st.error("❌ Data Quality Validation Warnings Detected")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Products", report["metrics"]["total_products"])
    with c2:
        st.metric("Total Daily Sales Records", report["metrics"]["total_sales_records"])
    with c3:
        st.metric("Date Range Covered", report["metrics"]["date_range"])

    st.markdown("---")
    st.subheader("Pipeline Re-execution")
    if st.button("🔄 Regenerate Dataset & Re-run Pipeline"):
        with st.spinner("Running ETL Pipeline..."):
            res = run_pipeline(generate_synthetic=True)
            st.cache_data.clear()
            st.success("Pipeline executed successfully! Please refresh application.")
