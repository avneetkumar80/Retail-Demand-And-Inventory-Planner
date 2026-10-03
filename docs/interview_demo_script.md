# 🎙️ StockWise: 3-Minute Interview Demo Script

This script is structured for technical interviewers (Data Science Managers, Senior Engineers, Product Leads) to demonstrate end-to-end data engineering, machine learning rigor, and business impact.

---

## ⏱️ Minute 1: Problem Definition & Business Context (The "Why")

> *"Hi! Today I’m excited to show you **StockWise**, a portfolio project I built to solve a critical operational pain point for small retailers and cafés: balancing stockout risks against holding costs.*
>
> *Most small retail managers rely on manual intuition or basic static thresholds to place inventory orders. This leads to either spoilage of perishable goods or lost revenue during demand spikes. StockWise creates a complete decision loop:*
>
> **Sales History ➔ Validated Data Pipeline ➔ Out-of-Sample Machine Learning Forecast ➔ Reorder Point Math ➔ Plain-Language Explanation**."*

---

## ⏱️ Minute 2: Engineering Architecture & ML Evaluation (The "How")

> *"Rather than building just another static dashboard, I structured StockWise with modular software design:*
>
> 1. **Data Pipeline & SQLite Database**: An ETL pipeline validates schema constraints, dtypes, and cleans missing calendar dates with zero-fill handling to ensure continuous time-series integrity.
> 2. **Leakage-Free Backtest Evaluation**: To prove model accuracy without inflating metrics, I built a rolling out-of-sample temporal validation engine (reserving the last 28 days). I benchmarked **Same Day Last Week**, **7-Day Moving Average**, **Simple Exponential Smoothing**, and **Croston's Intermittent Method** against a **LightGBM Regressor**.
> 3. **Metrics**: I evaluated models using **WAPE (Weighted Absolute Percentage Error)** rather than standard MAPE, because MAPE suffers from division-by-zero errors on days with zero sales. In backtesting, our LightGBM model reduced forecast error from **21.2% WAPE (naive baseline)** down to **17.7% WAPE**."*

---

## ⏱️ Minute 3: Interactive Demo & Explainability (The Impact)

> *"Let me show you the live Streamlit application:*
>
> - **Reorder Planning**: Here on the Reorder Planning page, StockWise calculates the **Reorder Point (ROP)** by combining Demand during Supplier Lead Time with a statistical **Safety Buffer** derived from demand volatility:
>   $$\\text{ROP} = D_{\\text{lead time}} + Z \\times \\sigma_d \\times \\sqrt{L}$$
>   It rounds recommended order quantities up to supplier **Minimum Order Quantities (MOQs)** and exports purchase orders to CSV.
> - **Plain-Language Explanations**: For every SKU, StockWise generates plain-English narratives explaining *why* an order is needed, breaking down safety stock math so non-technical managers can trust the recommendations.
> - **What-If Scenario Sandbox**: Managers can simulate supplier delivery delays or promotional demand surges (+20%) to visualize capital impact on inventory spend in real time."*

---

## 💡 Key Talking Points for Q&A

- **Q: How do you handle cold-start or low-volume products?**
  - *A: StockWise detects low volume / intermittent items and automatically falls back to Croston’s method or longer-window moving averages to prevent over-fitting.*
- **Q: How do you ensure no data leakage?**
  - *A: Lag features (`lag_7`, `lag_14`) and rolling statistics are calculated strictly on past data prior to the cutoff date. Multi-step forecasts use recursive auto-regression.*
