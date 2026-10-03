import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List

def render_metric_card(label: str, value: str, subtext: str = "", color_theme: str = "indigo"):
    """Renders styled metric card component."""
    html_code = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {'<div class="metric-subtext">' + subtext + '</div>' if subtext else ''}
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def render_explanation_box(headline: str, key_drivers: List[str], breakdown: str):
    """Renders plain language explanation box with decision drivers."""
    drivers_html = "".join([f"<li>{d}</li>" for d in key_drivers])
    
    st.markdown(f"""
    <div class="explanation-box">
        <div class="explanation-title">💡 Plain Language Reorder Breakdown</div>
        <p style="font-size: 1.05rem; margin-bottom: 0.75rem;">{headline}</p>
        <div style="margin-bottom: 0.75rem;">
            <strong>Key Decision Factors:</strong>
            <ul style="margin-top: 0.25rem; padding-left: 1.25rem;">
                {drivers_html}
            </ul>
        </div>
        <div style="font-size: 0.9rem; color: #cbd5e1; white-space: pre-line;">
            {breakdown}
        </div>
    </div>
    """, unsafe_allow_html=True)

def plot_demand_forecast(
    df_history: pd.DataFrame,
    product_name: str,
    forecast_7d: np.ndarray,
    forecast_14d: np.ndarray,
    baseline_naive: np.ndarray,
    baseline_ma: np.ndarray
) -> go.Figure:
    """
    Plots historical time series alongside 7-day and 14-day ML forecast vs Baselines.
    """
    df_hist = df_history.tail(60).copy()
    df_hist["date"] = pd.to_datetime(df_hist["date"])
    last_date = df_hist["date"].max()
    
    # Generate future dates
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 15)]
    
    fig = go.Figure()
    
    # 1. Historical Actual Sales
    fig.add_trace(go.Scatter(
        x=df_hist["date"],
        y=df_hist["units_sold"],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='#38bdf8', width=2),
        marker=dict(size=4)
    ))
    
    # 2. ML Forecast (14-day horizon)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast_14d,
        mode='lines+markers',
        name='ML Forecast (LightGBM)',
        line=dict(color='#818cf8', width=3, dash='solid'),
        marker=dict(size=6, symbol='diamond')
    ))

    # 3. Naive Baseline (Same Day Last Week)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=baseline_naive,
        mode='lines',
        name='Baseline: Same Day Last Wk',
        line=dict(color='#f59e0b', width=1.5, dash='dash')
    ))

    # 4. Moving Average Baseline
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=baseline_ma,
        mode='lines',
        name='Baseline: 7-Day Moving Avg',
        line=dict(color='#a855f7', width=1.5, dash='dot')
    ))

    # Highlight 7-day horizon boundary
    fig.add_vline(
        x=future_dates[6].timestamp() * 1000,
        line_width=1, line_dash="dash", line_color="#94a3b8"
    )
    
    fig.update_layout(
        title=dict(text=f"Demand Forecast for {product_name} (Next 14 Days)", font=dict(size=16, color="#f8fafc")),
        xaxis=dict(title="Date", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title="Units Sold / Forecast", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.8)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=420
    )
    
    return fig

def plot_model_evaluation_metrics(global_metrics: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Renders bar chart comparing WAPE (%) and MAE across models.
    """
    models = list(global_metrics.keys())
    wapes = [global_metrics[m]["WAPE_percent"] for m in models]
    maes = [global_metrics[m]["MAE"] for m in models]

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=models,
        y=wapes,
        name='WAPE (%)',
        marker_color='#6366f1',
        text=[f"{v}%" for v in wapes],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        x=models,
        y=maes,
        name='MAE (Units)',
        marker_color='#38bdf8',
        text=[f"±{v}" for v in maes],
        textposition='auto'
    ))

    fig.update_layout(
        title=dict(text="Out-of-Sample Backtest Accuracy (Lower is Better)", font=dict(size=16, color="#f8fafc")),
        barmode='group',
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.8)',
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Error Score", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380
    )
    
    return fig
