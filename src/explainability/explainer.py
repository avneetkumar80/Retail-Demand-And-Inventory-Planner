from typing import Dict, Any, List

class PlainLanguageExplainer:
    """
    Generates structured, human-readable explanations for demand forecasts
    and reorder recommendations to assist small business operators.
    """

    @staticmethod
    def explain_recommendation(rec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates a reorder recommendation object into a plain-language summary
        with key decision drivers, mathematical breakdown, and risk notes.
        """
        pname = rec["product_name"]
        stock = rec["current_stock"]
        lt_days = rec["lead_time_days"]
        lt_demand = rec["lead_time_demand"]
        safety_stock = rec["safety_stock"]
        rop = rec["reorder_point"]
        suggested_qty = rec["suggested_reorder_qty"]
        moq = rec["min_order_qty"]
        cost = rec["total_reorder_cost"]
        status_code = rec["status_code"]
        mult = rec["demand_multiplier"]

        drivers = []
        narrative_parts = []

        # 1. Primary Recommendation Statement
        if status_code == "CRITICAL_OUT_OF_STOCK":
            summary = f"🚨 **{pname}** is completely OUT OF STOCK! Place an immediate order of **{suggested_qty} units** (${cost:,.2f}) to avoid ongoing revenue loss."
            drivers.append(f"Current inventory is 0 units against an expected lead time demand of {lt_demand} units.")
        elif status_code == "URGENT_REORDER":
            summary = f"⚠️ **Urgent Action Required:** Current stock of **{stock} units** has breached the safety buffer of **{safety_stock} units**. Order **{suggested_qty} units** immediately."
            drivers.append(f"Stock is below the emergency buffer level ({stock} ≤ {safety_stock}). Supplier lead time is {lt_days} days.")
        elif status_code == "REORDER_SOON":
            summary = f"📦 **Reorder Advisory:** Current stock (**{stock} units**) is below the Reorder Point of **{rop} units**. We recommend ordering **{suggested_qty} units** soon."
            drivers.append(f"Stock level ({stock}) is insufficient to cover lead time demand ({lt_demand} units) plus safety stock ({safety_stock} units).")
        elif status_code == "OVERSTOCKED":
            summary = f"🔵 **Overstocked Alert:** Current stock (**{stock} units**) significantly exceeds the 14-day reorder buffer ({rop} units). Do NOT reorder now."
            drivers.append(f"Excess inventory detected. Holding cost optimization recommended.")
        else:
            summary = f"✅ **Optimal Inventory:** Current stock (**{stock} units**) is healthy and comfortably above the Reorder Point (**{rop} units**). No action needed."
            drivers.append(f"Stock ({stock} units) covers expected demand through supplier delivery period.")

        # 2. Key Driver Highlights
        if mult > 1.0:
            drivers.append(f"Demand is boosted by a **+{int((mult-1.0)*100)}% promotional/seasonal uplift** scenario.")
        elif mult < 1.0:
            drivers.append(f"Demand is adjusted down by **-{int((1.0-mult)*100)}%** for slow period simulation.")

        if suggested_qty > 0 and (rop - stock) < moq:
            drivers.append(f"Order quantity was rounded up from {int(max(0, rop - stock))} to the supplier's **Minimum Order Quantity (MOQ) of {moq} units**.")

        # 3. Transparent Calculation Breakdown
        breakdown_text = (
            f"**Calculation Logic:**\n"
            f"- **Expected Demand during Supplier Delivery ({lt_days} days):** {lt_demand} units\n"
            f"- **Safety Buffer (Z = {rec['safety_factor']}):** +{safety_stock} units\n"
            f"- **Target Reorder Point (ROP):** {lt_demand} + {safety_stock} = **{rop} units**\n"
            f"- **Inventory Deficit:** {rop} - {stock} = {max(0, rop - stock)} units (rounded to MOQ of {moq} = **{suggested_qty} units**)"
        )

        return {
            "headline": summary,
            "key_drivers": drivers,
            "calculation_breakdown": breakdown_text,
            "status_code": status_code
        }

    @staticmethod
    def explain_forecast_model(
        product_name: str,
        best_model_name: str,
        wape_score: float,
        mae_score: float,
        top_features: List[str]
    ) -> str:
        """
        Explains forecast model selection and accuracy in plain language.
        """
        explanation = (
            f"For **{product_name}**, the **{best_model_name}** model delivered the highest historical out-of-sample accuracy "
            f"with a **WAPE of {wape_score:.1f}%** and an average daily error of **±{mae_score:.1f} units**.\n\n"
            f"**Primary Factors Influencing Forecast:**\n"
        )
        for feat in top_features[:3]:
            explanation += f"- `{feat}` (historical pattern indicator)\n"
            
        explanation += "\n*Note: Forecasts represent statistical projections based on historical patterns and should be combined with manager judgment during unexpected events.*"
        return explanation
