"""
Shared Streamlit UI components for the GBE Electricity Trading dashboard.
Reusable widgets: metric cards, action tables, charts, urgency badges.
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

from state.game_state import Recommendation
from models.demand import demand_contracts


# ============================================================
# CSS Injection - Large, scannable metrics
# ============================================================

CUSTOM_CSS = """
<style>
/* Make st.metric values large and bold */
[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}
/* Urgency badges */
.urgency-critical {
    background-color: #dc3545;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 0.85rem;
}
.urgency-high {
    background-color: #fd7e14;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 0.85rem;
}
.urgency-medium {
    background-color: #ffc107;
    color: #333;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 0.85rem;
}
.urgency-low {
    background-color: #6c757d;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.85rem;
}
/* Action colors */
.action-buy {
    color: #28a745;
    font-weight: bold;
}
.action-sell {
    color: #dc3545;
    font-weight: bold;
}
/* Big alert box */
.big-alert {
    font-size: 1.3rem;
    padding: 12px 20px;
    border-radius: 8px;
    margin: 8px 0;
    font-weight: bold;
}
.alert-danger {
    background-color: #f8d7da;
    color: #721c24;
    border: 2px solid #dc3545;
}
.alert-success {
    background-color: #d4edda;
    color: #155724;
    border: 2px solid #28a745;
}
.alert-warning {
    background-color: #fff3cd;
    color: #856404;
    border: 2px solid #ffc107;
}
</style>
"""


def inject_css():
    """Inject custom CSS for the dashboard."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# Urgency Badge
# ============================================================

def urgency_badge(urgency: str) -> str:
    """Return HTML for a colored urgency badge."""
    css_class = f"urgency-{urgency.lower()}"
    return f'<span class="{css_class}">{urgency}</span>'


def action_badge(action: str) -> str:
    """Return HTML for a colored action badge."""
    if action in ("BUY", "ACCEPT_TENDER"):
        return f'<span class="action-buy">{action}</span>'
    elif action in ("SELL", "DECLINE_TENDER"):
        return f'<span class="action-sell">{action}</span>'
    return action


# ============================================================
# Action Table
# ============================================================

def action_table(recs: list[Recommendation], max_rows: int = 12, title: str = "Recommended Actions"):
    """Render a color-coded action queue."""
    if not recs:
        st.info("No recommendations at this time.")
        return

    st.subheader(title)

    rows = []
    for r in recs[:max_rows]:
        price_str = f"${r.price:.0f}" if r.price else "MKT"
        pnl_str = f"${r.expected_pnl:,.0f}" if r.expected_pnl else "-"
        risk_str = f"${r.penalty_risk:,.0f}" if r.penalty_risk else "-"

        rows.append({
            "Urgency": r.urgency,
            "Action": r.action,
            "Ticker": r.ticker,
            "Qty": r.quantity,
            "Price": price_str,
            "Reason": r.reason,
            "E[PnL]": pnl_str,
            "Risk": risk_str,
        })

    df = pd.DataFrame(rows)

    # Style the dataframe
    def style_urgency(val):
        colors = {
            "CRITICAL": "background-color: #dc3545; color: white; font-weight: bold",
            "HIGH": "background-color: #fd7e14; color: white; font-weight: bold",
            "MEDIUM": "background-color: #ffc107; color: #333; font-weight: bold",
            "LOW": "background-color: #e9ecef; color: #333",
        }
        return colors.get(val, "")

    def style_action(val):
        if val in ("BUY", "ACCEPT_TENDER"):
            return "color: #28a745; font-weight: bold"
        elif val in ("SELL", "DECLINE_TENDER"):
            return "color: #dc3545; font-weight: bold"
        return ""

    styled = df.style.map(style_urgency, subset=["Urgency"])
    styled = styled.map(style_action, subset=["Action"])

    st.dataframe(styled, use_container_width=True, hide_index=True)


# ============================================================
# Demand Curve Chart
# ============================================================

def demand_chart(current_temp: float | None = None):
    """Render the demand curve with optional current temperature marker."""
    temps = np.arange(10, 46, 0.5)
    demands = [demand_contracts(t) for t in temps]

    chart_data = pd.DataFrame({"Temperature (C)": temps, "Demand (contracts)": demands})

    st.line_chart(chart_data, x="Temperature (C)", y="Demand (contracts)")

    if current_temp is not None:
        current_demand = demand_contracts(current_temp)
        st.markdown(
            f"**Current forecast**: {current_temp:.1f}C "
            f"&rarr; **{current_demand:.0f} contracts** "
            f"({current_demand * 100:,.0f} MWh)"
        )


# ============================================================
# Production Table
# ============================================================

def production_table(schedule: dict):
    """Render a production schedule summary."""
    data = {
        "Component": ["Solar", "Gas", "Total"],
        "Contracts": [
            f"{schedule['solar_contracts']:.0f}",
            f"{schedule['gas_contracts']}",
            f"{schedule['total_production']:.0f}",
        ],
        "MWh": [
            f"{schedule['solar_contracts'] * 100:,.0f}",
            f"{schedule['gas_contracts'] * 100:,.0f}",
            f"{schedule['total_production'] * 100:,.0f}",
        ],
        "Cost": [
            "$0",
            f"${schedule['gas_cost_total']:,.0f}",
            f"${schedule['gas_cost_total']:,.0f}",
        ],
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


# ============================================================
# Alert Boxes
# ============================================================

def alert_box(message: str, level: str = "warning"):
    """Render a large, visible alert box."""
    css_class = f"alert-{level}"
    st.markdown(f'<div class="big-alert {css_class}">{message}</div>', unsafe_allow_html=True)


def penalty_warning(penalty_risk: float):
    """Show penalty risk if nonzero."""
    if penalty_risk > 0:
        alert_box(
            f"PENALTY RISK: ${penalty_risk:,.0f} "
            f"({int(penalty_risk / 20_000)} contracts at $20,000 each)",
            level="danger"
        )
