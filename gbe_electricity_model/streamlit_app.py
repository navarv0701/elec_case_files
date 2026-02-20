"""
GBE Electricity Trading - Streamlit Interactive Dashboard.

Usage:
    streamlit run streamlit_app.py

Sidebar: manual inputs for weather, market prices, positions.
Main area: 4 tabs (Producer, Distributor, Trader, Team Overview).
All existing optimizers and models reused without modification.
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd

from state.game_state import GameState, RAEBulletin, FactoryTender
from optimizer.producer_optimizer import ProducerOptimizer
from optimizer.distributor_optimizer import DistributorOptimizer
from optimizer.trader_optimizer import TraderOptimizer
from optimizer.team_coordinator import TeamCoordinator
from models.demand import demand_contracts, DemandForecaster
from models.production import solar_production, gas_profit_per_plant, gas_cost_per_elec_contract
from models.pricing import PricingModel
from ui.st_components import (
    inject_css, action_table, demand_chart, production_table,
    alert_box, penalty_warning,
)
from config import (
    CUSTOMER_PRICE_PER_MWH, DISPOSAL_PENALTY, SHORTFALL_PENALTY,
    ELEC_SPOT_CONTRACT_SIZE, ELEC_FWD_CONTRACT_SIZE,
    NG_TO_ELEC_RATIO, MAX_GAS_PLANTS, TICKS_PER_DAY,
)


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="GBE Electricity Trading",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


# ============================================================
# Build GameState from Sidebar Inputs
# ============================================================

def build_game_state() -> GameState:
    """Construct a GameState from st.session_state sidebar inputs."""
    state = GameState()

    # Time
    day = st.session_state.get("day", 1)
    tick_in_day = st.session_state.get("tick_in_day", 90)
    state.current_day = day
    state.current_tick = (day - 1) * TICKS_PER_DAY + tick_in_day
    state.case_status = "ACTIVE"

    # Weather
    sunshine = st.session_state.get("sunshine", 8.0)
    temperature = st.session_state.get("temperature", 28.0)
    forecast_num = st.session_state.get("forecast_num", 1)
    next_day = day + 1

    # Add forecasts for the next day
    for i in range(1, forecast_num + 1):
        state.add_sunshine_forecast(next_day, sunshine, i)
        state.add_temperature_forecast(next_day, temperature, i)

    # Market Prices
    state.elec_fwd_bid = st.session_state.get("elec_fwd_bid", 38.0)
    state.elec_fwd_ask = st.session_state.get("elec_fwd_ask", 40.0)
    state.elec_fwd_last = (state.elec_fwd_bid + state.elec_fwd_ask) / 2
    state.ng_bid = st.session_state.get("ng_bid", 4.50)
    state.ng_ask = st.session_state.get("ng_ask", 4.75)
    state.ng_last = (state.ng_bid + state.ng_ask) / 2

    # RAE Bulletin
    rae_low = st.session_state.get("rae_low", 0.0)
    rae_high = st.session_state.get("rae_high", 0.0)
    if rae_low > 0 and rae_high > 0:
        state.add_rae_bulletin(RAEBulletin(
            day=next_day, price_low=rae_low, price_high=rae_high,
            volume_buy=100, volume_sell=100,
            bulletin_number=1, tick_received=state.current_tick,
        ))

    # Positions
    state.elec_f_position = st.session_state.get("elec_f_pos", 0)
    state.ng_position = st.session_state.get("ng_pos", 0)
    elec_day_pos = st.session_state.get("elec_day_pos", 0)
    if elec_day_pos != 0:
        state.elec_day_positions[f"ELEC-day{day}"] = elec_day_pos

    # Gas production (from previous day's NG purchase)
    gas_plants_active = st.session_state.get("gas_plants_active", 0)
    state.gas_production[next_day] = gas_plants_active

    return state


# ============================================================
# Sidebar
# ============================================================

def render_sidebar():
    """Render all sidebar inputs."""
    st.sidebar.title("GBE Electricity Model")

    # Time
    st.sidebar.header("Simulation Time")
    st.sidebar.number_input("Day (1-5)", min_value=1, max_value=5, value=2, key="day")
    st.sidebar.slider("Tick within day", min_value=0, max_value=179, value=90, key="tick_in_day")

    ticks_left = TICKS_PER_DAY - st.session_state.get("tick_in_day", 90)
    st.sidebar.caption(f"{ticks_left} ticks ({ticks_left}s) remaining in day")

    # Weather
    st.sidebar.header("Weather Forecast")
    st.sidebar.number_input("Sunshine (hours)", min_value=0.0, max_value=16.0,
                            value=8.0, step=0.5, key="sunshine")
    st.sidebar.number_input("Temperature (C)", min_value=0.0, max_value=50.0,
                            value=28.0, step=0.5, key="temperature")
    st.sidebar.selectbox("Forecast #", options=[1, 2, 3], index=0, key="forecast_num",
                         help="1=Initial, 2=Noon update, 3=Final (most accurate)")

    # Market Prices
    st.sidebar.header("Market Prices")
    col1, col2 = st.sidebar.columns(2)
    col1.number_input("ELEC-F Bid", min_value=0.0, value=38.0, step=1.0, key="elec_fwd_bid")
    col2.number_input("ELEC-F Ask", min_value=0.0, value=40.0, step=1.0, key="elec_fwd_ask")
    col1.number_input("NG Bid", min_value=0.0, value=4.50, step=0.25, key="ng_bid")
    col2.number_input("NG Ask", min_value=0.0, value=4.75, step=0.25, key="ng_ask")

    st.sidebar.subheader("RAE Bulletin")
    col1, col2 = st.sidebar.columns(2)
    col1.number_input("RAE Low", min_value=0.0, value=0.0, step=1.0, key="rae_low",
                      help="Leave at 0 if no bulletin received yet")
    col2.number_input("RAE High", min_value=0.0, value=0.0, step=1.0, key="rae_high")

    # Positions
    st.sidebar.header("Positions")
    st.sidebar.number_input("ELEC-F Net Position", min_value=-60, max_value=60,
                            value=0, step=1, key="elec_f_pos",
                            help="+ = long (bought), - = short (sold)")
    st.sidebar.number_input("NG Net Position", min_value=-80, max_value=80,
                            value=0, step=8, key="ng_pos")
    st.sidebar.number_input("ELEC-dayX Net Position", min_value=-300, max_value=300,
                            value=0, step=1, key="elec_day_pos",
                            help="Position in today's ELEC-dayX")

    # Gas production
    st.sidebar.header("Production")
    st.sidebar.number_input("Gas Plants Active (for tomorrow)", min_value=0,
                            max_value=MAX_GAS_PLANTS, value=0, step=1,
                            key="gas_plants_active",
                            help="Number of gas plants running (from NG bought today)")


# ============================================================
# Producer Tab
# ============================================================

def render_producer_tab(state: GameState):
    """Producer role view."""
    optimizer = ProducerOptimizer()
    schedule = optimizer.get_production_schedule(state)
    recs = optimizer.optimize(state)
    next_day = state.current_day + 1

    # Key metrics
    st.subheader(f"Production for Day {next_day}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Solar Output", f"{schedule['solar_contracts']:.0f} contracts",
              f"{schedule['solar_contracts'] * 100:,.0f} MWh")
    c2.metric("Gas Plants", f"{schedule['gas_plants']}",
              f"{'Profitable' if schedule['gas_is_profitable'] else 'NOT Profitable'}")
    c3.metric("Total Production", f"{schedule['total_production']:.0f} contracts")

    disposal = schedule['surplus_contracts']
    if disposal > 0:
        c4.metric("Disposal Risk", f"${disposal * DISPOSAL_PENALTY:,.0f}",
                  f"{disposal:.0f} surplus contracts", delta_color="inverse")
    else:
        deficit = abs(disposal) if disposal < 0 else 0
        c4.metric("Supply Gap", f"{deficit:.0f} contracts",
                  "Team needs to buy from market" if deficit > 0 else "Balanced")

    # Production schedule
    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Production Schedule")
        production_table(schedule)

    with col_right:
        st.subheader("NG Purchase Analysis")
        ng_price = state.ng_ask if state.ng_ask > 0 else state.ng_last
        elec_price = state.elec_fwd_last if state.elec_fwd_last > 0 else 40.0

        if ng_price > 0:
            cost_per_plant = gas_cost_per_elec_contract(ng_price)
            profit = gas_profit_per_plant(ng_price, elec_price)
            revenue_per_plant = elec_price * ELEC_SPOT_CONTRACT_SIZE

            data = {
                "Metric": [
                    "NG Price",
                    "Cost per ELEC contract",
                    "ELEC Price (est.)",
                    "Revenue per ELEC contract",
                    "Profit per Plant",
                    "NG to Buy (8 per plant)",
                ],
                "Value": [
                    f"${ng_price:.2f}/MMBtu",
                    f"${cost_per_plant:,.0f}",
                    f"${elec_price:.0f}/MWh",
                    f"${revenue_per_plant:,.0f}",
                    f"${profit:,.0f}",
                    f"{schedule['ng_to_buy']:.0f} contracts",
                ],
            }
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

            if profit > 0:
                alert_box(f"Gas is PROFITABLE: ${profit:,.0f}/plant", "success")
            else:
                alert_box(f"Gas is UNPROFITABLE: ${profit:,.0f}/plant - rely on solar", "warning")
        else:
            st.info("No NG price available")

    # Disposal warning
    if disposal > 0:
        penalty_warning(disposal * DISPOSAL_PENALTY)

    # Recommendations
    st.divider()
    producer_recs = [r for r in recs if r.quantity and r.quantity > 0]
    action_table(producer_recs, title="Producer Actions")

    # Recommendation text
    st.caption(schedule['recommendation'])


# ============================================================
# Distributor Tab
# ============================================================

def render_distributor_tab(state: GameState):
    """Distributor role view."""
    optimizer = DistributorOptimizer()
    next_day = state.current_day + 1
    demand_info = optimizer.get_demand_forecast(state, next_day)
    recs = optimizer.optimize(state)

    # Key metrics
    st.subheader(f"Demand Forecast for Day {next_day}")
    c1, c2, c3, c4 = st.columns(4)

    demand_val = demand_info['demand_contracts']
    c1.metric("Expected Demand", f"{demand_val:.0f} contracts",
              f"Confidence: {demand_info['confidence']}")

    # Procurement status
    fwd_bought = max(0, state.elec_f_position)
    fwd_in_spot = fwd_bought * 5
    still_needed = max(0, demand_val - fwd_in_spot)
    c2.metric("Procured (fwd equiv)", f"{fwd_in_spot:.0f} contracts",
              f"{fwd_bought} ELEC-F contracts")

    if still_needed > 0:
        c3.metric("Still Needed", f"{still_needed:.0f} contracts",
                  f"-${still_needed * SHORTFALL_PENALTY:,.0f} if unmet", delta_color="inverse")
    else:
        surplus = fwd_in_spot - demand_val
        c3.metric("Surplus", f"{surplus:.0f} contracts",
                  f"-${surplus * DISPOSAL_PENALTY:,.0f} if unsold" if surplus > 0 else "Exact match",
                  delta_color="inverse" if surplus > 0 else "off")

    revenue = demand_val * CUSTOMER_PRICE_PER_MWH * ELEC_SPOT_CONTRACT_SIZE
    c4.metric("Customer Revenue", f"${revenue:,.0f}",
              f"@ ${CUSTOMER_PRICE_PER_MWH}/MWh")

    # Demand curve
    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Demand vs Temperature")
        temp = demand_info.get('temperature')
        demand_chart(current_temp=temp)

    with col_right:
        st.subheader("Demand Range")

        demand_low = demand_info.get('demand_low', 0)
        demand_high = demand_info.get('demand_high', 0)
        demand_exact = demand_info['demand_contracts']

        range_data = {
            "Scenario": ["Low (5th %ile)", "Expected", "High (95th %ile)"],
            "Demand (contracts)": [
                f"{demand_low:.0f}",
                f"{demand_exact:.0f}",
                f"{demand_high:.0f}",
            ],
            "MWh": [
                f"{demand_low * 100:,.0f}",
                f"{demand_exact * 100:,.0f}",
                f"{demand_high * 100:,.0f}",
            ],
            "Revenue": [
                f"${demand_low * CUSTOMER_PRICE_PER_MWH * ELEC_SPOT_CONTRACT_SIZE:,.0f}",
                f"${demand_exact * CUSTOMER_PRICE_PER_MWH * ELEC_SPOT_CONTRACT_SIZE:,.0f}",
                f"${demand_high * CUSTOMER_PRICE_PER_MWH * ELEC_SPOT_CONTRACT_SIZE:,.0f}",
            ],
        }
        st.dataframe(pd.DataFrame(range_data), use_container_width=True, hide_index=True)

        # Newsvendor optimal
        if temp is not None:
            temp_std = state.temperature_uncertainty(next_day)
            pricing = PricingModel()
            spot_est = pricing.estimate_spot_price(next_day, state)
            forecaster = DemandForecaster()
            for i in range(demand_info.get('num_updates', 1)):
                forecaster.update(temp)
            newsvendor = forecaster.optimal_procurement(temp, temp_std, spot_est['estimate'])

            st.subheader("Optimal Procurement (Newsvendor)")
            nv1, nv2 = st.columns(2)
            nv1.metric("Target Quantity", f"{newsvendor['target_contracts']} contracts")
            nv2.metric("Critical Ratio", f"{newsvendor['critical_ratio']:.2f}")
            st.caption(
                f"Underage cost: ${newsvendor['underage_cost']:,.0f}/contract | "
                f"Overage cost: ${newsvendor['overage_cost']:,.0f}/contract"
            )

    # Recommendations
    st.divider()
    dist_recs = [r for r in recs if r.quantity and r.quantity > 0]
    action_table(dist_recs, title="Distributor Actions")


# ============================================================
# Trader Tab
# ============================================================

def render_trader_tab(state: GameState):
    """Trader role view."""
    optimizer = TraderOptimizer()
    pricing = PricingModel()
    recs = optimizer.optimize(state)
    day = state.current_day
    next_day = day + 1

    # Key metrics
    st.subheader("Trading Dashboard")
    c1, c2, c3, c4 = st.columns(4)

    # Forward-Spot Spread
    premium = pricing.forward_vs_spot_premium(state, next_day)
    spread_str = f"${premium['premium_per_mwh']:.2f}/MWh"
    signal = premium['action_signal']
    c1.metric("Fwd-Spot Spread", spread_str,
              signal.replace("_", " "))

    # Arbitrage signal
    if signal == "SELL_FORWARD":
        c2.metric("Arbitrage", "SELL FORWARD",
                  f"+${premium['premium_per_mwh'] * ELEC_FWD_CONTRACT_SIZE:.0f}/contract")
    elif signal == "BUY_FORWARD":
        c2.metric("Arbitrage", "BUY FORWARD",
                  f"+${abs(premium['premium_per_mwh']) * ELEC_FWD_CONTRACT_SIZE:.0f}/contract")
    else:
        c2.metric("Arbitrage", "NEUTRAL", "No clear opportunity")

    # Open positions & penalty
    total_day_pos = sum(abs(v) for v in state.elec_day_positions.values())
    penalty_risk = total_day_pos * DISPOSAL_PENALTY
    c3.metric("Open Day Positions", f"{total_day_pos} contracts",
              f"${penalty_risk:,.0f} at risk" if penalty_risk > 0 else "Clean")

    ticks_left = state.ticks_remaining_in_day()
    c4.metric("Ticks Left", f"{ticks_left}",
              f"{ticks_left}s to day end")

    # Penalty warning
    if penalty_risk > 0 and ticks_left <= 60:
        if ticks_left <= 5:
            alert_box(f"CRITICAL: {total_day_pos} contracts open, {ticks_left}s left! "
                      f"Penalty: ${penalty_risk:,.0f}", "danger")
        elif ticks_left <= 20:
            alert_box(f"URGENT: Close {total_day_pos} contracts! {ticks_left}s left. "
                      f"Penalty: ${penalty_risk:,.0f}", "danger")
        else:
            alert_box(f"WARNING: {total_day_pos} open contracts, {ticks_left}s left. "
                      f"Plan closeout.", "warning")

    st.divider()

    # Tender Evaluator + Position Tracker
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Tender Evaluator")
        st.caption("Enter tender details to get instant accept/decline recommendation")

        t_action = st.selectbox("Factory wants to", ["BUY", "SELL"],
                                key="tender_action",
                                help="BUY = factory buys from you (you sell)")
        t_qty = st.number_input("Quantity (contracts)", min_value=1, max_value=100,
                                value=5, key="tender_qty")
        t_price = st.number_input("Tender Price ($/MWh)", min_value=0.0,
                                  value=40.0, step=1.0, key="tender_price")

        if st.button("Evaluate Tender", type="primary"):
            tender = FactoryTender(
                tender_id=999,
                action=t_action,
                ticker="ELEC-F",
                quantity=t_qty,
                price=t_price,
                expiration_tick=state.current_tick + 30,
            )
            result = optimizer._evaluate_tender(tender, state)

            if result['accept']:
                alert_box(
                    f"ACCEPT - Expected profit: ${result['expected_pnl']:,.0f}",
                    "success"
                )
            else:
                alert_box(
                    f"DECLINE - {result['decline_reason']}",
                    "danger"
                )

    with col_right:
        st.subheader("Position Closeout Tracker")

        if state.elec_day_positions:
            for ticker, pos in state.elec_day_positions.items():
                if pos != 0:
                    action = "SELL" if pos > 0 else "BUY"
                    st.markdown(
                        f"**{ticker}**: {pos:+d} contracts "
                        f"&rarr; {action} {abs(pos)} to close "
                        f"(penalty: ${abs(pos) * DISPOSAL_PENALTY:,.0f})"
                    )
        else:
            st.success("No open day positions - clean!")

        # Market data summary
        st.subheader("Market Snapshot")
        spot_est = pricing.estimate_spot_price(next_day, state)
        market_data = {
            "Metric": [
                "ELEC-F Bid / Ask",
                "NG Bid / Ask",
                "Expected Spot (Day " + str(next_day) + ")",
                "Spot Confidence",
                "ELEC-F Position",
            ],
            "Value": [
                f"${state.elec_fwd_bid:.0f} / ${state.elec_fwd_ask:.0f}",
                f"${state.ng_bid:.2f} / ${state.ng_ask:.2f}",
                f"${spot_est['estimate']:.1f}/MWh",
                spot_est['confidence'],
                f"{state.elec_f_position:+d}",
            ],
        }
        st.dataframe(pd.DataFrame(market_data), use_container_width=True, hide_index=True)

    # Recommendations
    st.divider()
    action_table(recs, title="Trader Actions")


# ============================================================
# Team Overview Tab
# ============================================================

def render_team_tab(state: GameState):
    """Team coordination view."""
    coordinator = TeamCoordinator()
    plan = coordinator.generate_team_plan(state)
    next_day = plan['next_day']

    # Key metrics
    st.subheader(f"Team Overview - Day {plan['day']} (Planning for Day {next_day})")
    c1, c2, c3, c4 = st.columns(4)

    supply = plan['supply']
    demand = plan['demand']
    balance = plan['balance']

    c1.metric("Total Supply", f"{supply['total_contracts']:.0f} contracts",
              f"Solar: {supply['solar_contracts']:.0f} + Gas: {supply['gas_contracts']}")
    c2.metric("Total Demand", f"{demand['demand_contracts']:.0f} contracts",
              f"Temp: {demand['temperature']:.1f}C")

    if balance['is_surplus']:
        c3.metric("Balance", f"+{balance['surplus']:.0f} SURPLUS",
                  "Sell excess on market", delta_color="normal")
    elif balance['is_deficit']:
        c3.metric("Balance", f"-{balance['deficit']:.0f} DEFICIT",
                  "Buy from market", delta_color="inverse")
    else:
        c3.metric("Balance", "BALANCED", "Supply ~ Demand")

    c4.metric("Internal Price", f"${plan['internal_price']:.0f}/MWh",
              f"Market mid: ${state.elec_fwd_last:.0f}")

    st.divider()

    # Role-specific plans
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Producer Plan")
        pp = plan['producer_plan']
        st.markdown(f"- **Sell Forward**: {pp['sell_forward_target']} ELEC-F contracts")
        st.markdown(f"- **Sell Spot**: {pp['sell_spot_target']:.0f} contracts")
        st.markdown(f"- **Target Price**: ${pp['target_price']}/MWh")
        st.markdown(f"- **Production**: {pp['production_total']:.0f} contracts total")

    with col2:
        st.subheader("Distributor Plan")
        dp = plan['distributor_plan']
        st.markdown(f"- **Buy Forward**: {dp['buy_forward_target']} ELEC-F contracts")
        st.markdown(f"- **Buy Spot Reserve**: {dp['buy_spot_reserve']:.0f} contracts")
        st.markdown(f"- **Target Price**: ${dp['target_price']}/MWh")
        st.markdown(f"- **Max Price**: ${dp['max_price']}/MWh")

    with col3:
        st.subheader("Trader Plan")
        tp = plan['trader_plan']
        st.markdown(f"- **Stance**: {tp['stance']}")
        st.markdown(f"- **Market Direction**: {tp['market_direction']}")
        st.markdown(f"- **Expected Spot**: ${tp['expected_spot']:.1f}/MWh")
        st.markdown(f"- **Arbitrage**: {'Active' if tp['arbitrage_active'] else 'Inactive'}")
        st.markdown(f"- **Internal Facilitation**: {'Yes' if tp['facilitate_internal'] else 'No'}")

    # Team exposure
    st.divider()
    exposure = plan['team_exposure']

    st.subheader("Team Exposure")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("ELEC-F", f"{exposure['forward_contracts']} contracts")
    e2.metric("NG", f"{exposure['ng_contracts']} contracts")
    e3.metric("ELEC-dayX", f"{exposure['elec_day_contracts']} contracts")
    e4.metric("Penalty at Risk", f"${exposure['penalty_at_risk']:,.0f}",
              delta_color="inverse" if exposure['penalty_at_risk'] > 0 else "off")

    if exposure['penalty_at_risk'] > 0:
        penalty_warning(exposure['penalty_at_risk'])

    # Full action queue
    st.divider()
    all_recs = []
    all_recs.extend(ProducerOptimizer().optimize(state))
    all_recs.extend(DistributorOptimizer().optimize(state))
    all_recs.extend(TraderOptimizer().optimize(state))

    urgency_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    all_recs.sort(key=lambda r: urgency_order.get(r.urgency, 3))

    action_table(all_recs, max_rows=15, title="All Team Actions (sorted by urgency)")


# ============================================================
# Main App
# ============================================================

def main():
    # Render sidebar
    render_sidebar()

    # Build state from inputs
    state = build_game_state()

    # Header
    day = state.current_day
    ticks_left = state.ticks_remaining_in_day()
    st.title(f"GBE Electricity Trading - Day {day}")

    header_cols = st.columns(4)
    header_cols[0].markdown(f"**Tick**: {state.current_tick} ({state.tick_within_day()}/{TICKS_PER_DAY})")
    header_cols[1].markdown(f"**Time Left**: {ticks_left}s")

    forecast_num = st.session_state.get("forecast_num", 1)
    forecast_labels = {1: "Initial", 2: "Noon", 3: "Final"}
    header_cols[2].markdown(f"**Forecast**: {forecast_labels.get(forecast_num, '?')}")

    sunshine = st.session_state.get("sunshine", 8.0)
    temp = st.session_state.get("temperature", 28.0)
    header_cols[3].markdown(f"**Weather**: {sunshine}h sun, {temp}C")

    # Tabs
    tab_producer, tab_distributor, tab_trader, tab_team = st.tabs(
        ["Producer", "Distributor", "Trader", "Team Overview"]
    )

    with tab_producer:
        render_producer_tab(state)

    with tab_distributor:
        render_distributor_tab(state)

    with tab_trader:
        render_trader_tab(state)

    with tab_team:
        render_team_tab(state)


if __name__ == "__main__":
    main()
