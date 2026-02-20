"""
Distributor Optimizer - decides procurement strategy to meet customer demand.
Buys electricity from forward/spot markets, sells to customers at $70/MWh.
"""

from __future__ import annotations

import numpy as np

from state.game_state import GameState, Recommendation
from models.demand import DemandForecaster, demand_contracts
from models.pricing import PricingModel
from config import (
    ELEC_F_MAX_TRADE, ELEC_F_MAX_NET_POSITION,
    CUSTOMER_REVENUE_PER_CONTRACT, CUSTOMER_PRICE_PER_MWH,
    DISPOSAL_PENALTY, SHORTFALL_PENALTY,
    ELEC_FWD_CONTRACT_SIZE, ELEC_SPOT_CONTRACT_SIZE,
    FORWARD_PROCUREMENT_RATIO,
)


class DistributorOptimizer:
    """Optimizes all Distributor decisions each tick."""

    def __init__(self):
        self.demand_forecaster = DemandForecaster()
        self.pricing = PricingModel()
        self._last_demand_day = 0

    def optimize(self, state: GameState) -> list[Recommendation]:
        """Generate all recommendations for the Distributor at current tick."""
        recs = []
        day = state.current_day
        next_day = day + 1

        # 1. Position closeout warnings (highest priority)
        recs.extend(self._closeout_recommendations(state, day))

        # 2. Procurement plan for tomorrow
        if next_day <= 6:
            recs.extend(self._procurement_recommendations(state, day, next_day))

        # 3. Excess electricity disposal
        recs.extend(self._excess_management(state, day))

        # Sort by urgency
        urgency_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        recs.sort(key=lambda r: urgency_order.get(r.urgency, 3))

        return recs

    def get_demand_forecast(self, state: GameState, day: int) -> dict:
        """Compute detailed demand forecast for a given day.

        Returns comprehensive demand analysis including uncertainty.
        """
        temp = state.best_temperature_forecast(day)

        if temp is None:
            # No forecast yet - use prior
            return {
                'demand_contracts': 180,  # Conservative mid-range
                'demand_low': 120,
                'demand_high': 240,
                'demand_mwh': 18_000,
                'temperature': None,
                'confidence': 'NONE',
                'num_updates': 0,
                'customer_revenue': 180 * CUSTOMER_REVENUE_PER_CONTRACT,
            }

        # Reset forecaster if new day
        if day != self._last_demand_day:
            self.demand_forecaster.reset()
            self._last_demand_day = day

        forecast = self.demand_forecaster.update(temp)

        demand_exact = demand_contracts(temp)
        customer_revenue = demand_exact * CUSTOMER_REVENUE_PER_CONTRACT

        return {
            'demand_contracts': forecast['demand_exact'],
            'demand_mean': forecast['demand_mean'],
            'demand_median': forecast['demand_median'],
            'demand_low': forecast['demand_low'],
            'demand_high': forecast['demand_high'],
            'demand_mwh': demand_exact * ELEC_SPOT_CONTRACT_SIZE,
            'temperature': temp,
            'temp_std': forecast['temp_std'],
            'confidence': 'HIGH' if forecast['num_updates'] >= 3 else
                         ('MEDIUM' if forecast['num_updates'] >= 2 else 'LOW'),
            'num_updates': forecast['num_updates'],
            'customer_revenue': customer_revenue,
        }

    def _procurement_recommendations(self, state: GameState, today: int,
                                      tomorrow: int) -> list[Recommendation]:
        """Generate procurement recommendations (buy electricity)."""
        recs = []

        demand = self.get_demand_forecast(state, tomorrow)
        target = demand['demand_contracts']

        if target <= 0:
            return recs

        # Current procurement status
        # Forward position: negative = we've committed to buy (which is what we want)
        fwd_bought = max(0, state.elec_f_position)  # Positive = long = bought forward
        fwd_in_spot_contracts = fwd_bought * 5  # 1 ELEC-F = 500 MWh = 5 spot contracts

        # How much more do we need?
        still_needed = target - fwd_in_spot_contracts

        # Get pricing info
        price_est = self.pricing.estimate_spot_price(tomorrow, state)
        expected_spot = price_est['estimate']

        # Newsvendor optimal quantity
        temp = state.best_temperature_forecast(tomorrow)
        temp_std = state.temperature_uncertainty(tomorrow)
        if temp is not None:
            newsvendor = self.demand_forecaster.optimal_procurement(
                temp, temp_std, expected_spot
            )
            optimal_target = newsvendor['target_contracts']
        else:
            optimal_target = int(target)

        # Forward purchase recommendations
        if still_needed > 0 and state.elec_fwd_ask > 0:
            fwd_ask = state.elec_fwd_ask

            # Profitable check: buy forward only if cost < customer price
            if fwd_ask < CUSTOMER_PRICE_PER_MWH:
                # How many forward contracts? (1 fwd = 5 spot-equiv)
                fwd_needed = int(still_needed / 5)
                fwd_needed = min(fwd_needed,
                               ELEC_F_MAX_TRADE,
                               ELEC_F_MAX_NET_POSITION - state.elec_f_position)

                if fwd_needed > 0:
                    # Determine urgency based on forecast confidence
                    if demand['num_updates'] >= 3:
                        urgency = "HIGH"  # Final forecast - buy now
                    elif demand['num_updates'] >= 2:
                        urgency = "MEDIUM"
                    else:
                        # Only buy partial amount before final forecast
                        fwd_needed = max(1, int(fwd_needed * FORWARD_PROCUREMENT_RATIO))
                        urgency = "LOW"

                    profit_per_fwd = (CUSTOMER_PRICE_PER_MWH - fwd_ask) * ELEC_FWD_CONTRACT_SIZE

                    recs.append(Recommendation(
                        action="BUY",
                        ticker="ELEC-F",
                        quantity=fwd_needed,
                        price=int(fwd_ask) + 1,  # Integer quotes, bid above ask to fill
                        reason=(f"Procure for Day {tomorrow} demand: "
                               f"{target:.0f} contracts needed, "
                               f"{fwd_in_spot_contracts:.0f} already covered, "
                               f"margin ${CUSTOMER_PRICE_PER_MWH - fwd_ask:.0f}/MWh"),
                        urgency=urgency,
                        expected_pnl=profit_per_fwd * fwd_needed,
                    ))
            else:
                recs.append(Recommendation(
                    action="BUY",
                    ticker="ELEC-F",
                    quantity=0,
                    reason=f"Forward price ${fwd_ask:.0f} > customer price "
                           f"${CUSTOMER_PRICE_PER_MWH} - wait for better price or use spot",
                    urgency="LOW",
                ))

        # Summary info
        deficit = max(0, target - fwd_in_spot_contracts)
        surplus = max(0, fwd_in_spot_contracts - target)

        if deficit > 0:
            recs.append(Recommendation(
                action="BUY",
                ticker=f"ELEC-day{tomorrow}",
                quantity=int(deficit),
                reason=f"Plan to buy {deficit:.0f} contracts on spot tomorrow "
                       f"(if not covered by forwards)",
                urgency="LOW",
                penalty_risk=deficit * SHORTFALL_PENALTY,
            ))

        return recs

    def _excess_management(self, state: GameState, day: int) -> list[Recommendation]:
        """Handle excess electricity (bought more than demand)."""
        recs = []

        if day < 2:
            return recs

        elec_ticker = state.elec_day_ticker(day)
        position = state.elec_day_positions.get(elec_ticker, 0)

        if position > 0:
            # We have excess electricity - sell on spot
            urgency = "HIGH" if state.is_day_close_warning() else "MEDIUM"
            recs.append(Recommendation(
                action="SELL",
                ticker=elec_ticker,
                quantity=position,
                price=None,
                reason=f"Sell {position} excess contracts to avoid "
                       f"${position * DISPOSAL_PENALTY:,} MECC disposal fee",
                urgency=urgency,
                penalty_risk=position * DISPOSAL_PENALTY,
            ))

        return recs

    def _closeout_recommendations(self, state: GameState,
                                   day: int) -> list[Recommendation]:
        """Generate urgent recommendations to close positions before day end."""
        recs = []

        if day < 2:
            return recs

        elec_ticker = state.elec_day_ticker(day)
        position = state.elec_day_positions.get(elec_ticker, 0)

        if position == 0:
            return recs

        ticks_left = state.ticks_remaining_in_day()

        if ticks_left <= 5:
            urgency = "CRITICAL"
        elif ticks_left <= 20:
            urgency = "HIGH"
        elif ticks_left <= 60:
            urgency = "MEDIUM"
        else:
            return recs

        action = "SELL" if position > 0 else "BUY"

        recs.append(Recommendation(
            action=action,
            ticker=elec_ticker,
            quantity=abs(position),
            reason=f"CLOSE {position} {elec_ticker}! {ticks_left} ticks left. "
                   f"Penalty: ${abs(position) * DISPOSAL_PENALTY:,}",
            urgency=urgency,
            penalty_risk=abs(position) * DISPOSAL_PENALTY,
        ))

        return recs
