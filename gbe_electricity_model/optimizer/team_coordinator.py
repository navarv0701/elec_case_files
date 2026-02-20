"""
Team Coordinator - cross-role optimization and coordination.
Aggregates information from all roles to generate team-wide strategy.
"""

from __future__ import annotations

from state.game_state import GameState
from models.demand import demand_contracts
from models.production import solar_production, gas_profit_per_plant
from models.pricing import PricingModel
from config import (
    CUSTOMER_PRICE_PER_MWH, DISPOSAL_PENALTY,
    ELEC_FWD_CONTRACT_SIZE, ELEC_SPOT_CONTRACT_SIZE,
)


class TeamCoordinator:
    """Coordinates decisions across all four team members."""

    def __init__(self):
        self.pricing = PricingModel()

    def generate_team_plan(self, state: GameState) -> dict:
        """Generate a comprehensive team plan for the current day.

        Aggregates supply, demand, and market information to produce
        coordinated recommendations for all roles.

        Returns:
            {
                'day': int,
                'supply': dict,     # Production summary
                'demand': dict,     # Demand summary
                'balance': dict,    # Supply-demand balance
                'pricing': dict,    # Price estimates
                'producer_plan': dict,
                'distributor_plan': dict,
                'trader_plan': dict,
                'internal_price': float,
                'team_exposure': dict,
            }
        """
        day = state.current_day
        next_day = day + 1

        # === SUPPLY SIDE (Producer) ===
        supply = self._compute_supply(state, next_day)

        # === DEMAND SIDE (Distributor) ===
        demand = self._compute_demand(state, next_day)

        # === MARKET PRICING ===
        pricing = self.pricing.estimate_spot_price(next_day, state)

        # === SUPPLY-DEMAND BALANCE ===
        balance = self._compute_balance(supply, demand)

        # === INTERNAL TRANSFER PRICE ===
        internal_price = self._optimal_internal_price(state, supply, demand)

        # === ROLE-SPECIFIC PLANS ===
        producer_plan = self._producer_plan(state, supply, demand, pricing, internal_price)
        distributor_plan = self._distributor_plan(state, supply, demand, pricing, internal_price)
        trader_plan = self._trader_plan(state, balance, pricing)

        # === TEAM EXPOSURE ===
        exposure = self._compute_exposure(state)

        return {
            'day': day,
            'next_day': next_day,
            'supply': supply,
            'demand': demand,
            'balance': balance,
            'pricing': pricing,
            'producer_plan': producer_plan,
            'distributor_plan': distributor_plan,
            'trader_plan': trader_plan,
            'internal_price': internal_price,
            'team_exposure': exposure,
        }

    def _compute_supply(self, state: GameState, day: int) -> dict:
        """Compute expected electricity supply for a given day."""
        sunshine = state.best_sunshine_forecast(day)
        solar = solar_production(sunshine) if sunshine is not None else 0
        solar_uncertainty = state.sunshine_uncertainty(day)

        # Gas production (from NG purchased the previous day)
        # This is already determined by previous day's NG purchase
        gas = state.gas_production.get(day, 0)

        return {
            'solar_contracts': solar,
            'solar_uncertainty': solar_uncertainty * 6,  # Convert hours to contracts
            'gas_contracts': gas,
            'total_contracts': solar + gas,
            'sunshine_hours': sunshine,
        }

    def _compute_demand(self, state: GameState, day: int) -> dict:
        """Compute expected customer demand for a given day."""
        temp = state.best_temperature_forecast(day)

        if temp is not None:
            demand = demand_contracts(temp)
            temp_uncertainty = state.temperature_uncertainty(day)
        else:
            demand = 180  # Default assumption
            temp_uncertainty = 8.0
            temp = 28.0

        # Demand sensitivity to temperature
        from models.demand import demand_derivative
        sensitivity = demand_derivative(temp)

        return {
            'demand_contracts': demand,
            'temperature': temp,
            'temp_uncertainty': temp_uncertainty,
            'demand_low': demand_contracts(temp - temp_uncertainty),
            'demand_high': demand_contracts(temp + temp_uncertainty),
            'sensitivity': sensitivity,
            'customer_revenue': demand * CUSTOMER_PRICE_PER_MWH * ELEC_SPOT_CONTRACT_SIZE,
        }

    def _compute_balance(self, supply: dict, demand: dict) -> dict:
        """Compute supply-demand balance."""
        total_supply = supply['total_contracts']
        total_demand = demand['demand_contracts']
        balance = total_supply - total_demand

        return {
            'surplus': max(0, balance),
            'deficit': max(0, -balance),
            'net': balance,
            'is_surplus': balance > 0,
            'is_deficit': balance < 0,
            'is_balanced': abs(balance) < 5,
            'market_direction': 'BEARISH' if balance > 10 else
                               ('BULLISH' if balance < -10 else 'NEUTRAL'),
        }

    def _optimal_internal_price(self, state: GameState,
                                 supply: dict, demand: dict) -> float:
        """Calculate the optimal internal transfer price.

        The internal price should be between:
        - Producer's marginal cost (floor)
        - Customer price of $70/MWh (ceiling for distributor)
        - Current forward market midpoint (market reference)

        Optimal = midpoint of (producer_cost, min(customer_price, market_price))
        """
        # Producer's marginal cost
        ng_price = state.ng_ask if state.ng_ask > 0 else state.ng_last
        if ng_price > 0:
            gas_cost_per_mwh = 8 * ng_price  # 8 * NG_price per MWh
        else:
            gas_cost_per_mwh = 0

        # Solar is free, so weighted average cost depends on mix
        total = supply['total_contracts']
        if total > 0:
            solar_share = supply['solar_contracts'] / total
            gas_share = supply['gas_contracts'] / total
            avg_cost = solar_share * 0 + gas_share * gas_cost_per_mwh
        else:
            avg_cost = 0

        # Market reference
        if state.elec_fwd_bid > 0 and state.elec_fwd_ask > 0:
            market_mid = (state.elec_fwd_bid + state.elec_fwd_ask) / 2
        elif state.elec_fwd_last > 0:
            market_mid = state.elec_fwd_last
        else:
            market_mid = 40  # Default

        # Internal price = midpoint between cost and min(customer_price, market)
        ceiling = min(CUSTOMER_PRICE_PER_MWH, market_mid * 1.1)
        internal = (avg_cost + ceiling) / 2

        # Ensure it's at least at market level to be fair
        internal = max(internal, market_mid * 0.95)

        return round(internal, 2)

    def _producer_plan(self, state: GameState, supply: dict, demand: dict,
                       pricing: dict, internal_price: float) -> dict:
        """Generate Producer-specific plan."""
        total = supply['total_contracts']
        # How many to sell via forward?
        fwd_target = min(total, demand['demand_contracts'])
        fwd_in_fwd_contracts = int(fwd_target / 5)  # 500 MWh per fwd contract

        return {
            'sell_forward_target': fwd_in_fwd_contracts,
            'sell_spot_target': max(0, total - fwd_target),
            'target_price': int(internal_price),
            'production_total': total,
        }

    def _distributor_plan(self, state: GameState, supply: dict, demand: dict,
                          pricing: dict, internal_price: float) -> dict:
        """Generate Distributor-specific plan."""
        target = demand['demand_contracts']
        fwd_target_contracts = int(target / 5)

        return {
            'buy_forward_target': fwd_target_contracts,
            'buy_spot_reserve': max(0, target - fwd_target_contracts * 5),
            'target_price': int(internal_price) + 1,  # Bid slightly above
            'demand_total': target,
            'max_price': CUSTOMER_PRICE_PER_MWH,
        }

    def _trader_plan(self, state: GameState, balance: dict, pricing: dict) -> dict:
        """Generate Trader-specific plan."""
        return {
            'stance': 'SELL' if balance['is_surplus'] else
                     ('BUY' if balance['is_deficit'] else 'NEUTRAL'),
            'facilitate_internal': True,
            'arbitrage_active': True,
            'market_direction': balance['market_direction'],
            'expected_spot': pricing['estimate'],
        }

    def _compute_exposure(self, state: GameState) -> dict:
        """Compute aggregate team exposure."""
        total_elec_day = sum(abs(v) for v in state.elec_day_positions.values())
        total_fwd = abs(state.elec_f_position)
        total_ng = abs(state.ng_position)

        penalty_at_risk = 0
        for ticker, pos in state.elec_day_positions.items():
            if pos != 0:
                penalty_at_risk += abs(pos) * DISPOSAL_PENALTY

        return {
            'elec_day_contracts': total_elec_day,
            'forward_contracts': total_fwd,
            'ng_contracts': total_ng,
            'penalty_at_risk': penalty_at_risk,
            'nlv': state.nlv,
        }
