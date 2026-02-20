"""
Production model for electricity generation.
Solar: ELEC_solar = 6 * H_day (zero cost, automatic)
Gas: 8 NG contracts -> 1 ELEC contract (via natural gas power plant)
"""

from __future__ import annotations

from config import (
    SOLAR_MULTIPLIER,
    NG_TO_ELEC_RATIO,
    MAX_GAS_PLANTS,
    NG_CONTRACT_SIZE,
    ELEC_SPOT_CONTRACT_SIZE,
    GAS_PROFITABILITY_MARGIN,
)


def solar_production(sunshine_hours: float) -> float:
    """Compute solar electricity production in contracts.

    ELEC_solar = 6 * H_day

    Args:
        sunshine_hours: Hours of sunshine during the day.

    Returns:
        Number of electricity contracts produced (can be fractional).
    """
    return SOLAR_MULTIPLIER * sunshine_hours


def gas_production(ng_contracts: int) -> int:
    """Compute gas electricity production in contracts.

    8 NG contracts -> 1 ELEC contract.

    Args:
        ng_contracts: Number of NG contracts purchased (must be multiple of 8).

    Returns:
        Number of ELEC contracts produced.
    """
    return ng_contracts // NG_TO_ELEC_RATIO


def gas_cost_per_elec_contract(ng_price_per_mmbtu: float) -> float:
    """Compute cost of producing 1 ELEC contract from natural gas.

    Cost = 8 NG contracts * 100 MMBtu/contract * NG_price/MMBtu

    Args:
        ng_price_per_mmbtu: Natural gas price per MMBtu.

    Returns:
        Total cost in dollars to produce 1 ELEC contract (100 MWh).
    """
    return NG_TO_ELEC_RATIO * NG_CONTRACT_SIZE * ng_price_per_mmbtu


def gas_revenue_per_elec_contract(elec_price_per_mwh: float) -> float:
    """Revenue from selling 1 ELEC contract.

    Revenue = price/MWh * 100 MWh/contract

    Args:
        elec_price_per_mwh: Electricity price per MWh.

    Returns:
        Revenue in dollars per ELEC contract.
    """
    return elec_price_per_mwh * ELEC_SPOT_CONTRACT_SIZE


def gas_profit_per_plant(ng_price: float, elec_price: float) -> float:
    """Profit from running one gas plant (produces 1 ELEC contract).

    Args:
        ng_price: NG price per MMBtu.
        elec_price: Expected electricity price per MWh.

    Returns:
        Profit in dollars per plant.
    """
    cost = gas_cost_per_elec_contract(ng_price)
    revenue = gas_revenue_per_elec_contract(elec_price)
    return revenue - cost


class ProductionScheduler:
    """Determines optimal production mix across solar and gas plants."""

    def compute_schedule(
        self,
        sunshine_forecast: float,
        sunshine_uncertainty: float,
        ng_ask_price: float,
        expected_elec_price: float,
        team_demand: float,
        current_ng_position: int = 0,
    ) -> dict:
        """Compute optimal production schedule for the next day.

        Decision logic:
        1. Solar production is automatic (cannot be shut down).
        2. Gas production is profitable if revenue > cost * margin threshold.
        3. Produce gas to fill gap between demand and solar, up to 10 plants.
        4. Don't overproduce (disposal penalty is $20,000/contract).

        Args:
            sunshine_forecast: Best forecast of sunshine hours for next day.
            sunshine_uncertainty: Std dev of sunshine forecast.
            ng_ask_price: Current NG ask price (per MMBtu).
            expected_elec_price: Expected electricity price (per MWh) for next day.
            team_demand: Expected team demand in contracts for next day.
            current_ng_position: NG contracts already held.

        Returns:
            Production schedule dict with quantities, costs, and recommendations.
        """
        # Solar (automatic, certain within forecast uncertainty)
        solar_expected = solar_production(sunshine_forecast)
        solar_low = solar_production(max(0, sunshine_forecast - sunshine_uncertainty))
        solar_high = solar_production(sunshine_forecast + sunshine_uncertainty)

        # Gas profitability
        profit_per_plant = gas_profit_per_plant(ng_ask_price, expected_elec_price)
        cost_per_plant = gas_cost_per_elec_contract(ng_ask_price)
        revenue_per_plant = gas_revenue_per_elec_contract(expected_elec_price)
        is_profitable = profit_per_plant > 0 and revenue_per_plant > cost_per_plant * GAS_PROFITABILITY_MARGIN

        # Gap to fill with gas
        gap = max(0, team_demand - solar_expected)

        # Optimal number of gas plants
        if is_profitable and gap > 0:
            # Don't overproduce: limit to gap
            gas_plants = min(MAX_GAS_PLANTS, int(gap) + 1)  # +1 for safety buffer
            # But don't exceed what we can sell
            gas_plants = min(gas_plants, MAX_GAS_PLANTS)
        else:
            gas_plants = 0

        ng_to_buy = gas_plants * NG_TO_ELEC_RATIO
        gas_elec = gas_plants  # 1 plant = 1 ELEC contract

        # Account for NG already held
        additional_ng_needed = max(0, ng_to_buy - current_ng_position)

        total_production = solar_expected + gas_elec
        surplus = total_production - team_demand
        disposal_risk = max(0, surplus) * 20_000  # Penalty if can't sell

        return {
            # Solar
            'solar_contracts': solar_expected,
            'solar_low': solar_low,
            'solar_high': solar_high,
            'solar_cost': 0.0,

            # Gas
            'gas_plants': gas_plants,
            'gas_contracts': gas_elec,
            'ng_to_buy': additional_ng_needed,
            'ng_total_needed': ng_to_buy,
            'gas_cost_total': gas_plants * cost_per_plant,
            'gas_revenue_total': gas_plants * revenue_per_plant,
            'gas_profit_per_plant': profit_per_plant,
            'gas_is_profitable': is_profitable,

            # Totals
            'total_production': total_production,
            'team_demand': team_demand,
            'surplus_contracts': surplus,
            'disposal_risk': disposal_risk,

            # Recommendation
            'recommendation': self._format_recommendation(
                additional_ng_needed, ng_ask_price, gas_plants, is_profitable,
                solar_expected, total_production, team_demand
            ),
        }

    def _format_recommendation(self, ng_to_buy, ng_price, gas_plants, is_profitable,
                                solar, total, demand) -> str:
        """Generate human-readable recommendation string."""
        parts = []

        if ng_to_buy > 0 and is_profitable:
            parts.append(f"BUY {ng_to_buy} NG @ ${ng_price:.2f}/MMBtu "
                        f"({gas_plants} plants, total cost ${ng_to_buy * ng_price * 100:,.0f})")
        elif not is_profitable and gas_plants == 0:
            parts.append("Gas NOT profitable - rely on solar only")

        parts.append(f"Expected production: {total:.0f} contracts "
                    f"(solar={solar:.0f}, gas={gas_plants})")

        if total > demand:
            parts.append(f"WARNING: Surplus of {total - demand:.0f} contracts - "
                        f"sell aggressively on forward market")
        elif total < demand:
            parts.append(f"Shortfall of {demand - total:.0f} contracts - "
                        f"team Distributor will need to buy from external")

        return " | ".join(parts)
