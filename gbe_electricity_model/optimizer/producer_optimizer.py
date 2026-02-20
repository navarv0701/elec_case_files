"""
Producer Optimizer - decides NG purchases, forward sales, and spot sales.
The Producer owns solar + gas plants, produces electricity, and sells it.
"""

from __future__ import annotations

from state.game_state import GameState, Recommendation
from models.production import ProductionScheduler, solar_production, gas_profit_per_plant
from models.pricing import PricingModel
from config import (
    ELEC_F_MAX_TRADE, ELEC_F_MAX_NET_POSITION, NG_TO_ELEC_RATIO,
    MAX_GAS_PLANTS, DISPOSAL_PENALTY, ELEC_FWD_CONTRACT_SIZE,
    ELEC_SPOT_CONTRACT_SIZE, FORWARD_PREMIUM_THRESHOLD,
)


class ProducerOptimizer:
    """Optimizes all Producer decisions each tick."""

    def __init__(self):
        self.scheduler = ProductionScheduler()
        self.pricing = PricingModel()

    def optimize(self, state: GameState) -> list[Recommendation]:
        """Generate all recommendations for the Producer at current tick.

        Called every polling cycle. Returns a prioritized list of actions.
        """
        recs = []
        day = state.current_day
        next_day = day + 1

        # 1. Position closeout warnings (highest priority)
        recs.extend(self._closeout_recommendations(state, day))

        # 2. NG purchase decision (for tomorrow's production)
        if next_day <= 6:
            recs.extend(self._ng_purchase_recommendations(state, day, next_day))

        # 3. Forward sales (sell production on ELEC-F)
        recs.extend(self._forward_sale_recommendations(state, day))

        # 4. Spot market recommendations
        recs.extend(self._spot_recommendations(state, day))

        # Sort by urgency
        urgency_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        recs.sort(key=lambda r: urgency_order.get(r.urgency, 3))

        return recs

    def get_production_schedule(self, state: GameState) -> dict:
        """Compute the full production schedule for display purposes."""
        day = state.current_day
        next_day = day + 1

        sunshine = state.best_sunshine_forecast(next_day)
        if sunshine is None:
            sunshine = 3.0  # Conservative default

        ng_price = state.ng_ask if state.ng_ask > 0 else state.ng_last

        # Get expected electricity price
        price_est = self.pricing.estimate_spot_price(next_day, state)
        expected_price = price_est['estimate']

        # Estimate team demand (use Distributor's demand model if available)
        temp = state.best_temperature_forecast(next_day)
        if temp is not None:
            from models.demand import demand_contracts
            team_demand = demand_contracts(temp)
        else:
            team_demand = 180  # Conservative estimate

        return self.scheduler.compute_schedule(
            sunshine_forecast=sunshine,
            sunshine_uncertainty=state.sunshine_uncertainty(next_day),
            ng_ask_price=ng_price,
            expected_elec_price=expected_price,
            team_demand=team_demand,
            current_ng_position=state.ng_position,
        )

    def _ng_purchase_recommendations(self, state: GameState, today: int,
                                      tomorrow: int) -> list[Recommendation]:
        """Decide how much NG to buy for tomorrow's gas production."""
        recs = []

        schedule = self.get_production_schedule(state)

        ng_to_buy = int(schedule['ng_to_buy'])
        if ng_to_buy <= 0:
            return recs

        ng_price = state.ng_ask
        if ng_price <= 0:
            return recs

        # Only recommend if profitable
        if not schedule['gas_is_profitable']:
            return recs

        recs.append(Recommendation(
            action="BUY",
            ticker="NG",
            quantity=ng_to_buy,
            price=ng_price,
            reason=(f"Gas production for Day {tomorrow}: "
                   f"{schedule['gas_plants']} plants, "
                   f"profit ${schedule['gas_profit_per_plant']:,.0f}/plant"),
            urgency="MEDIUM",
            expected_pnl=schedule['gas_profit_per_plant'] * schedule['gas_plants'],
        ))

        return recs

    def _forward_sale_recommendations(self, state: GameState,
                                       day: int) -> list[Recommendation]:
        """Recommend forward sales when profitable."""
        recs = []

        fwd_bid = state.elec_fwd_bid
        if fwd_bid <= 0:
            return recs

        next_day = day + 1
        price_est = self.pricing.estimate_spot_price(next_day, state)
        expected_spot = price_est['estimate']

        if expected_spot <= 0:
            return recs

        # Check forward premium
        premium = self.pricing.forward_vs_spot_premium(state, next_day)

        if premium['action_signal'] == 'SELL_FORWARD':
            # How much can we sell?
            max_sell = ELEC_F_MAX_NET_POSITION + state.elec_f_position  # Room to go more short
            max_sell = min(max_sell, ELEC_F_MAX_TRADE)

            if max_sell > 0:
                recs.append(Recommendation(
                    action="SELL",
                    ticker="ELEC-F",
                    quantity=max_sell,
                    price=int(fwd_bid),  # Forward only accepts integer quotes
                    reason=(f"Forward premium: ${premium['premium_per_mwh']:.2f}/MWh "
                           f"({premium['premium_pct']:.1f}%) above expected spot"),
                    urgency="MEDIUM",
                    expected_pnl=premium['premium_per_mwh'] * ELEC_FWD_CONTRACT_SIZE * max_sell,
                ))

        # Also sell forward to lock in solar production revenue
        sunshine = state.best_sunshine_forecast(next_day)
        if sunshine is not None:
            solar = solar_production(sunshine)
            # Convert solar contracts (100 MWh) to forward contracts (500 MWh)
            solar_in_fwd = int(solar / 5)  # 5 spot contracts = 1 fwd contract

            # How many forwards needed to cover solar?
            current_short_fwd = -state.elec_f_position if state.elec_f_position < 0 else 0
            solar_uncovered_fwd = max(0, solar_in_fwd - current_short_fwd)

            if solar_uncovered_fwd > 0 and fwd_bid > 5:  # Sanity check on price
                sell_qty = min(solar_uncovered_fwd, ELEC_F_MAX_TRADE,
                             ELEC_F_MAX_NET_POSITION + state.elec_f_position)
                if sell_qty > 0:
                    recs.append(Recommendation(
                        action="SELL",
                        ticker="ELEC-F",
                        quantity=sell_qty,
                        price=int(fwd_bid),
                        reason=f"Lock in solar revenue: {sell_qty} fwd contracts "
                               f"({sell_qty * 5} spot-equiv) @ ${fwd_bid:.0f}/MWh",
                        urgency="LOW",
                        expected_pnl=sell_qty * fwd_bid * ELEC_FWD_CONTRACT_SIZE,
                    ))

        return recs

    def _spot_recommendations(self, state: GameState, day: int) -> list[Recommendation]:
        """Recommend spot market trades."""
        recs = []

        # Check if we have excess production that needs selling on spot
        elec_ticker = state.elec_day_ticker(day)
        position = state.elec_day_positions.get(elec_ticker, 0)

        if position > 0:
            # We have unsold electricity - sell on spot to avoid disposal
            spot_bid = state.elec_spot_bid.get(day, 0)
            urgency = "HIGH" if state.is_day_close_warning() else "MEDIUM"

            recs.append(Recommendation(
                action="SELL",
                ticker=elec_ticker,
                quantity=position,
                price=spot_bid if spot_bid > 0 else None,
                reason=f"Sell {position} excess contracts to avoid "
                       f"${position * DISPOSAL_PENALTY:,} disposal penalty",
                urgency=urgency,
                penalty_risk=position * DISPOSAL_PENALTY,
            ))

        return recs

    def _closeout_recommendations(self, state: GameState,
                                   day: int) -> list[Recommendation]:
        """Generate urgent recommendations to close positions before day end."""
        recs = []

        if day < 2:
            return recs  # No closeout needed on day 1

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
            return recs  # Not yet in closeout zone

        action = "SELL" if position > 0 else "BUY"
        spot_price = state.elec_spot_bid.get(day, 0) if position > 0 else state.elec_spot_ask.get(day, 0)

        recs.append(Recommendation(
            action=action,
            ticker=elec_ticker,
            quantity=abs(position),
            price=spot_price if spot_price > 0 else None,
            reason=f"CLOSE {position} {elec_ticker} position! "
                   f"{ticks_left} ticks left. "
                   f"Penalty risk: ${abs(position) * DISPOSAL_PENALTY:,}",
            urgency=urgency,
            penalty_risk=abs(position) * DISPOSAL_PENALTY,
        ))

        return recs
