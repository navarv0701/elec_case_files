"""
Trader Optimizer - arbitrage, tender evaluation, and position management.
Traders act as intermediaries and shock absorbers in the electricity market.
"""

from __future__ import annotations

from state.game_state import GameState, Recommendation, FactoryTender
from models.pricing import PricingModel
from config import (
    ELEC_F_MAX_TRADE, ELEC_F_MAX_NET_POSITION,
    ELEC_DAY_MAX_NET_POSITION, ELEC_FWD_CONTRACT_SIZE,
    ELEC_SPOT_CONTRACT_SIZE, DISPOSAL_PENALTY,
    TENDER_MIN_PROFIT,
)


class TraderOptimizer:
    """Optimizes all Trader decisions each tick."""

    def __init__(self):
        self.pricing = PricingModel()

    def optimize(self, state: GameState) -> list[Recommendation]:
        """Generate all recommendations for the Trader at current tick."""
        recs = []
        day = state.current_day

        # 1. Position closeout (highest priority)
        recs.extend(self._closeout_recommendations(state, day))

        # 2. Factory tenders
        recs.extend(self._tender_recommendations(state))

        # 3. Arbitrage opportunities
        recs.extend(self._arbitrage_recommendations(state, day))

        # 4. Team facilitation
        recs.extend(self._team_facilitation_recommendations(state, day))

        urgency_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        recs.sort(key=lambda r: urgency_order.get(r.urgency, 3))

        return recs

    def _arbitrage_recommendations(self, state: GameState,
                                    day: int) -> list[Recommendation]:
        """Identify forward-spot arbitrage opportunities."""
        recs = []
        next_day = day + 1
        if next_day > 6:
            return recs

        premium = self.pricing.forward_vs_spot_premium(state, next_day)

        if premium['action_signal'] == 'SELL_FORWARD':
            # Forward overpriced: sell forward, plan to buy spot tomorrow
            available = ELEC_F_MAX_NET_POSITION + state.elec_f_position
            qty = min(available, ELEC_F_MAX_TRADE)
            if qty > 0:
                recs.append(Recommendation(
                    action="SELL",
                    ticker="ELEC-F",
                    quantity=qty,
                    price=int(state.elec_fwd_bid),
                    reason=(f"Arbitrage: forward premium "
                           f"${premium['premium_per_mwh']:.2f}/MWh. "
                           f"Sell forward, buy spot tomorrow."),
                    urgency="MEDIUM",
                    expected_pnl=premium['premium_per_mwh'] * ELEC_FWD_CONTRACT_SIZE * qty,
                ))

        elif premium['action_signal'] == 'BUY_FORWARD':
            # Forward underpriced: buy forward, plan to sell spot tomorrow
            available = ELEC_F_MAX_NET_POSITION - state.elec_f_position
            qty = min(available, ELEC_F_MAX_TRADE)
            if qty > 0:
                recs.append(Recommendation(
                    action="BUY",
                    ticker="ELEC-F",
                    quantity=qty,
                    price=int(state.elec_fwd_ask) + 1,
                    reason=(f"Arbitrage: forward discount "
                           f"${abs(premium['premium_per_mwh']):.2f}/MWh. "
                           f"Buy forward, sell spot tomorrow."),
                    urgency="MEDIUM",
                    expected_pnl=abs(premium['premium_per_mwh']) * ELEC_FWD_CONTRACT_SIZE * qty,
                ))

        return recs

    def _tender_recommendations(self, state: GameState) -> list[Recommendation]:
        """Evaluate factory tender offers."""
        recs = []

        for tender in state.pending_tenders:
            evaluation = self._evaluate_tender(tender, state)

            if evaluation['accept']:
                recs.append(Recommendation(
                    action="ACCEPT_TENDER",
                    ticker=tender.ticker,
                    quantity=tender.quantity,
                    price=tender.price,
                    reason=(f"Tender #{tender.tender_id}: {tender.action} "
                           f"{tender.quantity} @ ${tender.price:.2f}, "
                           f"expected profit ${evaluation['expected_pnl']:,.0f}"),
                    urgency="HIGH",
                    expected_pnl=evaluation['expected_pnl'],
                ))
            else:
                recs.append(Recommendation(
                    action="DECLINE_TENDER",
                    ticker=tender.ticker,
                    quantity=tender.quantity,
                    price=tender.price,
                    reason=(f"Tender #{tender.tender_id}: {evaluation['decline_reason']}"),
                    urgency="LOW",
                ))

        return recs

    def _evaluate_tender(self, tender: FactoryTender, state: GameState) -> dict:
        """Evaluate whether to accept a factory tender.

        Decision framework:
        1. Compare tender price to market price
        2. Check position limits
        3. Assess closeout risk
        4. Calculate expected P&L
        """
        tender_price = tender.price

        if tender.action == "BUY":
            # Factory wants to buy from us -> we sell
            # We need to procure at market price and sell at tender price
            market_price = state.elec_fwd_ask if state.elec_fwd_ask > 0 else state.elec_fwd_last
            if market_price <= 0:
                return {'accept': False, 'expected_pnl': 0,
                        'decline_reason': 'No market price available'}
            spread = tender_price - market_price
            action_side = "SELL"
        else:
            # Factory wants to sell to us -> we buy
            market_price = state.elec_fwd_bid if state.elec_fwd_bid > 0 else state.elec_fwd_last
            if market_price <= 0:
                return {'accept': False, 'expected_pnl': 0,
                        'decline_reason': 'No market price available'}
            spread = market_price - tender_price
            action_side = "BUY"

        expected_pnl = spread * tender.quantity * ELEC_SPOT_CONTRACT_SIZE

        # Position limit check
        if action_side == "BUY":
            room = ELEC_F_MAX_NET_POSITION - state.elec_f_position
        else:
            room = ELEC_F_MAX_NET_POSITION + state.elec_f_position

        if room < tender.quantity:
            return {'accept': False, 'expected_pnl': expected_pnl,
                    'decline_reason': f'Position limit: only {room} room, need {tender.quantity}'}

        # Closeout risk
        if tender.quantity > 50:  # Large tender
            closeout_risk = tender.quantity * DISPOSAL_PENALTY
            if expected_pnl < closeout_risk * 0.1:  # Not worth the risk
                return {'accept': False, 'expected_pnl': expected_pnl,
                        'decline_reason': f'Closeout risk ${closeout_risk:,} too high for '
                                         f'expected profit ${expected_pnl:,.0f}'}

        # Minimum profit threshold
        if expected_pnl < TENDER_MIN_PROFIT:
            return {'accept': False, 'expected_pnl': expected_pnl,
                    'decline_reason': f'Profit ${expected_pnl:,.0f} below '
                                     f'${TENDER_MIN_PROFIT:,} threshold'}

        return {'accept': True, 'expected_pnl': expected_pnl, 'decline_reason': ''}

    def _team_facilitation_recommendations(self, state: GameState,
                                            day: int) -> list[Recommendation]:
        """Recommend trades to facilitate between team Producer and Distributor.

        When the team's producer needs to sell and distributor needs to buy,
        intermediate the trade to capture the spread.
        """
        recs = []

        # If forward bid-ask spread is wide, there's room to intermediate
        if state.elec_fwd_bid > 0 and state.elec_fwd_ask > 0:
            spread = state.elec_fwd_ask - state.elec_fwd_bid
            if spread >= 2:  # At least $2/MWh spread
                midpoint = (state.elec_fwd_bid + state.elec_fwd_ask) / 2
                recs.append(Recommendation(
                    action="BUY",
                    ticker="ELEC-F",
                    quantity=1,
                    price=int(midpoint),
                    reason=f"Team facilitation: buy at ${midpoint:.0f} (mid), "
                           f"coordinate with Producer to sell at same price. "
                           f"Spread ${spread:.0f}/MWh.",
                    urgency="LOW",
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
