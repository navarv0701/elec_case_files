"""
Position manager for merger arbitrage.
Enforces position limits, manages hedge ratios, and tracks exposure.
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state.market_state import MarketState, TradeRecommendation
from config import (
    GROSS_POSITION_LIMIT, NET_POSITION_LIMIT, MAX_ORDER_SIZE,
    MAX_POSITION_PER_DEAL, HEDGE_RATIO_TOLERANCE,
)

logger = logging.getLogger(__name__)


class PositionManager:
    """Validates and adjusts trade recommendations to respect all limits."""

    def validate_and_adjust(self, recs: list[TradeRecommendation],
                            state: MarketState) -> list[TradeRecommendation]:
        """Filter and adjust recommendations to respect all limits.

        Checks:
        1. Gross position limit (100,000)
        2. Net position limit (50,000)
        3. Max order size (5,000)
        4. No duplicate orders for same ticker+direction
        """
        validated: list[TradeRecommendation] = []

        # Simulate positions forward to check cumulative impact
        simulated_positions: dict[str, int] = dict(state.positions)
        simulated_gross = state.gross_position()
        seen_ticker_actions: set[tuple[str, str]] = set()

        for rec in recs:
            # Skip duplicate ticker+action combos (except CRITICAL urgency)
            key = (rec.ticker, rec.action)
            if key in seen_ticker_actions and rec.urgency != "CRITICAL":
                continue

            adjusted = self._adjust_for_limits(
                rec, simulated_positions, simulated_gross
            )
            if adjusted and adjusted.quantity > 0:
                # Update simulated state
                delta = adjusted.quantity if adjusted.action == "BUY" else -adjusted.quantity
                simulated_positions[adjusted.ticker] = (
                    simulated_positions.get(adjusted.ticker, 0) + delta
                )
                simulated_gross = sum(abs(p) for p in simulated_positions.values())
                validated.append(adjusted)
                seen_ticker_actions.add(key)

        return validated

    def _adjust_for_limits(self, rec: TradeRecommendation,
                           positions: dict[str, int],
                           current_gross: int) -> Optional[TradeRecommendation]:
        """Adjust a single recommendation to fit within limits."""
        qty = rec.quantity

        # Max order size
        qty = min(qty, MAX_ORDER_SIZE)

        # Net position limit
        current_net = positions.get(rec.ticker, 0)
        if rec.action == "BUY":
            max_allowed = NET_POSITION_LIMIT - current_net
        else:
            max_allowed = NET_POSITION_LIMIT + current_net
        qty = min(qty, max(0, max_allowed))

        # Gross position limit
        remaining_gross = GROSS_POSITION_LIMIT - current_gross
        qty = min(qty, max(0, remaining_gross))

        if qty <= 0:
            return None

        # Create adjusted recommendation
        scale = qty / max(1, rec.quantity)
        return TradeRecommendation(
            action=rec.action,
            ticker=rec.ticker,
            quantity=qty,
            price=rec.price,
            order_type=rec.order_type,
            reason=rec.reason,
            urgency=rec.urgency,
            deal_id=rec.deal_id,
            expected_profit=rec.expected_profit * scale,
            is_hedge_leg=rec.is_hedge_leg,
        )

    def check_hedge_drift(self, state: MarketState) -> list[TradeRecommendation]:
        """Generate rebalancing signals when hedge ratios drift too far."""
        recs: list[TradeRecommendation] = []

        for deal_id, deal in state.deals.items():
            if deal.resolved or deal.config.structure == "all_cash":
                continue

            target_pos = deal.target_position
            acquirer_pos = deal.acquirer_position

            if target_pos == 0:
                # No target position - close any orphaned hedge
                if acquirer_pos != 0:
                    action = "SELL" if acquirer_pos > 0 else "BUY"
                    recs.append(TradeRecommendation(
                        action=action,
                        ticker=deal.config.acquirer_ticker,
                        quantity=min(abs(acquirer_pos), MAX_ORDER_SIZE),
                        order_type="MARKET",
                        reason=f"Close orphaned hedge in {deal.config.deal_id}",
                        urgency="HIGH",
                        deal_id=deal_id,
                        is_hedge_leg=True,
                    ))
                continue

            # Expected hedge: if long target, short acquirer proportionally
            # If target_pos > 0 (long), expected acquirer = -target_pos * ratio (short)
            # If target_pos < 0 (short), expected acquirer = +|target_pos| * ratio (long)
            expected_acquirer = -int(target_pos * deal.ideal_hedge_ratio)
            drift = acquirer_pos - expected_acquirer

            tolerance_shares = abs(target_pos) * HEDGE_RATIO_TOLERANCE
            if abs(drift) <= max(tolerance_shares, 100):
                continue  # Within tolerance

            if drift > 0:
                # Have too many acquirer shares (need to sell)
                action = "SELL"
            else:
                # Need more acquirer shares (need to buy)
                action = "BUY"

            qty = min(abs(drift), MAX_ORDER_SIZE)
            # Round to 100
            qty = (qty // 100) * 100
            if qty <= 0:
                continue

            recs.append(TradeRecommendation(
                action=action,
                ticker=deal.config.acquirer_ticker,
                quantity=qty,
                order_type="LIMIT",
                reason=(
                    f"Hedge rebal {deal.config.deal_id}: "
                    f"drift={drift:+d}, target={target_pos}, "
                    f"acq={acquirer_pos}, expected={expected_acquirer}"
                ),
                urgency="MEDIUM",
                deal_id=deal_id,
                is_hedge_leg=True,
            ))

        return recs
