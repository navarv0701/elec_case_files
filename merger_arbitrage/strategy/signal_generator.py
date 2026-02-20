"""
Trade signal generation for merger arbitrage.
Compares intrinsic values to market prices across all 5 deals.
Generates TradeRecommendation objects for the order executor.
"""

from __future__ import annotations

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state.market_state import MarketState, TradeRecommendation
from models.deal import DealState
from config import (
    MIN_MISPRICING_THRESHOLD, SPREAD_SAFETY_FACTOR,
    MAX_POSITION_PER_DEAL, MAX_ORDER_SIZE, COMMISSION_PER_SHARE,
    GROSS_POSITION_LIMIT, NET_POSITION_LIMIT,
    URGENCY_CLOSE_TICKS, URGENCY_CRITICAL_TICKS,
)

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates trade signals for all 5 deals based on mispricing analysis."""

    def generate_signals(self, state: MarketState) -> list[TradeRecommendation]:
        """Analyze all deals and generate trade recommendations.

        Pipeline per deal:
        1. Skip resolved deals
        2. Compute mispricing (intrinsic - market)
        3. Filter by minimum threshold and spread safety
        4. Generate target leg (buy undervalued / sell overvalued)
        5. Generate hedge leg for stock-for-stock / mixed deals
        6. Size position based on mispricing magnitude and limits
        """
        all_recs: list[TradeRecommendation] = []

        for deal_id, deal in state.deals.items():
            if deal.resolved:
                recs = self._handle_resolved_deal(deal, state)
                all_recs.extend(recs)
                continue

            recs = self._analyze_deal(deal, state)
            all_recs.extend(recs)

        # Add end-of-heat closeout signals
        all_recs.extend(self._closeout_signals(state))

        # Sort by urgency
        urgency_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        all_recs.sort(key=lambda r: urgency_order.get(r.urgency, 3))

        return all_recs

    def _analyze_deal(self, deal: DealState, state: MarketState) -> list[TradeRecommendation]:
        """Analyze a single deal for trading opportunities."""
        recs: list[TradeRecommendation] = []
        cfg = deal.config

        mispricing = deal.target_mispricing
        if abs(mispricing) < MIN_MISPRICING_THRESHOLD:
            return recs

        # Check bid-ask spread safety
        spread = 0.0
        if deal.target_ask > 0 and deal.target_bid > 0:
            spread = deal.target_ask - deal.target_bid
        if spread > 0 and abs(mispricing) < spread * SPREAD_SAFETY_FACTOR:
            return recs

        # Determine direction
        if mispricing > 0:
            # Target undervalued -> BUY target
            target_action = "BUY"
            target_price = deal.target_ask if deal.target_ask > 0 else deal.target_price
            acquirer_action = "SELL"  # Hedge by shorting acquirer
            acquirer_price = deal.acquirer_bid if deal.acquirer_bid > 0 else deal.acquirer_price
        else:
            # Target overvalued -> SELL target
            target_action = "SELL"
            target_price = deal.target_bid if deal.target_bid > 0 else deal.target_price
            acquirer_action = "BUY"
            acquirer_price = deal.acquirer_ask if deal.acquirer_ask > 0 else deal.acquirer_price

        # Position sizing
        target_qty = self._compute_position_size(deal, state, mispricing)
        if target_qty <= 0:
            return recs

        # Urgency based on mispricing magnitude
        mispricing_pct = abs(deal.target_mispricing_pct)
        if mispricing_pct > 3.0:
            urgency = "HIGH"
        elif mispricing_pct > 1.5:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"

        # Target leg
        recs.append(TradeRecommendation(
            action=target_action,
            ticker=cfg.target_ticker,
            quantity=target_qty,
            price=round(target_price, 2) if target_price > 0 else None,
            order_type="LIMIT" if target_price > 0 else "MARKET",
            reason=(
                f"{cfg.deal_id}: P*={deal.intrinsic_target_price:.2f} vs "
                f"Mkt={deal.target_price:.2f} (misp={mispricing:+.2f}, "
                f"p={deal.probability:.2f})"
            ),
            urgency=urgency,
            deal_id=cfg.deal_id,
            expected_profit=abs(mispricing) * target_qty - COMMISSION_PER_SHARE * target_qty * 2,
        ))

        # Hedge leg (only for stock-for-stock and mixed deals)
        if cfg.structure != "all_cash" and deal.ideal_hedge_ratio > 0:
            hedge_qty = int(target_qty * deal.ideal_hedge_ratio)
            hedge_qty = min(hedge_qty, MAX_ORDER_SIZE)

            if hedge_qty > 0:
                recs.append(TradeRecommendation(
                    action=acquirer_action,
                    ticker=cfg.acquirer_ticker,
                    quantity=hedge_qty,
                    price=round(acquirer_price, 2) if acquirer_price > 0 else None,
                    order_type="LIMIT" if acquirer_price > 0 else "MARKET",
                    reason=(
                        f"Hedge {cfg.deal_id}: {acquirer_action} {hedge_qty} "
                        f"{cfg.acquirer_ticker} (ratio={cfg.exchange_ratio})"
                    ),
                    urgency=urgency,
                    deal_id=cfg.deal_id,
                    is_hedge_leg=True,
                ))

        return recs

    def _compute_position_size(self, deal: DealState, state: MarketState,
                                mispricing: float) -> int:
        """Size position based on mispricing magnitude, confidence, and limits.

        Sizing factors:
        1. Mispricing percentage (larger = bigger position, up to 5% = full size)
        2. Probability confidence (trade more when p near extremes)
        3. Gross and net limit headroom
        4. Existing position (avoid doubling down excessively)
        """
        mispricing_pct = abs(deal.target_mispricing_pct) / 100.0

        # Scale linearly with mispricing, capped at 1.0
        scale = min(1.0, mispricing_pct / 0.05)

        # Probability confidence: higher when p near 0 or 1
        p = deal.probability
        p_confidence = max(p, 1.0 - p)

        base_size = int(scale * p_confidence * MAX_POSITION_PER_DEAL)

        # Cap at max order size
        base_size = min(base_size, MAX_ORDER_SIZE)

        # Check gross position limit headroom
        current_gross = state.gross_position()
        headroom_gross = GROSS_POSITION_LIMIT - current_gross
        # Reserve room for hedge leg
        reserve = int(base_size * deal.ideal_hedge_ratio) if deal.ideal_hedge_ratio > 0 else 0
        base_size = min(base_size, max(0, headroom_gross - reserve))

        # Check net position limit for this ticker
        ticker = deal.config.target_ticker
        current_net = state.net_position(ticker)
        if mispricing > 0:  # BUY
            headroom_net = NET_POSITION_LIMIT - current_net
        else:  # SELL
            headroom_net = NET_POSITION_LIMIT + current_net
        base_size = min(base_size, max(0, headroom_net))

        # Reduce if we already have a large position in the same direction
        existing = abs(deal.target_position)
        if existing > MAX_POSITION_PER_DEAL * 0.8:
            base_size = min(base_size, MAX_ORDER_SIZE // 4)

        # Round to nearest 100 for cleaner orders
        base_size = max(0, (base_size // 100) * 100)

        return base_size

    def _handle_resolved_deal(self, deal: DealState,
                               state: MarketState) -> list[TradeRecommendation]:
        """Close all positions in a resolved deal immediately."""
        recs: list[TradeRecommendation] = []

        if deal.target_position != 0:
            action = "SELL" if deal.target_position > 0 else "BUY"
            qty = abs(deal.target_position)
            while qty > 0:
                chunk = min(qty, MAX_ORDER_SIZE)
                recs.append(TradeRecommendation(
                    action=action,
                    ticker=deal.config.target_ticker,
                    quantity=chunk,
                    order_type="MARKET",
                    reason=f"{deal.config.deal_id} RESOLVED ({deal.resolution}) - close target",
                    urgency="CRITICAL",
                    deal_id=deal.config.deal_id,
                ))
                qty -= chunk

        if deal.acquirer_position != 0:
            action = "SELL" if deal.acquirer_position > 0 else "BUY"
            qty = abs(deal.acquirer_position)
            while qty > 0:
                chunk = min(qty, MAX_ORDER_SIZE)
                recs.append(TradeRecommendation(
                    action=action,
                    ticker=deal.config.acquirer_ticker,
                    quantity=chunk,
                    order_type="MARKET",
                    reason=f"{deal.config.deal_id} RESOLVED - close hedge",
                    urgency="CRITICAL",
                    deal_id=deal.config.deal_id,
                    is_hedge_leg=True,
                ))
                qty -= chunk

        return recs

    def _closeout_signals(self, state: MarketState) -> list[TradeRecommendation]:
        """Generate urgency signals to close all positions near heat end."""
        recs: list[TradeRecommendation] = []
        ticks_left = state.ticks_remaining()

        if ticks_left > URGENCY_CLOSE_TICKS:
            return recs

        urgency = "CRITICAL" if ticks_left <= URGENCY_CRITICAL_TICKS else "HIGH"
        order_type = "MARKET" if ticks_left <= URGENCY_CRITICAL_TICKS else "LIMIT"

        for ticker, position in state.positions.items():
            if position == 0:
                continue

            action = "SELL" if position > 0 else "BUY"

            # For LIMIT, use aggressive pricing to get filled quickly
            price = None
            if order_type == "LIMIT" and ticker in state.prices:
                p = state.prices[ticker]
                if action == "SELL":
                    price = round(p.get("bid", 0) * 0.995, 2)  # Sell slightly below bid
                else:
                    price = round(p.get("ask", 0) * 1.005, 2)  # Buy slightly above ask

            qty = abs(position)
            while qty > 0:
                chunk = min(qty, MAX_ORDER_SIZE)
                recs.append(TradeRecommendation(
                    action=action,
                    ticker=ticker,
                    quantity=chunk,
                    price=price,
                    order_type=order_type,
                    reason=f"CLOSEOUT: {ticks_left}s left, {position} shares open",
                    urgency=urgency,
                ))
                qty -= chunk

        return recs
