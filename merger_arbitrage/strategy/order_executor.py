"""
Order executor - submits, monitors, and manages orders via the RIT API.
Converts TradeRecommendation objects into actual order submissions.
Handles order lifecycle: submit -> monitor fills -> cancel stale -> resubmit.
"""

from __future__ import annotations

import logging
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.rit_client import RITClient
from state.market_state import MarketState, TradeRecommendation, OpenOrder
from config import MAX_ORDER_SIZE, ORDER_REFRESH_INTERVAL_TICKS

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Manages order submission and lifecycle."""

    def __init__(self, client: RITClient):
        self.client = client
        self._last_order_tick: dict[str, int] = {}
        self._pending_order_ids: set[int] = set()
        self._rate_limited_until: float = 0.0

    def execute_recommendations(self, recs: list[TradeRecommendation],
                                 state: MarketState):
        """Execute validated trade recommendations via the API."""
        for rec in recs:
            if time.time() < self._rate_limited_until:
                logger.debug("Rate limited - skipping order")
                continue
            self._execute_single(rec, state)

    def _execute_single(self, rec: TradeRecommendation, state: MarketState):
        """Execute a single trade recommendation, chunking if needed."""
        remaining = rec.quantity

        while remaining > 0:
            chunk = min(remaining, MAX_ORDER_SIZE)

            result = self.client.submit_order(
                ticker=rec.ticker,
                order_type=rec.order_type,
                quantity=chunk,
                action=rec.action,
                price=rec.price if rec.order_type == "LIMIT" else None,
            )

            if result:
                if result.get("rate_limited"):
                    wait = result.get("wait", 0.5)
                    self._rate_limited_until = time.time() + wait
                    logger.warning(f"Rate limited for {wait}s")
                    break

                order_id = result.get("order_id")
                if order_id:
                    self._pending_order_ids.add(order_id)
                    self._last_order_tick[rec.ticker] = state.current_tick

                price_str = f"${rec.price:.2f}" if rec.price else "MKT"
                logger.info(
                    f"ORDER: {rec.action} {chunk} {rec.ticker} @ {price_str} "
                    f"[{rec.urgency}] {rec.reason[:60]}"
                )
            else:
                logger.warning(f"Order FAILED: {rec.action} {chunk} {rec.ticker}")
                break

            remaining -= chunk

    def cancel_stale_orders(self, state: MarketState):
        """Cancel open orders that are too old."""
        open_orders = self.client.get_open_orders()
        if not open_orders:
            return

        cancelled = 0
        for order in open_orders:
            order_id = order.get("order_id")
            tick_submitted = order.get("tick", 0)
            age = state.current_tick - tick_submitted

            if age > ORDER_REFRESH_INTERVAL_TICKS:
                if self.client.cancel_order(order_id):
                    self._pending_order_ids.discard(order_id)
                    cancelled += 1

        if cancelled:
            logger.info(f"Cancelled {cancelled} stale orders")

    def cancel_all(self):
        """Emergency cancellation of all open orders."""
        result = self.client.cancel_all_orders()
        if result:
            logger.info("All orders cancelled")
            self._pending_order_ids.clear()
        else:
            logger.warning("Failed to cancel all orders")

    def sync_open_orders(self, state: MarketState):
        """Sync the open order list from the API into state."""
        open_orders = self.client.get_open_orders()
        if open_orders is None:
            return

        state.open_orders = [
            OpenOrder(
                order_id=o.get("order_id", 0),
                ticker=o.get("ticker", ""),
                action=o.get("action", ""),
                quantity=o.get("quantity", 0),
                filled=o.get("quantity_filled", 0),
                price=o.get("price", 0),
                order_type=o.get("type", ""),
                tick_submitted=o.get("tick", 0),
            )
            for o in open_orders
        ]
