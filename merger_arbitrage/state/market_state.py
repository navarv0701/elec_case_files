"""
Central market state for merger arbitrage trading.
Single source of truth. Thread-safe via Lock. Event-driven recomputation.
"""

from __future__ import annotations

import logging
import sys
import os
from dataclasses import dataclass, field
from threading import Lock, Event
from typing import Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deal import DealState, initialize_all_deals
from models.probability import ProbabilityTracker, NewsImpact
from config import DEALS, HEAT_DURATION_TICKS, URGENCY_CLOSE_TICKS

logger = logging.getLogger(__name__)


@dataclass
class TradeRecommendation:
    """A single actionable trade recommendation."""
    action: str              # BUY, SELL
    ticker: str              # e.g., "TGX", "PHR"
    quantity: int
    price: Optional[float] = None    # None = market order
    order_type: str = "LIMIT"        # LIMIT or MARKET
    reason: str = ""
    urgency: str = "LOW"             # LOW, MEDIUM, HIGH, CRITICAL
    deal_id: str = ""
    expected_profit: float = 0.0
    is_hedge_leg: bool = False       # True if this is the hedge side of a pair


@dataclass
class OpenOrder:
    """Tracks an open order in the market."""
    order_id: int
    ticker: str
    action: str
    quantity: int
    filled: int
    price: float
    order_type: str
    tick_submitted: int


class MarketState:
    """Single source of truth for merger arbitrage trading.
    All fields are protected by a lock for thread-safe access.
    """

    def __init__(self):
        self._lock = Lock()

        # Time
        self.current_tick: int = 0
        self.current_period: int = 1
        self.case_status: str = "STOPPED"

        # Deals
        self.deals: dict[str, DealState] = {}
        self.probability_tracker: Optional[ProbabilityTracker] = None

        # Market Data
        self.prices: dict[str, dict] = {}

        # Positions
        self.positions: dict[str, int] = {}

        # Open Orders
        self.open_orders: list[OpenOrder] = []

        # P&L
        self.nlv: float = 0.0

        # News
        self.news_history: list[dict] = []
        self.classified_news: list[dict] = []
        self.last_news_id: int = 0

        # Event system
        self._event_callbacks: list[Callable] = []
        self._recompute_event = Event()
        self._news_version: int = 0
        self._market_version: int = 0
        self._position_version: int = 0
        self._last_recompute_version: int = 0

        # Recommendations
        self.active_recommendations: list[TradeRecommendation] = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        """Set up all 5 deals with initial probabilities and standalone values."""
        self.deals = initialize_all_deals()
        initial_probs = {d: cfg.initial_probability for d, cfg in DEALS.items()}
        self.probability_tracker = ProbabilityTracker(initial_probs)
        logger.info("MarketState initialized with 5 deals")
        for deal_id, deal in self.deals.items():
            logger.info(
                f"  {deal_id} ({deal.config.target_ticker}/{deal.config.acquirer_ticker}): "
                f"V={deal.standalone_value:.2f}, K0={deal.deal_value_K:.2f}, "
                f"P*={deal.intrinsic_target_price:.2f}, p0={deal.probability:.2f}"
            )

    # ------------------------------------------------------------------
    # Event System
    # ------------------------------------------------------------------

    def on_event(self, callback: Callable):
        """Register a callback for state change events."""
        self._event_callbacks.append(callback)

    def _fire_event(self, event_type: str, data: Optional[dict] = None):
        """Notify all listeners of a state change."""
        self._recompute_event.set()
        for cb in self._event_callbacks:
            try:
                cb(event_type, data or {})
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def wait_for_change(self, timeout: float = 1.0) -> bool:
        """Block until a state change occurs or timeout."""
        triggered = self._recompute_event.wait(timeout)
        self._recompute_event.clear()
        return triggered

    def needs_recompute(self) -> bool:
        """Check if state has changed since last strategy run."""
        total = self._news_version + self._market_version + self._position_version
        return total > self._last_recompute_version

    def mark_recomputed(self):
        """Mark that the strategy has consumed the latest state."""
        self._last_recompute_version = (
            self._news_version + self._market_version + self._position_version
        )

    # ------------------------------------------------------------------
    # Thread-Safe Updates
    # ------------------------------------------------------------------

    def update_tick(self, tick: int, period: int, status: str):
        """Update time and case status."""
        with self._lock:
            self.current_tick = tick
            self.current_period = period
            self.case_status = status

    def update_prices(self, price_data: dict[str, dict]):
        """Update all security prices and propagate to DealState objects."""
        with self._lock:
            self.prices = price_data
            for deal_id, deal in self.deals.items():
                t = deal.config.target_ticker
                a = deal.config.acquirer_ticker
                if t in price_data:
                    deal.target_price = price_data[t].get("last", deal.target_price)
                    deal.target_bid = price_data[t].get("bid", 0)
                    deal.target_ask = price_data[t].get("ask", 0)
                if a in price_data:
                    deal.acquirer_price = price_data[a].get("last", deal.acquirer_price)
                    deal.acquirer_bid = price_data[a].get("bid", 0)
                    deal.acquirer_ask = price_data[a].get("ask", 0)
            self._market_version += 1
        self._fire_event("MARKET_UPDATE")

    def update_positions(self, positions: dict[str, int]):
        """Update positions and propagate to DealState objects."""
        with self._lock:
            self.positions = positions
            for deal_id, deal in self.deals.items():
                deal.target_position = positions.get(deal.config.target_ticker, 0)
                deal.acquirer_position = positions.get(deal.config.acquirer_ticker, 0)
            self._position_version += 1

    def apply_news_impact(self, impact: NewsImpact):
        """Apply a classified news impact to probability and deal state."""
        with self._lock:
            if self.probability_tracker:
                new_p = self.probability_tracker.apply_news(impact)
                if impact.deal_id in self.deals:
                    self.deals[impact.deal_id].probability = new_p
            self._news_version += 1
        self._fire_event("NEWS_IMPACT", {
            "deal_id": impact.deal_id,
            "delta_p": impact.delta_p,
        })

    def mark_deal_resolved(self, deal_id: str, completed: bool):
        """Mark a deal as completed or failed."""
        with self._lock:
            if deal_id in self.deals:
                self.deals[deal_id].resolved = True
                self.deals[deal_id].resolution = "completed" if completed else "failed"
                self.deals[deal_id].probability = 1.0 if completed else 0.0
            if self.probability_tracker:
                self.probability_tracker.mark_resolved(deal_id, completed)
            self._news_version += 1
        self._fire_event("DEAL_RESOLVED", {
            "deal_id": deal_id,
            "completed": completed,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def ticks_remaining(self) -> int:
        return max(0, HEAT_DURATION_TICKS - self.current_tick)

    def gross_position(self) -> int:
        """Sum of absolute positions across all securities."""
        return sum(abs(p) for p in self.positions.values())

    def net_position(self, ticker: str) -> int:
        return self.positions.get(ticker, 0)

    def is_near_end(self, threshold: int = URGENCY_CLOSE_TICKS) -> bool:
        return self.ticks_remaining() <= threshold

    def get_deal_summary(self) -> list[dict]:
        """Get a summary of all deals for display."""
        summaries = []
        for deal_id, deal in self.deals.items():
            summaries.append({
                "deal_id": deal_id,
                "target": deal.config.target_ticker,
                "acquirer": deal.config.acquirer_ticker,
                "structure": deal.config.structure,
                "probability": deal.probability,
                "deal_value_K": deal.deal_value_K,
                "standalone_V": deal.standalone_value,
                "intrinsic_P": deal.intrinsic_target_price,
                "market_price": deal.target_price,
                "mispricing": deal.target_mispricing,
                "mispricing_pct": deal.target_mispricing_pct,
                "spread": deal.spread_to_deal,
                "target_pos": deal.target_position,
                "acquirer_pos": deal.acquirer_position,
                "resolved": deal.resolved,
                "resolution": deal.resolution,
            })
        return summaries
