"""
Background data polling for merger arbitrage.
Polls: /case, /securities, /news
Parses news through the NLP classifier and updates MarketState.
"""

from __future__ import annotations

import logging
import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.rit_client import RITClient
from state.market_state import MarketState
from nlp.news_classifier import NewsClassifier
from models.probability import compute_delta_p, NewsImpact
from config import POLL_CASE_MS, POLL_SECURITIES_MS, POLL_NEWS_MS

logger = logging.getLogger(__name__)


class DataPoller:
    """Continuously polls RIT API and updates MarketState."""

    def __init__(self, client: RITClient, state: MarketState):
        self.client = client
        self.state = state
        self.classifier = NewsClassifier()
        self.running = False
        self._thread: threading.Thread | None = None
        self._last_poll: dict[str, float] = {}

    def start(self):
        """Start the polling thread."""
        self.running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="poller")
        self._thread.start()
        logger.info("DataPoller started")

    def stop(self):
        """Stop the polling thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("DataPoller stopped")

    def _should_poll(self, key: str, interval_ms: int) -> bool:
        """Check if enough time has elapsed since last poll for this key."""
        now = time.time()
        last = self._last_poll.get(key, 0)
        if (now - last) * 1000 >= interval_ms:
            self._last_poll[key] = now
            return True
        return False

    def _poll_loop(self):
        """Main polling loop."""
        while self.running:
            try:
                if self._should_poll("case", POLL_CASE_MS):
                    self._poll_case()

                if self._should_poll("securities", POLL_SECURITIES_MS):
                    self._poll_securities()

                if self._should_poll("news", POLL_NEWS_MS):
                    self._poll_news()

                time.sleep(0.05)  # 50ms sleep to avoid busy-waiting
            except Exception as e:
                logger.error(f"Polling error: {e}", exc_info=True)
                time.sleep(1)

    def _poll_case(self):
        """Poll /case for tick, period, status."""
        case = self.client.get_case()
        if case:
            self.state.update_tick(
                case.get("tick", 0),
                case.get("period", 1),
                case.get("status", "STOPPED"),
            )

    def _poll_securities(self):
        """Poll /securities for prices and positions."""
        securities = self.client.get_securities()
        if not securities:
            return

        price_data: dict[str, dict] = {}
        positions: dict[str, int] = {}

        for s in securities:
            ticker = s.get("ticker", "")
            price_data[ticker] = {
                "bid": s.get("bid", 0) or 0,
                "ask": s.get("ask", 0) or 0,
                "last": s.get("last", 0) or 0,
                "volume": s.get("volume", 0) or 0,
            }
            positions[ticker] = s.get("position", 0)

        self.state.update_prices(price_data)
        self.state.update_positions(positions)

        # Also update P&L
        trader = self.client.get_trader()
        if trader:
            self.state.nlv = trader.get("nlv", 0)

    def _poll_news(self):
        """Poll /news, classify each item, compute probability impact."""
        news_items = self.client.get_news(since=self.state.last_news_id)
        if not news_items:
            return

        for item in news_items:
            news_id = item.get("news_id", 0)
            if news_id <= self.state.last_news_id:
                continue

            self.state.last_news_id = news_id
            headline = item.get("headline", "")
            body = item.get("body", "")
            tick = item.get("tick", self.state.current_tick)

            # Store raw news
            self.state.news_history.append(item)

            # Classify
            classified = self.classifier.classify(headline, body, news_id, tick)
            self.state.classified_news.append({
                "news_id": news_id,
                "tick": tick,
                "headline": headline,
                "classified": classified,
            })

            if classified.deal_id is None:
                logger.debug(f"News not matched to deal: {headline[:80]}")
                continue

            # Handle resolution events
            if classified.is_resolution:
                completed = classified.resolution_type == "completed"
                self.state.mark_deal_resolved(classified.deal_id, completed)
                logger.info(
                    f"*** DEAL RESOLUTION: {classified.deal_id} "
                    f"{'COMPLETED' if completed else 'FAILED'} ***"
                )
                continue

            # Compute and apply probability impact
            delta_p = compute_delta_p(
                deal_id=classified.deal_id,
                category=classified.category,
                direction=classified.direction,
                severity=classified.severity,
            )

            impact = NewsImpact(
                deal_id=classified.deal_id,
                category=classified.category,
                direction=classified.direction,
                severity=classified.severity,
                delta_p=delta_p,
                raw_headline=headline,
                tick=tick,
            )

            self.state.apply_news_impact(impact)

    def poll_once(self):
        """Run a single poll cycle (useful for testing)."""
        self._poll_case()
        self._poll_securities()
        self._poll_news()
