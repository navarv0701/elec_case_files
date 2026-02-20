"""
Background data polling - continuously fetches from RIT API and updates GameState.
Parses news items to extract weather forecasts, RAE bulletins, and factory tenders.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

from api.rit_client import RITClient
from state.game_state import GameState, RAEBulletin, FactoryTender
from config import (
    POLL_CASE_MS, POLL_SECURITIES_MS, POLL_NEWS_MS,
    TICKS_PER_DAY,
)

logger = logging.getLogger(__name__)


class NewsParser:
    """Parses RIT news text to extract structured data."""

    # Regex patterns for different news types
    SUNSHINE_PATTERNS = [
        # "X hours of sunshine" or "sunshine ... X hours"
        re.compile(r'(\d+\.?\d*)\s*hours?\s*of\s*sunshine', re.IGNORECASE),
        re.compile(r'sunshine.*?(\d+\.?\d*)\s*hours?', re.IGNORECASE),
        re.compile(r'sun.*?(\d+\.?\d*)\s*h', re.IGNORECASE),
    ]

    TEMPERATURE_PATTERNS = [
        # "average temperature ... X degrees" or "X degrees Celsius"
        re.compile(r'(?:average\s+)?temperature.*?(\d+\.?\d*)\s*(?:degrees?|°|C)', re.IGNORECASE),
        re.compile(r'(\d+\.?\d*)\s*(?:degrees?\s*)?(?:Celsius|C)\b', re.IGNORECASE),
    ]

    RAE_PRICE_PATTERN = re.compile(
        r'price.*?\$(\d+\.?\d*).*?(?:and|to)\s*\$(\d+\.?\d*)', re.IGNORECASE
    )
    RAE_VOLUME_PATTERN = re.compile(
        r'(\d+)\s*contracts?\s*(?:available|for\s+buying)', re.IGNORECASE
    )
    RAE_VOLUME_BUY_SELL = re.compile(
        r'(\d+)\s*contracts?\s*for\s*buying.*?(\d+)\s*contracts?\s*for\s*selling', re.IGNORECASE
    )

    TENDER_PATTERN = re.compile(
        r'(?:buy|sell|purchase)\s+(\d+)\s*contracts?.*?\$(\d+\.?\d*)', re.IGNORECASE
    )

    SUNSHINE_RANGE_PATTERN = re.compile(
        r'sunshine.*?between\s*(\d+\.?\d*)\s*(?:and|to)\s*(\d+\.?\d*)', re.IGNORECASE
    )

    TEMPERATURE_RANGE_PATTERN = re.compile(
        r'temperature.*?between\s*(\d+\.?\d*)\s*(?:and|to)\s*(\d+\.?\d*)', re.IGNORECASE
    )

    @classmethod
    def parse(cls, headline: str, body: str = "") -> dict:
        """Parse a news item and return structured data.

        Returns:
            {
                'type': str,          # 'sunshine', 'temperature', 'rae_bulletin',
                                      # 'factory_tender', 'sunshine_range',
                                      # 'temperature_range', 'unknown'
                'value': float/None,  # Primary numeric value
                'low': float/None,    # Range low (for range forecasts)
                'high': float/None,   # Range high
                'data': dict,         # Additional parsed data
            }
        """
        text = f"{headline} {body}".strip()

        # Try sunshine exact
        for pat in cls.SUNSHINE_PATTERNS:
            m = pat.search(text)
            if m:
                return {'type': 'sunshine', 'value': float(m.group(1)),
                        'low': None, 'high': None, 'data': {}}

        # Try sunshine range
        m = cls.SUNSHINE_RANGE_PATTERN.search(text)
        if m:
            low, high = float(m.group(1)), float(m.group(2))
            return {'type': 'sunshine_range', 'value': (low + high) / 2,
                    'low': low, 'high': high, 'data': {}}

        # Try temperature exact
        for pat in cls.TEMPERATURE_PATTERNS:
            m = pat.search(text)
            if m:
                return {'type': 'temperature', 'value': float(m.group(1)),
                        'low': None, 'high': None, 'data': {}}

        # Try temperature range
        m = cls.TEMPERATURE_RANGE_PATTERN.search(text)
        if m:
            low, high = float(m.group(1)), float(m.group(2))
            return {'type': 'temperature_range', 'value': (low + high) / 2,
                    'low': low, 'high': high, 'data': {}}

        # Try RAE bulletin
        price_match = cls.RAE_PRICE_PATTERN.search(text)
        if price_match:
            price_low = float(price_match.group(1))
            price_high = float(price_match.group(2))
            vol_buy, vol_sell = 0, 0
            vol_match = cls.RAE_VOLUME_BUY_SELL.search(text)
            if vol_match:
                vol_buy = int(vol_match.group(1))
                vol_sell = int(vol_match.group(2))
            else:
                vol_match = cls.RAE_VOLUME_PATTERN.search(text)
                if vol_match:
                    vol_buy = vol_sell = int(vol_match.group(1))
            return {
                'type': 'rae_bulletin',
                'value': (price_low + price_high) / 2,
                'low': price_low,
                'high': price_high,
                'data': {'volume_buy': vol_buy, 'volume_sell': vol_sell},
            }

        # Try factory tender
        tender_match = cls.TENDER_PATTERN.search(text)
        if tender_match:
            qty = int(tender_match.group(1))
            price = float(tender_match.group(2))
            action = "BUY" if re.search(r'\bbuy\b', text, re.IGNORECASE) else "SELL"
            return {
                'type': 'factory_tender',
                'value': price,
                'low': None,
                'high': None,
                'data': {'quantity': qty, 'action': action, 'price': price},
            }

        return {'type': 'unknown', 'value': None, 'low': None, 'high': None, 'data': {}}


class DataPoller:
    """Continuously polls RIT API and updates GameState."""

    def __init__(self, client: RITClient, state: GameState):
        self.client = client
        self.state = state
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._last_poll_times: dict[str, float] = {}
        self._previous_day: int = 0
        self._news_forecast_counts: dict[int, dict[str, int]] = {}  # day -> {type -> count}

    def start(self):
        """Start the background polling thread."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("DataPoller started")

    def stop(self):
        """Stop the background polling thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("DataPoller stopped")

    def _should_poll(self, key: str, interval_ms: int) -> bool:
        """Check if enough time has elapsed for this poll type."""
        now = time.time()
        last = self._last_poll_times.get(key, 0)
        if (now - last) * 1000 >= interval_ms:
            self._last_poll_times[key] = now
            return True
        return False

    def _poll_loop(self):
        """Main polling loop."""
        while self.running:
            try:
                # Case status (tick, day)
                if self._should_poll("case", POLL_CASE_MS):
                    self._poll_case()

                # Securities (prices, positions)
                if self._should_poll("securities", POLL_SECURITIES_MS):
                    self._poll_securities()

                # News (forecasts, bulletins, tenders)
                if self._should_poll("news", POLL_NEWS_MS):
                    self._poll_news()

                time.sleep(0.05)  # 50ms minimum between cycles

            except Exception as e:
                logger.error(f"Polling error: {e}")
                time.sleep(1)

    def _poll_case(self):
        """Poll /case for tick and status."""
        case_data = self.client.get_case()
        if not case_data:
            return

        tick = case_data.get("tick", 0)
        status = case_data.get("status", "STOPPED")
        self.state.update_tick(tick, status)

        # Detect day transition
        new_day = self.state.current_day
        if new_day != self._previous_day:
            logger.info(f"Day transition: {self._previous_day} -> {new_day}")
            self._previous_day = new_day

    def _poll_securities(self):
        """Poll /securities for prices and positions."""
        securities = self.client.get_securities()
        if not securities:
            return

        elec_f_pos = 0
        ng_pos = 0
        elec_days = {}
        fwd_bid = fwd_ask = fwd_last = 0.0
        ng_bid_val = ng_ask_val = ng_last_val = 0.0

        for sec in securities:
            ticker = sec.get("ticker", "")
            position = sec.get("position", 0)
            bid = sec.get("bid", 0) or 0
            ask = sec.get("ask", 0) or 0
            last = sec.get("last", 0) or 0

            if ticker == "ELEC-F":
                elec_f_pos = position
                fwd_bid, fwd_ask, fwd_last = bid, ask, last
            elif ticker == "NG":
                ng_pos = position
                ng_bid_val, ng_ask_val, ng_last_val = bid, ask, last
            elif ticker.startswith("ELEC-day"):
                elec_days[ticker] = position
                # Also store spot prices
                day_num = int(ticker.replace("ELEC-day", ""))
                if bid > 0:
                    self.state.elec_spot_bid[day_num] = bid
                if ask > 0:
                    self.state.elec_spot_ask[day_num] = ask

        self.state.update_market_prices(
            fwd_bid, fwd_ask, fwd_last,
            ng_bid_val, ng_ask_val, ng_last_val,
        )
        self.state.update_positions(elec_f_pos, ng_pos, elec_days)

        # Also update P&L from trader endpoint
        trader_data = self.client.get_trader()
        if trader_data:
            nlv = trader_data.get("nlv", 0)
            self.state.update_pnl(
                realized=0,  # Not directly available; NLV is the key metric
                unrealized=0,
                nlv=nlv,
            )

    def _poll_news(self):
        """Poll /news for weather forecasts, RAE bulletins, and tenders."""
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

            # Parse
            parsed = NewsParser.parse(headline, body)
            news_type = parsed['type']

            # Determine which day this forecast is for
            # Forecasts received on day X are about day X+1
            current_day = max(1, (tick // TICKS_PER_DAY) + 1)
            forecast_target_day = current_day + 1

            if news_type == 'sunshine' or news_type == 'sunshine_range':
                if forecast_target_day not in self._news_forecast_counts:
                    self._news_forecast_counts[forecast_target_day] = {'sunshine': 0, 'temperature': 0}
                self._news_forecast_counts[forecast_target_day]['sunshine'] += 1
                update_num = self._news_forecast_counts[forecast_target_day]['sunshine']
                self.state.add_sunshine_forecast(forecast_target_day, parsed['value'], update_num)
                logger.info(f"Sunshine forecast for day {forecast_target_day}: "
                           f"{parsed['value']} hours (update #{update_num})")

            elif news_type == 'temperature' or news_type == 'temperature_range':
                if forecast_target_day not in self._news_forecast_counts:
                    self._news_forecast_counts[forecast_target_day] = {'sunshine': 0, 'temperature': 0}
                self._news_forecast_counts[forecast_target_day]['temperature'] += 1
                update_num = self._news_forecast_counts[forecast_target_day]['temperature']
                self.state.add_temperature_forecast(forecast_target_day, parsed['value'], update_num)
                logger.info(f"Temperature forecast for day {forecast_target_day}: "
                           f"{parsed['value']}C (update #{update_num})")

            elif news_type == 'rae_bulletin':
                bulletin_day = forecast_target_day
                existing = self.state.rae_bulletins.get(bulletin_day, [])
                bulletin_num = len(existing) + 1
                bulletin = RAEBulletin(
                    day=bulletin_day,
                    price_low=parsed['low'],
                    price_high=parsed['high'],
                    volume_buy=parsed['data'].get('volume_buy', 0),
                    volume_sell=parsed['data'].get('volume_sell', 0),
                    bulletin_number=bulletin_num,
                    tick_received=tick,
                )
                self.state.add_rae_bulletin(bulletin)
                logger.info(f"RAE bulletin for day {bulletin_day}: "
                           f"${parsed['low']:.2f}-${parsed['high']:.2f}, "
                           f"vol buy={bulletin.volume_buy}, sell={bulletin.volume_sell}")

            elif news_type == 'factory_tender':
                tender = FactoryTender(
                    tender_id=news_id,
                    action=parsed['data']['action'],
                    ticker="ELEC-F",
                    quantity=parsed['data']['quantity'],
                    price=parsed['data']['price'],
                    expiration_tick=tick + 90,  # Default: ~90 seconds to decide
                )
                self.state.add_tender(tender)
                logger.info(f"Factory tender: {tender.action} {tender.quantity} "
                           f"contracts @ ${tender.price:.2f}")

    def poll_once(self):
        """Execute one full polling cycle (useful for testing)."""
        self._poll_case()
        self._poll_securities()
        self._poll_news()
