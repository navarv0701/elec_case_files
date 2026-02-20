"""
RIT REST API wrapper - read-only.
Order submission is DISABLED for the GBE Electricity case.
This client retrieves market data, positions, news, and tenders.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from config import RIT_BASE_URL

logger = logging.getLogger(__name__)


class RITClient:
    """Read-only RIT REST API client."""

    def __init__(self, api_key: str, base_url: str = RIT_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict | list | None:
        """Make a GET request. Returns parsed JSON or None on error."""
        url = f"{self.base_url}{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=2)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                logger.error("API authentication failed. Check API key.")
            else:
                logger.warning(f"API {endpoint} returned {resp.status_code}")
        except requests.ConnectionError:
            logger.debug("RIT server not reachable")
        except requests.Timeout:
            logger.debug(f"API timeout on {endpoint}")
        except Exception as e:
            logger.error(f"API error on {endpoint}: {e}")
        return None

    # ------------------------------------------------------------------
    # Case Information
    # ------------------------------------------------------------------

    def get_case(self) -> Optional[dict]:
        """GET /case - tick, period, status, name."""
        return self._get("/case")

    # ------------------------------------------------------------------
    # Securities
    # ------------------------------------------------------------------

    def get_securities(self, ticker: Optional[str] = None) -> Optional[list[dict]]:
        """GET /securities - all security data including prices and positions.
        If ticker is provided, returns data for that specific security.
        """
        params = {"ticker": ticker} if ticker else None
        return self._get("/securities", params)

    def get_security_book(self, ticker: str, limit: int = 20) -> Optional[dict]:
        """GET /securities/book - order book for a ticker.
        Returns {'bids': [...], 'asks': [...]}.
        """
        return self._get("/securities/book", {"ticker": ticker, "limit": limit})

    def get_security_history(self, ticker: str, period: Optional[int] = None,
                              limit: Optional[int] = None) -> Optional[list[dict]]:
        """GET /securities/history - OHLC price history."""
        params = {"ticker": ticker}
        if period is not None:
            params["period"] = period
        if limit is not None:
            params["limit"] = limit
        return self._get("/securities/history", params)

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def get_news(self, since: Optional[int] = None, limit: int = 50) -> Optional[list[dict]]:
        """GET /news - news items (weather forecasts, RAE bulletins, tenders).
        Use `since` to only get news newer than a specific news_id.
        """
        params = {"limit": limit}
        if since is not None:
            params["since"] = since
        return self._get("/news", params)

    # ------------------------------------------------------------------
    # Tenders
    # ------------------------------------------------------------------

    def get_tenders(self) -> Optional[list[dict]]:
        """GET /tenders - active tender offers (factory tenders for Traders)."""
        return self._get("/tenders")

    # ------------------------------------------------------------------
    # Trader Information
    # ------------------------------------------------------------------

    def get_trader(self) -> Optional[dict]:
        """GET /trader - trader info including NLV, first/last name."""
        return self._get("/trader")

    # ------------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------------

    def get_limits(self) -> Optional[list[dict]]:
        """GET /limits - position limits and fine structures."""
        return self._get("/limits")

    # ------------------------------------------------------------------
    # Assets
    # ------------------------------------------------------------------

    def get_assets(self, ticker: Optional[str] = None) -> Optional[list[dict]]:
        """GET /assets - power plant assets, conversion ratios."""
        params = {"ticker": ticker} if ticker else None
        return self._get("/assets", params)

    # ------------------------------------------------------------------
    # Convenience Methods
    # ------------------------------------------------------------------

    def get_all_positions(self) -> dict[str, int]:
        """Get positions for all securities as {ticker: quantity}."""
        securities = self.get_securities()
        if not securities:
            return {}
        return {s["ticker"]: s.get("position", 0) for s in securities}

    def get_price_data(self) -> dict[str, dict]:
        """Get bid/ask/last for all securities as {ticker: {bid, ask, last}}."""
        securities = self.get_securities()
        if not securities:
            return {}
        result = {}
        for s in securities:
            result[s["ticker"]] = {
                "bid": s.get("bid", 0),
                "ask": s.get("ask", 0),
                "last": s.get("last", 0),
                "position": s.get("position", 0),
                "volume": s.get("volume", 0),
            }
        return result

    def is_connected(self) -> bool:
        """Check if the RIT server is reachable and case is active."""
        case = self.get_case()
        return case is not None
