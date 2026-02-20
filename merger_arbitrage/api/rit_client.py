"""
RIT REST API wrapper - full read/write for Merger Arbitrage case.
Extends the base pattern with order submission, cancellation, and bulk operations.
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Optional

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RIT_BASE_URL

logger = logging.getLogger(__name__)


class RITClient:
    """Full-featured RIT REST API client with order execution."""

    def __init__(self, api_key: str, base_url: str = RIT_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})

    # ------------------------------------------------------------------
    # Base HTTP Methods
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: Optional[dict] = None):
        """Make a GET request. Returns parsed JSON or None on error."""
        url = f"{self.base_url}{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=2)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                logger.error("API authentication failed. Check API key.")
            else:
                logger.warning(f"GET {endpoint} returned {resp.status_code}")
        except requests.ConnectionError:
            logger.debug("RIT server not reachable")
        except requests.Timeout:
            logger.debug(f"GET timeout on {endpoint}")
        except Exception as e:
            logger.error(f"GET error on {endpoint}: {e}")
        return None

    def _post(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        """Make a POST request. Returns parsed JSON or None on error."""
        url = f"{self.base_url}{endpoint}"
        try:
            resp = self.session.post(url, params=params, timeout=2)
            if resp.status_code in (200, 201):
                return resp.json()
            elif resp.status_code == 429:
                wait = resp.json().get("wait", 0.5)
                logger.warning(f"Rate limited on POST {endpoint} - wait {wait}s")
                return {"rate_limited": True, "wait": wait}
            elif resp.status_code == 401:
                logger.error("API authentication failed on POST. Check API key.")
            else:
                logger.warning(f"POST {endpoint} returned {resp.status_code}: {resp.text[:200]}")
        except requests.ConnectionError:
            logger.debug("RIT server not reachable")
        except requests.Timeout:
            logger.debug(f"POST timeout on {endpoint}")
        except Exception as e:
            logger.error(f"POST error on {endpoint}: {e}")
        return None

    def _delete(self, endpoint: str, params: Optional[dict] = None) -> bool:
        """Make a DELETE request. Returns True on success."""
        url = f"{self.base_url}{endpoint}"
        try:
            resp = self.session.delete(url, params=params, timeout=2)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"DELETE error on {endpoint}: {e}")
            return False

    # ------------------------------------------------------------------
    # Case Information
    # ------------------------------------------------------------------

    def get_case(self) -> Optional[dict]:
        """GET /case - tick, period, status, name."""
        return self._get("/case")

    # ------------------------------------------------------------------
    # Securities
    # ------------------------------------------------------------------

    def get_securities(self, ticker: Optional[str] = None) -> Optional[list]:
        """GET /securities - all security data including prices and positions."""
        params = {"ticker": ticker} if ticker else None
        return self._get("/securities", params)

    def get_security_book(self, ticker: str, limit: int = 20) -> Optional[dict]:
        """GET /securities/book - order book for a ticker."""
        return self._get("/securities/book", {"ticker": ticker, "limit": limit})

    def get_security_history(self, ticker: str, period: Optional[int] = None,
                              limit: Optional[int] = None) -> Optional[list]:
        """GET /securities/history - OHLC price history."""
        params: dict = {"ticker": ticker}
        if period is not None:
            params["period"] = period
        if limit is not None:
            params["limit"] = limit
        return self._get("/securities/history", params)

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def get_news(self, since: Optional[int] = None, limit: int = 50) -> Optional[list]:
        """GET /news - news items. Use `since` for incremental polling."""
        params: dict = {"limit": limit}
        if since is not None:
            params["since"] = since
        return self._get("/news", params)

    # ------------------------------------------------------------------
    # Trader Information
    # ------------------------------------------------------------------

    def get_trader(self) -> Optional[dict]:
        """GET /trader - trader info including NLV."""
        return self._get("/trader")

    # ------------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------------

    def get_limits(self) -> Optional[list]:
        """GET /limits - position limits and fine structures."""
        return self._get("/limits")

    # ------------------------------------------------------------------
    # Order Execution (NEW for Merger Arb)
    # ------------------------------------------------------------------

    def submit_order(self, ticker: str, order_type: str, quantity: int,
                     action: str, price: Optional[float] = None) -> Optional[dict]:
        """POST /orders - Submit a new order.

        Args:
            ticker: Security ticker (e.g., "TGX")
            order_type: "MARKET" or "LIMIT"
            quantity: Number of shares (max 5000)
            action: "BUY" or "SELL"
            price: Required for LIMIT orders
        """
        params: dict = {
            "ticker": ticker,
            "type": order_type,
            "quantity": quantity,
            "action": action,
        }
        if order_type == "LIMIT" and price is not None:
            params["price"] = round(price, 2)
        return self._post("/orders", params)

    def get_open_orders(self, status: str = "OPEN") -> Optional[list]:
        """GET /orders - List orders by status."""
        return self._get("/orders", {"status": status})

    def cancel_order(self, order_id: int) -> bool:
        """DELETE /orders/{id} - Cancel a specific order."""
        return self._delete(f"/orders/{order_id}")

    def cancel_all_orders(self, ticker: Optional[str] = None) -> Optional[dict]:
        """POST /commands/cancel - Bulk cancel open orders."""
        params: dict = {}
        if ticker:
            params["ticker"] = ticker
        else:
            params["all"] = 1
        return self._post("/commands/cancel", params)

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
        """Get bid/ask/last for all securities."""
        securities = self.get_securities()
        if not securities:
            return {}
        return {
            s["ticker"]: {
                "bid": s.get("bid", 0) or 0,
                "ask": s.get("ask", 0) or 0,
                "last": s.get("last", 0) or 0,
                "position": s.get("position", 0),
                "volume": s.get("volume", 0) or 0,
            }
            for s in securities
        }

    def is_connected(self) -> bool:
        """Check if the RIT server is reachable and case is active."""
        case = self.get_case()
        return case is not None
