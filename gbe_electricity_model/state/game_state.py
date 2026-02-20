"""
Central game state - single source of truth for all real-time data.
Updated by the DataPoller, consumed by all optimizers and UI components.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from threading import Lock, Event
from typing import Optional, Callable
import time

from config import (
    TICKS_PER_DAY, DAY_START_TICK, DAY_END_TICK,
    FORECAST_INITIAL_OFFSET, FORECAST_NOON_OFFSET, FORECAST_FINAL_OFFSET,
)


@dataclass
class Recommendation:
    """A single actionable trade recommendation."""
    action: str           # BUY, SELL, ACCEPT_TENDER, DECLINE_TENDER, CONVERT
    ticker: str           # e.g. "ELEC-F", "NG", "ELEC-day3"
    quantity: int          # Number of contracts
    price: Optional[float] = None  # Suggested price (None for market order)
    reason: str = ""
    urgency: str = "LOW"   # LOW, MEDIUM, HIGH, CRITICAL
    expected_pnl: float = 0.0
    penalty_risk: float = 0.0


@dataclass
class ForecastUpdate:
    """A single weather forecast update."""
    day: int              # Which day this forecast is for
    update_number: int    # 1=initial, 2=noon, 3=final
    value: float          # Sunshine hours or temperature
    tick_received: int    # When this was received
    confidence: str       # LOW, MEDIUM, HIGH


@dataclass
class RAEBulletin:
    """RAE Price and Volume Bulletin."""
    day: int
    price_low: float
    price_high: float
    volume_buy: int       # Contracts available to buy on spot
    volume_sell: int      # Contracts available to sell on spot
    bulletin_number: int  # 1 or 2
    tick_received: int


@dataclass
class FactoryTender:
    """A factory tender offer received by Traders."""
    tender_id: int
    action: str           # BUY or SELL (from factory's perspective)
    ticker: str
    quantity: int
    price: float
    expiration_tick: int
    is_fixed_price: bool = True


@dataclass
class GameState:
    """
    Single source of truth for all game state.
    Thread-safe via a lock for concurrent access from poller and optimizer threads.
    """
    _lock: Lock = field(default_factory=Lock, repr=False)

    # === Time ===
    current_tick: int = 0
    current_day: int = 1
    case_status: str = "STOPPED"  # STOPPED, ACTIVE, PAUSED

    # === Weather Forecasts ===
    # Keyed by target day -> list of ForecastUpdate in chronological order
    sunshine_forecasts: dict[int, list[ForecastUpdate]] = field(default_factory=dict)
    temperature_forecasts: dict[int, list[ForecastUpdate]] = field(default_factory=dict)

    # === Market Data ===
    elec_fwd_bid: float = 0.0
    elec_fwd_ask: float = 0.0
    elec_fwd_last: float = 0.0
    ng_bid: float = 0.0
    ng_ask: float = 0.0
    ng_last: float = 0.0

    # Spot prices set by RAE (per day)
    elec_spot_bid: dict[int, float] = field(default_factory=dict)
    elec_spot_ask: dict[int, float] = field(default_factory=dict)

    # === RAE Bulletins ===
    rae_bulletins: dict[int, list[RAEBulletin]] = field(default_factory=dict)

    # === Positions ===
    elec_f_position: int = 0        # Net ELEC-F contracts (+ = long, - = short)
    ng_position: int = 0            # Net NG contracts
    elec_day_positions: dict[str, int] = field(default_factory=dict)  # "ELEC-day2" -> qty

    # === P&L ===
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    nlv: float = 0.0

    # === Production Tracking ===
    ng_purchased_today: int = 0     # NG contracts bought today (for tomorrow's gas production)
    gas_plants_active: int = 0      # Number of gas plants leased for conversion
    solar_production: dict[int, float] = field(default_factory=dict)  # day -> contracts produced
    gas_production: dict[int, int] = field(default_factory=dict)      # day -> contracts produced
    total_production: dict[int, float] = field(default_factory=dict)  # day -> total contracts

    # === Distributor Tracking ===
    customer_demand: dict[int, float] = field(default_factory=dict)   # day -> contracts demanded
    electricity_purchased: dict[int, float] = field(default_factory=dict)  # day -> contracts bought

    # === Factory Tenders ===
    pending_tenders: list[FactoryTender] = field(default_factory=list)
    completed_tenders: list[FactoryTender] = field(default_factory=list)

    # === News History ===
    news_history: list[dict] = field(default_factory=list)
    last_news_id: int = 0

    # === Recommendations (output from optimizer) ===
    active_recommendations: list[Recommendation] = field(default_factory=list)

    # === Event System ===
    # Callbacks fired when high-priority data arrives (forecast, tender, etc.)
    _event_callbacks: list[Callable] = field(default_factory=list, repr=False)
    # Threading event: set whenever new data arrives that should trigger recompute
    _recompute_event: Event = field(default_factory=Event, repr=False)
    # Change tracking: monotonically increasing version counters
    _forecast_version: int = 0     # Bumped on any new forecast
    _market_version: int = 0       # Bumped on market price change
    _position_version: int = 0     # Bumped on position change
    _last_recompute_version: int = 0  # Last version seen by optimizer

    # ------------------------------------------------------------------
    # Event System
    # ------------------------------------------------------------------

    def on_event(self, callback: Callable):
        """Register a callback to fire on high-priority events.
        Callback receives (event_type: str, data: dict).
        """
        self._event_callbacks.append(callback)

    def _fire_event(self, event_type: str, data: dict = None):
        """Fire all registered callbacks. Non-blocking."""
        self._recompute_event.set()
        for cb in self._event_callbacks:
            try:
                cb(event_type, data or {})
            except Exception:
                pass

    def wait_for_change(self, timeout: float = 1.0) -> bool:
        """Block until new data arrives or timeout. Returns True if data arrived."""
        triggered = self._recompute_event.wait(timeout=timeout)
        self._recompute_event.clear()
        return triggered

    def needs_recompute(self) -> bool:
        """Check if any data has changed since last optimizer run."""
        current = self._forecast_version + self._market_version + self._position_version
        return current != self._last_recompute_version

    def mark_recomputed(self):
        """Mark that the optimizer has consumed all current data."""
        self._last_recompute_version = (
            self._forecast_version + self._market_version + self._position_version
        )

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def current_day_from_tick(self) -> int:
        """Convert current tick to simulation day (1-5)."""
        return min(5, max(1, (self.current_tick // TICKS_PER_DAY) + 1))

    def tick_within_day(self) -> int:
        """Current tick offset within the current day (0-179)."""
        return self.current_tick % TICKS_PER_DAY

    def ticks_remaining_in_day(self) -> int:
        """Ticks left in the current day."""
        return TICKS_PER_DAY - self.tick_within_day()

    def ticks_remaining_total(self) -> int:
        """Ticks left in the entire simulation."""
        return max(0, 900 - self.current_tick)

    def is_forecast_time(self) -> Optional[str]:
        """Check if current tick aligns with a forecast update time.
        Returns 'initial', 'noon', 'final', or None.
        """
        offset = self.tick_within_day()
        if offset == FORECAST_INITIAL_OFFSET:
            return "initial"
        if offset == FORECAST_NOON_OFFSET:
            return "noon"
        if offset == FORECAST_FINAL_OFFSET:
            return "final"
        return None

    def day_for_next_production(self) -> int:
        """Production decided today is delivered the next day."""
        return self.current_day + 1

    def is_day_close_warning(self) -> bool:
        """True if we're within closeout warning zone (60 ticks before day end)."""
        return self.ticks_remaining_in_day() <= 60

    def is_day_close_urgent(self) -> bool:
        """True if we're within urgent closeout zone (20 ticks before day end)."""
        return self.ticks_remaining_in_day() <= 20

    def is_day_close_critical(self) -> bool:
        """True if we're within critical closeout zone (5 ticks before day end)."""
        return self.ticks_remaining_in_day() <= 5

    def elec_day_ticker(self, day: int) -> str:
        """Get the ELEC-dayX ticker for a given day."""
        return f"ELEC-day{day}"

    def get_elec_day_position(self, day: int) -> int:
        """Get net position for a specific day's electricity spot."""
        ticker = self.elec_day_ticker(day)
        return self.elec_day_positions.get(ticker, 0)

    # ------------------------------------------------------------------
    # Best Forecast Accessors
    # ------------------------------------------------------------------

    def best_sunshine_forecast(self, day: int) -> Optional[float]:
        """Get the most recent sunshine forecast for a given day."""
        forecasts = self.sunshine_forecasts.get(day, [])
        if not forecasts:
            return None
        return forecasts[-1].value

    def best_temperature_forecast(self, day: int) -> Optional[float]:
        """Get the most recent temperature forecast for a given day."""
        forecasts = self.temperature_forecasts.get(day, [])
        if not forecasts:
            return None
        return forecasts[-1].value

    def sunshine_uncertainty(self, day: int) -> float:
        """Uncertainty in sunshine forecast (hours).
        Decreases with each update: initial=3.0, noon=1.5, final=0.2
        """
        forecasts = self.sunshine_forecasts.get(day, [])
        n = len(forecasts)
        return {0: 5.0, 1: 3.0, 2: 1.5, 3: 0.2}.get(n, 5.0)

    def temperature_uncertainty(self, day: int) -> float:
        """Uncertainty in temperature forecast (degrees C).
        Decreases with each update: initial=5.0, noon=2.5, final=0.5
        """
        forecasts = self.temperature_forecasts.get(day, [])
        n = len(forecasts)
        return {0: 8.0, 1: 5.0, 2: 2.5, 3: 0.5}.get(n, 8.0)

    def latest_rae_bulletin(self, day: int) -> Optional[RAEBulletin]:
        """Get the most recent RAE bulletin for a given day."""
        bulletins = self.rae_bulletins.get(day, [])
        if not bulletins:
            return None
        return bulletins[-1]

    # ------------------------------------------------------------------
    # Thread-Safe Update Methods
    # ------------------------------------------------------------------

    def update_tick(self, tick: int, status: str = "ACTIVE"):
        """Update the current tick and derived day."""
        with self._lock:
            self.current_tick = tick
            self.current_day = self.current_day_from_tick()
            self.case_status = status

    def update_market_prices(self, fwd_bid: float, fwd_ask: float, fwd_last: float,
                              ng_bid: float, ng_ask: float, ng_last: float):
        """Update forward and NG market prices."""
        with self._lock:
            self.elec_fwd_bid = fwd_bid
            self.elec_fwd_ask = fwd_ask
            self.elec_fwd_last = fwd_last
            self.ng_bid = ng_bid
            self.ng_ask = ng_ask
            self.ng_last = ng_last
            self._market_version += 1

    def update_positions(self, elec_f: int, ng: int, elec_days: dict[str, int]):
        """Update position data."""
        with self._lock:
            self.elec_f_position = elec_f
            self.ng_position = ng
            self.elec_day_positions = elec_days
            self._position_version += 1

    def add_sunshine_forecast(self, day: int, value: float, update_num: int):
        """Record a new sunshine forecast. Fires FORECAST event."""
        with self._lock:
            if day not in self.sunshine_forecasts:
                self.sunshine_forecasts[day] = []
            confidence = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}.get(update_num, "LOW")
            self.sunshine_forecasts[day].append(ForecastUpdate(
                day=day, update_number=update_num, value=value,
                tick_received=self.current_tick, confidence=confidence
            ))
            self._forecast_version += 1
        self._fire_event("FORECAST", {"type": "sunshine", "day": day,
                                       "value": value, "update": update_num})

    def add_temperature_forecast(self, day: int, value: float, update_num: int):
        """Record a new temperature forecast. Fires FORECAST event."""
        with self._lock:
            if day not in self.temperature_forecasts:
                self.temperature_forecasts[day] = []
            confidence = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}.get(update_num, "LOW")
            self.temperature_forecasts[day].append(ForecastUpdate(
                day=day, update_number=update_num, value=value,
                tick_received=self.current_tick, confidence=confidence
            ))
            self._forecast_version += 1
        self._fire_event("FORECAST", {"type": "temperature", "day": day,
                                       "value": value, "update": update_num})

    def add_rae_bulletin(self, bulletin: RAEBulletin):
        """Record a new RAE bulletin. Fires BULLETIN event."""
        with self._lock:
            if bulletin.day not in self.rae_bulletins:
                self.rae_bulletins[bulletin.day] = []
            self.rae_bulletins[bulletin.day].append(bulletin)
            self._market_version += 1
        self._fire_event("BULLETIN", {"day": bulletin.day})

    def add_tender(self, tender: FactoryTender):
        """Add a new factory tender. Fires TENDER event."""
        with self._lock:
            self.pending_tenders.append(tender)
        self._fire_event("TENDER", {"tender_id": tender.tender_id})

    def set_recommendations(self, recs: list[Recommendation]):
        """Replace the active recommendation list."""
        with self._lock:
            self.active_recommendations = recs

    def update_pnl(self, realized: float, unrealized: float, nlv: float):
        """Update P&L tracking."""
        with self._lock:
            self.realized_pnl = realized
            self.unrealized_pnl = unrealized
            self.nlv = nlv
