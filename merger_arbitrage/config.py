"""
RITC 2026 Merger Arbitrage Case - Central Configuration
All constants from the case package.
"""

from __future__ import annotations

from dataclasses import dataclass

# ============================================================
# API CONFIGURATION
# ============================================================
RIT_HOST = "localhost"
RIT_PORT = 9999
RIT_BASE_URL = f"http://{RIT_HOST}:{RIT_PORT}/v1"
API_KEY = ""  # Set via command line or environment variable

# ============================================================
# TIMING
# ============================================================
HEAT_DURATION_TICKS = 600       # 10 minutes = 600 seconds per sub-heat
NUM_HEATS = 5                   # 5 sub-heats per session
CALENDAR_MONTHS_PER_HEAT = 6   # Each heat represents 6 calendar months

# ============================================================
# POSITION LIMITS
# ============================================================
GROSS_POSITION_LIMIT = 100_000   # Max sum of |long| + |short| across all securities
NET_POSITION_LIMIT = 50_000      # Max net position
MAX_ORDER_SIZE = 5_000           # Max shares per order
COMMISSION_PER_SHARE = 0.02      # $0.02/share transaction cost

# ============================================================
# DEAL DEFINITIONS
# ============================================================


@dataclass(frozen=True)
class DealConfig:
    """Immutable deal parameters from the case package."""
    deal_id: str
    target_ticker: str
    acquirer_ticker: str
    structure: str           # "all_cash", "stock_for_stock", "mixed"
    cash_component: float    # Cash per share (0 for pure stock deals)
    exchange_ratio: float    # Shares of acquirer per target (0 for pure cash)
    target_start_price: float
    acquirer_start_price: float
    initial_probability: float
    sensitivity_multiplier: float
    industry: str


DEALS: dict[str, DealConfig] = {
    "D1": DealConfig(
        deal_id="D1", target_ticker="TGX", acquirer_ticker="PHR",
        structure="all_cash", cash_component=50.0, exchange_ratio=0.0,
        target_start_price=43.70, acquirer_start_price=47.50,
        initial_probability=0.70, sensitivity_multiplier=1.00,
        industry="Pharmaceuticals",
    ),
    "D2": DealConfig(
        deal_id="D2", target_ticker="BYL", acquirer_ticker="CLD",
        structure="stock_for_stock", cash_component=0.0, exchange_ratio=0.75,
        target_start_price=43.50, acquirer_start_price=79.30,
        initial_probability=0.55, sensitivity_multiplier=1.05,
        industry="Cloud Software",
    ),
    "D3": DealConfig(
        deal_id="D3", target_ticker="GGD", acquirer_ticker="PNR",
        structure="mixed", cash_component=33.0, exchange_ratio=0.20,
        target_start_price=31.50, acquirer_start_price=59.80,
        initial_probability=0.50, sensitivity_multiplier=1.10,
        industry="Energy / Infrastructure",
    ),
    "D4": DealConfig(
        deal_id="D4", target_ticker="FSR", acquirer_ticker="ATB",
        structure="all_cash", cash_component=40.0, exchange_ratio=0.0,
        target_start_price=30.50, acquirer_start_price=62.20,
        initial_probability=0.38, sensitivity_multiplier=1.30,
        industry="Banking",
    ),
    "D5": DealConfig(
        deal_id="D5", target_ticker="SPK", acquirer_ticker="EEC",
        structure="stock_for_stock", cash_component=0.0, exchange_ratio=1.20,
        target_start_price=52.80, acquirer_start_price=48.00,
        initial_probability=0.45, sensitivity_multiplier=1.15,
        industry="Renewable Energy",
    ),
}

# Ticker -> Deal lookup
TICKER_TO_DEAL: dict[str, str] = {}
for _deal_id, _dc in DEALS.items():
    TICKER_TO_DEAL[_dc.target_ticker] = _deal_id
    TICKER_TO_DEAL[_dc.acquirer_ticker] = _deal_id

# Company name -> Deal lookup (for news classification)
COMPANY_NAME_TO_DEAL: dict[str, str] = {
    "targenix": "D1", "pharmaco": "D1",
    "bytelayer": "D2", "cloudsys": "D2",
    "greengrid": "D3", "petronorth": "D3",
    "finsure": "D4", "atlas bank": "D4", "atlas": "D4",
    "solarpeak": "D5", "eastenergy": "D5",
}

# All tickers
ALL_TARGET_TICKERS = [d.target_ticker for d in DEALS.values()]
ALL_ACQUIRER_TICKERS = [d.acquirer_ticker for d in DEALS.values()]
ALL_TICKERS = ALL_TARGET_TICKERS + ALL_ACQUIRER_TICKERS

# ============================================================
# NEWS IMPACT PARAMETERS
# ============================================================
BASELINE_IMPACT: dict[tuple[str, str], float] = {
    ("positive", "small"):   +0.03,
    ("positive", "medium"):  +0.07,
    ("positive", "large"):   +0.14,
    ("negative", "small"):   -0.04,
    ("negative", "medium"):  -0.09,
    ("negative", "large"):   -0.18,
}

AMBIGUOUS_RANGE: dict[str, tuple[float, float]] = {
    "small":  (-0.03, +0.03),
    "medium": (-0.05, +0.05),
    "large":  (-0.07, +0.07),
}

CATEGORY_MULTIPLIERS: dict[str, float] = {
    "REG": 1.25,   # Regulatory / Antitrust
    "FIN": 1.00,   # Financing / Capital Markets
    "SHR": 0.90,   # Shareholder / Board Actions
    "ALT": 1.40,   # Strategic Alternatives / Competing Bids
    "PRC": 0.70,   # Process / Timing / Conditions
}

# ============================================================
# POLLING INTERVALS (milliseconds)
# ============================================================
POLL_CASE_MS = 500
POLL_SECURITIES_MS = 200
POLL_NEWS_MS = 300           # News is critical - poll aggressively
POLL_ORDERS_MS = 500

# ============================================================
# TRADING PARAMETERS (tunable)
# ============================================================
MIN_MISPRICING_THRESHOLD = 0.15  # $ minimum mispricing to trade
SPREAD_SAFETY_FACTOR = 0.5       # Only trade if mispricing > 50% of bid-ask spread
MAX_POSITION_PER_DEAL = 10_000   # Max shares in any single deal leg
URGENCY_CLOSE_TICKS = 60         # Start unwinding positions 60 ticks before heat end
URGENCY_CRITICAL_TICKS = 15      # Aggressive market orders 15 ticks before end
HEDGE_RATIO_TOLERANCE = 0.10     # Acceptable deviation from perfect hedge ratio
ORDER_REFRESH_INTERVAL_TICKS = 5 # Re-evaluate and replace stale orders every 5 ticks
