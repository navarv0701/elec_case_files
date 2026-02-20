"""
GBE Electricity Trading Case - Central Configuration
All constants from the RITC 2026 Case Package.
"""

# ============================================================
# TIMING
# ============================================================
TOTAL_TICKS = 900              # 15 minutes = 900 seconds
TICKS_PER_DAY = 180            # 3 minutes per simulated day
NUM_DAYS = 5

# Day boundaries (inclusive tick ranges)
DAY_START_TICK = {
    1: 0,
    2: 180,
    3: 360,
    4: 540,
    5: 720,
}
DAY_END_TICK = {
    1: 179,
    2: 359,
    3: 539,
    4: 719,
    5: 899,
}

# Forecast timing (ticks relative to day start)
FORECAST_INITIAL_OFFSET = 0      # Beginning of day
FORECAST_NOON_OFFSET = 90        # 1:30 into the day (noon in sim)
FORECAST_FINAL_OFFSET = 150      # 30 seconds before day end (evening)

# ============================================================
# SECURITIES
# ============================================================
ELEC_SPOT_CONTRACT_SIZE = 100    # MWh per ELEC-dayX contract
ELEC_FWD_CONTRACT_SIZE = 500     # MWh per ELEC-F contract
NG_CONTRACT_SIZE = 100           # MMBtu per NG contract

# Conversion
NG_TO_ELEC_RATIO = 8             # 8 NG contracts -> 1 ELEC contract
MAX_GAS_PLANTS = 10              # Max simultaneous gas plants

# ============================================================
# POSITION / TRADE LIMITS
# ============================================================
ELEC_F_MAX_TRADE = 10            # Max contracts per ELEC-F order
ELEC_F_MAX_NET_POSITION = 60     # Max net position in ELEC-F
NG_MAX_TRADE = 80                # Max contracts per NG order
NG_MAX_NET_POSITION = 80         # Max net position in NG
ELEC_DAY_MAX_NET_POSITION = 300  # Max net position in any ELEC-dayX

# ============================================================
# COSTS & PENALTIES
# ============================================================
DISPOSAL_PENALTY = 20_000        # $/contract for unsold electricity (MECC fee)
SHORTFALL_PENALTY = 20_000       # $/contract for unmet customer demand
CLOSEOUT_PENALTY = 20_000        # $/contract for unclosed day 2-5 positions
CUSTOMER_PRICE_PER_MWH = 70      # $/MWh that Distributors sell to customers
CUSTOMER_REVENUE_PER_CONTRACT = CUSTOMER_PRICE_PER_MWH * ELEC_SPOT_CONTRACT_SIZE  # $7,000

# No transaction costs on ELEC-F and NG
ELEC_F_TRANSACTION_COST = 0
NG_TRANSACTION_COST = 0
RAE_BID_ASK_SPREAD = 0.01       # 1 cent spread on spot market

# ============================================================
# PRODUCTION
# ============================================================
SOLAR_MULTIPLIER = 6             # ELEC_solar = 6 * H_day (contracts)
SOLAR_COST_PER_MWH = 0           # Free

# ============================================================
# MARKET STRUCTURE (approximate, from case description)
# ============================================================
APPROX_NUM_PRODUCERS = 28
APPROX_NUM_DISTRIBUTORS = 28
APPROX_NUM_TRADERS = 56
APPROX_TOTAL_PARTICIPANTS = 112

# ============================================================
# OPTIMIZER PARAMETERS (tunable)
# ============================================================
GAS_PROFITABILITY_MARGIN = 1.05   # Require 5% margin to produce gas electricity
FORWARD_PREMIUM_THRESHOLD = 1.02  # Sell forward when fwd > 2% above expected spot
FORWARD_PROCUREMENT_RATIO = 0.70  # Buy 70% of demand via forwards initially
TENDER_MIN_PROFIT = 5_000         # Minimum $ profit to accept a factory tender
CLOSEOUT_WARNING_TICKS = 60       # Start closeout warnings 60 ticks before day end
CLOSEOUT_URGENT_TICKS = 20        # Urgent closeout 20 ticks before day end
CLOSEOUT_CRITICAL_TICKS = 5       # Critical alert 5 ticks before day end

# ============================================================
# API CONFIGURATION
# ============================================================
RIT_HOST = "localhost"
RIT_PORT = 9999
RIT_BASE_URL = f"http://{RIT_HOST}:{RIT_PORT}/v1"
API_KEY = ""  # Set via command line or environment variable

# Polling intervals (milliseconds)
POLL_CASE_MS = 500
POLL_SECURITIES_MS = 250
POLL_NEWS_MS = 1000
POLL_TENDERS_MS = 2000
POLL_LIMITS_MS = 5000
POLL_TRADER_MS = 2000

# ============================================================
# DEMAND MODEL REFERENCE
# ============================================================
# ELEC_customers = 200 - 15*AT + 0.8*AT^2 - 0.01*AT^3
# where AT = average temperature in Celsius
# Output is in contracts (each contract = 100 MWh)
# This is the formula exactly as given in the case package.

# ============================================================
# DAY-BY-DAY RULES
# ============================================================
# Day 1: No electricity produced for delivery. Market opens.
#         Production decisions for Day 2 are made.
# Day 2-4: Normal production/trading cycle.
#           ELEC-dayX positions MUST be zero at day end ($20K/contract fine).
# Day 5: Last day. Can produce for Day 6 and buy ELEC-F for Day 6.
#         ELEC-day5 must close at zero (fine applies).
#         ELEC-day6 closes at final RAE price, NO fine.
DAYS_WITH_CLOSEOUT_FINE = {2, 3, 4, 5}  # ELEC-dayX fined if nonzero at day end
DAY6_NO_FINE = True                       # ELEC-day6 settles at RAE price, no fine
