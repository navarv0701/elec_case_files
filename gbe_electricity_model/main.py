"""
GBE Electricity Trading - Main Entry Point.

Architecture:
  - Poller thread: fetches API data -> updates GameState -> fires events
  - Optimizer thread: wakes on events -> recomputes recommendations instantly
  - Display thread: renders dashboard on its own cadence (doesn't block optimizer)
  - Alert system: prints one-line quick-actions on CRITICAL events (forecast, closeout)

The optimizer reacts to the final forecast in <5ms, well within the
5-second window before day end.

Usage:
    python main.py --role producer --api-key XXXX
    python main.py --role distributor --api-key XXXX
    python main.py --role trader --api-key XXXX
    python main.py --role all               # All roles combined view
    python main.py --demo                   # Demo mode with mock data
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

from state.game_state import GameState, Recommendation
from api.rit_client import RITClient
from api.data_poller import DataPoller
from optimizer.producer_optimizer import ProducerOptimizer
from optimizer.distributor_optimizer import DistributorOptimizer
from optimizer.trader_optimizer import TraderOptimizer
from optimizer.team_coordinator import TeamCoordinator
from ui.console_display import ConsoleDisplay
from ui.excel_writer import ExcelWriter
from config import RIT_BASE_URL, CUSTOMER_PRICE_PER_MWH, DISPOSAL_PENALTY


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("gbe_trading.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


URGENCY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GBE Electricity Trading Optimizer")
    parser.add_argument(
        "--role", choices=["producer", "distributor", "trader", "all"],
        default="all", help="Team role to optimize for (default: all)"
    )
    parser.add_argument("--api-key", default="", help="RIT API key")
    parser.add_argument(
        "--api-url", default=RIT_BASE_URL,
        help=f"RIT API base URL (default: {RIT_BASE_URL})"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run in demo mode with mock data (no API connection)"
    )
    parser.add_argument("--no-excel", action="store_true", help="Disable Excel output")
    parser.add_argument(
        "--excel-path", default=None,
        help="Path for Excel dashboard output"
    )
    parser.add_argument(
        "--refresh-rate", type=float, default=1.0,
        help="Display refresh rate in seconds (default: 1.0)"
    )
    return parser.parse_args()


# ============================================================
# Quick-Action Alert System
# ============================================================

def print_quick_actions(recs: list[Recommendation], state: GameState):
    """Print a compact one-line-per-action summary for rapid human execution.

    This is what the team reads in the 5-second window after the final forecast.
    Format: [URGENCY] ACTION QTY TICKER @ PRICE | reason
    """
    if not recs:
        return

    # Only show actionable items (skip DECLINE_TENDER, zero-quantity info recs)
    actionable = [r for r in recs
                  if r.quantity and r.quantity > 0
                  and r.action in ("BUY", "SELL", "ACCEPT_TENDER")]

    if not actionable:
        return

    print("\n" + "=" * 70)
    print(f"  QUICK ACTIONS  |  Day {state.current_day}  |  "
          f"{state.ticks_remaining_in_day()}s left  |  "
          f"Tick {state.current_tick}")
    print("=" * 70)

    for r in actionable[:8]:  # Top 8 most urgent
        price_str = f"@ ${r.price:.0f}" if r.price else "@ MKT"
        pnl_str = f"  E[PnL] ${r.expected_pnl:,.0f}" if r.expected_pnl > 0 else ""
        risk_str = f"  RISK ${r.penalty_risk:,.0f}" if r.penalty_risk > 0 else ""

        # Color via ANSI codes for terminal
        if r.urgency == "CRITICAL":
            prefix = "\033[1;41;37m CRITICAL \033[0m"
        elif r.urgency == "HIGH":
            prefix = "\033[1;31m HIGH \033[0m"
        elif r.urgency == "MEDIUM":
            prefix = "\033[1;33m MED  \033[0m"
        else:
            prefix = "\033[2m LOW  \033[0m"

        action_color = "\033[32m" if r.action == "BUY" else "\033[31m"
        print(f"  {prefix} {action_color}{r.action:4s}\033[0m "
              f"{r.quantity:3d} {r.ticker:<12s} {price_str}"
              f"{pnl_str}{risk_str}")

    print("=" * 70 + "\n")


# ============================================================
# Optimizer Thread
# ============================================================

class OptimizerThread:
    """Runs optimizers on a dedicated thread, triggered by events.

    Instead of polling on a timer, it wakes up when:
    1. A new forecast arrives (event-driven, <1ms latency)
    2. Market data changes significantly
    3. A tender arrives
    4. Approaching day-end closeout zone
    """

    def __init__(self, state: GameState, role: str):
        self.state = state
        self.role = role
        self.running = False
        self._thread = None

        # Persistent optimizer instances (maintain internal state between calls)
        self.producer_opt = ProducerOptimizer()
        self.distributor_opt = DistributorOptimizer()
        self.trader_opt = TraderOptimizer()
        self.coordinator = TeamCoordinator()

        # Track last computation time for the quick-action alert
        self._last_compute_ms = 0.0

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="optimizer")
        self._thread.start()

    def stop(self):
        self.running = False
        self.state._recompute_event.set()  # Wake up to exit
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        """Event-driven optimizer loop."""
        while self.running:
            # Block until new data arrives (or timeout for periodic refresh)
            self.state.wait_for_change(timeout=0.5)

            if not self.running:
                break

            if self.state.case_status != "ACTIVE":
                continue

            # Recompute
            t0 = time.perf_counter()
            recs = self._compute()
            self._last_compute_ms = (time.perf_counter() - t0) * 1000

            self.state.set_recommendations(recs)
            self.state.mark_recomputed()

            # Print quick-actions if there are CRITICAL or HIGH urgency items
            has_critical = any(r.urgency in ("CRITICAL", "HIGH") for r in recs)
            if has_critical:
                print_quick_actions(recs, self.state)

    def _compute(self) -> list[Recommendation]:
        """Run all relevant optimizers and return sorted recommendations."""
        all_recs = []

        if self.role in ("producer", "all"):
            all_recs.extend(self.producer_opt.optimize(self.state))
        if self.role in ("distributor", "all"):
            all_recs.extend(self.distributor_opt.optimize(self.state))
        if self.role in ("trader", "all"):
            all_recs.extend(self.trader_opt.optimize(self.state))

        all_recs.sort(key=lambda r: URGENCY_ORDER.get(r.urgency, 3))
        return all_recs

    def compute_once(self) -> list[Recommendation]:
        """Single synchronous compute (for demo mode)."""
        recs = self._compute()
        self.state.set_recommendations(recs)
        return recs


# ============================================================
# Display Thread
# ============================================================

class DisplayThread:
    """Renders the dashboard on its own cadence without blocking the optimizer."""

    def __init__(self, display: ConsoleDisplay, state: GameState,
                 refresh_rate: float = 1.0, excel: ExcelWriter = None):
        self.display = display
        self.state = state
        self.refresh_rate = refresh_rate
        self.excel = excel
        self.running = False
        self._thread = None
        self._excel_counter = 0

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="display")
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while self.running:
            try:
                self.display.render(self.state)

                # Excel update every 5 display cycles
                self._excel_counter += 1
                if self.excel and self._excel_counter % 5 == 0:
                    try:
                        self.excel.update(self.state)
                    except Exception:
                        pass

            except Exception as e:
                logger.debug(f"Display error: {e}")

            time.sleep(self.refresh_rate)


# ============================================================
# Demo State
# ============================================================

def create_demo_state() -> GameState:
    """Create a GameState populated with demo data for testing the UI."""
    from state.game_state import RAEBulletin

    state = GameState()
    state.current_tick = 275
    state.current_day = 2
    state.case_status = "ACTIVE"

    # Market prices
    state.elec_fwd_bid = 38.0
    state.elec_fwd_ask = 40.0
    state.elec_fwd_last = 39.0
    state.ng_bid = 4.50
    state.ng_ask = 4.75
    state.ng_last = 4.60
    state.elec_spot_bid[2] = 37.50
    state.elec_spot_ask[2] = 38.50

    # Positions
    state.elec_f_position = -5
    state.ng_position = 16
    state.elec_day_positions = {"ELEC-day2": 3}

    # P&L
    state.nlv = 125_000.0
    state.realized_pnl = 45_000.0
    state.unrealized_pnl = 80_000.0

    # Weather forecasts for day 3
    state.add_sunshine_forecast(3, 8.5, 1)
    state.add_sunshine_forecast(3, 9.2, 2)
    state.add_temperature_forecast(3, 28.0, 1)
    state.add_temperature_forecast(3, 30.5, 2)

    # Weather for day 2 (fully resolved)
    state.add_sunshine_forecast(2, 7.0, 1)
    state.add_sunshine_forecast(2, 7.5, 2)
    state.add_sunshine_forecast(2, 7.3, 3)
    state.add_temperature_forecast(2, 25.0, 1)
    state.add_temperature_forecast(2, 26.5, 2)
    state.add_temperature_forecast(2, 27.0, 3)

    # RAE bulletin
    state.add_rae_bulletin(RAEBulletin(
        day=2, price_low=35.0, price_high=42.0,
        volume_buy=150, volume_sell=200,
        bulletin_number=1, tick_received=200,
    ))

    # Production tracking
    state.gas_production[2] = 2
    state.gas_production[3] = 0

    return state


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    logger.info(f"Starting GBE Electricity Trading Optimizer")
    logger.info(f"Role: {args.role} | Demo: {args.demo}")

    # Initialize state
    state = GameState()

    # Initialize display
    display = ConsoleDisplay(role=args.role)

    # Initialize Excel writer
    excel = None
    if not args.no_excel:
        excel = ExcelWriter(filepath=args.excel_path)
        try:
            excel.initialize()
            logger.info("Excel dashboard initialized")
        except Exception as e:
            logger.warning(f"Excel initialization failed: {e}")
            excel = None

    # Initialize optimizer thread
    opt_thread = OptimizerThread(state, args.role)

    # Initialize display thread
    disp_thread = DisplayThread(display, state, args.refresh_rate, excel)

    # Setup polling (or demo mode)
    poller = None
    if args.demo:
        state = create_demo_state()
        # Update references in threads
        opt_thread.state = state
        disp_thread.state = state
        logger.info("Running in DEMO mode with mock data")

        # Run optimizer once to populate recommendations
        opt_thread.compute_once()

        # In demo mode, just render once and exit
        display.render(state)
        print_quick_actions(state.active_recommendations, state)
        return
    else:
        client = RITClient(base_url=args.api_url, api_key=args.api_key)
        poller = DataPoller(client, state)

    # Register event callback for logging
    def on_event(event_type, data):
        logger.info(f"Event: {event_type} {data}")
    state.on_event(on_event)

    # Graceful shutdown
    shutdown = False

    def signal_handler(sig, frame):
        nonlocal shutdown
        shutdown = True
        logger.info("Shutdown signal received")

    signal.signal(signal.SIGINT, signal_handler)

    # Start all threads
    poller.start()
    opt_thread.start()
    disp_thread.start()
    logger.info(f"All threads started. Connected to RIT API at {args.api_url}")

    # Main thread just waits for shutdown
    try:
        while not shutdown:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        opt_thread.stop()
        disp_thread.stop()
        poller.stop()
        logger.info("GBE Electricity Trading Optimizer stopped.")


if __name__ == "__main__":
    main()
