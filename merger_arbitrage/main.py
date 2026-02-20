"""
Merger Arbitrage Trading Bot - Main Entry Point.

Architecture:
  - Poller thread: fetches API data -> updates MarketState -> fires events
  - Strategy thread: wakes on events -> generate signals -> validate -> execute
  - Display thread: renders dashboard on its own cadence

Usage:
    python main.py --api-key XXXX
    python main.py --api-key XXXX --api-url http://localhost:9999/v1
    python main.py --demo    (runs with mock data for testing)
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from state.market_state import MarketState
from api.rit_client import RITClient
from api.data_poller import DataPoller
from strategy.signal_generator import SignalGenerator
from strategy.position_manager import PositionManager
from strategy.order_executor import OrderExecutor
from ui.console_display import ConsoleDisplay
from config import RIT_BASE_URL, URGENCY_CRITICAL_TICKS, HEAT_DURATION_TICKS


class StrategyThread:
    """Event-driven strategy execution loop."""

    def __init__(self, state: MarketState, executor: OrderExecutor):
        self.state = state
        self.signal_gen = SignalGenerator()
        self.pos_mgr = PositionManager()
        self.executor = executor
        self.running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="strategy"
        )
        self._thread.start()
        logging.getLogger(__name__).info("StrategyThread started")

    def stop(self):
        self.running = False
        self.state._recompute_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        logger = logging.getLogger(__name__)

        while self.running:
            # Wait for new data
            self.state.wait_for_change(timeout=0.3)
            if not self.running:
                break
            if self.state.case_status != "ACTIVE":
                continue

            try:
                # Generate signals from mispricing analysis
                raw_signals = self.signal_gen.generate_signals(self.state)

                # Check hedge drift and add rebalancing signals
                hedge_signals = self.pos_mgr.check_hedge_drift(self.state)
                raw_signals.extend(hedge_signals)

                # Validate against position limits
                validated = self.pos_mgr.validate_and_adjust(raw_signals, self.state)

                # Store for display
                self.state.active_recommendations = validated

                if validated:
                    # Execute orders
                    self.executor.execute_recommendations(validated, self.state)

                # Cancel stale orders
                self.executor.cancel_stale_orders(self.state)

                # Sync open orders for display
                self.executor.sync_open_orders(self.state)

                # Near end of heat: aggressive closeout
                ticks_left = self.state.ticks_remaining()
                if 0 < ticks_left <= URGENCY_CRITICAL_TICKS:
                    self.executor.cancel_all()
                    # Resubmit closeout as market orders
                    closeout = [
                        r for r in validated if r.urgency == "CRITICAL"
                    ]
                    for r in closeout:
                        r.order_type = "MARKET"
                        r.price = None
                    if closeout:
                        self.executor.execute_recommendations(closeout, self.state)

                self.state.mark_recomputed()

            except Exception as e:
                logger.error(f"Strategy error: {e}", exc_info=True)
                time.sleep(1)


def run_demo(state: MarketState):
    """Run a demo showing initial deal valuations without API connection."""
    state.initialize()
    display = ConsoleDisplay()

    print("\n" + "=" * 80)
    print("  MERGER ARBITRAGE BOT - DEMO MODE")
    print("  (No RIT server connection - showing initial valuations)")
    print("=" * 80)

    print("\n  INITIAL DEAL VALUATIONS:")
    print(f"  {'Deal':<5} {'Target':<5} {'Acq':<5} {'Type':<12} "
          f"{'p0':>6} {'K0':>8} {'V':>8} {'P*':>8} {'Start':>8} {'Spread':>7}")
    print("  " + "-" * 85)

    for deal_id, deal in state.deals.items():
        struct = deal.config.structure.replace("_", " ")
        K0 = deal.deal_value_K
        V = deal.standalone_value
        Pstar = deal.intrinsic_target_price
        spread = deal.spread_to_deal * 100

        print(
            f"  {deal_id:<5} {deal.config.target_ticker:<5} "
            f"{deal.config.acquirer_ticker:<5} {struct:<12} "
            f"{deal.probability:>5.0%} {K0:>8.2f} {V:>8.2f} "
            f"{Pstar:>8.2f} {deal.target_price:>8.2f} {spread:>+6.1f}%"
        )

    # Show example news impact
    print("\n  EXAMPLE NEWS IMPACTS:")
    from models.probability import compute_delta_p
    examples = [
        ("D4", "REG", "positive", "medium",
         "Regulators indicate remedies framework is acceptable"),
        ("D3", "FIN", "negative", "large",
         "Credit conditions deteriorate; lenders seek repricing"),
        ("D1", "SHR", "positive", "small",
         "Proxy advisor supports deal"),
        ("D5", "ALT", "negative", "medium",
         "Rival interest reported in target"),
        ("D2", "PRC", "positive", "large",
         "Timeline accelerated, closing expected ahead of schedule"),
    ]

    print(f"  {'Deal':<5} {'Cat':<5} {'Dir':<9} {'Sev':<7} {'dp':>8} {'Headline':<50}")
    print("  " + "-" * 90)
    for deal_id, cat, dirn, sev, headline in examples:
        dp = compute_delta_p(deal_id, cat, dirn, sev)
        print(f"  {deal_id:<5} {cat:<5} {dirn:<9} {sev:<7} {dp:>+7.4f} {headline:<50}")

    print("\n  To run live: python main.py --api-key YOUR_KEY")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="RITC 2026 Merger Arbitrage Trading Bot"
    )
    parser.add_argument("--api-key", default="", help="RIT API key")
    parser.add_argument("--api-url", default=RIT_BASE_URL, help="RIT API base URL")
    parser.add_argument("--demo", action="store_true", help="Run demo mode (no server)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler("merger_arb.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Initialize state
    state = MarketState()
    state.initialize()

    # Demo mode
    if args.demo:
        run_demo(state)
        return

    if not args.api_key:
        print("Error: --api-key is required (or use --demo for demo mode)")
        sys.exit(1)

    # API client
    client = RITClient(api_key=args.api_key, base_url=args.api_url)

    # Check connection
    if not client.is_connected():
        logger.warning("Cannot connect to RIT server. Will keep trying...")

    # Components
    poller = DataPoller(client, state)
    executor = OrderExecutor(client)
    strategy = StrategyThread(state, executor)
    display = ConsoleDisplay()

    # Graceful shutdown
    shutdown_event = threading.Event()

    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        shutdown_event.set()
        executor.cancel_all()

    signal.signal(signal.SIGINT, signal_handler)

    # Start threads
    poller.start()
    strategy.start()
    logger.info("Merger Arbitrage Bot started")

    try:
        while not shutdown_event.is_set():
            try:
                display.render(state)
            except Exception as e:
                logger.error(f"Display error: {e}")
            shutdown_event.wait(timeout=1.0)
    finally:
        logger.info("Shutting down...")
        executor.cancel_all()
        strategy.stop()
        poller.stop()
        logger.info("Merger Arbitrage Bot stopped")


if __name__ == "__main__":
    main()
