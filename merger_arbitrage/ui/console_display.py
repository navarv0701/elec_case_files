"""
Rich terminal dashboard for merger arbitrage trading bot.
Displays deal status, probabilities, positions, and recent activity.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state.market_state import MarketState


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


class ConsoleDisplay:
    """Terminal-based dashboard showing real-time merger arb state."""

    def __init__(self):
        self._render_count = 0

    def render(self, state: MarketState):
        """Render the full dashboard to terminal."""
        self._render_count += 1
        clear_screen()

        self._print_header(state)
        self._print_deals(state)
        self._print_positions(state)
        self._print_recent_news(state)
        self._print_recommendations(state)
        self._print_footer(state)

    def _print_header(self, state: MarketState):
        print("=" * 100)
        print("  RITC 2026 MERGER ARBITRAGE BOT")
        print("=" * 100)
        ticks_left = state.ticks_remaining()
        status_char = "LIVE" if state.case_status == "ACTIVE" else state.case_status
        print(
            f"  Status: {status_char}  |  "
            f"Tick: {state.current_tick}/{600}  |  "
            f"Time Left: {ticks_left}s  |  "
            f"Period: {state.current_period}  |  "
            f"NLV: ${state.nlv:,.2f}"
        )
        gross = state.gross_position()
        print(
            f"  Gross Position: {gross:,}/100,000  |  "
            f"Open Orders: {len(state.open_orders)}"
        )
        print("-" * 100)

    def _print_deals(self, state: MarketState):
        print()
        print(
            f"  {'Deal':<5} {'Target':<5} {'Acquirer':<5} {'Type':<8} "
            f"{'Prob':>6} {'K':>8} {'V':>8} {'P*':>8} {'Mkt':>8} "
            f"{'Misp':>8} {'Misp%':>7} {'Spread':>7} {'Status':<10}"
        )
        print("  " + "-" * 96)

        for deal_id, deal in state.deals.items():
            status = ""
            if deal.resolved:
                status = deal.resolution or "resolved"
            elif deal.probability > 0.9:
                status = "likely"
            elif deal.probability < 0.1:
                status = "unlikely"

            struct = deal.config.structure.replace("_", " ")[:7]
            misp = deal.target_mispricing
            misp_pct = deal.target_mispricing_pct
            spread_pct = deal.spread_to_deal * 100

            # Color indicators (using text markers)
            misp_marker = "+" if misp > 0.15 else ("-" if misp < -0.15 else " ")

            print(
                f"  {deal_id:<5} "
                f"{deal.config.target_ticker:<5} "
                f"{deal.config.acquirer_ticker:<5} "
                f"{struct:<8} "
                f"{deal.probability:>5.1%} "
                f"{deal.deal_value_K:>8.2f} "
                f"{deal.standalone_value:>8.2f} "
                f"{deal.intrinsic_target_price:>8.2f} "
                f"{deal.target_price:>8.2f} "
                f"{misp_marker}{abs(misp):>7.2f} "
                f"{misp_pct:>+6.1f}% "
                f"{spread_pct:>+6.1f}% "
                f"{status:<10}"
            )

    def _print_positions(self, state: MarketState):
        print()
        print("  POSITIONS:")
        has_positions = False
        for deal_id, deal in state.deals.items():
            t_pos = deal.target_position
            a_pos = deal.acquirer_position
            if t_pos == 0 and a_pos == 0:
                continue
            has_positions = True

            # Check hedge ratio
            hedge_status = ""
            if deal.config.structure != "all_cash" and t_pos != 0:
                expected_acq = -int(t_pos * deal.ideal_hedge_ratio)
                drift = a_pos - expected_acq
                if abs(drift) > abs(t_pos) * 0.10:
                    hedge_status = f" DRIFT={drift:+d}"
                else:
                    hedge_status = " hedged"

            print(
                f"    {deal_id}: "
                f"{deal.config.target_ticker}={t_pos:+,d}  "
                f"{deal.config.acquirer_ticker}={a_pos:+,d}"
                f"{hedge_status}"
            )

        if not has_positions:
            print("    (no positions)")

    def _print_recent_news(self, state: MarketState):
        print()
        print("  RECENT NEWS (last 5):")
        recent = state.classified_news[-5:] if state.classified_news else []
        if not recent:
            print("    (no news yet)")
            return

        for item in reversed(recent):
            c = item.get("classified")
            if c is None:
                continue
            deal = c.deal_id or "???"
            cat = c.category
            dirn = c.direction[:3]
            sev = c.severity[:3]
            tick = item.get("tick", 0)
            headline = item.get("headline", "")[:60]
            print(
                f"    t={tick:>3d} [{deal}] {cat}/{dirn}/{sev} "
                f"conf={c.confidence:.1f} | {headline}"
            )

    def _print_recommendations(self, state: MarketState):
        print()
        print("  ACTIVE SIGNALS:")
        recs = state.active_recommendations
        if not recs:
            print("    (none)")
            return

        for rec in recs[:8]:
            price_str = f"${rec.price:.2f}" if rec.price else "MKT"
            hedge = " [H]" if rec.is_hedge_leg else ""
            print(
                f"    [{rec.urgency:<8}] {rec.action} {rec.quantity:>5,d} "
                f"{rec.ticker:<5} @ {price_str:<8} "
                f"E[P]=${rec.expected_profit:>8,.0f}{hedge} | {rec.reason[:50]}"
            )

        if len(recs) > 8:
            print(f"    ... and {len(recs) - 8} more signals")

    def _print_footer(self, state: MarketState):
        print()
        print("-" * 100)
        print("  Press Ctrl+C to stop")
