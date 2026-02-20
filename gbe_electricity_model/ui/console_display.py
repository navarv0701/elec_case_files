"""
Rich terminal dashboard for GBE Electricity Trading.
Displays role-specific panels, action queue, and real-time market data.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.live import Live
from rich.columns import Columns
from rich import box

from state.game_state import GameState, Recommendation
from models.demand import demand_contracts
from config import (
    TICKS_PER_DAY, CUSTOMER_PRICE_PER_MWH, DISPOSAL_PENALTY,
    ELEC_FWD_CONTRACT_SIZE, ELEC_SPOT_CONTRACT_SIZE,
    DAYS_WITH_CLOSEOUT_FINE,
)


URGENCY_COLORS = {
    "CRITICAL": "bold white on red",
    "HIGH": "bold red",
    "MEDIUM": "yellow",
    "LOW": "dim",
}


class ConsoleDisplay:
    """Rich terminal dashboard with role-specific views."""

    def __init__(self, role: str = "all"):
        self.console = Console()
        self.role = role.lower()

    def render(self, state: GameState) -> None:
        """Render the full dashboard to the console (clears screen)."""
        self.console.clear()
        self.console.print(self._build_dashboard(state))

    def render_string(self, state: GameState) -> str:
        """Return the dashboard as a string (for Live display)."""
        with self.console.capture() as capture:
            self.console.print(self._build_dashboard(state))
        return capture.get()

    def _build_dashboard(self, state: GameState) -> Table:
        """Build the complete dashboard as a Rich renderable."""
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)

        # Header
        grid.add_row(self._header_panel(state))

        # Market + Positions row
        market_table = Table.grid(expand=True)
        market_table.add_column(ratio=1)
        market_table.add_column(ratio=1)
        market_table.add_row(
            self._market_panel(state),
            self._position_panel(state),
        )
        grid.add_row(market_table)

        # Weather + Supply/Demand
        info_table = Table.grid(expand=True)
        info_table.add_column(ratio=1)
        info_table.add_column(ratio=1)
        info_table.add_row(
            self._weather_panel(state),
            self._supply_demand_panel(state),
        )
        grid.add_row(info_table)

        # Action Queue (the most important section)
        grid.add_row(self._action_panel(state))

        # P&L footer
        grid.add_row(self._pnl_panel(state))

        return grid

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _header_panel(self, state: GameState) -> Panel:
        """Top bar showing day, tick, time remaining, status."""
        day = state.current_day
        tick = state.current_tick
        ticks_left_day = state.ticks_remaining_in_day()
        ticks_left_total = state.ticks_remaining_total()

        # Progress bar
        day_progress = (TICKS_PER_DAY - ticks_left_day) / TICKS_PER_DAY
        bar_width = 30
        filled = int(bar_width * day_progress)
        bar = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_width - filled)}[/dim]"

        # Status color
        if state.case_status == "ACTIVE":
            status_str = "[bold green]ACTIVE[/bold green]"
        elif state.case_status == "PAUSED":
            status_str = "[bold yellow]PAUSED[/bold yellow]"
        else:
            status_str = "[bold red]STOPPED[/bold red]"

        # Warning indicators
        warnings = ""
        if state.is_day_close_critical():
            warnings = " [bold white on red] !! CRITICAL: <5 ticks left !! [/bold white on red]"
        elif state.is_day_close_urgent():
            warnings = " [bold red] ! URGENT: <20 ticks left ! [/bold red]"
        elif state.is_day_close_warning():
            warnings = " [yellow] Closeout zone: <60 ticks left [/yellow]"

        # Forecast indicator
        forecast = state.is_forecast_time()
        if forecast:
            warnings += f" [bold cyan] FORECAST: {forecast.upper()} [/bold cyan]"

        header = (
            f"  Day [bold]{day}[/bold]/5  |  "
            f"Tick [bold]{tick}[/bold]  |  "
            f"Day: {bar} {ticks_left_day}s left  |  "
            f"Total: {ticks_left_total}s  |  "
            f"{status_str}{warnings}"
        )

        role_label = f"  Role: [bold cyan]{self.role.upper()}[/bold cyan]" if self.role != "all" else ""
        title = f"GBE Electricity Trading{role_label}"

        return Panel(Text.from_markup(header), title=title, box=box.HEAVY)

    # ------------------------------------------------------------------
    # Market Panel
    # ------------------------------------------------------------------

    def _market_panel(self, state: GameState) -> Panel:
        """Show current market prices."""
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Security", style="cyan")
        table.add_column("Bid", justify="right")
        table.add_column("Ask", justify="right")
        table.add_column("Last", justify="right")
        table.add_column("Spread", justify="right")

        # ELEC-F
        fwd_spread = state.elec_fwd_ask - state.elec_fwd_bid if state.elec_fwd_bid > 0 and state.elec_fwd_ask > 0 else 0
        table.add_row(
            "ELEC-F",
            f"${state.elec_fwd_bid:.0f}" if state.elec_fwd_bid > 0 else "-",
            f"${state.elec_fwd_ask:.0f}" if state.elec_fwd_ask > 0 else "-",
            f"${state.elec_fwd_last:.0f}" if state.elec_fwd_last > 0 else "-",
            f"${fwd_spread:.0f}" if fwd_spread > 0 else "-",
        )

        # NG
        ng_spread = state.ng_ask - state.ng_bid if state.ng_bid > 0 and state.ng_ask > 0 else 0
        table.add_row(
            "NG",
            f"${state.ng_bid:.2f}" if state.ng_bid > 0 else "-",
            f"${state.ng_ask:.2f}" if state.ng_ask > 0 else "-",
            f"${state.ng_last:.2f}" if state.ng_last > 0 else "-",
            f"${ng_spread:.2f}" if ng_spread > 0 else "-",
        )

        # ELEC-dayX spot (current day)
        day = state.current_day
        spot_bid = state.elec_spot_bid.get(day, 0)
        spot_ask = state.elec_spot_ask.get(day, 0)
        spot_spread = spot_ask - spot_bid if spot_bid > 0 and spot_ask > 0 else 0
        table.add_row(
            f"ELEC-day{day}",
            f"${spot_bid:.2f}" if spot_bid > 0 else "-",
            f"${spot_ask:.2f}" if spot_ask > 0 else "-",
            "-",
            f"${spot_spread:.2f}" if spot_spread > 0 else "-",
        )

        # RAE bulletin info
        rae = state.latest_rae_bulletin(day)
        if rae:
            table.add_row(
                "[dim]RAE range[/dim]",
                f"[dim]${rae.price_low:.0f}[/dim]",
                f"[dim]${rae.price_high:.0f}[/dim]",
                f"[dim]B:{rae.volume_buy}/S:{rae.volume_sell}[/dim]",
                "",
            )

        return Panel(table, title="Market Prices", border_style="blue")

    # ------------------------------------------------------------------
    # Position Panel
    # ------------------------------------------------------------------

    def _position_panel(self, state: GameState) -> Panel:
        """Show current positions."""
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Security", style="cyan")
        table.add_column("Position", justify="right")
        table.add_column("Value", justify="right")
        table.add_column("Risk", justify="right")

        # ELEC-F
        fwd_val = state.elec_f_position * state.elec_fwd_last * ELEC_FWD_CONTRACT_SIZE if state.elec_fwd_last > 0 else 0
        table.add_row(
            "ELEC-F",
            _color_position(state.elec_f_position),
            f"${fwd_val:,.0f}" if fwd_val != 0 else "-",
            "",
        )

        # NG
        table.add_row(
            "NG",
            _color_position(state.ng_position),
            "",
            "",
        )

        # ELEC-dayX positions
        for day_num in range(1, 7):
            ticker = f"ELEC-day{day_num}"
            pos = state.elec_day_positions.get(ticker, 0)
            if pos != 0 or day_num == state.current_day:
                risk = ""
                if pos != 0 and day_num in DAYS_WITH_CLOSEOUT_FINE:
                    risk = f"[red]${abs(pos) * DISPOSAL_PENALTY:,}[/red]"
                table.add_row(
                    ticker,
                    _color_position(pos),
                    "",
                    risk,
                )

        return Panel(table, title="Positions", border_style="green")

    # ------------------------------------------------------------------
    # Weather Panel
    # ------------------------------------------------------------------

    def _weather_panel(self, state: GameState) -> Panel:
        """Show weather forecasts across days."""
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Day", style="cyan", justify="center")
        table.add_column("Sun (hrs)", justify="right")
        table.add_column("Temp (C)", justify="right")
        table.add_column("Updates", justify="center")
        table.add_column("Uncertainty", justify="right")

        for day in range(1, 7):
            sun = state.best_sunshine_forecast(day)
            temp = state.best_temperature_forecast(day)
            sun_updates = len(state.sunshine_forecasts.get(day, []))
            temp_updates = len(state.temperature_forecasts.get(day, []))
            sun_unc = state.sunshine_uncertainty(day)
            temp_unc = state.temperature_uncertainty(day)

            day_style = "bold" if day == state.current_day + 1 else ""
            marker = " <" if day == state.current_day + 1 else ""

            table.add_row(
                f"[{day_style}]{day}{marker}[/{day_style}]" if day_style else f"{day}",
                f"{sun:.1f} ±{sun_unc:.1f}" if sun is not None else "[dim]-[/dim]",
                f"{temp:.1f} ±{temp_unc:.1f}" if temp is not None else "[dim]-[/dim]",
                f"S:{sun_updates}/T:{temp_updates}",
                _uncertainty_bar(max(sun_unc / 5.0, temp_unc / 8.0)),
            )

        return Panel(table, title="Weather Forecasts", border_style="cyan")

    # ------------------------------------------------------------------
    # Supply / Demand Panel
    # ------------------------------------------------------------------

    def _supply_demand_panel(self, state: GameState) -> Panel:
        """Show supply-demand analysis for tomorrow."""
        next_day = state.current_day + 1
        lines = []

        # Supply
        sun = state.best_sunshine_forecast(next_day)
        if sun is not None:
            from models.production import solar_production
            solar = solar_production(sun)
        else:
            solar = 0
        gas = state.gas_production.get(next_day, 0)
        total_supply = solar + gas

        lines.append(f"[bold]Day {next_day} Supply[/bold]")
        lines.append(f"  Solar:  {solar:.0f} contracts ({sun:.1f}h)" if sun else "  Solar:  ? (no forecast)")
        lines.append(f"  Gas:    {gas} contracts")
        lines.append(f"  Total:  [bold]{total_supply:.0f}[/bold] contracts")
        lines.append("")

        # Demand
        temp = state.best_temperature_forecast(next_day)
        if temp is not None:
            demand = demand_contracts(temp)
            lines.append(f"[bold]Day {next_day} Demand[/bold]")
            lines.append(f"  Demand: {demand:.0f} contracts ({temp:.1f}°C)")
            lines.append(f"  Revenue: ${demand * CUSTOMER_PRICE_PER_MWH * ELEC_SPOT_CONTRACT_SIZE:,.0f}")
        else:
            demand = 180
            lines.append(f"[bold]Day {next_day} Demand[/bold]")
            lines.append("  Demand: ~180 (no forecast)")

        lines.append("")

        # Balance
        balance = total_supply - demand
        if balance > 0:
            lines.append(f"  Balance: [green]+{balance:.0f} SURPLUS[/green]")
        elif balance < 0:
            lines.append(f"  Balance: [red]{balance:.0f} DEFICIT[/red]")
        else:
            lines.append("  Balance: [yellow]BALANCED[/yellow]")

        return Panel("\n".join(lines), title="Supply / Demand", border_style="magenta")

    # ------------------------------------------------------------------
    # Action Queue (most important section)
    # ------------------------------------------------------------------

    def _action_panel(self, state: GameState) -> Panel:
        """Display prioritized action recommendations."""
        recs = state.active_recommendations

        if not recs:
            return Panel(
                "[dim]No recommendations at this time. Waiting for data...[/dim]",
                title="Action Queue",
                border_style="white",
            )

        table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
        table.add_column("#", justify="center", width=3)
        table.add_column("Urgency", justify="center", width=10)
        table.add_column("Action", width=8)
        table.add_column("Ticker", width=12)
        table.add_column("Qty", justify="right", width=5)
        table.add_column("Price", justify="right", width=8)
        table.add_column("Reason", ratio=1)
        table.add_column("E[PnL]", justify="right", width=10)
        table.add_column("Risk", justify="right", width=10)

        for i, rec in enumerate(recs[:12], 1):  # Show top 12
            style = URGENCY_COLORS.get(rec.urgency, "")
            urgency_text = Text(rec.urgency, style=style)

            action_style = "green" if rec.action == "BUY" else "red" if rec.action == "SELL" else "yellow"

            table.add_row(
                str(i),
                urgency_text,
                f"[{action_style}]{rec.action}[/{action_style}]",
                rec.ticker,
                str(rec.quantity) if rec.quantity else "-",
                f"${rec.price:.0f}" if rec.price else "MKT",
                Text(rec.reason[:60] + ("..." if len(rec.reason) > 60 else ""), overflow="ellipsis"),
                f"[green]${rec.expected_pnl:,.0f}[/green]" if rec.expected_pnl > 0 else "-",
                f"[red]${rec.penalty_risk:,.0f}[/red]" if rec.penalty_risk > 0 else "-",
            )

        border = "red" if any(r.urgency == "CRITICAL" for r in recs) else \
                 "yellow" if any(r.urgency == "HIGH" for r in recs) else "white"

        return Panel(table, title=f"Action Queue ({len(recs)} recommendations)", border_style=border)

    # ------------------------------------------------------------------
    # P&L Footer
    # ------------------------------------------------------------------

    def _pnl_panel(self, state: GameState) -> Panel:
        """Show P&L summary."""
        nlv_color = "green" if state.nlv >= 0 else "red"
        rpnl_color = "green" if state.realized_pnl >= 0 else "red"

        # Penalty exposure
        total_penalty = 0
        for ticker, pos in state.elec_day_positions.items():
            if pos != 0:
                # Extract day number
                try:
                    day_num = int(ticker.split("day")[1])
                    if day_num in DAYS_WITH_CLOSEOUT_FINE:
                        total_penalty += abs(pos) * DISPOSAL_PENALTY
                except (IndexError, ValueError):
                    pass

        penalty_str = f"  |  Penalty Exposure: [red]${total_penalty:,}[/red]" if total_penalty > 0 else ""

        text = (
            f"  NLV: [{nlv_color}]${state.nlv:,.0f}[/{nlv_color}]  |  "
            f"Realized: [{rpnl_color}]${state.realized_pnl:,.0f}[/{rpnl_color}]  |  "
            f"Unrealized: ${state.unrealized_pnl:,.0f}"
            f"{penalty_str}"
        )

        return Panel(Text.from_markup(text), title="P&L", box=box.HEAVY)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _color_position(pos: int) -> str:
    """Color-code a position value."""
    if pos > 0:
        return f"[green]+{pos}[/green]"
    elif pos < 0:
        return f"[red]{pos}[/red]"
    return "[dim]0[/dim]"


def _uncertainty_bar(ratio: float) -> str:
    """Visual uncertainty indicator (0=certain, 1=max uncertainty)."""
    ratio = max(0, min(1, ratio))
    blocks = 5
    filled = int(blocks * (1 - ratio))
    if filled >= 4:
        color = "green"
    elif filled >= 2:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{'●' * filled}{'○' * (blocks - filled)}[/{color}]"
