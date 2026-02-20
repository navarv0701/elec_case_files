"""
Deal valuation model for merger arbitrage.
Computes deal value (K), standalone value (V), and intrinsic target price (P*).

Formulas:
  K = deal offer value (depends on structure: cash, stock, mixed)
  V = standalone value = (P0 - p0 * K0) / (1 - p0)   [computed once at t=0, fixed]
  P* = p * K + (1-p) * V                               [updates with p and acquirer price]
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DealConfig, DEALS


@dataclass
class DealState:
    """Runtime state for a single deal. Created once, updated as data flows in."""
    config: DealConfig
    probability: float                  # Current probability of deal closing
    standalone_value: float             # V, computed at initialization and fixed

    # Live market prices
    target_price: float = 0.0           # Current market price of target
    acquirer_price: float = 0.0         # Current market price of acquirer
    target_bid: float = 0.0
    target_ask: float = 0.0
    acquirer_bid: float = 0.0
    acquirer_ask: float = 0.0

    # Positions
    target_position: int = 0
    acquirer_position: int = 0

    # Tracking
    news_history: list = field(default_factory=list)
    resolved: bool = False              # True if deal has completed or failed
    resolution: Optional[str] = None    # "completed" or "failed"

    @property
    def deal_value_K(self) -> float:
        """Compute current deal value K based on structure and live acquirer price.

        All-cash:        K = cash_component (fixed, independent of acquirer)
        Stock-for-stock: K = exchange_ratio * acquirer_price (varies with acquirer)
        Mixed:           K = cash_component + exchange_ratio * acquirer_price
        """
        cfg = self.config
        if cfg.structure == "all_cash":
            return cfg.cash_component
        elif cfg.structure == "stock_for_stock":
            price = self.acquirer_price if self.acquirer_price > 0 else cfg.acquirer_start_price
            return cfg.exchange_ratio * price
        elif cfg.structure == "mixed":
            price = self.acquirer_price if self.acquirer_price > 0 else cfg.acquirer_start_price
            return cfg.cash_component + cfg.exchange_ratio * price
        else:
            raise ValueError(f"Unknown deal structure: {cfg.structure}")

    @property
    def intrinsic_target_price(self) -> float:
        """P* = p * K + (1-p) * V"""
        return self.probability * self.deal_value_K + (1.0 - self.probability) * self.standalone_value

    @property
    def target_mispricing(self) -> float:
        """Difference between intrinsic and market price.
        Positive = target is undervalued (buy signal).
        Negative = target is overvalued (sell signal).
        """
        if self.target_price <= 0:
            return 0.0
        return self.intrinsic_target_price - self.target_price

    @property
    def target_mispricing_pct(self) -> float:
        """Mispricing as percentage of market price."""
        if self.target_price <= 0:
            return 0.0
        return self.target_mispricing / self.target_price * 100.0

    @property
    def spread_to_deal(self) -> float:
        """Current merger spread: (K - target_price) / target_price."""
        K = self.deal_value_K
        if self.target_price <= 0:
            return 0.0
        return (K - self.target_price) / self.target_price

    @property
    def ideal_hedge_ratio(self) -> float:
        """For stock/mixed deals, the exchange ratio defines the hedge.
        For all-cash deals, no hedge is needed.
        """
        if self.config.structure == "all_cash":
            return 0.0
        return self.config.exchange_ratio


def compute_standalone_value(config: DealConfig) -> float:
    """Compute standalone value V at t=0 from initial conditions.

    V = (P0 - p0 * K0) / (1 - p0)

    where K0 = deal value at initial acquirer price.
    """
    p0 = config.initial_probability
    P0 = config.target_start_price

    # K0: deal value using start-of-day acquirer price
    if config.structure == "all_cash":
        K0 = config.cash_component
    elif config.structure == "stock_for_stock":
        K0 = config.exchange_ratio * config.acquirer_start_price
    elif config.structure == "mixed":
        K0 = config.cash_component + config.exchange_ratio * config.acquirer_start_price
    else:
        raise ValueError(f"Unknown structure: {config.structure}")

    if p0 >= 1.0:
        return P0  # Degenerate case

    V = (P0 - p0 * K0) / (1.0 - p0)
    return V


def initialize_all_deals() -> dict[str, DealState]:
    """Create DealState objects for all 5 deals with standalone values computed."""
    deals = {}
    for deal_id, cfg in DEALS.items():
        V = compute_standalone_value(cfg)
        deals[deal_id] = DealState(
            config=cfg,
            probability=cfg.initial_probability,
            standalone_value=V,
            target_price=cfg.target_start_price,
            acquirer_price=cfg.acquirer_start_price,
        )
    return deals
