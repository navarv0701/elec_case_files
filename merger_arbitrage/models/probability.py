"""
Probability tracker for deal completion.
Applies news impacts using the formula:
  delta_p = baseline_impact * category_multiplier * deal_sensitivity
Clamps to [0, 1] after each update.
"""

from __future__ import annotations

import logging
import sys
import os
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASELINE_IMPACT, AMBIGUOUS_RANGE, CATEGORY_MULTIPLIERS, DEALS

logger = logging.getLogger(__name__)


@dataclass
class NewsImpact:
    """Structured representation of a news item's impact on a deal."""
    deal_id: str
    category: str          # REG, FIN, SHR, ALT, PRC
    direction: str         # positive, negative, ambiguous
    severity: str          # small, medium, large
    delta_p: float         # Computed probability change
    raw_headline: str      # Original text for logging
    tick: int              # When received


class ProbabilityTracker:
    """Maintains running deal probabilities and applies news impacts."""

    def __init__(self, initial_probs: dict[str, float]):
        self.probabilities: dict[str, float] = dict(initial_probs)
        self.history: dict[str, list[NewsImpact]] = {d: [] for d in initial_probs}

    def apply_news(self, impact: NewsImpact) -> float:
        """Apply a classified news event to update deal probability.

        Returns the new probability after update.
        """
        deal_id = impact.deal_id
        if deal_id not in self.probabilities:
            logger.warning(f"Unknown deal_id in news impact: {deal_id}")
            return 0.0

        old_p = self.probabilities[deal_id]
        new_p = max(0.0, min(1.0, old_p + impact.delta_p))

        self.probabilities[deal_id] = new_p
        self.history[deal_id].append(impact)

        logger.info(
            f"Deal {deal_id}: p {old_p:.3f} -> {new_p:.3f} "
            f"(delta={impact.delta_p:+.4f}, "
            f"{impact.category}/{impact.direction}/{impact.severity})"
        )
        return new_p

    def get_probability(self, deal_id: str) -> float:
        return self.probabilities.get(deal_id, 0.0)

    def get_all_probabilities(self) -> dict[str, float]:
        return dict(self.probabilities)

    def mark_resolved(self, deal_id: str, completed: bool):
        """Set probability to 1.0 (completed) or 0.0 (failed) on resolution."""
        self.probabilities[deal_id] = 1.0 if completed else 0.0
        logger.info(f"Deal {deal_id} resolved: {'COMPLETED' if completed else 'FAILED'}")


def compute_delta_p(
    deal_id: str,
    category: str,
    direction: str,
    severity: str,
    ambiguous_estimate: float = 0.0,
) -> float:
    """Compute the probability change for a news event.

    delta_p = baseline_impact * category_multiplier * deal_sensitivity
    """
    deal_cfg = DEALS.get(deal_id)
    if deal_cfg is None:
        return 0.0

    # Baseline impact
    if direction == "ambiguous":
        low, high = AMBIGUOUS_RANGE.get(severity, (0.0, 0.0))
        baseline = ambiguous_estimate if ambiguous_estimate != 0.0 else (low + high) / 2.0
    else:
        baseline = BASELINE_IMPACT.get((direction, severity), 0.0)

    # Category multiplier
    cat_mult = CATEGORY_MULTIPLIERS.get(category, 1.0)

    # Deal sensitivity
    deal_sens = deal_cfg.sensitivity_multiplier

    return baseline * cat_mult * deal_sens
