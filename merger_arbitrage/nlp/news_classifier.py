"""
News classification engine for merger arbitrage.
Classifies news headlines/bodies into:
  - Deal (D1-D5) by ticker/company mention
  - Category (REG/FIN/SHR/ALT/PRC) by keyword matching
  - Direction (positive/negative/ambiguous) by sentiment keywords
  - Severity (small/medium/large) by intensity keywords

Strategy: Rule-based keyword matching with fallback heuristics.
RIT news follows structured patterns, making keyword matching reliable.
"""

from __future__ import annotations

import logging
import sys
import os
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEALS, TICKER_TO_DEAL, COMPANY_NAME_TO_DEAL
from nlp.keyword_lexicon import (
    CATEGORY_KEYWORDS, POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS,
    SEVERITY_LARGE_KEYWORDS, SEVERITY_MEDIUM_KEYWORDS,
    DEAL_RESOLUTION_KEYWORDS,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedNews:
    """Result of news classification."""
    deal_id: Optional[str]          # D1-D5 or None if unrelated
    category: str                   # REG, FIN, SHR, ALT, PRC
    direction: str                  # positive, negative, ambiguous
    severity: str                   # small, medium, large
    is_resolution: bool             # True if this is a deal completion/failure event
    resolution_type: Optional[str]  # "completed" or "failed" or None
    confidence: float               # 0.0 to 1.0
    raw_headline: str
    raw_body: str


class NewsClassifier:
    """Classifies merger arbitrage news into structured categories."""

    def classify(self, headline: str, body: str = "",
                 news_id: int = 0, tick: int = 0) -> ClassifiedNews:
        """Classify a news item.

        Pipeline:
        1. Identify which deal (by ticker/company mention)
        2. Check for resolution events (deal completed/failed)
        3. Classify category (REG/FIN/SHR/ALT/PRC)
        4. Classify direction (positive/negative/ambiguous)
        5. Classify severity (small/medium/large)
        """
        text = f"{headline} {body}".strip().lower()

        deal_id = self._identify_deal(text)
        is_resolution, resolution_type = self._check_resolution(text)
        category = self._classify_category(text)
        direction = self._classify_direction(text)
        severity = self._classify_severity(text)
        confidence = self._compute_confidence(deal_id, category, direction)

        result = ClassifiedNews(
            deal_id=deal_id,
            category=category,
            direction=direction,
            severity=severity,
            is_resolution=is_resolution,
            resolution_type=resolution_type,
            confidence=confidence,
            raw_headline=headline,
            raw_body=body,
        )

        logger.debug(
            f"News classified: deal={deal_id} cat={category} "
            f"dir={direction} sev={severity} conf={confidence:.2f} "
            f"res={is_resolution} | {headline[:80]}"
        )
        return result

    def _identify_deal(self, text: str) -> Optional[str]:
        """Match deal by ticker, company name, or deal ID mention."""
        # Check for explicit deal ID (e.g., "D1", "D2")
        for deal_id in DEALS:
            # Match "d1" as a word boundary, not inside other words
            lower_id = deal_id.lower()
            if f" {lower_id} " in f" {text} ":
                return deal_id

        # Check for ticker mentions (e.g., "TGX", "PHR")
        for ticker, deal_id in TICKER_TO_DEAL.items():
            if ticker.lower() in text:
                return deal_id

        # Check for company name mentions
        for name, deal_id in COMPANY_NAME_TO_DEAL.items():
            if name in text:
                return deal_id

        return None

    def _check_resolution(self, text: str) -> tuple[bool, Optional[str]]:
        """Check if news indicates deal completion or failure."""
        for keyword in DEAL_RESOLUTION_KEYWORDS["completed"]:
            if keyword in text:
                return True, "completed"
        for keyword in DEAL_RESOLUTION_KEYWORDS["failed"]:
            if keyword in text:
                return True, "failed"
        return False, None

    def _classify_category(self, text: str) -> str:
        """Determine news category by keyword matching.
        Returns the category with the most keyword hits.
        """
        scores: dict[str, int] = {}
        for cat, keywords in CATEGORY_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw in text:
                    score += 1
            scores[cat] = score

        best = max(scores, key=lambda k: scores[k])
        if scores[best] > 0:
            return best
        return "FIN"  # Default to FIN (multiplier=1.0, neutral)

    def _classify_direction(self, text: str) -> str:
        """Determine positive/negative/ambiguous by keyword balance."""
        pos_score = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
        neg_score = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)

        # Require a clear margin to avoid false signals
        if pos_score > neg_score and pos_score > 0:
            return "positive"
        elif neg_score > pos_score and neg_score > 0:
            return "negative"
        else:
            return "ambiguous"

    def _classify_severity(self, text: str) -> str:
        """Determine small/medium/large by keyword presence."""
        for kw in SEVERITY_LARGE_KEYWORDS:
            if kw in text:
                return "large"
        for kw in SEVERITY_MEDIUM_KEYWORDS:
            if kw in text:
                return "medium"
        return "small"

    def _compute_confidence(self, deal_id: Optional[str],
                            category: str, direction: str) -> float:
        """Heuristic confidence score based on classification quality."""
        score = 0.0
        if deal_id is not None:
            score += 0.4
        if category != "FIN":  # Non-default category match
            score += 0.3
        if direction != "ambiguous":
            score += 0.3
        return min(1.0, score)
