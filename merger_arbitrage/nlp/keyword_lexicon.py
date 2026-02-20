"""
Keyword dictionaries for news classification.
Organized by category, sentiment, and severity.
Tuned for RIT competition news text patterns.
"""

# ============================================================
# CATEGORY KEYWORDS
# Maps news category (REG/FIN/SHR/ALT/PRC) to keyword stems
# ============================================================
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "REG": [
        "regulat", "antitrust", "ftc", "doj", "sec ",
        "commission", "approval", "review", "clearance",
        "compliance", "authority", "blocked", "injunction",
        "consent decree", "investigation", "monopoly",
        "competition", "merger review", "remedy", "remedies",
        "litigation", "court", "ruling", "jurisdiction",
        "regulatory body", "regulatory approval",
    ],
    "FIN": [
        "financ", "earnings", "revenue", "profit", "loss",
        "debt", "credit", "rating", "downgrad", "upgrad",
        "funding", "capital", "dividend", "cash flow",
        "balance sheet", "quarterly", "annual report",
        "leverage", "refinanc", "lender", "loan",
        "credit spread", "bond", "interest rate",
        "cash position", "liquidity",
    ],
    "SHR": [
        "shareholder", "vote", "proxy", "board", "director",
        "activist", "dissent", "meeting", "approval vote",
        "opposition", "institutional investor", "stake",
        "tender offer", "tender", "proxy advisor",
        "iss ", "glass lewis", "recommendation",
        "board of directors", "special committee",
    ],
    "ALT": [
        "alternative", "competing", "rival", "counter",
        "hostile", "topping", "white knight", "bidding war",
        "third party", "unsolicited", "superior proposal",
        "competing bid", "rival bid", "counter offer",
        "new bidder", "potential acquirer", "renegotiat",
    ],
    "PRC": [
        "timeline", "deadline", "extension", "delay",
        "schedule", "closing date", "expected close",
        "condition", "procedural", "filing",
        "process", "timing", "closing condition",
        "material adverse", "mac clause",
        "walk-away", "drop-dead", "outside date",
        "long stop", "termination date",
    ],
}

# ============================================================
# SENTIMENT KEYWORDS
# ============================================================
POSITIVE_KEYWORDS: list[str] = [
    "approv", "clear", "support", "favor", "progress",
    "advanc", "green light", "proceed", "close to",
    "confident", "on track", "positive", "strong",
    "better", "exceed", "success", "agreed", "unanim",
    "recommend", "endorse", "higher bid", "raised offer",
    "increased", "no objection", "unconditional",
    "constructive", "encouraging", "breakthrough",
    "commitment", "reaffirm", "on schedule",
    "smooth", "satisf", "resolution", "settled",
    "compliant", "in favor", "voting in favor",
    "acceptable", "welcomed",
]

NEGATIVE_KEYWORDS: list[str] = [
    "block", "reject", "oppos", "concern", "delay",
    "fail", "terminat", "withdrawn", "collapse",
    "doubt", "uncertain", "negative", "weak", "worse",
    "miss", "shortfall", "lawsuit", "litigation",
    "challenged", "hostile", "lower offer", "reduced",
    "downgrade", "condition", "obstacle", "hurdle",
    "risk", "investigation", "probe", "subpoena",
    "deteriorat", "repricing", "less certain",
    "impasse", "stalled", "deadlock", "threaten",
    "revoke", "suspend", "impediment", "setback",
    "objection", "dissatisf", "walk away",
    "unlikely", "jeopard",
]

# ============================================================
# SEVERITY KEYWORDS
# ============================================================
SEVERITY_LARGE_KEYWORDS: list[str] = [
    "major", "significant", "critical", "decisive",
    "dramatic", "breakthrough", "collapse", "terminat",
    "definitive", "unanimous", "blocked", "final",
    "completed", "failed", "abandoned", "hostile takeover",
    "transformative", "unprecedented", "sweeping",
    "fundamental", "absolute", "unconditional",
    "landmark", "overwhelming",
]

SEVERITY_MEDIUM_KEYWORDS: list[str] = [
    "notable", "material", "substantial", "important",
    "preliminary", "initial", "expected", "reported",
    "announcement", "update", "development",
    "meaningful", "considerable", "moderate",
    "evolving", "ongoing", "emerging",
]
# Small severity is the default when no large/medium keywords match

# ============================================================
# DEAL RESOLUTION KEYWORDS
# ============================================================
DEAL_RESOLUTION_KEYWORDS: dict[str, list[str]] = {
    "completed": [
        "deal completed", "merger completed",
        "acquisition completed", "deal closed",
        "merger closed", "transaction closed",
        "deal finalized", "merger finalized",
        "shareholders approved", "deal approved",
        "merger approved", "deal consummated",
        "transaction consummated", "closing completed",
    ],
    "failed": [
        "deal terminated", "merger terminated",
        "deal cancelled", "deal collapsed",
        "merger failed", "deal abandoned",
        "deal withdrawn", "offer withdrawn",
        "deal blocked", "merger blocked",
        "transaction terminated", "deal fell through",
        "merger called off", "acquisition cancelled",
    ],
}
