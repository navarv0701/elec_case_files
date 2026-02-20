"""
RITC 2026 Merger Arbitrage Dashboard
=====================================
Streamlit app with two analysis modes:
  Option A - Auto: NLTK VADER sentiment + keyword NLP classifies news automatically
  Option B - Manual: User selects sentiment/severity for each news item

Imports core valuation math from the merger_arbitrage package.

Usage:
    streamlit run app.py
    streamlit run app.py -- --api-key YOUR_KEY
"""

from __future__ import annotations

import sys
import os
import math
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path setup: import from the sibling merger_arbitrage package
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_MA_DIR = os.path.join(_PROJECT_ROOT, "merger_arbitrage")
if _MA_DIR not in sys.path:
    sys.path.insert(0, _MA_DIR)

from config import (
    DEALS, DealConfig, TICKER_TO_DEAL, COMPANY_NAME_TO_DEAL,
    BASELINE_IMPACT, AMBIGUOUS_RANGE, CATEGORY_MULTIPLIERS,
    RIT_BASE_URL,
)
from models.deal import compute_standalone_value, initialize_all_deals, DealState
from models.probability import compute_delta_p, NewsImpact, ProbabilityTracker
from nlp.news_classifier import NewsClassifier, ClassifiedNews
from nlp.keyword_lexicon import CATEGORY_KEYWORDS

# ---------------------------------------------------------------------------
# VADER setup (Option A) — lazy-loaded, graceful fallback
# ---------------------------------------------------------------------------
_vader_analyzer = None


def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is not None:
        return _vader_analyzer
    try:
        import nltk
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
        return _vader_analyzer
    except Exception:
        return None


# ---------------------------------------------------------------------------
# API helper — optional, works without RIT server
# ---------------------------------------------------------------------------
def _try_api_fetch(api_key: str, base_url: str):
    """Attempt to pull live data from RIT. Returns (case, securities, news) or Nones."""
    if not api_key:
        return None, None, None
    try:
        from api.rit_client import RITClient
        client = RITClient(api_key=api_key, base_url=base_url)
        case_info = client.get_case()
        securities = client.get_securities()
        news = client.get_news(limit=100)
        return case_info, securities, news
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# Confidence interval computation
# ---------------------------------------------------------------------------
def compute_confidence_interval(
    probability: float,
    deal_value_K: float,
    standalone_value: float,
    confidence_level: float = 0.95,
    p_uncertainty: float = 0.08,
) -> tuple[float, float, float]:
    """Compute intrinsic target price with confidence interval.

    Uses a Beta-distribution approximation around the current probability
    to derive low/high bounds on P*.

    Returns: (P*_low, P*_mid, P*_high)
    """
    p_mid = probability
    P_mid = p_mid * deal_value_K + (1.0 - p_mid) * standalone_value

    # z-score for confidence level
    from scipy.stats import norm
    z = norm.ppf(0.5 + confidence_level / 2.0)

    # Probability bounds (clamped to [0, 1])
    p_low = max(0.0, p_mid - z * p_uncertainty)
    p_high = min(1.0, p_mid + z * p_uncertainty)

    P_low = p_low * deal_value_K + (1.0 - p_low) * standalone_value
    P_high = p_high * deal_value_K + (1.0 - p_high) * standalone_value

    # Ensure low < high (if V > K, the relationship inverts)
    if P_low > P_high:
        P_low, P_high = P_high, P_low

    return P_low, P_mid, P_high


# ---------------------------------------------------------------------------
# VADER-enhanced classification (Option A)
# ---------------------------------------------------------------------------
DIRECTION_MAP_VADER = {
    "Strong Positive": ("positive", "large"),
    "Positive":        ("positive", "medium"),
    "Weak Positive":   ("positive", "small"),
    "Neutral":         ("ambiguous", "small"),
    "Weak Negative":   ("negative", "small"),
    "Negative":        ("negative", "medium"),
    "Strong Negative": ("negative", "large"),
}


def classify_with_vader(headline: str, body: str = "") -> dict:
    """Use VADER + keyword NLP to classify news automatically.

    Returns a dict with keys: deal_id, category, direction, severity,
    vader_compound, vader_label, confidence.
    """
    vader = _get_vader()
    classifier = NewsClassifier()
    text = f"{headline} {body}".strip()

    # Keyword-based classification for deal, category, resolution
    kw_result = classifier.classify(headline, body)

    # VADER sentiment scoring
    if vader:
        scores = vader.polarity_scores(text)
        compound = scores["compound"]
    else:
        compound = 0.0

    # Map compound score to direction/severity
    if compound >= 0.6:
        vader_label = "Strong Positive"
    elif compound >= 0.3:
        vader_label = "Positive"
    elif compound >= 0.05:
        vader_label = "Weak Positive"
    elif compound > -0.05:
        vader_label = "Neutral"
    elif compound > -0.3:
        vader_label = "Weak Negative"
    elif compound > -0.6:
        vader_label = "Negative"
    else:
        vader_label = "Strong Negative"

    direction, severity = DIRECTION_MAP_VADER[vader_label]

    # If keyword classifier found a clear direction, blend with VADER
    # Give priority to keyword classifier for direction, VADER for severity
    if kw_result.direction != "ambiguous":
        direction = kw_result.direction
    # Use keyword severity if VADER is neutral
    if kw_result.severity != "small" and vader_label == "Neutral":
        severity = kw_result.severity

    # Use keyword category (much more reliable than VADER for this)
    category = kw_result.category

    return {
        "deal_id": kw_result.deal_id,
        "category": category,
        "direction": direction,
        "severity": severity,
        "vader_compound": compound,
        "vader_label": vader_label,
        "confidence": kw_result.confidence,
        "is_resolution": kw_result.is_resolution,
        "resolution_type": kw_result.resolution_type,
    }


# =====================================================================
# STREAMLIT APP
# =====================================================================

st.set_page_config(
    page_title="Merger Arbitrage Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "deals" not in st.session_state:
    st.session_state.deals = initialize_all_deals()
if "prob_tracker" not in st.session_state:
    initial_probs = {d: cfg.initial_probability for d, cfg in DEALS.items()}
    st.session_state.prob_tracker = ProbabilityTracker(initial_probs)
if "news_log" not in st.session_state:
    st.session_state.news_log = []  # List of dicts for display
if "last_news_id" not in st.session_state:
    st.session_state.last_news_id = 0
if "api_news_cache" not in st.session_state:
    st.session_state.api_news_cache = []
if "confidence_level" not in st.session_state:
    st.session_state.confidence_level = 0.95
if "p_uncertainty" not in st.session_state:
    st.session_state.p_uncertainty = 0.08

# ---------------------------------------------------------------------------
# Sidebar — Settings & API Connection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    mode = st.radio(
        "Analysis Mode",
        ["Option A: Auto (VADER + NLP)", "Option B: Manual Entry"],
        help="Option A uses NLTK VADER sentiment + keyword NLP to auto-classify news.\n"
             "Option B lets you manually select sentiment for each news item.",
    )
    is_auto_mode = mode.startswith("Option A")

    st.divider()
    st.subheader("API Connection")
    api_key = st.text_input("RIT API Key", type="password", value="")
    api_url = st.text_input("API URL", value=RIT_BASE_URL)

    if st.button("🔄 Fetch Live Data", use_container_width=True):
        case_info, securities, news = _try_api_fetch(api_key, api_url)
        if case_info:
            st.success(f"Connected — Tick {case_info.get('tick', '?')}, "
                       f"Period {case_info.get('period', '?')}, "
                       f"Status: {case_info.get('status', '?')}")
        else:
            st.error("Could not connect to RIT server")

        if securities:
            for s in securities:
                ticker = s.get("ticker", "")
                deal_id = TICKER_TO_DEAL.get(ticker)
                if deal_id and deal_id in st.session_state.deals:
                    deal = st.session_state.deals[deal_id]
                    cfg = deal.config
                    if ticker == cfg.target_ticker:
                        deal.target_price = s.get("last", deal.target_price) or deal.target_price
                        deal.target_bid = s.get("bid", 0) or 0
                        deal.target_ask = s.get("ask", 0) or 0
                    elif ticker == cfg.acquirer_ticker:
                        deal.acquirer_price = s.get("last", deal.acquirer_price) or deal.acquirer_price
                        deal.acquirer_bid = s.get("bid", 0) or 0
                        deal.acquirer_ask = s.get("ask", 0) or 0

        if news:
            st.session_state.api_news_cache = [
                n for n in news
                if n.get("news_id", 0) > st.session_state.last_news_id
            ]
            if st.session_state.api_news_cache:
                st.info(f"Fetched {len(st.session_state.api_news_cache)} new news items")
                st.session_state.last_news_id = max(
                    n.get("news_id", 0) for n in news
                )

    st.divider()
    st.subheader("Confidence Interval")
    st.session_state.confidence_level = st.select_slider(
        "Confidence Level",
        options=[0.90, 0.95, 0.99],
        value=st.session_state.confidence_level,
        format_func=lambda x: f"{x:.0%}",
    )
    st.session_state.p_uncertainty = st.slider(
        "Probability Uncertainty (σ)",
        min_value=0.02, max_value=0.20, value=0.08, step=0.01,
        help="Standard deviation of probability estimate. "
             "Larger = wider confidence intervals.",
    )

    st.divider()
    if st.button("🔄 Reset All Probabilities", use_container_width=True):
        st.session_state.deals = initialize_all_deals()
        initial_probs = {d: cfg.initial_probability for d, cfg in DEALS.items()}
        st.session_state.prob_tracker = ProbabilityTracker(initial_probs)
        st.session_state.news_log = []
        st.session_state.api_news_cache = []
        st.session_state.last_news_id = 0
        st.rerun()

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("📊 RITC 2026 Merger Arbitrage Dashboard")

mode_label = "🤖 Auto (VADER + NLP)" if is_auto_mode else "✋ Manual Entry"
st.caption(f"Mode: **{mode_label}** — Switch in the sidebar")

# =====================================================================
# Section 1: Deal Overview — always visible
# =====================================================================
st.header("Deal Overview")

deals = st.session_state.deals
conf_level = st.session_state.confidence_level
p_unc = st.session_state.p_uncertainty

# Build overview dataframe
overview_rows = []
for deal_id in ["D1", "D2", "D3", "D4", "D5"]:
    deal = deals[deal_id]
    cfg = deal.config
    K = deal.deal_value_K
    V = deal.standalone_value
    p = deal.probability

    P_low, P_mid, P_high = compute_confidence_interval(
        p, K, V, conf_level, p_unc
    )

    struct_label = cfg.structure.replace("_", "-").title()
    if cfg.structure == "all_cash":
        terms = f"${cfg.cash_component:.0f}/share"
    elif cfg.structure == "stock_for_stock":
        terms = f"{cfg.exchange_ratio:.2f} {cfg.acquirer_ticker}/share"
    else:
        terms = f"${cfg.cash_component:.0f} + {cfg.exchange_ratio:.2f} {cfg.acquirer_ticker}"

    overview_rows.append({
        "Deal": deal_id,
        "Target": cfg.target_ticker,
        "Acquirer": cfg.acquirer_ticker,
        "Industry": cfg.industry,
        "Structure": struct_label,
        "Terms": terms,
        "p₀ (Analyst)": cfg.initial_probability,
        "p (Current)": p,
        "V (Standalone)": V,
        "K (Deal Value)": K,
        f"P* Low ({conf_level:.0%})": P_low,
        "P* (Intrinsic)": P_mid,
        f"P* High ({conf_level:.0%})": P_high,
        "Market Price": deal.target_price,
        "Mispricing": deal.target_mispricing,
        "Sensitivity": cfg.sensitivity_multiplier,
    })

df_overview = pd.DataFrame(overview_rows)

# Format and display
st.dataframe(
    df_overview.style.format({
        "p₀ (Analyst)": "{:.0%}",
        "p (Current)": "{:.1%}",
        "V (Standalone)": "${:.2f}",
        "K (Deal Value)": "${:.2f}",
        f"P* Low ({conf_level:.0%})": "${:.2f}",
        "P* (Intrinsic)": "${:.2f}",
        f"P* High ({conf_level:.0%})": "${:.2f}",
        "Market Price": "${:.2f}",
        "Mispricing": "${:+.2f}",
        "Sensitivity": "{:.2f}",
    }).map(
        lambda v: "color: green" if isinstance(v, (int, float)) and v > 0.001
        else ("color: red" if isinstance(v, (int, float)) and v < -0.001 else ""),
        subset=["Mispricing"],
    ),
    use_container_width=True,
    hide_index=True,
)

# =====================================================================
# Section 2: Per-Deal Detail Cards
# =====================================================================
st.header("Deal Details & Valuations")

cols = st.columns(5)
for i, deal_id in enumerate(["D1", "D2", "D3", "D4", "D5"]):
    deal = deals[deal_id]
    cfg = deal.config
    K = deal.deal_value_K
    V = deal.standalone_value
    p = deal.probability
    P_low, P_mid, P_high = compute_confidence_interval(p, K, V, conf_level, p_unc)
    misp = deal.target_mispricing

    with cols[i]:
        # Color-coded header
        if deal.resolved:
            status_emoji = "✅" if deal.resolution == "completed" else "❌"
        elif abs(misp) > 0.5:
            status_emoji = "🟢" if misp > 0 else "🔴"
        else:
            status_emoji = "⚪"

        st.subheader(f"{status_emoji} {deal_id}")
        st.caption(f"{cfg.target_ticker} / {cfg.acquirer_ticker} — {cfg.industry}")

        st.metric("Probability", f"{p:.1%}", f"{p - cfg.initial_probability:+.1%}")
        st.metric("Intrinsic P*", f"${P_mid:.2f}")
        st.metric("Market Price", f"${deal.target_price:.2f}", f"${misp:+.2f}")

        st.markdown(f"""
        | | |
        |---|---|
        | **V (Standalone)** | ${V:.2f} |
        | **K (Deal Value)** | ${K:.2f} |
        | **CI Low** | ${P_low:.2f} |
        | **CI High** | ${P_high:.2f} |
        | **Spread** | {deal.spread_to_deal:.1%} |
        | **Hedge Ratio** | {deal.ideal_hedge_ratio:.2f} |
        """)

# =====================================================================
# Section 3: Probability Impact Reference
# =====================================================================
with st.expander("📖 Probability Impact Reference (Case Package)", expanded=False):
    ref_col1, ref_col2, ref_col3 = st.columns(3)

    with ref_col1:
        st.markdown("**Baseline Impact (pp)**")
        base_df = pd.DataFrame({
            "Direction": ["Positive", "Positive", "Positive",
                          "Negative", "Negative", "Negative"],
            "Severity": ["Small", "Medium", "Large",
                         "Small", "Medium", "Large"],
            "Δp": ["+0.03", "+0.07", "+0.14",
                   "-0.04", "-0.09", "-0.18"],
        })
        st.dataframe(base_df, hide_index=True)

    with ref_col2:
        st.markdown("**Category Multipliers**")
        cat_df = pd.DataFrame({
            "Category": ["REG (Regulatory)", "FIN (Financing)",
                         "SHR (Shareholder)", "ALT (Alternatives)",
                         "PRC (Process)"],
            "Multiplier": ["1.25", "1.00", "0.90", "1.40", "0.70"],
        })
        st.dataframe(cat_df, hide_index=True)

    with ref_col3:
        st.markdown("**Deal Sensitivity**")
        sens_df = pd.DataFrame({
            "Deal": [f"{d} ({DEALS[d].target_ticker})" for d in DEALS],
            "Multiplier": [f"{DEALS[d].sensitivity_multiplier:.2f}" for d in DEALS],
        })
        st.dataframe(sens_df, hide_index=True)


# =====================================================================
# Section 4: News Analysis — Option A or Option B
# =====================================================================
st.header("📰 News Analysis")

# --- Sub-section: Process API news (if any) ---
api_news = st.session_state.api_news_cache
if api_news:
    st.subheader(f"Unprocessed API News ({len(api_news)} items)")

    if is_auto_mode:
        # Option A: Auto-process all
        if st.button("🤖 Auto-Classify All News", use_container_width=True):
            processed = 0
            for item in api_news:
                headline = item.get("headline", "")
                body = item.get("body", "")
                tick = item.get("tick", 0)
                news_id = item.get("news_id", 0)

                result = classify_with_vader(headline, body)
                deal_id = result["deal_id"]

                if deal_id:
                    dp = compute_delta_p(
                        deal_id, result["category"],
                        result["direction"], result["severity"],
                    )

                    if result["is_resolution"]:
                        completed = result["resolution_type"] == "completed"
                        deals[deal_id].resolved = True
                        deals[deal_id].resolution = result["resolution_type"]
                        deals[deal_id].probability = 1.0 if completed else 0.0
                        st.session_state.prob_tracker.mark_resolved(deal_id, completed)
                    else:
                        impact = NewsImpact(deal_id, result["category"],
                                           result["direction"], result["severity"],
                                           dp, headline, tick)
                        new_p = st.session_state.prob_tracker.apply_news(impact)
                        deals[deal_id].probability = new_p

                    st.session_state.news_log.append({
                        "Tick": tick,
                        "News ID": news_id,
                        "Deal": deal_id,
                        "Category": result["category"],
                        "Direction": result["direction"],
                        "Severity": result["severity"],
                        "Δp": dp,
                        "New p": deals[deal_id].probability if deal_id else None,
                        "VADER": result["vader_compound"],
                        "Headline": headline[:80],
                        "Mode": "Auto",
                    })
                    processed += 1

            st.session_state.api_news_cache = []
            st.success(f"Processed {processed} news items")
            st.rerun()
    else:
        # Option B: Show each for manual classification
        for idx, item in enumerate(api_news):
            headline = item.get("headline", "")
            body = item.get("body", "")
            tick = item.get("tick", 0)
            news_id = item.get("news_id", 0)

            # Auto-detect deal via keywords
            classifier = NewsClassifier()
            kw = classifier.classify(headline, body)
            auto_deal = kw.deal_id
            auto_cat = kw.category

            with st.container(border=True):
                st.markdown(f"**t={tick}** | ID={news_id} | `{headline}`")
                if body:
                    st.caption(body[:200])

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    deal_options = ["(none)"] + list(DEALS.keys())
                    default_idx = deal_options.index(auto_deal) if auto_deal in deal_options else 0
                    sel_deal = st.selectbox(
                        "Deal", deal_options, index=default_idx,
                        key=f"api_deal_{news_id}",
                    )
                with c2:
                    cat_options = list(CATEGORY_MULTIPLIERS.keys())
                    default_cat = cat_options.index(auto_cat) if auto_cat in cat_options else 1
                    sel_cat = st.selectbox(
                        "Category", cat_options, index=default_cat,
                        key=f"api_cat_{news_id}",
                    )
                with c3:
                    sel_sentiment = st.selectbox(
                        "Sentiment",
                        ["Strong Positive", "Positive", "Weak Positive",
                         "Neutral", "Weak Negative", "Negative", "Strong Negative"],
                        index=3,
                        key=f"api_sent_{news_id}",
                    )
                with c4:
                    if st.button("Apply", key=f"api_apply_{news_id}", use_container_width=True):
                        if sel_deal != "(none)":
                            direction, severity = DIRECTION_MAP_VADER[sel_sentiment]
                            dp = compute_delta_p(sel_deal, sel_cat, direction, severity)

                            impact = NewsImpact(sel_deal, sel_cat, direction, severity,
                                                dp, headline, tick)
                            new_p = st.session_state.prob_tracker.apply_news(impact)
                            deals[sel_deal].probability = new_p

                            st.session_state.news_log.append({
                                "Tick": tick,
                                "News ID": news_id,
                                "Deal": sel_deal,
                                "Category": sel_cat,
                                "Direction": direction,
                                "Severity": severity,
                                "Δp": dp,
                                "New p": new_p,
                                "VADER": None,
                                "Headline": headline[:80],
                                "Mode": "Manual",
                            })

                            # Remove from cache
                            st.session_state.api_news_cache = [
                                n for n in st.session_state.api_news_cache
                                if n.get("news_id") != news_id
                            ]
                            st.rerun()

# --- Sub-section: Manual news entry (always available) ---
st.subheader("Enter News Manually")

if is_auto_mode:
    # Option A: Paste headline, auto-classify
    with st.form("auto_news_form", clear_on_submit=True):
        col_a1, col_a2 = st.columns([3, 1])
        with col_a1:
            input_headline = st.text_input("Headline", placeholder="Paste news headline here...")
            input_body = st.text_area("Body (optional)", height=68,
                                      placeholder="Additional context...")
        with col_a2:
            st.markdown("")  # spacing
            override_deal = st.selectbox(
                "Override Deal (optional)",
                ["Auto-detect"] + list(DEALS.keys()),
            )

        submitted = st.form_submit_button("🤖 Classify & Apply", use_container_width=True)
        if submitted and input_headline.strip():
            result = classify_with_vader(input_headline, input_body)

            deal_id = result["deal_id"]
            if override_deal != "Auto-detect":
                deal_id = override_deal

            if deal_id:
                dp = compute_delta_p(
                    deal_id, result["category"],
                    result["direction"], result["severity"],
                )

                if result["is_resolution"]:
                    completed = result["resolution_type"] == "completed"
                    deals[deal_id].resolved = True
                    deals[deal_id].resolution = result["resolution_type"]
                    deals[deal_id].probability = 1.0 if completed else 0.0
                    st.session_state.prob_tracker.mark_resolved(deal_id, completed)
                else:
                    impact = NewsImpact(deal_id, result["category"],
                                       result["direction"], result["severity"],
                                       dp, input_headline, 0)
                    new_p = st.session_state.prob_tracker.apply_news(impact)
                    deals[deal_id].probability = new_p

                st.session_state.news_log.append({
                    "Tick": 0,
                    "News ID": "-",
                    "Deal": deal_id,
                    "Category": result["category"],
                    "Direction": result["direction"],
                    "Severity": result["severity"],
                    "Δp": dp,
                    "New p": deals[deal_id].probability,
                    "VADER": result["vader_compound"],
                    "Headline": input_headline[:80],
                    "Mode": "Auto",
                })
                st.rerun()
            else:
                st.warning("Could not identify deal. Use the override selector.")

else:
    # Option B: Full manual controls per deal
    st.markdown("Select the deal, then choose your qualitative assessment.")

    manual_tabs = st.tabs([
        f"{d} ({DEALS[d].target_ticker}/{DEALS[d].acquirer_ticker})"
        for d in ["D1", "D2", "D3", "D4", "D5"]
    ])

    for tab_idx, deal_id in enumerate(["D1", "D2", "D3", "D4", "D5"]):
        cfg = DEALS[deal_id]
        deal = deals[deal_id]

        with manual_tabs[tab_idx]:
            with st.form(f"manual_form_{deal_id}", clear_on_submit=True):
                st.markdown(f"**{deal_id}: {cfg.target_ticker} / {cfg.acquirer_ticker}** "
                            f"— {cfg.industry} — Current p = **{deal.probability:.1%}**")

                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    m_headline = st.text_input("Headline / Description",
                                               key=f"m_head_{deal_id}",
                                               placeholder="Brief news description")
                with m_col2:
                    m_category = st.selectbox(
                        "Category",
                        list(CATEGORY_MULTIPLIERS.keys()),
                        format_func=lambda c: f"{c} (×{CATEGORY_MULTIPLIERS[c]:.2f})",
                        key=f"m_cat_{deal_id}",
                    )
                with m_col3:
                    m_sentiment = st.selectbox(
                        "Your Assessment",
                        ["Strong Positive", "Positive", "Weak Positive",
                         "Neutral", "Weak Negative", "Negative", "Strong Negative"],
                        index=3,
                        key=f"m_sent_{deal_id}",
                    )

                # Preview the delta_p before applying
                preview_dir, preview_sev = DIRECTION_MAP_VADER[m_sentiment]
                preview_dp = compute_delta_p(deal_id, m_category, preview_dir, preview_sev)
                preview_new_p = max(0.0, min(1.0, deal.probability + preview_dp))

                st.markdown(
                    f"**Preview:** Δp = `{preview_dp:+.4f}` → "
                    f"p would become `{preview_new_p:.3f}` "
                    f"(baseline {BASELINE_IMPACT.get((preview_dir, preview_sev), 0.0):+.2f} "
                    f"× cat {CATEGORY_MULTIPLIERS[m_category]:.2f} "
                    f"× deal {cfg.sensitivity_multiplier:.2f})"
                )

                m_submitted = st.form_submit_button("✅ Apply", use_container_width=True)
                if m_submitted:
                    direction, severity = DIRECTION_MAP_VADER[m_sentiment]
                    dp = compute_delta_p(deal_id, m_category, direction, severity)

                    impact = NewsImpact(deal_id, m_category, direction, severity,
                                       dp, m_headline or f"Manual: {m_sentiment}", 0)
                    new_p = st.session_state.prob_tracker.apply_news(impact)
                    deals[deal_id].probability = new_p

                    st.session_state.news_log.append({
                        "Tick": 0,
                        "News ID": "-",
                        "Deal": deal_id,
                        "Category": m_category,
                        "Direction": direction,
                        "Severity": severity,
                        "Δp": dp,
                        "New p": new_p,
                        "VADER": None,
                        "Headline": (m_headline or f"Manual: {m_sentiment}")[:80],
                        "Mode": "Manual",
                    })
                    st.rerun()

# =====================================================================
# Section 5: Probability Trajectory Visualization
# =====================================================================
st.header("📈 Probability Trajectories")

if st.session_state.news_log:
    # Build trajectory data
    traj_data: dict[str, list[float]] = {d: [DEALS[d].initial_probability] for d in DEALS}
    traj_labels: list[str] = ["Start"]

    for entry in st.session_state.news_log:
        d = entry.get("Deal")
        if d and d in traj_data:
            label = f"{entry.get('Headline', '')[:25]}..."
            traj_labels.append(label)
            for deal_id in DEALS:
                if deal_id == d:
                    traj_data[deal_id].append(entry["New p"])
                else:
                    traj_data[deal_id].append(traj_data[deal_id][-1])

    traj_df = pd.DataFrame(traj_data, index=range(len(traj_labels)))
    traj_df.columns = [f"{d} ({DEALS[d].target_ticker})" for d in DEALS]
    st.line_chart(traj_df, height=350)
else:
    st.info("No news processed yet. Probabilities are at initial analyst anchors.")

# =====================================================================
# Section 6: Intrinsic Value Chart
# =====================================================================
st.header("💰 Intrinsic Value Summary")

val_cols = st.columns(5)
for i, deal_id in enumerate(["D1", "D2", "D3", "D4", "D5"]):
    deal = deals[deal_id]
    cfg = deal.config
    K = deal.deal_value_K
    V = deal.standalone_value
    p = deal.probability
    P_low, P_mid, P_high = compute_confidence_interval(p, K, V, conf_level, p_unc)

    with val_cols[i]:
        st.markdown(f"**{deal_id} — {cfg.target_ticker}**")

        chart_data = pd.DataFrame({
            "Value": [V, P_low, P_mid, P_high, K, deal.target_price],
            "Label": ["Standalone (V)", f"CI Low", "Intrinsic (P*)",
                       f"CI High", "Deal Value (K)", "Market"],
        })
        st.dataframe(
            chart_data.style.format({"Value": "${:.2f}"}).map(
                lambda v: "font-weight: bold" if v == P_mid else "",
                subset=["Value"]
            ),
            hide_index=True,
            use_container_width=True,
        )

        # Visual bar
        # Normalize all values relative to K for the bar
        max_val = max(K, P_high, deal.target_price, V) * 1.05
        if max_val > 0:
            bar_data = pd.DataFrame({
                "Metric": ["V", "P* Low", "P*", "P* High", "K", "Market"],
                "Value": [V, P_low, P_mid, P_high, K, deal.target_price],
            })
            st.bar_chart(bar_data.set_index("Metric"), height=200)

# =====================================================================
# Section 7: Acquirer Valuations (for stock/mixed deals)
# =====================================================================
st.header("🔗 Acquirer Impact (Stock & Mixed Deals)")

acq_deals = {d: deals[d] for d in ["D2", "D3", "D5"]}
acq_cols = st.columns(3)

for i, (deal_id, deal) in enumerate(acq_deals.items()):
    cfg = deal.config
    with acq_cols[i]:
        st.markdown(f"**{deal_id} — {cfg.acquirer_ticker}**")
        st.metric("Acquirer Price", f"${deal.acquirer_price:.2f}")
        st.metric("Exchange Ratio", f"{cfg.exchange_ratio:.2f}")
        st.metric("Deal Value K", f"${deal.deal_value_K:.2f}")

        # Show how K changes with acquirer price
        acq_range = np.linspace(deal.acquirer_price * 0.85, deal.acquirer_price * 1.15, 20)
        k_values = []
        for ap in acq_range:
            if cfg.structure == "stock_for_stock":
                k_values.append(cfg.exchange_ratio * ap)
            else:
                k_values.append(cfg.cash_component + cfg.exchange_ratio * ap)

        sens_df = pd.DataFrame({
            f"{cfg.acquirer_ticker} Price": acq_range,
            "Deal Value K": k_values,
        })
        st.line_chart(sens_df.set_index(f"{cfg.acquirer_ticker} Price"), height=180)

# =====================================================================
# Section 8: News Log
# =====================================================================
st.header("📋 News Log")

if st.session_state.news_log:
    log_df = pd.DataFrame(st.session_state.news_log)
    display_cols = ["Deal", "Category", "Direction", "Severity", "Δp",
                    "New p", "Headline", "Mode"]
    if "VADER" in log_df.columns and log_df["VADER"].notna().any():
        display_cols.insert(6, "VADER")

    st.dataframe(
        log_df[display_cols].style.format({
            "Δp": "{:+.4f}",
            "New p": "{:.3f}",
            "VADER": "{:+.3f}",
        }).map(
            lambda v: "color: green" if isinstance(v, (int, float)) and v > 0.001
            else ("color: red" if isinstance(v, (int, float)) and v < -0.001 else ""),
            subset=["Δp"],
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No news has been processed yet.")

# =====================================================================
# Footer
# =====================================================================
st.divider()
st.caption(
    "RITC 2026 Merger Arbitrage Dashboard | "
    "Formulas: P* = p × K + (1-p) × V | "
    "Δp = baseline × category_mult × deal_sensitivity | "
    "Packages: streamlit, pandas, numpy, scipy, nltk (VADER)"
)
