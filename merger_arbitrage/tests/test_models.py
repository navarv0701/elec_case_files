"""
Tests for deal valuation models, probability tracking, and news classification.
Verifies calculations against hand-computed reference values from the case package.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deal import compute_standalone_value, initialize_all_deals, DealState
from models.probability import ProbabilityTracker, compute_delta_p, NewsImpact
from nlp.news_classifier import NewsClassifier
from state.market_state import MarketState, TradeRecommendation
from strategy.signal_generator import SignalGenerator
from strategy.position_manager import PositionManager
from config import DEALS


class TestStandaloneValues:
    """Verify standalone value V for all 5 deals against hand-calculated references."""

    def test_d1_standalone_value(self):
        """D1: All-cash $50, p0=0.70, P0=43.70
        V = (43.70 - 0.70*50) / (1-0.70) = (43.70-35)/0.30 = 29.00
        """
        V = compute_standalone_value(DEALS["D1"])
        assert abs(V - 29.00) < 0.01, f"D1 V={V}, expected 29.00"

    def test_d2_standalone_value(self):
        """D2: Stock 0.75*CLD, p0=0.55, P0=43.50, CLD0=79.30
        K0 = 0.75*79.30 = 59.475
        V = (43.50 - 0.55*59.475) / (1-0.55) = (43.50-32.71125)/0.45 = 23.975
        """
        V = compute_standalone_value(DEALS["D2"])
        assert abs(V - 23.975) < 0.05, f"D2 V={V}, expected ~23.975"

    def test_d3_standalone_value(self):
        """D3: Mixed $33+0.20*PNR, p0=0.50, P0=31.50, PNR0=59.80
        K0 = 33 + 0.20*59.80 = 44.96
        V = (31.50 - 0.50*44.96) / (1-0.50) = (31.50-22.48)/0.50 = 18.04
        """
        V = compute_standalone_value(DEALS["D3"])
        assert abs(V - 18.04) < 0.01, f"D3 V={V}, expected 18.04"

    def test_d4_standalone_value(self):
        """D4: All-cash $40, p0=0.38, P0=30.50
        V = (30.50 - 0.38*40) / (1-0.38) = (30.50-15.20)/0.62 = 24.677
        This is the case package example (page 36): V ≈ $24.68
        """
        V = compute_standalone_value(DEALS["D4"])
        assert abs(V - 24.677) < 0.01, f"D4 V={V}, expected ~24.68"

    def test_d5_standalone_value(self):
        """D5: Stock 1.20*EEC, p0=0.45, P0=52.80, EEC0=48.00
        K0 = 1.20*48.00 = 57.60
        V = (52.80 - 0.45*57.60) / (1-0.45) = (52.80-25.92)/0.55 = 48.873
        """
        V = compute_standalone_value(DEALS["D5"])
        assert abs(V - 48.873) < 0.01, f"D5 V={V}, expected ~48.87"


class TestDealValueK:
    """Verify deal value K computation for different structures."""

    def test_all_cash_K(self):
        deals = initialize_all_deals()
        d1 = deals["D1"]
        assert d1.deal_value_K == 50.0, f"D1 K should be $50 (all-cash)"

    def test_stock_K(self):
        deals = initialize_all_deals()
        d2 = deals["D2"]
        # K = 0.75 * 79.30 = 59.475
        assert abs(d2.deal_value_K - 59.475) < 0.01

    def test_mixed_K(self):
        deals = initialize_all_deals()
        d3 = deals["D3"]
        # K = 33 + 0.20 * 59.80 = 44.96
        assert abs(d3.deal_value_K - 44.96) < 0.01

    def test_stock_K_updates_with_acquirer_price(self):
        """Stock deal value should change when acquirer price changes."""
        deals = initialize_all_deals()
        d2 = deals["D2"]
        K_initial = d2.deal_value_K

        d2.acquirer_price = 85.00
        K_new = d2.deal_value_K
        # K = 0.75 * 85.00 = 63.75
        assert abs(K_new - 63.75) < 0.01
        assert K_new != K_initial

    def test_cash_K_does_not_change(self):
        """All-cash deal value should NOT change when acquirer price changes."""
        deals = initialize_all_deals()
        d1 = deals["D1"]
        K_initial = d1.deal_value_K

        d1.acquirer_price = 55.00
        assert d1.deal_value_K == K_initial


class TestIntrinsicPrice:
    """Verify P* = p * K + (1-p) * V"""

    def test_initial_pstar_equals_start_price(self):
        """At t=0, P* should approximately equal the starting price."""
        deals = initialize_all_deals()
        for deal_id, deal in deals.items():
            Pstar = deal.intrinsic_target_price
            P0 = deal.config.target_start_price
            assert abs(Pstar - P0) < 0.02, (
                f"{deal_id}: P*={Pstar:.4f} should ≈ P0={P0:.4f}"
            )

    def test_pstar_at_p_equals_1(self):
        """When p=1.0, P* should equal K (deal value)."""
        deals = initialize_all_deals()
        d1 = deals["D1"]
        d1.probability = 1.0
        assert abs(d1.intrinsic_target_price - 50.0) < 0.01

    def test_pstar_at_p_equals_0(self):
        """When p=0.0, P* should equal V (standalone value)."""
        deals = initialize_all_deals()
        d1 = deals["D1"]
        d1.probability = 0.0
        assert abs(d1.intrinsic_target_price - d1.standalone_value) < 0.01


class TestMispricing:
    """Verify mispricing detection."""

    def test_no_mispricing_at_start(self):
        """At initialization, mispricing should be near zero."""
        deals = initialize_all_deals()
        for deal_id, deal in deals.items():
            assert abs(deal.target_mispricing) < 0.02, (
                f"{deal_id}: mispricing={deal.target_mispricing:.4f} should be ~0"
            )

    def test_positive_mispricing_when_undervalued(self):
        """If market price drops below P*, mispricing should be positive."""
        deals = initialize_all_deals()
        d1 = deals["D1"]
        d1.target_price = 40.00  # Below P* of ~43.70
        assert d1.target_mispricing > 0

    def test_negative_mispricing_when_overvalued(self):
        """If market price rises above P*, mispricing should be negative."""
        deals = initialize_all_deals()
        d1 = deals["D1"]
        d1.target_price = 48.00  # Above P* of ~43.70
        assert d1.target_mispricing < 0


class TestProbabilityDeltaP:
    """Verify delta_p computation matches case package examples."""

    def test_case_package_example_d4_reg_positive_medium(self):
        """Case package example (p.36):
        D4 Banking, REG category, positive medium
        delta_p = 0.07 * 1.25 * 1.30 = 0.11375
        """
        dp = compute_delta_p("D4", "REG", "positive", "medium")
        assert abs(dp - 0.11375) < 0.0001, f"dp={dp}, expected 0.11375"

    def test_case_package_example_d3_fin_negative_large(self):
        """Case package example (p.37):
        D3 Energy, FIN category, negative large
        delta_p = -0.18 * 1.00 * 1.10 = -0.198
        """
        dp = compute_delta_p("D3", "FIN", "negative", "large")
        assert abs(dp - (-0.198)) < 0.0001, f"dp={dp}, expected -0.198"

    def test_d1_small_positive(self):
        """D1: +0.03 * 1.25 (REG) * 1.00 = 0.0375"""
        dp = compute_delta_p("D1", "REG", "positive", "small")
        assert abs(dp - 0.0375) < 0.0001

    def test_ambiguous_returns_zero_by_default(self):
        """Ambiguous with default midpoint should be 0.0."""
        dp = compute_delta_p("D1", "FIN", "ambiguous", "small")
        assert dp == 0.0

    def test_unknown_deal_returns_zero(self):
        dp = compute_delta_p("D99", "REG", "positive", "large")
        assert dp == 0.0


class TestProbabilityTracker:
    """Test probability tracking with clamping and accumulation."""

    def test_apply_positive_news(self):
        tracker = ProbabilityTracker({"D4": 0.38})
        impact = NewsImpact("D4", "REG", "positive", "medium", 0.11375, "test", 0)
        new_p = tracker.apply_news(impact)
        assert abs(new_p - 0.49375) < 0.0001

    def test_clamping_at_1(self):
        tracker = ProbabilityTracker({"D1": 0.95})
        impact = NewsImpact("D1", "REG", "positive", "large", 0.20, "test", 0)
        new_p = tracker.apply_news(impact)
        assert new_p == 1.0

    def test_clamping_at_0(self):
        tracker = ProbabilityTracker({"D1": 0.05})
        impact = NewsImpact("D1", "FIN", "negative", "large", -0.20, "test", 0)
        new_p = tracker.apply_news(impact)
        assert new_p == 0.0

    def test_cumulative_impacts(self):
        tracker = ProbabilityTracker({"D4": 0.38})
        impact1 = NewsImpact("D4", "REG", "positive", "medium", 0.11375, "test1", 0)
        impact2 = NewsImpact("D4", "FIN", "negative", "small", -0.052, "test2", 10)
        tracker.apply_news(impact1)
        p = tracker.apply_news(impact2)
        expected = 0.38 + 0.11375 - 0.052
        assert abs(p - expected) < 0.0001


class TestNewsClassifier:
    """Test news classification pipeline."""

    def setup_method(self):
        self.classifier = NewsClassifier()

    def test_identify_deal_by_ticker(self):
        result = self.classifier.classify("TGX announces quarterly results")
        assert result.deal_id == "D1"

    def test_identify_deal_by_company_name(self):
        result = self.classifier.classify("Targenix receives regulatory approval")
        assert result.deal_id == "D1"

    def test_identify_deal_by_acquirer_ticker(self):
        result = self.classifier.classify("CLD reports strong earnings growth")
        assert result.deal_id == "D2"

    def test_regulatory_category(self):
        result = self.classifier.classify(
            "Regulators indicate remedies framework is acceptable"
        )
        assert result.category == "REG"

    def test_financing_category(self):
        result = self.classifier.classify(
            "Credit conditions deteriorate; lenders seek repricing"
        )
        assert result.category == "FIN"

    def test_shareholder_category(self):
        result = self.classifier.classify(
            "Proxy advisor ISS recommends shareholders vote in favor"
        )
        assert result.category == "SHR"

    def test_alternative_bid_category(self):
        result = self.classifier.classify(
            "A competing bid has been submitted by a rival firm"
        )
        assert result.category == "ALT"

    def test_positive_direction(self):
        result = self.classifier.classify(
            "Regulators approve the merger unconditionally"
        )
        assert result.direction == "positive"

    def test_negative_direction(self):
        result = self.classifier.classify(
            "Deal blocked by antitrust regulators"
        )
        assert result.direction == "negative"

    def test_large_severity(self):
        result = self.classifier.classify(
            "Major breakthrough in regulatory approval process"
        )
        assert result.severity == "large"

    def test_deal_completed_resolution(self):
        result = self.classifier.classify("TGX deal completed successfully")
        assert result.is_resolution is True
        assert result.resolution_type == "completed"
        assert result.deal_id == "D1"

    def test_deal_terminated_resolution(self):
        result = self.classifier.classify("FSR deal terminated by both parties")
        assert result.is_resolution is True
        assert result.resolution_type == "failed"
        assert result.deal_id == "D4"

    def test_unrelated_news(self):
        result = self.classifier.classify("Weather forecast for Toronto is sunny")
        assert result.deal_id is None


class TestHedgeRatios:
    """Verify hedge ratio computation for each deal type."""

    def test_all_cash_no_hedge(self):
        deals = initialize_all_deals()
        assert deals["D1"].ideal_hedge_ratio == 0.0
        assert deals["D4"].ideal_hedge_ratio == 0.0

    def test_stock_for_stock_hedge(self):
        deals = initialize_all_deals()
        assert deals["D2"].ideal_hedge_ratio == 0.75
        assert deals["D5"].ideal_hedge_ratio == 1.20

    def test_mixed_hedge(self):
        deals = initialize_all_deals()
        assert deals["D3"].ideal_hedge_ratio == 0.20


class TestSignalGenerator:
    """Test trade signal generation."""

    def setup_method(self):
        self.state = MarketState()
        self.state.initialize()
        self.gen = SignalGenerator()

    def test_no_signals_when_no_mispricing(self):
        """At initialization, prices match P*, so no signals."""
        signals = self.gen.generate_signals(self.state)
        # Filter out closeout signals
        trade_signals = [s for s in signals if "CLOSEOUT" not in s.reason]
        assert len(trade_signals) == 0

    def test_buy_signal_on_undervalued_target(self):
        """When target is undervalued, should generate BUY signal."""
        self.state.deals["D1"].target_price = 38.00  # Well below P* of ~43.70
        self.state.deals["D1"].target_ask = 38.05
        self.state.deals["D1"].target_bid = 37.95
        signals = self.gen.generate_signals(self.state)
        target_signals = [s for s in signals if s.ticker == "TGX"]
        assert len(target_signals) > 0
        assert target_signals[0].action == "BUY"

    def test_stock_deal_generates_hedge(self):
        """Stock-for-stock deal should generate a hedge leg."""
        self.state.deals["D2"].target_price = 38.00  # Below P*
        self.state.deals["D2"].target_ask = 38.05
        self.state.deals["D2"].target_bid = 37.95
        signals = self.gen.generate_signals(self.state)
        byl_signals = [s for s in signals if s.ticker == "BYL"]
        cld_signals = [s for s in signals if s.ticker == "CLD"]
        assert len(byl_signals) > 0  # Target leg
        assert len(cld_signals) > 0  # Hedge leg
        assert cld_signals[0].is_hedge_leg is True

    def test_cash_deal_no_hedge(self):
        """All-cash deal should NOT generate hedge leg."""
        self.state.deals["D1"].target_price = 38.00
        self.state.deals["D1"].target_ask = 38.05
        self.state.deals["D1"].target_bid = 37.95
        signals = self.gen.generate_signals(self.state)
        phr_signals = [s for s in signals if s.ticker == "PHR"]
        assert len(phr_signals) == 0  # No hedge needed


class TestPositionManager:
    """Test position limit enforcement."""

    def setup_method(self):
        self.state = MarketState()
        self.state.initialize()
        self.mgr = PositionManager()

    def test_max_order_size_enforced(self):
        rec = TradeRecommendation(
            action="BUY", ticker="TGX", quantity=10000,
            order_type="MARKET", deal_id="D1",
        )
        validated = self.mgr.validate_and_adjust([rec], self.state)
        assert len(validated) == 1
        assert validated[0].quantity <= 5000

    def test_net_position_limit_enforced(self):
        self.state.positions["TGX"] = 48000
        rec = TradeRecommendation(
            action="BUY", ticker="TGX", quantity=5000,
            order_type="MARKET", deal_id="D1",
        )
        validated = self.mgr.validate_and_adjust([rec], self.state)
        if validated:
            # Should be reduced to stay under 50,000 net
            assert self.state.positions["TGX"] + validated[0].quantity <= 50000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
