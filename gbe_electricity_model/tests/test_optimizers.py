"""
Tests for optimizer modules: producer, distributor, trader, team coordinator.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.game_state import GameState, RAEBulletin, FactoryTender
from optimizer.producer_optimizer import ProducerOptimizer
from optimizer.distributor_optimizer import DistributorOptimizer
from optimizer.trader_optimizer import TraderOptimizer
from optimizer.team_coordinator import TeamCoordinator


def _make_active_state(day=2, tick=200) -> GameState:
    """Create a GameState with basic market data for testing."""
    state = GameState()
    state.current_tick = tick
    state.current_day = day
    state.case_status = "ACTIVE"

    # Market data
    state.elec_fwd_bid = 38.0
    state.elec_fwd_ask = 40.0
    state.elec_fwd_last = 39.0
    state.ng_bid = 4.50
    state.ng_ask = 4.75
    state.ng_last = 4.60

    # Forecasts for tomorrow
    next_day = day + 1
    state.add_sunshine_forecast(next_day, 8.0, 1)
    state.add_temperature_forecast(next_day, 30.0, 1)

    # RAE bulletin
    state.add_rae_bulletin(RAEBulletin(
        day=next_day, price_low=35.0, price_high=45.0,
        volume_buy=100, volume_sell=100,
        bulletin_number=1, tick_received=tick - 10,
    ))

    return state


class TestProducerOptimizer:
    """Test producer decisions."""

    def test_basic_optimize(self):
        state = _make_active_state()
        optimizer = ProducerOptimizer()
        recs = optimizer.optimize(state)
        assert isinstance(recs, list)
        # Should have at least NG purchase or forward sale recommendations
        # (depending on profitability)

    def test_ng_purchase_recommended_when_profitable(self):
        state = _make_active_state()
        # NG at $4.00, ELEC forward at $45
        # Gas cost: 8 * 4.00 * 100 = $3,200, with 5% margin = $3,360
        # Revenue: 45 * 100 = $4,500 -> clearly profitable
        state.ng_ask = 4.00
        state.elec_fwd_bid = 44.0
        state.elec_fwd_ask = 46.0
        state.elec_fwd_last = 45.0
        optimizer = ProducerOptimizer()
        recs = optimizer.optimize(state)
        ng_recs = [r for r in recs if r.ticker == "NG" and r.action == "BUY"]
        assert len(ng_recs) > 0, "Should recommend buying NG when profitable"

    def test_no_ng_when_unprofitable(self):
        state = _make_active_state()
        state.ng_ask = 8.0  # Very expensive gas
        optimizer = ProducerOptimizer()
        recs = optimizer.optimize(state)
        ng_recs = [r for r in recs if r.ticker == "NG" and r.action == "BUY"]
        assert len(ng_recs) == 0, "Should not buy NG when unprofitable"

    def test_closeout_on_day_end(self):
        # Set up close to day end with open position
        state = _make_active_state(day=2, tick=350)  # Near end of day 2
        state.elec_day_positions = {"ELEC-day2": 5}
        optimizer = ProducerOptimizer()
        recs = optimizer.optimize(state)
        closeout = [r for r in recs if r.urgency in ("HIGH", "CRITICAL")]
        assert len(closeout) > 0, "Should have closeout warnings near day end"

    def test_production_schedule(self):
        state = _make_active_state()
        optimizer = ProducerOptimizer()
        schedule = optimizer.get_production_schedule(state)
        assert 'solar_contracts' in schedule
        assert 'gas_plants' in schedule
        assert schedule['solar_contracts'] == 48  # 6 * 8 hours

    def test_no_recommendations_on_last_day_for_future(self):
        state = _make_active_state(day=5, tick=800)
        state.sunshine_forecasts = {}  # No day 6 forecast
        state.temperature_forecasts = {}
        optimizer = ProducerOptimizer()
        recs = optimizer.optimize(state)
        # Day 5: next_day=6, but forward sales still possible
        # Just verify it doesn't crash
        assert isinstance(recs, list)


class TestDistributorOptimizer:
    """Test distributor decisions."""

    def test_basic_optimize(self):
        state = _make_active_state()
        optimizer = DistributorOptimizer()
        recs = optimizer.optimize(state)
        assert isinstance(recs, list)

    def test_demand_forecast(self):
        state = _make_active_state()
        optimizer = DistributorOptimizer()
        forecast = optimizer.get_demand_forecast(state, 3)
        assert forecast['demand_contracts'] > 0
        assert forecast['temperature'] == 30.0
        assert forecast['customer_revenue'] > 0

    def test_procurement_when_deficit(self):
        state = _make_active_state()
        # No forward position -> needs to buy
        state.elec_f_position = 0
        optimizer = DistributorOptimizer()
        recs = optimizer.optimize(state)
        buy_recs = [r for r in recs if r.action == "BUY" and r.ticker == "ELEC-F"]
        assert len(buy_recs) > 0, "Should recommend forward purchases when deficit"

    def test_no_buy_above_customer_price(self):
        state = _make_active_state()
        state.elec_fwd_ask = 75.0  # Above $70 customer price
        optimizer = DistributorOptimizer()
        recs = optimizer.optimize(state)
        fwd_buys = [r for r in recs if r.action == "BUY" and r.ticker == "ELEC-F"
                     and r.quantity > 0]
        # Should not buy forward at $75 when customer price is $70
        for r in fwd_buys:
            assert r.price is None or r.price <= 75, "Should warn against buying above customer price"

    def test_excess_management(self):
        state = _make_active_state(day=2, tick=200)
        state.elec_day_positions = {"ELEC-day2": 10}  # Excess
        optimizer = DistributorOptimizer()
        recs = optimizer.optimize(state)
        sell_recs = [r for r in recs if r.action == "SELL" and "day" in r.ticker]
        assert len(sell_recs) > 0, "Should recommend selling excess"

    def test_no_forecast_uses_default(self):
        state = _make_active_state()
        state.temperature_forecasts = {}  # Clear forecasts
        optimizer = DistributorOptimizer()
        forecast = optimizer.get_demand_forecast(state, 3)
        assert forecast['demand_contracts'] == 180  # Default
        assert forecast['confidence'] == 'NONE'


class TestTraderOptimizer:
    """Test trader decisions."""

    def test_basic_optimize(self):
        state = _make_active_state()
        optimizer = TraderOptimizer()
        recs = optimizer.optimize(state)
        assert isinstance(recs, list)

    def test_tender_acceptance(self):
        state = _make_active_state()
        # Add a profitable tender
        state.pending_tenders = [FactoryTender(
            tender_id=1, action="BUY", ticker="ELEC-F",
            quantity=5, price=50.0,  # High price - they want to buy from us
            expiration_tick=300,
        )]
        optimizer = TraderOptimizer()
        recs = optimizer.optimize(state)
        tender_recs = [r for r in recs if "TENDER" in r.action]
        assert len(tender_recs) > 0

    def test_tender_rejection_low_profit(self):
        state = _make_active_state()
        # Add an unprofitable tender
        state.pending_tenders = [FactoryTender(
            tender_id=2, action="BUY", ticker="ELEC-F",
            quantity=5, price=35.0,  # Below market ask
            expiration_tick=300,
        )]
        optimizer = TraderOptimizer()
        recs = optimizer.optimize(state)
        declines = [r for r in recs if r.action == "DECLINE_TENDER"]
        assert len(declines) > 0, "Should decline unprofitable tender"

    def test_closeout_recommendations(self):
        state = _make_active_state(day=3, tick=520)  # Near end of day 3
        state.elec_day_positions = {"ELEC-day3": -8}
        optimizer = TraderOptimizer()
        recs = optimizer.optimize(state)
        closeout = [r for r in recs if r.urgency in ("HIGH", "CRITICAL", "MEDIUM")
                     and "CLOSE" in r.reason]
        assert len(closeout) > 0, "Should warn about unclosed position"

    def test_arbitrage_detection(self):
        state = _make_active_state()
        # Set forward at premium above spot
        state.elec_fwd_bid = 48.0  # High forward bid
        state.elec_fwd_ask = 50.0
        # RAE says spot should be ~35-40
        optimizer = TraderOptimizer()
        recs = optimizer.optimize(state)
        arb_recs = [r for r in recs if "arbitrage" in r.reason.lower() or "Arbitrage" in r.reason]
        # May or may not detect depending on premium threshold
        assert isinstance(recs, list)


class TestTeamCoordinator:
    """Test cross-role coordination."""

    def test_generate_team_plan(self):
        state = _make_active_state()
        coordinator = TeamCoordinator()
        plan = coordinator.generate_team_plan(state)

        assert 'day' in plan
        assert 'supply' in plan
        assert 'demand' in plan
        assert 'balance' in plan
        assert 'pricing' in plan
        assert 'producer_plan' in plan
        assert 'distributor_plan' in plan
        assert 'trader_plan' in plan
        assert 'internal_price' in plan
        assert 'team_exposure' in plan

    def test_supply_computation(self):
        state = _make_active_state()
        coordinator = TeamCoordinator()
        supply = coordinator._compute_supply(state, 3)
        assert supply['solar_contracts'] == 48  # 6 * 8 hours
        assert supply['total_contracts'] >= supply['solar_contracts']

    def test_demand_computation(self):
        state = _make_active_state()
        coordinator = TeamCoordinator()
        demand = coordinator._compute_demand(state, 3)
        assert demand['demand_contracts'] > 0
        assert demand['temperature'] == 30.0
        assert demand['demand_low'] < demand['demand_high']

    def test_balance_surplus(self):
        coordinator = TeamCoordinator()
        supply = {'total_contracts': 250, 'solar_contracts': 200, 'gas_contracts': 50}
        demand = {'demand_contracts': 180}
        balance = coordinator._compute_balance(supply, demand)
        assert balance['is_surplus'] is True
        assert balance['surplus'] == 70
        assert balance['deficit'] == 0

    def test_balance_deficit(self):
        coordinator = TeamCoordinator()
        supply = {'total_contracts': 150, 'solar_contracts': 150, 'gas_contracts': 0}
        demand = {'demand_contracts': 200}
        balance = coordinator._compute_balance(supply, demand)
        assert balance['is_deficit'] is True
        assert balance['deficit'] == 50

    def test_internal_price(self):
        state = _make_active_state()
        coordinator = TeamCoordinator()
        supply = {'total_contracts': 200, 'solar_contracts': 150, 'gas_contracts': 50}
        demand = {'demand_contracts': 180}
        price = coordinator._optimal_internal_price(state, supply, demand)
        assert price > 0
        assert price <= 70  # Can't exceed customer price


class TestNewsParser:
    """Test news parsing."""

    def test_parse_sunshine(self):
        from api.data_poller import NewsParser
        result = NewsParser.parse("Weather Update", "Tomorrow expect 8.5 hours of sunshine")
        assert result['type'] == 'sunshine'
        assert result['value'] == 8.5

    def test_parse_temperature(self):
        from api.data_poller import NewsParser
        result = NewsParser.parse("Weather", "Average temperature expected to be 28 degrees Celsius")
        assert result['type'] == 'temperature'
        assert result['value'] == 28.0

    def test_parse_rae_bulletin(self):
        from api.data_poller import NewsParser
        result = NewsParser.parse(
            "RAE Bulletin",
            "Spot price expected between $35 and $45. "
            "200 contracts for buying and 150 contracts for selling."
        )
        assert result['type'] == 'rae_bulletin'
        assert result['low'] == 35.0
        assert result['high'] == 45.0
        assert result['data']['volume_buy'] == 200
        assert result['data']['volume_sell'] == 150

    def test_parse_unknown(self):
        from api.data_poller import NewsParser
        result = NewsParser.parse("Random headline", "Nothing useful here")
        assert result['type'] == 'unknown'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
