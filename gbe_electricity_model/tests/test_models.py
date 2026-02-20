"""
Tests for mathematical models: demand, production, pricing, weather.
"""

import pytest
import sys
import os

# Ensure the parent package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.demand import demand_contracts, demand_derivative, DemandForecaster
from models.production import (
    solar_production, gas_production, gas_cost_per_elec_contract,
    gas_profit_per_plant, ProductionScheduler,
)
from models.pricing import PricingModel
from models.weather import ForecastSeries, WeatherTracker
from state.game_state import GameState, RAEBulletin


# ============================================================
# Demand Model Tests
# ============================================================

class TestDemandModel:
    """Test the demand curve: ELEC = 200 - 15*AT + 0.8*AT^2 - 0.01*AT^3"""

    def test_demand_at_20c(self):
        """At 20C: 200 - 300 + 320 - 80 = 140"""
        result = demand_contracts(20.0)
        assert abs(result - 140.0) < 0.1

    def test_demand_at_30c(self):
        """At 30C: 200 - 450 + 720 - 270 = 200"""
        result = demand_contracts(30.0)
        assert abs(result - 200.0) < 0.1

    def test_demand_at_40c(self):
        """At 40C: 200 - 600 + 1280 - 640 = 240"""
        result = demand_contracts(40.0)
        assert abs(result - 240.0) < 0.1

    def test_demand_at_12c(self):
        """At 12C: approximately 118"""
        result = demand_contracts(12.0)
        # 200 - 180 + 115.2 - 17.28 = 117.92
        assert abs(result - 117.92) < 0.1

    def test_demand_monotonically_increasing_in_summer_range(self):
        """Demand should increase from 15C to 41C (cubic peaks near 41C)."""
        prev = demand_contracts(15.0)
        for temp in range(16, 42):
            current = demand_contracts(float(temp))
            assert current >= prev, f"Demand decreased from {temp-1}C to {temp}C"
            prev = current

    def test_demand_derivative_positive_in_range(self):
        """Derivative should be positive in 20-40C range."""
        for temp in [20, 25, 30, 35, 40]:
            d = demand_derivative(float(temp))
            assert d > 0, f"Derivative negative at {temp}C: {d}"

    def test_demand_clamped_to_nonnegative(self):
        """Demand should never go negative."""
        for temp in range(0, 60):
            d = demand_contracts(float(temp))
            assert d >= 0, f"Negative demand at {temp}C: {d}"


class TestDemandForecaster:
    """Test Bayesian demand forecasting."""

    def test_single_update(self):
        forecaster = DemandForecaster()
        result = forecaster.update(30.0)
        assert result['demand_exact'] == pytest.approx(200.0, abs=0.1)
        assert result['num_updates'] == 1
        assert result['demand_mean'] > 0

    def test_multiple_updates_narrow_uncertainty(self):
        forecaster = DemandForecaster()
        r1 = forecaster.update(28.0)
        r2 = forecaster.update(30.0)
        r3 = forecaster.update(29.5)
        # Uncertainty should decrease
        assert r3['temp_std'] < r1['temp_std']
        assert r3['num_updates'] == 3

    def test_optimal_procurement(self):
        forecaster = DemandForecaster()
        forecaster.update(30.0)
        result = forecaster.optimal_procurement(30.0, 2.5, 40.0)
        assert result['target_contracts'] > 0
        assert result['critical_ratio'] > 0
        assert result['critical_ratio'] <= 1

    def test_reset(self):
        forecaster = DemandForecaster()
        forecaster.update(30.0)
        forecaster.reset()
        result = forecaster.update(25.0)
        assert result['num_updates'] == 1


# ============================================================
# Production Model Tests
# ============================================================

class TestProductionModel:
    """Test solar and gas production calculations."""

    def test_solar_production_basic(self):
        """Solar = 6 * hours of sunshine."""
        assert solar_production(0) == 0
        assert solar_production(6) == 36
        assert solar_production(12) == 72
        assert solar_production(10) == 60

    def test_solar_production_max(self):
        """Max sunshine ~24h -> 144 contracts."""
        assert solar_production(24) == 144

    def test_gas_production(self):
        """8 NG contracts -> 1 ELEC contract."""
        assert gas_production(0) == 0
        assert gas_production(7) == 0   # Needs 8 for 1 plant
        assert gas_production(8) == 1
        assert gas_production(16) == 2
        assert gas_production(80) == 10  # Max

    def test_gas_production_high_input(self):
        """Gas conversion is pure division; capping happens in scheduler."""
        assert gas_production(160) == 20  # 160 // 8 = 20

    def test_gas_cost_per_contract(self):
        """Cost = 8 * NG_price * 100 (NG_contract_size)."""
        cost = gas_cost_per_elec_contract(5.0)
        # 8 contracts * 5.0 $/MMBtu * 100 MMBtu/contract = $4,000
        assert cost == 4000.0

    def test_gas_profit_per_plant(self):
        """Profit = revenue - cost per ELEC contract from gas."""
        profit = gas_profit_per_plant(5.0, 50.0)
        # Revenue: 50 * 100 = $5,000
        # Cost: 8 * 5.0 * 100 = $4,000
        # Profit: $1,000
        assert profit == 1000.0

    def test_gas_unprofitable(self):
        """Gas is unprofitable when elec price is too low."""
        profit = gas_profit_per_plant(6.0, 40.0)
        # Revenue: 40 * 100 = $4,000
        # Cost: 8 * 6.0 * 100 = $4,800
        # Profit: -$800
        assert profit == -800.0


class TestProductionScheduler:
    """Test the production scheduler."""

    def test_compute_schedule_profitable(self):
        scheduler = ProductionScheduler()
        schedule = scheduler.compute_schedule(
            sunshine_forecast=8.0,
            sunshine_uncertainty=1.5,
            ng_ask_price=5.0,
            expected_elec_price=50.0,
            team_demand=200,
            current_ng_position=0,
        )
        assert schedule['solar_contracts'] == 48  # 6 * 8
        assert schedule['gas_is_profitable'] is True
        assert schedule['ng_to_buy'] > 0
        assert schedule['gas_plants'] > 0

    def test_compute_schedule_unprofitable_gas(self):
        scheduler = ProductionScheduler()
        schedule = scheduler.compute_schedule(
            sunshine_forecast=8.0,
            sunshine_uncertainty=1.5,
            ng_ask_price=10.0,
            expected_elec_price=30.0,
            team_demand=200,
            current_ng_position=0,
        )
        # Gas cost = 8 * 10 * 100 = $8,000 but revenue = 30 * 100 = $3,000
        assert schedule['gas_is_profitable'] is False


# ============================================================
# Pricing Model Tests
# ============================================================

class TestPricingModel:
    """Test spot price estimation."""

    def test_estimate_from_rae_bulletin(self):
        pricing = PricingModel()
        state = GameState()
        state.current_day = 2
        state.add_rae_bulletin(RAEBulletin(
            day=3, price_low=35.0, price_high=45.0,
            volume_buy=100, volume_sell=100,
            bulletin_number=1, tick_received=200,
        ))
        result = pricing.estimate_spot_price(3, state)
        assert result['estimate'] > 0
        assert result['confidence'] in ('LOW', 'MEDIUM', 'HIGH')

    def test_estimate_from_forward(self):
        pricing = PricingModel()
        state = GameState()
        state.elec_fwd_bid = 38.0
        state.elec_fwd_ask = 42.0
        state.elec_fwd_last = 40.0
        result = pricing.estimate_spot_price(3, state)
        assert result['estimate'] > 0

    def test_forward_vs_spot_premium(self):
        pricing = PricingModel()
        state = GameState()
        state.elec_fwd_bid = 45.0
        state.elec_fwd_ask = 47.0
        state.elec_fwd_last = 46.0
        # Set a lower RAE price so forward is at a premium
        state.add_rae_bulletin(RAEBulletin(
            day=3, price_low=35.0, price_high=40.0,
            volume_buy=100, volume_sell=100,
            bulletin_number=1, tick_received=200,
        ))
        result = pricing.forward_vs_spot_premium(state, 3)
        assert 'premium_per_mwh' in result
        assert 'action_signal' in result


# ============================================================
# Weather Tracker Tests
# ============================================================

class TestWeatherTracker:
    """Test weather forecast tracking."""

    def test_forecast_series_updates(self):
        fs = ForecastSeries(day=2, variable='sunshine')
        assert fs.num_updates == 0
        assert fs.latest is None
        assert fs.uncertainty == 5.0

        fs.add_update(8.0)
        assert fs.num_updates == 1
        assert fs.latest == 8.0
        assert fs.uncertainty == 3.0

        fs.add_update(8.5)
        assert fs.num_updates == 2
        assert fs.latest == 8.5
        assert fs.uncertainty == 1.5

        fs.add_update(8.3)
        assert fs.is_final
        assert fs.uncertainty == 0.2

    def test_temperature_uncertainty(self):
        fs = ForecastSeries(day=2, variable='temperature')
        assert fs.uncertainty == 8.0  # No updates

        fs.add_update(25.0)
        assert fs.uncertainty == 5.0

        fs.add_update(26.0)
        assert fs.uncertainty == 2.5

        fs.add_update(25.5)
        assert fs.uncertainty == 0.5

    def test_range_method(self):
        fs = ForecastSeries(day=2, variable='sunshine')
        fs.add_update(10.0)
        low, high = fs.range()
        assert low == 7.0   # 10 - 3
        assert high == 13.0  # 10 + 3

    def test_weather_tracker(self):
        tracker = WeatherTracker()
        tracker.update_sunshine(2, 8.0)
        tracker.update_sunshine(2, 8.5)
        tracker.update_temperature(2, 25.0)

        sun = tracker.get_sunshine(2)
        assert sun.num_updates == 2
        assert sun.latest == 8.5

        temp = tracker.get_temperature(2)
        assert temp.num_updates == 1
        assert temp.latest == 25.0

    def test_summary(self):
        tracker = WeatherTracker()
        tracker.update_sunshine(2, 8.0)
        tracker.update_temperature(2, 25.0)
        summary = tracker.summary()
        assert 2 in summary
        assert summary[2]['sunshine']['latest'] == 8.0
        assert summary[2]['temperature']['latest'] == 25.0


# ============================================================
# Integration: GameState
# ============================================================

class TestGameState:
    """Test GameState helper methods."""

    def test_current_day_from_tick(self):
        state = GameState()
        state.current_tick = 0
        assert state.current_day_from_tick() == 1
        state.current_tick = 179
        assert state.current_day_from_tick() == 1
        state.current_tick = 180
        assert state.current_day_from_tick() == 2
        state.current_tick = 899
        assert state.current_day_from_tick() == 5

    def test_ticks_remaining(self):
        state = GameState()
        state.current_tick = 0
        assert state.ticks_remaining_in_day() == 180
        state.current_tick = 90
        assert state.ticks_remaining_in_day() == 90
        state.current_tick = 179
        assert state.ticks_remaining_in_day() == 1

    def test_is_forecast_time(self):
        state = GameState()
        state.current_tick = 0
        assert state.is_forecast_time() == "initial"
        state.current_tick = 90
        assert state.is_forecast_time() == "noon"
        state.current_tick = 150
        assert state.is_forecast_time() == "final"
        state.current_tick = 50
        assert state.is_forecast_time() is None

    def test_closeout_warnings(self):
        state = GameState()
        state.current_tick = 120  # 60 ticks into day 1
        assert state.is_day_close_warning() is True
        assert state.is_day_close_urgent() is False

        state.current_tick = 160  # 160 % 180 = 160, 20 left
        assert state.is_day_close_urgent() is True
        assert state.is_day_close_critical() is False

        state.current_tick = 176  # 176 % 180 = 176, 4 left
        assert state.is_day_close_critical() is True

    def test_add_forecasts(self):
        state = GameState()
        state.add_sunshine_forecast(3, 8.0, 1)
        state.add_sunshine_forecast(3, 8.5, 2)
        assert state.best_sunshine_forecast(3) == 8.5
        assert state.sunshine_uncertainty(3) == 1.5  # 2 updates

    def test_no_forecast_returns_none(self):
        state = GameState()
        assert state.best_sunshine_forecast(5) is None
        assert state.best_temperature_forecast(5) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
