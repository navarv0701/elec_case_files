"""
Weather forecast tracking and uncertainty management.
Tracks sunshine (hours) and temperature (Celsius) forecasts
across multiple updates per day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ForecastSeries:
    """Tracks a series of forecast updates for one variable on one day."""
    day: int
    variable: str  # 'sunshine' or 'temperature'
    updates: list[float] = field(default_factory=list)

    @property
    def num_updates(self) -> int:
        return len(self.updates)

    @property
    def latest(self) -> Optional[float]:
        return self.updates[-1] if self.updates else None

    @property
    def uncertainty(self) -> float:
        """Standard deviation of forecast uncertainty.
        Decreases with each update (initial, noon, final).
        """
        if self.variable == 'sunshine':
            return {0: 5.0, 1: 3.0, 2: 1.5, 3: 0.2}.get(self.num_updates, 5.0)
        else:  # temperature
            return {0: 8.0, 1: 5.0, 2: 2.5, 3: 0.5}.get(self.num_updates, 8.0)

    @property
    def is_final(self) -> bool:
        """True if we have the final (evening) forecast."""
        return self.num_updates >= 3

    def add_update(self, value: float):
        """Add a new forecast update."""
        self.updates.append(value)

    def range(self) -> tuple[float, float]:
        """Return (low, high) range based on latest + uncertainty."""
        if self.latest is None:
            return (0.0, 24.0) if self.variable == 'sunshine' else (10.0, 45.0)
        u = self.uncertainty
        return (max(0, self.latest - u), self.latest + u)


class WeatherTracker:
    """Manages all weather forecasts across all days."""

    def __init__(self):
        self._sunshine: dict[int, ForecastSeries] = {}
        self._temperature: dict[int, ForecastSeries] = {}

    def update_sunshine(self, day: int, value: float):
        """Record a sunshine forecast update for a day."""
        if day not in self._sunshine:
            self._sunshine[day] = ForecastSeries(day=day, variable='sunshine')
        self._sunshine[day].add_update(value)

    def update_temperature(self, day: int, value: float):
        """Record a temperature forecast update for a day."""
        if day not in self._temperature:
            self._temperature[day] = ForecastSeries(day=day, variable='temperature')
        self._temperature[day].add_update(value)

    def get_sunshine(self, day: int) -> ForecastSeries:
        """Get sunshine forecast series for a day."""
        if day not in self._sunshine:
            self._sunshine[day] = ForecastSeries(day=day, variable='sunshine')
        return self._sunshine[day]

    def get_temperature(self, day: int) -> ForecastSeries:
        """Get temperature forecast series for a day."""
        if day not in self._temperature:
            self._temperature[day] = ForecastSeries(day=day, variable='temperature')
        return self._temperature[day]

    def summary(self) -> dict:
        """Get a summary of all forecasts."""
        result = {}
        for day in sorted(set(list(self._sunshine.keys()) + list(self._temperature.keys()))):
            sun = self._sunshine.get(day)
            temp = self._temperature.get(day)
            result[day] = {
                'sunshine': {
                    'latest': sun.latest if sun else None,
                    'uncertainty': sun.uncertainty if sun else 5.0,
                    'updates': sun.num_updates if sun else 0,
                    'is_final': sun.is_final if sun else False,
                },
                'temperature': {
                    'latest': temp.latest if temp else None,
                    'uncertainty': temp.uncertainty if temp else 8.0,
                    'updates': temp.num_updates if temp else 0,
                    'is_final': temp.is_final if temp else False,
                },
            }
        return result
