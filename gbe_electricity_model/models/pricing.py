"""
Electricity price estimation model.
Combines RAE bulletins, forward prices, and supply-demand signals
to estimate next-day spot prices.
"""

from __future__ import annotations

from typing import Optional

from state.game_state import GameState, RAEBulletin


class PricingModel:
    """Estimates future electricity spot prices from available market data."""

    def estimate_spot_price(self, day: int, state: GameState) -> dict:
        """Estimate the spot price for a given day using multiple signals.

        Signal weights:
        1. RAE Bulletin midpoint: 0.5 (first) or 0.7 (second)
        2. Forward price: 0.3
        3. Supply-demand balance: 0.2

        Args:
            day: The target day for price estimation.
            state: Current game state.

        Returns:
            {
                'estimate': float,      # Best estimate $/MWh
                'low': float,           # Low estimate
                'high': float,          # High estimate
                'confidence': str,      # LOW, MEDIUM, HIGH
                'signals': dict,        # Component signals
            }
        """
        signals = {}

        # Signal 1: RAE Bulletin
        rae_estimate = None
        rae_low = None
        rae_high = None
        bulletin = state.latest_rae_bulletin(day)
        if bulletin:
            rae_estimate = (bulletin.price_low + bulletin.price_high) / 2
            rae_low = bulletin.price_low
            rae_high = bulletin.price_high
            signals['rae'] = {
                'estimate': rae_estimate,
                'low': rae_low,
                'high': rae_high,
                'bulletin_number': bulletin.bulletin_number,
            }

        # Signal 2: Forward price
        fwd_mid = None
        if state.elec_fwd_bid > 0 and state.elec_fwd_ask > 0:
            fwd_mid = (state.elec_fwd_bid + state.elec_fwd_ask) / 2
            signals['forward'] = {
                'bid': state.elec_fwd_bid,
                'ask': state.elec_fwd_ask,
                'mid': fwd_mid,
            }
        elif state.elec_fwd_last > 0:
            fwd_mid = state.elec_fwd_last
            signals['forward'] = {'last': fwd_mid}

        # Combine signals with weights
        if rae_estimate is not None and fwd_mid is not None:
            # RAE weight depends on which bulletin (second is more accurate)
            rae_weight = 0.7 if (bulletin and bulletin.bulletin_number >= 2) else 0.5
            fwd_weight = 1.0 - rae_weight
            estimate = rae_estimate * rae_weight + fwd_mid * fwd_weight
            confidence = "HIGH" if bulletin.bulletin_number >= 2 else "MEDIUM"
        elif rae_estimate is not None:
            estimate = rae_estimate
            confidence = "MEDIUM"
        elif fwd_mid is not None:
            estimate = fwd_mid
            confidence = "LOW"
        else:
            # No data: use a default mid-range price
            estimate = 40.0
            confidence = "NONE"

        # Bounds
        low = rae_low if rae_low is not None else estimate * 0.85
        high = rae_high if rae_high is not None else estimate * 1.15

        return {
            'estimate': estimate,
            'low': low,
            'high': high,
            'confidence': confidence,
            'signals': signals,
        }

    def forward_vs_spot_premium(self, state: GameState, day: int) -> dict:
        """Compute the forward premium (or discount) relative to expected spot.

        Positive premium = forward is overpriced (consider selling forward).
        Negative premium = forward is underpriced (consider buying forward).

        Returns:
            {
                'premium_per_mwh': float,
                'premium_pct': float,
                'action_signal': str,   # 'SELL_FORWARD', 'BUY_FORWARD', 'NEUTRAL'
            }
        """
        spot_est = self.estimate_spot_price(day, state)
        spot = spot_est['estimate']

        if spot == 0:
            return {'premium_per_mwh': 0, 'premium_pct': 0, 'action_signal': 'NEUTRAL'}

        fwd_bid = state.elec_fwd_bid
        fwd_ask = state.elec_fwd_ask

        # Selling perspective: compare forward bid vs spot
        sell_premium = fwd_bid - spot if fwd_bid > 0 else 0
        # Buying perspective: compare spot vs forward ask
        buy_premium = spot - fwd_ask if fwd_ask > 0 else 0

        if sell_premium > 1.0:  # >$1/MWh premium
            action = "SELL_FORWARD"
            premium = sell_premium
        elif buy_premium > 1.0:
            action = "BUY_FORWARD"
            premium = -buy_premium
        else:
            action = "NEUTRAL"
            premium = sell_premium if fwd_bid > 0 else 0

        return {
            'premium_per_mwh': premium,
            'premium_pct': (premium / spot * 100) if spot > 0 else 0,
            'action_signal': action,
            'spot_estimate': spot,
            'forward_bid': fwd_bid,
            'forward_ask': fwd_ask,
        }

    def gas_conversion_value(self, state: GameState, day: int) -> dict:
        """Compare the cost of producing electricity from gas vs. buying on market.

        Helps Producer decide whether to run gas plants.

        Returns:
            {
                'ng_cost_per_elec': float,    # Cost to produce 1 ELEC contract from gas
                'market_elec_price': float,   # Expected market price for 1 ELEC contract
                'spread': float,              # Market price - gas cost (positive = profitable)
                'profitable': bool,
            }
        """
        ng_price = state.ng_ask if state.ng_ask > 0 else state.ng_last
        if ng_price <= 0:
            return {'ng_cost_per_elec': 0, 'market_elec_price': 0, 'spread': 0, 'profitable': False}

        # Cost of 1 ELEC contract from gas: 8 NG contracts * 100 MMBtu * NG_price
        ng_cost = 8 * 100 * ng_price

        # Expected revenue: spot estimate * 100 MWh
        spot_est = self.estimate_spot_price(day, state)
        market_revenue = spot_est['estimate'] * 100

        spread = market_revenue - ng_cost

        return {
            'ng_cost_per_elec': ng_cost,
            'market_elec_price': market_revenue,
            'spread': spread,
            'profitable': spread > 0,
            'margin_pct': (spread / ng_cost * 100) if ng_cost > 0 else 0,
        }
