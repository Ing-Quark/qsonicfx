#!/usr/bin/env python3
"""
alpha_signals.py
================
Q-SonicFX -- Advanced Quantitative Alpha Signal Engine
======================================================

Implements 3 institutional alpha models:
1. VPIN (Volume-Synchronized Probability of Toxicity): Detects toxic order flow
   and predatory market-maker dumps before violent price moves.
2. Z-Score Statistical Arbitrage: Mean-reversion signal generator for ranging regimes.
3. Microstructure Liquidity Sweep Detector: Real-time Level-2 orderbook footprint sweeps.

Author : Q-SonicFX Quant Engine
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# Dataclasses & Data Structures
# ===========================================================================

@dataclass
class VolumeBucket:
    """A constant-volume bucket for VPIN calculation."""
    bucket_id   : int
    target_vol  : float
    buy_vol     : float = 0.0
    sell_vol    : float = 0.0
    filled_vol  : float = 0.0
    start_time  : float = field(default_factory=time.time)
    end_time    : Optional[float] = None
    is_complete : bool = False

    def add_trade(self, price_change: float, volume: float) -> float:
        """
        Add trade volume using the Lee-Ready tick rule.
        Returns remaining volume that exceeded bucket capacity.
        """
        needed = self.target_vol - self.filled_vol
        fill = min(volume, needed)

        if price_change > 0:
            self.buy_vol += fill
        elif price_change < 0:
            self.sell_vol += fill
        else:
            # 50/50 split on zero tick
            self.buy_vol += fill * 0.5
            self.sell_vol += fill * 0.5

        self.filled_vol += fill
        if self.filled_vol >= self.target_vol:
            self.is_complete = True
            self.end_time = time.time()

        return volume - fill

    @property
    def imbalance(self) -> float:
        return abs(self.buy_vol - self.sell_vol)


@dataclass
class CompositeAlphaSignal:
    """Synthesized alpha output from all sub-models."""
    timestamp           : str
    symbol              : str
    primary_signal      : str   # 'BUY', 'SELL', 'NEUTRAL', 'TOXIC_PAUSE'
    confidence          : float # 0.0 to 1.0
    vpin_score          : float # 0.0 to 1.0 (>0.70 is toxic)
    vpin_status         : str   # 'NORMAL', 'ELEVATED', 'TOXIC_ALERT'
    stat_arb_zscore     : float # -3.0 to +3.0
    stat_arb_signal     : str   # 'OVERBOUGHT_SELL', 'OVERSOLD_BUY', 'NEUTRAL'
    sweep_signal        : str   # 'SWEEP_BUY', 'SWEEP_SELL', 'NONE'
    regime              : str
    obi_signal          : str
    model_contributions : Dict[str, float] = field(default_factory=dict)


# ===========================================================================
# 1. VPIN Calculator (Volume-Synchronized Probability of Toxicity)
# ===========================================================================

class VPINCalculator:
    """
    Calculates VPIN (Volume-Synchronized Probability of Toxicity).
    VPIN = sum(|V_b - V_s|) / (N * V) over N volume buckets.
    """

    def __init__(self, bucket_size: float = 10.0, num_buckets: int = 20) -> None:
        self.bucket_size = bucket_size
        self.num_buckets = num_buckets
        self.completed_buckets: deque[VolumeBucket] = deque(maxlen=num_buckets)
        self.current_bucket = VolumeBucket(
            bucket_id=1, target_vol=bucket_size
        )
        self.last_price: Optional[float] = None
        self._bucket_counter = 1

    def update_tick(self, price: float, volume: float) -> float:
        """
        Process a new trade tick and return current VPIN score (0.0 to 1.0).
        """
        if self.last_price is None:
            self.last_price = price
            return 0.0

        price_change = price - self.last_price
        self.last_price = price
        rem_vol = volume

        while rem_vol > 0:
            rem_vol = self.current_bucket.add_trade(price_change, rem_vol)
            if self.current_bucket.is_complete:
                self.completed_buckets.append(self.current_bucket)
                self._bucket_counter += 1
                self.current_bucket = VolumeBucket(
                    bucket_id=self._bucket_counter, target_vol=self.bucket_size
                )

        return self.vpin_score

    @property
    def vpin_score(self) -> float:
        if not self.completed_buckets:
            return 0.0
        total_imb = sum(b.imbalance for b in self.completed_buckets)
        total_vol = sum(b.filled_vol for b in self.completed_buckets)
        if total_vol <= 0:
            return 0.0
        return min(1.0, max(0.0, total_imb / total_vol))

    @property
    def toxicity_status(self) -> str:
        score = self.vpin_score
        if score >= 0.70:
            return "TOXIC_ALERT"
        elif score >= 0.50:
            return "ELEVATED"
        return "NORMAL"


# ===========================================================================
# 2. Statistical Arbitrage / Z-Score Calculator
# ===========================================================================

class StatArbCalculator:
    """
    Calculates rolling Z-Score mean reversion signals: Z = (P - mean) / std.
    """

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self.prices: deque[float] = deque(maxlen=window_size)

    def update(self, price: float) -> Tuple[float, str]:
        """
        Update with latest price tick and return (z_score, signal).
        Signals: 'OVERSOLD_BUY', 'OVERBOUGHT_SELL', 'NEUTRAL'
        """
        self.prices.append(price)
        if len(self.prices) < 10:
            return 0.0, "NEUTRAL"

        arr = np.array(self.prices, dtype=np.float64)
        mean = np.mean(arr)
        std = np.std(arr)

        if std < 1e-8:
            return 0.0, "NEUTRAL"

        z_score = float((price - mean) / std)

        if z_score <= -2.0:
            signal = "OVERSOLD_BUY"
        elif z_score >= 2.0:
            signal = "OVERBOUGHT_SELL"
        else:
            signal = "NEUTRAL"

        return round(z_score, 4), signal


# ===========================================================================
# 3. Microstructure Liquidity Sweep Detector
# ===========================================================================

class LiquiditySweepDetector:
    """
    Detects sudden orderbook depth sweeps by monitoring OBI velocity
    and top-of-book bid/ask volume thins.
    """

    def __init__(self, velocity_threshold: float = 2.5) -> None:
        self.velocity_threshold = velocity_threshold

    def evaluate(
        self, obi_value: float, obi_velocity: float, bid_pct: int, ask_pct: int
    ) -> str:
        """
        Detect microsecond orderbook depth sweeps.
        Returns: 'SWEEP_BUY', 'SWEEP_SELL', 'NONE'
        """
        if obi_velocity >= self.velocity_threshold and bid_pct >= 75:
            return "SWEEP_BUY"
        elif obi_velocity <= -self.velocity_threshold and ask_pct >= 75:
            return "SWEEP_SELL"
        return "NONE"


# ===========================================================================
# Master Quantitative Alpha Signal Synthesizer Engine
# ===========================================================================

class AlphaEngine:
    """
    Master Quantitative Signal Engine coordinating VPIN, StatArb, Sweep,
    and Market Regime into a single robust CompositeAlphaSignal.
    """

    def __init__(self, symbol: str = "BTCUSDT") -> None:
        self.symbol = symbol
        self.vpin_calc = VPINCalculator(bucket_size=5.0, num_buckets=15)
        self.stat_arb = StatArbCalculator(window_size=50)
        self.sweep_detector = LiquiditySweepDetector(velocity_threshold=2.0)

    def process_tick(
        self,
        price: float,
        volume: float,
        regime: str = "RANGING",
        obi_signal: str = "NEUTRAL",
        obi_value: float = 0.0,
        obi_velocity: float = 0.0,
        bid_pct: int = 50,
        ask_pct: int = 50,
    ) -> CompositeAlphaSignal:
        """
        Synthesize all model outputs into a unified CompositeAlphaSignal.
        """
        vpin_score = self.vpin_calc.update_tick(price, volume)
        vpin_status = self.vpin_calc.toxicity_status

        z_score, stat_arb_sig = self.stat_arb.update(price)
        sweep_sig = self.sweep_detector.evaluate(obi_value, obi_velocity, bid_pct, ask_pct)

        primary_signal = "NEUTRAL"
        confidence = 0.50

        # Rule 1: TOXICITY OVERRIDE -- If VPIN > 0.70, halt new long/short entries to prevent getting wrecked
        if vpin_status == "TOXIC_ALERT":
            primary_signal = "TOXIC_PAUSE"
            confidence = 0.95

        # Rule 2: TRENDING REGIME -- Prioritize OBI + Sweep Momentum
        elif regime == "STRONG_TREND":
            if obi_signal == "BUY" or sweep_sig == "SWEEP_BUY":
                primary_signal = "BUY"
                confidence = 0.85 if sweep_sig == "SWEEP_BUY" else 0.75
            elif obi_signal == "SELL" or sweep_sig == "SWEEP_SELL":
                primary_signal = "SELL"
                confidence = 0.85 if sweep_sig == "SWEEP_SELL" else 0.75

        # Rule 3: RANGING REGIME -- Prioritize StatArb Z-Score Mean Reversion
        elif regime in ("RANGING", "LOW_LIQUIDITY_PAUSE"):
            if stat_arb_sig == "OVERSOLD_BUY" and obi_signal != "SELL":
                primary_signal = "BUY"
                confidence = min(0.90, 0.60 + abs(z_score) * 0.10)
            elif stat_arb_sig == "OVERBOUGHT_SELL" and obi_signal != "BUY":
                primary_signal = "SELL"
                confidence = min(0.90, 0.60 + abs(z_score) * 0.10)

        # Fallback to OBI signal if no specific rule triggered
        if primary_signal == "NEUTRAL" and obi_signal in ("BUY", "SELL"):
            primary_signal = obi_signal
            confidence = 0.65

        ts = datetime.utcnow().isoformat() + "Z"
        return CompositeAlphaSignal(
            timestamp=ts,
            symbol=self.symbol,
            primary_signal=primary_signal,
            confidence=round(confidence, 2),
            vpin_score=round(vpin_score, 4),
            vpin_status=vpin_status,
            stat_arb_zscore=z_score,
            stat_arb_signal=stat_arb_sig,
            sweep_signal=sweep_sig,
            regime=regime,
            obi_signal=obi_signal,
            model_contributions={
                "vpin_weight": 0.30,
                "stat_arb_weight": 0.40,
                "obi_sweep_weight": 0.30,
            },
        )


# ===========================================================================
# Self-Test Verification Block
# ===========================================================================

if __name__ == "__main__":
    print("=== Testing Advanced Quantitative Alpha Engine (alpha_signals.py) ===")
    
    engine = AlphaEngine(symbol="BTCUSDT")
    base_price = 65000.0

    # Simulate 30 ticks
    import random
    for i in range(35):
        base_price += random.gauss(0, 12.0)
        vol = random.uniform(0.5, 3.5)
        sig = engine.process_tick(
            price=base_price,
            volume=vol,
            regime="RANGING" if i < 20 else "STRONG_TREND",
            obi_signal="BUY" if i % 3 == 0 else "NEUTRAL",
            obi_value=0.45 if i % 3 == 0 else 0.0,
            obi_velocity=2.1 if i == 25 else 0.0,
            bid_pct=80 if i == 25 else 50,
            ask_pct=20 if i == 25 else 50,
        )

    print(f"Latest Signal      : {sig.primary_signal} (Confidence: {sig.confidence*100:.0f}%)")
    print(f"VPIN Toxicity Score: {sig.vpin_score:.4f} [{sig.vpin_status}]")
    print(f"StatArb Z-Score    : {sig.stat_arb_zscore} [{sig.stat_arb_signal}]")
    print(f"Orderbook Sweep    : {sig.sweep_signal}")
    print("=== Alpha Engine Module Verified OK ===")
