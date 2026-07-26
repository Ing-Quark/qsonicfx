"""
coin_scanner.py
===============
Q-SonicFX — Autonomous Multi-Coin Scanner, Ranker & Selector
=============================================================

Scans all active USDT linear perpetuals on Bybit, filters by
liquidity/min-notional/balance-fit, ranks by weighted composite score
(volatility × volume × OBI confluence), and auto-selects the best
tradeable pair for the current market regime and wallet size.

Author : Q-SonicFX Quant Engine
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("QSonicFX.CoinScanner")

# ── Composite Score Weights ─────────────────────────────────────────────
WEIGHT_VOLATILITY_24H = 0.35   # Higher vol = more opportunity
WEIGHT_VOLUME_Z_SCORE = 0.35   # Normalized 24h volume rank
WEIGHT_OBI_STRENGTH   = 0.30   # |OBI| magnitude (direction-agnostic strength)
PENALTY_MEME_COINS    = 0.85   # 15% penalty on meme/hype coins (reduce rug risk)

# ── Filters ─────────────────────────────────────────────────────────────
MIN_24H_VOLUME_USD  = 10_000   # $10k minimum daily volume
MIN_OPEN_INTEREST   = 100_000  # $100k minimum open interest (optional)
MAX_PRICE_FRACTION  = 0.95     # Max 95% of balance for 1 unit


@dataclass
class CoinCandidate:
    symbol:            str
    price:             float
    volume_24h:        float
    volatility_24h:    float     # std of 1h returns over last 24 periods
    obi_raw:           float     # [-1, +1] raw orderbook imbalance
    obi_score:         float     # normalized |OBI| magnitude [0, 1]
    composite_score:   float     # final weighted rank
    min_qty:           float     # minimum order quantity from instruments info
    min_notional:      float     # minimum notional value
    is_meme:           bool      # True for PEPE, DOGE, WIF, SHIB, etc.
    rejection_reason:  Optional[str] = None

    def fits_balance(self, balance: float) -> bool:
        """Can we buy at least 1 unit without exceeding MAX_PRICE_FRACTION?"""
        cost_one_unit = self.price * self.min_qty
        return cost_one_unit <= balance * MAX_PRICE_FRACTION

    def required_qty(self, balance: float, risk_pct: float = 0.02) -> float:
        """Suggested quantity given balance and risk per trade."""
        max_risk_dollars = balance * risk_pct
        # Use 10x leverage as default assumption
        buying_power = balance * 10
        raw_qty = buying_power / self.price
        # Round to min_qty step
        qty = max(self.min_qty, round(raw_qty / self.min_qty) * self.min_qty)
        return qty


@dataclass
class ScanResult:
    candidates:      List[CoinCandidate]
    total_pairs:     int
    after_filters:   int
    top_pick:        Optional[CoinCandidate] = None
    scan_duration_ms: float = 0.0


class CoinScanner:
    """
    Full life-cycle scanner:
      1. Fetch all active linear USDT instruments from Bybit instruments-info
      2. Fetch tickers for live prices + 24h volume
      3. Fetch 24 hourly candles for volatility calculation
      4. Fetch orderbook for OBI calculation
      5. Filter by volume, balance-fit, notional
      6. Apply composite score ranking
      7. Return top-3 sorted candidates
    """

    def __init__(self, max_pairs: int = 100) -> None:
        self.max_pairs = max_pairs
        self._meme_set = {"PEPEUSDT", "DOGEUSDT", "WIFUSDT", "SHIBUSDT",
                          "FLOKIUSDT", "BONKUSDT", "BABYDOGEUSDT", "MEMEUSDT",
                          "DOGEGOVUSDT"}

    async def scan(
        self,
        exchange: Any,          # BybitLinearConnector instance
        balance: float,         # Current wallet balance
        current_regime: str = "RANGING",
    ) -> ScanResult:
        """Full scan pipeline. Returns ranked candidates."""
        t0 = time.perf_counter()

        # ── Step 1: Fetch all active linear USDT instruments ──────────
        instruments = await asyncio.to_thread(
            exchange.fetch_instruments_info, "linear"
        )
        total_pairs = len(instruments)
        logger.info("[Scanner] Fetched %d total linear USDT pairs", total_pairs)

        # ── Step 2: Fetch live tickers for all pairs ──────────────────
        tickers = {}
        for inst in instruments[:self.max_pairs]:
            sym = inst["symbol"]
            try:
                tkr = await asyncio.to_thread(exchange.fetch_ticker, sym)
                tickers[sym] = tkr
            except Exception:
                continue

        # ── Step 3: Fetch 1h klines & orderbook for top volume pairs ──
        # Sort by volume descending, take top self.max_pairs
        volume_sorted = sorted(
            [(sym, t.get("volume_24h", 0)) for sym, t in tickers.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:self.max_pairs]

        candidates = []
        for sym, vol in volume_sorted:
            try:
                # Volume filter
                if vol < MIN_24H_VOLUME_USD:
                    continue

                tkr = tickers[sym]
                price = float(tkr.get("last_price", 0))
                if price <= 0:
                    continue

                # ── Volatility: fetch 24 hourly klines ────────────────
                df = await asyncio.to_thread(
                    exchange.fetch_klines, sym, "60", 24
                )
                if df is not None and len(df) >= 5:
                    returns = df["close"].pct_change().dropna().values
                    vol_24h = float(np.std(returns)) if len(returns) > 1 else 0.001
                else:
                    vol_24h = 0.001

                # ── OBI scan ───────────────────────────────────────────
                ob = await asyncio.to_thread(
                    exchange.fetch_orderbook, sym, 10
                )
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                b_vol = sum(q for p, q in bids[:5])
                a_vol = sum(q for p, q in asks[:5])
                total = b_vol + a_vol
                obi_raw = (b_vol - a_vol) / total if total > 0 else 0.0

                # ── Build candidate ────────────────────────────────────
                min_qty = float(instruments[0].get("min_qty", 0.001)) if instruments else 0.001
                # Better: find matching instrument
                matched_inst = next(
                    (i for i in instruments if i["symbol"] == sym),
                    None,
                )
                if matched_inst:
                    min_qty = matched_inst["min_qty"]
                    min_notional = matched_inst["min_notional"]
                else:
                    min_notional = 1.0

                is_meme = sym in self._meme_set

                cand = CoinCandidate(
                    symbol=sym,
                    price=price,
                    volume_24h=vol,
                    volatility_24h=vol_24h,
                    obi_raw=round(obi_raw, 4),
                    obi_score=abs(obi_raw),
                    composite_score=0.0,  # computed below
                    min_qty=min_qty,
                    min_notional=min_notional,
                    is_meme=is_meme,
                )

                # Balance fit filter
                if not cand.fits_balance(balance):
                    cand.rejection_reason = "Exceeds balance fraction"
                    continue

                candidates.append(cand)

            except Exception as e:
                logger.debug("[Scanner] Skip %s: %s", sym, e)
                continue

        # ── Step 4: Compute composite scores & rank ───────────────────
        after_filters = len(candidates)

        if not candidates:
            return ScanResult(
                candidates=[],
                total_pairs=total_pairs,
                after_filters=0,
                scan_duration_ms=(time.perf_counter() - t0) * 1000,
            )

        # Z-score normalize volume
        volumes = np.array([c.volume_24h for c in candidates])
        vol_mean, vol_std = volumes.mean(), volumes.std()
        if vol_std == 0:
            vol_std = 1.0

        # Z-score normalize volatility
        vols = np.array([c.volatility_24h for c in candidates])
        v_mean, v_std = vols.mean(), vols.std()
        if v_std == 0:
            v_std = 1.0

        for c in candidates:
            vol_z = (c.volume_24h - vol_mean) / vol_std
            vol_z = max(-3.0, min(3.0, vol_z))  # clip outliers
            vol_z_norm = (vol_z + 3.0) / 6.0     # map to [0, 1]

            vola_z = (c.volatility_24h - v_mean) / v_std
            vola_z = max(-3.0, min(3.0, vola_z))
            vola_z_norm = (vola_z + 3.0) / 6.0

            # Meme penalty
            meme_factor = PENALTY_MEME_COINS if c.is_meme else 1.0

            c.composite_score = round(
                (WEIGHT_VOLATILITY_24H * vola_z_norm +
                 WEIGHT_VOLUME_Z_SCORE * vol_z_norm +
                 WEIGHT_OBI_STRENGTH * c.obi_score)
                * meme_factor,
                4,
            )

        # Sort descending
        candidates.sort(key=lambda c: c.composite_score, reverse=True)

        # Regime-aware top pick
        top = candidates[0] if candidates else None
        if top and current_regime in ("STRONG_TREND", "HIGH_VOLATILITY"):
            # In strong trend, prefer higher volatility pairs
            top = candidates[0]  # already sorted, this is fine
        elif top and current_regime == "RANGING":
            # In ranging market, prefer lower vol / higher volume
            candidates.sort(
                key=lambda c: (c.composite_score * 0.6 + c.volume_24h * 0.4),  # type: ignore
                reverse=True,
            )
            top = candidates[0]

        scan_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "[Scanner] Scan complete: %d → %d filtered → top=%s (score=%.3f) in %.0fms",
            total_pairs, after_filters, top.symbol if top else "NONE",
            top.composite_score if top else 0, scan_ms,
        )

        return ScanResult(
            candidates=candidates[:10],
            total_pairs=total_pairs,
            after_filters=after_filters,
            top_pick=top,
            scan_duration_ms=round(scan_ms, 1),
        )
