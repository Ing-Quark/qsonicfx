#!/usr/bin/env python3
"""
coin_scanner.py
===============
Q-SonicFX Multi-Coin Auto-Scanner & Ranker
==========================================

Scans all active USDT linear perpetual pairs on Bybit, scores them by
composite signal strength (OBI magnitude + volume + volatility), and
returns a ranked list of tradeable candidates that fit the current balance.

Author : Q-SonicFX Quant Engine
"""

from __future__ import annotations
import logging
import time
import math
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("qsonicfx.coin_scanner")

# ---------------------------------------------------------------------------
# Meme / micro-cap symbols (different risk params)
# ---------------------------------------------------------------------------
MEME_SYMBOLS = {
    "PEPE", "FLOKI", "SHIB", "BONK", "WIF", "DOGE",
    "PENGU", "TURBO", "NEIRO", "MOODENG", "BABYDOGE",
    "POPCAT", "MYRO", "BOME", "BRETT", "MOG",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CoinCandidate:
    """A single ranked coin candidate from the scanner."""
    symbol          : str
    price           : float
    volume_24h      : float
    volatility_24h  : float
    liquidity_score : float
    obi_raw         : float
    obi_weighted    : float
    composite_score : float
    min_qty         : float
    min_notional    : float
    qty_step        : float
    tick_size       : float
    max_leverage    : float = 10.0
    is_meme         : bool  = False

    def fits_balance(self, balance: float, leverage: float = 10.0) -> bool:
        """Return True if this coin can be entered with the given balance + leverage."""
        purchasing_power = balance * leverage * 0.95
        min_cost = self.min_notional if self.min_notional > 0 else 5.0
        return purchasing_power >= min_cost and self.price > 0


@dataclass
class ScanResult:
    """Output from a full scan cycle."""
    candidates    : List[CoinCandidate]
    top_pick      : Optional[CoinCandidate] = None
    total_scanned : int   = 0
    duration_ms   : float = 0.0


# ---------------------------------------------------------------------------
# CoinScanner
# ---------------------------------------------------------------------------

class CoinScanner:
    """
    Scans Bybit USDT perpetual pairs, scores them by a composite signal,
    and returns a ScanResult with the best balance-fitting candidate.

    Parameters
    ----------
    max_pairs       : int    Maximum number of instruments to scan (default 100).
    min_volume_usd  : float  Minimum 24h volume in USD to consider (default 10 000).
    """

    def __init__(
        self,
        max_pairs      : int   = 100,
        min_volume_usd : float = 10_000.0,
    ) -> None:
        self.max_pairs      = max_pairs
        self.min_volume_usd = min_volume_usd

    async def scan(
        self,
        exchange       : Any,
        balance        : float,
        current_regime : str = "RANGING",
    ) -> ScanResult:
        """
        Run a full scan cycle.

        Parameters
        ----------
        exchange        : BaseExchangeClient  Live or simulated connector.
        balance         : float               Current USDT balance.
        current_regime  : str                 Current market regime string.

        Returns
        -------
        ScanResult
        """
        t0 = time.perf_counter()

        # ── 1. Fetch all active instruments ──────────────────────────────
        try:
            if hasattr(exchange, "fetch_instruments_info"):
                instruments = await asyncio.to_thread(
                    exchange.fetch_instruments_info, "linear"
                )
            else:
                instruments = []
        except Exception as exc:
            logger.warning("[Scanner] fetch_instruments_info failed: %s", exc)
            instruments = []

        if not instruments:
            logger.warning("[Scanner] No instruments returned — aborting scan.")
            return ScanResult(candidates=[], total_scanned=0, duration_ms=0.0)

        # ── 2. Score each coin ────────────────────────────────────────────
        candidates: List[CoinCandidate] = []

        for inst in instruments[: self.max_pairs]:
            sym = inst.get("symbol", "")
            if not sym:
                continue

            # ── Fetch ticker ──────────────────────────────────────────────
            try:
                ticker    = await asyncio.to_thread(exchange.fetch_ticker, sym)
                price     = float(ticker.get("last_price", 0.0) or 0.0)
                vol_base  = float(ticker.get("volume_24h", 0.0) or 0.0)
                vol_usd   = vol_base * price
            except Exception:
                continue

            if vol_usd < self.min_volume_usd or price <= 0:
                continue

            # ── Fetch orderbook for OBI ───────────────────────────────────
            try:
                ob       = await asyncio.to_thread(exchange.fetch_orderbook, sym, 10)
                bids     = ob.get("bids", [])
                asks     = ob.get("asks", [])

                bv  = sum(q for _, q in bids[:3])
                av  = sum(q for _, q in asks[:3])
                tot = bv + av
                obi_raw = (bv - av) / tot if tot > 0 else 0.0

                w_bv  = sum(q * (1.0 / (1 + i)) for i, (_, q) in enumerate(bids[:3]))
                w_av  = sum(q * (1.0 / (1 + i)) for i, (_, q) in enumerate(asks[:3]))
                w_tot = w_bv + w_av
                obi_w = (w_bv - w_av) / w_tot if w_tot > 0 else 0.0
            except Exception:
                obi_raw = obi_w = 0.0

            # ── Derived metrics ───────────────────────────────────────────
            vol_est   = abs(obi_raw) * 0.5 + 0.001
            liq_score = min(1.0, vol_usd / 1_000_000.0)
            is_meme   = any(m in sym.upper() for m in MEME_SYMBOLS)

            # ── Composite score (regime-sensitive) ────────────────────────
            if current_regime == "STRONG_TREND":
                score = abs(obi_raw) * 3.0 + liq_score * 2.0 + vol_est * 5.0
            else:
                score = abs(obi_raw) * 2.0 + liq_score * 1.5 + vol_est * 3.0

            # Boost for micro-price coins (small capital friendly)
            if price < 1.0:
                score += 2.0
            if price < 0.01:
                score += 3.0
            if price < 0.0001:
                score += 2.0

            candidate = CoinCandidate(
                symbol          = sym,
                price           = price,
                volume_24h      = vol_usd,
                volatility_24h  = vol_est,
                liquidity_score = liq_score,
                obi_raw         = obi_raw,
                obi_weighted    = obi_w,
                composite_score = score,
                min_qty         = float(inst.get("min_qty",      0.001)),
                min_notional    = float(inst.get("min_notional",  1.0)),
                qty_step        = float(inst.get("qty_step",      0.001)),
                tick_size       = float(inst.get("tick_size",     0.01)),
                is_meme         = is_meme,
            )
            candidates.append(candidate)

        # ── 3. Sort by composite score ────────────────────────────────────
        candidates.sort(key=lambda c: c.composite_score, reverse=True)

        # ── 4. Pick best balance-fitting coin ─────────────────────────────
        top: Optional[CoinCandidate] = None
        for c in candidates:
            if c.fits_balance(balance):
                top = c
                break

        # If none fit balance, fall back to overall top
        if top is None and candidates:
            top = candidates[0]
            logger.warning(
                "[Scanner] No coin fits balance $%.4f — using top score pick: %s",
                balance, top.symbol,
            )

        duration_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "[Scanner] Scan complete | %d scanned | top=%s | %.0fms",
            len(candidates),
            top.symbol if top else "NONE",
            duration_ms,
        )

        return ScanResult(
            candidates    = candidates[:10],
            top_pick      = top,
            total_scanned = len(candidates),
            duration_ms   = duration_ms,
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def _test() -> None:
        from exchange_connector import get_exchange_client
        import os
        client = get_exchange_client(
            mode       = "BYBIT_LIVE",
            api_key    = os.getenv("BYBIT_API_KEY", ""),
            secret_key = os.getenv("BYBIT_SECRET_KEY", ""),
        )
        scanner = CoinScanner(max_pairs=20, min_volume_usd=50_000)
        result  = await scanner.scan(client, balance=1.20, current_regime="RANGING")
        print(f"Top pick : {result.top_pick.symbol if result.top_pick else 'NONE'}")
        print(f"Scanned  : {result.total_scanned}")
        print(f"Duration : {result.duration_ms:.0f} ms")
        for i, c in enumerate(result.candidates[:5], 1):
            print(f"  #{i} {c.symbol:<15} score={c.composite_score:.3f} "
                  f"obi={c.obi_raw:+.4f} price={c.price:.6f}")

    asyncio.run(_test())
