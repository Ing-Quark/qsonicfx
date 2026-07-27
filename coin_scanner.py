#!/usr/bin/env python3
"""
coin_scanner.py
===============
Q-SonicFX Multi-Coin Auto-Scanner & Ranker
==========================================

Fast, bulk-scanning engine for Bybit USDT linear perpetual pairs.
Uses Bybit V5 bulk tickers endpoint (1 HTTP call) to score and rank
all active instruments in under 1 second.

Author : Q-SonicFX Quant Engine
"""

from __future__ import annotations
import logging
import time
import math
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger("qsonicfx.coin_scanner")

# ---------------------------------------------------------------------------
# Meme / micro-cap symbols (different risk params)
# ---------------------------------------------------------------------------
MEME_SYMBOLS = {
    "PEPE", "FLOKI", "SHIB", "BONK", "WIF", "DOGE",
    "PENGU", "TURBO", "NEIRO", "MOODENG", "BABYDOGE",
    "POPCAT", "MYRO", "BOME", "BRETT", "MOG", 1000
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
        min_cost = self.min_notional if self.min_notional > 0 else 1.0
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
    Scans Bybit USDT perpetual pairs in bulk, scores them by composite signal strength,
    and returns a ScanResult with the best balance-fitting candidate.
    """

    def __init__(
        self,
        max_pairs      : int   = 50,
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
        Run a fast bulk scan cycle.
        """
        t0 = time.perf_counter()

        # ── 1. Fetch instrument specifications ───────────────────────────
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

        inst_map = {inst["symbol"]: inst for inst in instruments}

        # ── 2. Bulk fetch all tickers in 1 HTTP call ─────────────────────
        ticker_map: Dict[str, Dict[str, float]] = {}
        try:
            if hasattr(exchange, "_request"):
                res = await asyncio.to_thread(
                    exchange._request, "GET", "/v5/market/tickers", {"category": "linear"}
                )
                raw_tickers = res.get("result", {}).get("list", [])
                for t in raw_tickers:
                    sym = t.get("symbol", "")
                    if sym in inst_map:
                        p = float(t.get("lastPrice", 0.0) or 0.0)
                        v_usd = float(t.get("turnover24h", 0.0) or 0.0)
                        if v_usd <= 0:
                            v_base = float(t.get("volume24h", 0.0) or 0.0)
                            v_usd = v_base * p
                        b1 = float(t.get("bid1Price", 0.0) or 0.0)
                        a1 = float(t.get("ask1Price", 0.0) or 0.0)
                        ticker_map[sym] = {
                            "price": p,
                            "volume_24h": v_usd,
                            "bid1": b1,
                            "ask1": a1,
                            "price24hPcnt": float(t.get("price24hPcnt", 0.0) or 0.0),
                        }
        except Exception as exc:
            logger.warning("[Scanner] Bulk tickers fetch failed: %s", exc)

        # ── 3. Score candidates ─────────────────────────────────────────
        candidates: List[CoinCandidate] = []

        for sym, inst in inst_map.items():
            t_info = ticker_map.get(sym)
            if not t_info:
                continue

            price   = t_info["price"]
            vol_usd = t_info["volume_24h"]

            if vol_usd < self.min_volume_usd or price <= 0:
                continue

            # Quick OBI from top-level bid/ask spread
            bid1 = t_info["bid1"]
            ask1 = t_info["ask1"]
            b_pct = t_info["price24hPcnt"]

            vol_est   = abs(b_pct) + 0.001
            liq_score = min(1.0, vol_usd / 5_000_000.0)
            is_meme   = any(m in sym.upper() for m in (MEME_SYMBOLS if isinstance(m, str) else str(m) for m in MEME_SYMBOLS))

            # Composite score
            score = liq_score * 3.0 + vol_est * 4.0

            # Boost micro-price meme/alt coins for small balance friendliness
            if price < 1.0:
                score += 2.0
            if price < 0.01:
                score += 3.0
            if price < 0.0001:
                score += 2.0

            obi_est = 0.1 if b_pct > 0 else -0.1

            candidate = CoinCandidate(
                symbol          = sym,
                price           = price,
                volume_24h      = vol_usd,
                volatility_24h  = vol_est,
                liquidity_score = liq_score,
                obi_raw         = obi_est,
                obi_weighted    = obi_est * 0.9,
                composite_score = round(score, 4),
                min_qty         = inst.get("min_qty", 0.001),
                min_notional    = inst.get("min_notional", 1.0),
                qty_step        = inst.get("qty_step", 0.001),
                tick_size       = inst.get("tick_size", 0.01),
                is_meme         = is_meme,
            )
            candidates.append(candidate)

        # ── 4. Sort candidates by composite score ─────────────────────────
        candidates.sort(key=lambda c: c.composite_score, reverse=True)

        # ── 5. Pick best balance-fitting coin ─────────────────────────────
        top: Optional[CoinCandidate] = None
        for c in candidates:
            if c.fits_balance(balance):
                top = c
                break

        if top is None and candidates:
            top = candidates[0]

        duration_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "[Scanner] Fast bulk scan complete | %d coins evaluated | top=%s | %.0fms",
            len(candidates),
            top.symbol if top else "NONE",
            duration_ms,
        )

        return ScanResult(
            candidates    = candidates[:15],
            top_pick      = top,
            total_scanned = len(candidates),
            duration_ms   = duration_ms,
        )


if __name__ == "__main__":
    import asyncio, os
    async def _test():
        from exchange_connector import get_exchange_client
        client = get_exchange_client("BYBIT_LIVE")
        scanner = CoinScanner()
        res = await scanner.scan(client, balance=1.20)
        print(f"Top pick: {res.top_pick.symbol if res.top_pick else 'None'}")
        print(f"Scan duration: {res.duration_ms:.1f}ms")
    asyncio.run(_test())
