#!/usr/bin/env python3
"""
orderbook_imbalance.py
======================
Q-SonicFX — Real-Time Order Book Imbalance (OBI) Engine
========================================================

Connects to any WebSocketFeed, processes each order book update in < 100μs,
and fires a registered callback whenever the directional signal changes or a
LIQUIDITY_VANISH event is detected.

Core signals
------------
    BUY              — bid walls dominate; price likely to rise.
    SELL             — ask walls dominate; price likely to fall.
    NEUTRAL          — no significant imbalance.
    LIQUIDITY_VANISH — sudden depth collapse + sign flip (spoof / iceberg).

OBI formula (per update)
------------------------
    raw_obi = (bid_vol - ask_vol) / (bid_vol + ask_vol)   ∈ [-1, +1]

Threshold mapping
-----------------
    Constructor `threshold` is expressed as a bid/ask RATIO (e.g. 1.5 means
    bids must be 1.5x asks).  Internally this is converted to OBI scale:
        obi_threshold = (ratio - 1) / (ratio + 1)
    so threshold=1.5 → obi_threshold ≈ 0.200
       threshold=1.2 → obi_threshold ≈ 0.091

Author : Q-SonicFX Quant Engine
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Level          = List[float]                  # [price, size]
Book           = List[Level]
SignalCallback = Callable[
    [str, str, float, float, float, float, "DepthSnapshot"], None
]

# Signal string constants — kept as module-level to avoid repeated allocation
SIGNAL_BUY              = "BUY"
SIGNAL_SELL             = "SELL"
SIGNAL_NEUTRAL          = "NEUTRAL"
SIGNAL_LIQUIDITY_VANISH = "LIQUIDITY_VANISH"

_ALL_SIGNALS = (SIGNAL_BUY, SIGNAL_SELL, SIGNAL_NEUTRAL, SIGNAL_LIQUIDITY_VANISH)


# ---------------------------------------------------------------------------
# DepthSnapshot — frozen, passed to signal callbacks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DepthSnapshot:
    """
    Immutable snapshot of top-N bid/ask levels at the moment a signal fires.

    Attributes
    ----------
    symbol             : str    Instrument identifier.
    bids               : tuple  ((price, size), ...) — best bid first.
    asks               : tuple  ((price, size), ...) — best ask first.
    timestamp          : float  UNIX epoch seconds (UTC).
    imbalance          : float  Raw OBI ∈ [-1, +1].
    weighted_imbalance : float  Distance-weighted OBI ∈ [-1, +1].
    """

    symbol             : str
    bids               : Tuple[Tuple[float, float], ...]
    asks               : Tuple[Tuple[float, float], ...]
    timestamp          : float
    imbalance          : float
    weighted_imbalance : float


# ---------------------------------------------------------------------------
# WebSocketFeed — Abstract base class
# ---------------------------------------------------------------------------

class WebSocketFeed(ABC):
    """
    Abstract interface for exchange WebSocket order book feeds.

    All concrete implementations must override the four abstract methods.

    The ``on_message`` callback receives dicts of the form::

        {
            'symbol'    : str,
            'bids'      : [[price: float, size: float], ...],  # best first
            'asks'      : [[price: float, size: float], ...],  # best first
            'timestamp' : float   # UNIX epoch seconds, UTC
        }

    Implementor notes
    -----------------
    - ``connect`` should be idempotent (safe to call more than once).
    - ``run`` is the async event loop; it should run until stopped.
    - Wire ``on_message`` callbacks *before* calling ``run``.
    """

    @abstractmethod
    async def connect(self, url: str, symbols: List[str]) -> None:
        """
        Establish WebSocket connection to the exchange.

        Parameters
        ----------
        url     : str          WebSocket endpoint (e.g. ``wss://stream.exchange/ws``).
        symbols : list of str  Instruments to track (e.g. ``['BTCUSDT']``).
        """

    @abstractmethod
    async def subscribe(self, orderbook_depth: int = 10) -> None:
        """
        Send order book subscription message(s) to the exchange.

        Parameters
        ----------
        orderbook_depth : int
            Number of price levels per side to stream (default 10).
        """

    @abstractmethod
    def on_message(self, callback: Callable[[dict], None]) -> None:
        """
        Register a callback that fires on every order book update.

        Parameters
        ----------
        callback : Callable[[dict], None]
            Function called with the parsed book message dict.
        """

    @abstractmethod
    async def run(self) -> None:
        """Start the async event loop, emitting messages indefinitely."""


# ---------------------------------------------------------------------------
# MockWebSocketFeed — realistic simulated feed for testing
# ---------------------------------------------------------------------------

class MockWebSocketFeed(WebSocketFeed):
    """
    Simulated WebSocket order book feed using geometric Brownian motion.

    Generates realistic bid/ask updates with:
      - Mid-price following a log-normal random walk.
      - Mean-reverting bid/ask spread.
      - Log-normal level sizes (realistic microstructure).
      - Configurable spoof events (sudden large walls, then disappear).

    Parameters
    ----------
    volatility : float  Log-return sigma per tick (default 0.0003 ≈ 3 bps).
    depth      : int    Number of bid/ask levels generated per update (default 10).
    update_hz  : float  Target updates per second per symbol (default 200).
    spoof_prob : float  Probability of a spoof wall event per tick (default 0.02).
    seed       : int    Optional random seed for reproducibility (default None).
    """

    def __init__(
        self,
        volatility : float           = 0.0003,
        depth      : int             = 10,
        update_hz  : float           = 200.0,
        spoof_prob : float           = 0.02,
        seed       : Optional[int]   = None,
    ) -> None:
        self._volatility  = volatility
        self._depth       = depth
        self._update_hz   = update_hz
        self._spoof_prob  = spoof_prob
        self._rng         = random.Random(seed)
        self._callback    : Optional[Callable[[dict], None]] = None
        self._symbols     : List[str] = []
        self._mid_prices  : Dict[str, float] = {}
        self._running     : bool = False
        self._update_count: int = 0

    # -- WebSocketFeed interface ----------------------------------------

    async def connect(self, url: str, symbols: List[str]) -> None:
        """Initialise internal mid-price state. ``url`` is ignored in mock mode."""
        self._symbols = list(symbols)
        # Assign plausible starting mid-prices per symbol
        _defaults = {"BTCUSDT": 65_000.0, "ETHUSDT": 3_500.0, "SOLUSDT": 180.0}
        self._mid_prices = {
            s: _defaults.get(s, self._rng.uniform(100.0, 5_000.0))
            for s in symbols
        }
        logger.info("[MockWSFeed] Connected (mock). symbols=%s", symbols)

    async def subscribe(self, orderbook_depth: int = 10) -> None:
        """No-op in mock — depth is set at construction time."""
        logger.info("[MockWSFeed] Subscribed (mock). depth=%d", orderbook_depth)

    def on_message(self, callback: Callable[[dict], None]) -> None:
        """Register the message callback fired on every synthetic update."""
        self._callback = callback

    async def run(self) -> None:
        """
        Emit synthetic order book updates at ``update_hz`` frequency.

        Stops cleanly when ``stop()`` is called.
        """
        self._running = True
        interval = 1.0 / self._update_hz

        while self._running:
            for symbol in self._symbols:
                if self._callback is not None:
                    msg = self._generate_book(symbol)
                    self._callback(msg)
                    self._update_count += 1
            await asyncio.sleep(interval)

    def stop(self) -> None:
        """Signal the run loop to exit after the current iteration."""
        self._running = False

    # -- Internal -------------------------------------------------------

    def _generate_book(self, symbol: str) -> dict:
        """
        Generate a single realistic order book snapshot for *symbol*.

        Returns
        -------
        dict
            ``{'symbol', 'bids', 'asks', 'timestamp'}``
        """
        rng = self._rng

        # ── Mid-price log-normal random walk ───────────────────────────
        self._mid_prices[symbol] *= math.exp(rng.gauss(0.0, self._volatility))
        mid = self._mid_prices[symbol]

        # ── Spread: uniform jitter around 0.015% of mid ────────────────
        tick = mid * rng.uniform(0.00005, 0.0003)

        # ── Spoof event (random large wall on one side) ─────────────────
        spoof_side  = rng.choice(("bid", "ask")) if rng.random() < self._spoof_prob else None
        spoof_level = rng.randint(0, min(2, self._depth - 1))

        bids: Book = []
        asks: Book = []

        for i in range(self._depth):
            bid_px  = round(mid - tick * (1 + i * 0.5), 6)
            ask_px  = round(mid + tick * (1 + i * 0.5), 6)
            bid_sz  = round(rng.lognormvariate(2.8, 1.1), 4)
            ask_sz  = round(rng.lognormvariate(2.8, 1.1), 4)

            if spoof_side == "bid"  and i == spoof_level:
                bid_sz = round(bid_sz * rng.uniform(8.0, 20.0), 4)
            if spoof_side == "ask"  and i == spoof_level:
                ask_sz = round(ask_sz * rng.uniform(8.0, 20.0), 4)

            bids.append([bid_px, bid_sz])
            asks.append([ask_px, ask_sz])

        return {
            "symbol"   : symbol,
            "bids"     : bids,
            "asks"     : asks,
            "timestamp": time.time(),
        }


# ---------------------------------------------------------------------------
# _SymbolState — per-symbol hot-path state (__slots__ for speed)
# ---------------------------------------------------------------------------

class _SymbolState:
    """
    Mutable per-symbol computation state.

    Uses ``__slots__`` to eliminate ``__dict__`` overhead on frequent attribute
    access in the hot path (message handler runs up to 200+ times / second).
    """

    __slots__ = (
        "bids",
        "asks",
        "imbalance_history",    # deque[float] — last velocity_window OBI values
        "abs_imbalance_buf",    # np.ndarray   — circular buffer for dynamic threshold
        "abs_buf_idx",          # int           — current write index in circular buf
        "abs_buf_filled",       # bool          — True once buffer is fully populated
        "top_depth_history",    # deque[float]  — last 3 top-level depths (liq vanish)
        "last_signal",          # str           — most recently emitted signal
        "prev_imbalance_sign",  # int           — sign of OBI on previous tick
        "update_count",         # int           — total updates received
    )

    def __init__(self, velocity_window: int, dyn_buf_size: int) -> None:
        self.bids                : Book            = []
        self.asks                : Book            = []
        self.imbalance_history   : Deque[float]   = deque(maxlen=velocity_window)
        self.abs_imbalance_buf   : np.ndarray     = np.zeros(dyn_buf_size, dtype=np.float32)
        self.abs_buf_idx         : int            = 0
        self.abs_buf_filled      : bool           = False
        self.top_depth_history   : Deque[float]   = deque(maxlen=3)
        self.last_signal         : str            = SIGNAL_NEUTRAL
        self.prev_imbalance_sign : int            = 0
        self.update_count        : int            = 0


# ---------------------------------------------------------------------------
# OrderBookImbalance — main class
# ---------------------------------------------------------------------------

class OrderBookImbalance:
    """
    Real-time Order Book Imbalance calculator and directional signal emitter.

    Parameters
    ----------
    feed              : WebSocketFeed   Exchange feed (real or mock).
    depth_levels      : int             Top-N levels used for OBI (default 3).
    threshold         : float           Bid/ask volume RATIO to trigger BUY/SELL
                                        (default 1.5 → bids 1.5x asks).
                                        Internally converted to OBI scale:
                                        obi_t = (ratio-1)/(ratio+1).
    symbols           : list of str     Instruments to monitor.
    velocity_window   : int             Updates tracked for velocity (default 5).
    dynamic_threshold : bool            Auto-adjust threshold to 90th percentile of
                                        rolling |OBI| history (default False).
    dyn_buf_size      : int             Rolling window size for dynamic threshold
                                        (default 1000 updates).

    Notes
    -----
    ``__slots__`` is used for all instance attributes to minimise attribute
    lookup overhead.  Heavy allocations (numpy buffers, deques) are performed
    once in ``__init__`` and reused throughout.
    """

    __slots__ = (
        "_feed",
        "_depth_levels",
        "_threshold",         # static bid/ask ratio threshold
        "_obi_threshold",     # static threshold converted to OBI scale
        "_symbols",
        "_velocity_window",
        "_dynamic_threshold",
        "_dyn_buf_size",
        "_weights",           # pre-computed level-decay weight tuple
        "_states",            # Dict[str, _SymbolState]
        "_signal_callback",
        "_stats",
    )

    def __init__(
        self,
        feed              : WebSocketFeed,
        depth_levels      : int            = 3,
        threshold         : float          = 1.5,
        symbols           : Optional[List[str]] = None,
        velocity_window   : int            = 5,
        dynamic_threshold : bool           = False,
        dyn_buf_size      : int            = 1000,
    ) -> None:
        if symbols is None:
            symbols = ["BTCUSDT"]
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        if depth_levels < 1:
            raise ValueError(f"depth_levels must be >= 1, got {depth_levels}")

        self._feed              = feed
        self._depth_levels      = depth_levels
        self._threshold         = threshold
        # Convert ratio → OBI scale: obi_t = (r-1)/(r+1)
        self._obi_threshold     = (threshold - 1.0) / (threshold + 1.0)
        self._symbols           = list(symbols)
        self._velocity_window   = velocity_window
        self._dynamic_threshold = dynamic_threshold
        self._dyn_buf_size      = dyn_buf_size

        # Pre-compute level-decay weights once (1/(1+i) for i=0..N-1)
        self._weights: Tuple[float, ...] = tuple(
            1.0 / (1.0 + i) for i in range(depth_levels)
        )

        # Per-symbol state — pre-allocated
        self._states: Dict[str, _SymbolState] = {
            sym: _SymbolState(velocity_window, dyn_buf_size)
            for sym in self._symbols
        }

        self._signal_callback: Optional[SignalCallback] = None

        # Signal distribution counters
        self._stats: Dict[str, Dict[str, int]] = {
            sym: {s: 0 for s in _ALL_SIGNALS}
            for sym in self._symbols
        }

    # ── Public API ────────────────────────────────────────────────────

    def register_signal_callback(self, callback: SignalCallback) -> None:
        """
        Register a user callback invoked on signal change or LIQUIDITY_VANISH.

        Callback signature::

            def on_signal(
                symbol              : str,
                signal              : str,           # BUY/SELL/NEUTRAL/LIQUIDITY_VANISH
                imbalance_value     : float,         # raw OBI ∈ [-1, +1]
                weighted_imbalance  : float,         # distance-weighted OBI ∈ [-1, +1]
                imbalance_velocity  : float,         # rate of change over velocity_window
                timestamp           : float,         # UNIX epoch seconds
                depth_snapshot      : DepthSnapshot, # frozen top-N snapshot
            ) -> None: ...

        Parameters
        ----------
        callback : SignalCallback
        """
        self._signal_callback = callback

    async def start(self, url: str = "wss://mock.exchange/ws") -> None:
        """
        Connect to the feed, subscribe, and start processing updates.

        This coroutine runs until the feed's ``run()`` method returns (or is
        cancelled via ``asyncio.CancelledError``).

        Parameters
        ----------
        url : str  Exchange WebSocket URL.
        """
        await self._feed.connect(url, self._symbols)
        await self._feed.subscribe(orderbook_depth=self._depth_levels + 5)
        self._feed.on_message(self._on_book_update)
        logger.info(
            "[OBI] Engine started | symbols=%s depth=%d ratio_threshold=%.3f "
            "obi_threshold=%.4f dynamic=%s velocity_window=%d",
            self._symbols, self._depth_levels, self._threshold,
            self._obi_threshold, self._dynamic_threshold, self._velocity_window,
        )
        await self._feed.run()

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Return signal distribution counts per symbol.

        Returns
        -------
        dict
            ``{symbol: {signal_type: count, ...}, ...}``
        """
        return {sym: dict(counts) for sym, counts in self._stats.items()}

    def get_latest_obi(self, symbol: str) -> Optional[float]:
        """
        Return the most recent raw OBI value for *symbol*, or None if not yet computed.

        Parameters
        ----------
        symbol : str

        Returns
        -------
        float or None
        """
        state = self._states.get(symbol)
        if state is None or len(state.imbalance_history) == 0:
            return None
        return state.imbalance_history[-1]

    # ── Hot path — message handler (target: < 100μs) ─────────────────

    def _on_book_update(self, msg: dict) -> None:
        """
        Core hot-path handler — called synchronously on every book update.

        Pipeline
        --------
        1.  Trim to depth_levels.
        2.  Compute raw OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol).
        3.  Compute distance-weighted OBI (level-decay weights 1/(1+i)).
        4.  Append OBI to velocity history; compute imbalance_velocity.
        5.  Update dynamic threshold circular buffer.
        6.  Classify signal using effective threshold.
        7.  Check LIQUIDITY_VANISH (>50% top-depth drop in 3 ticks + sign flip).
        8.  Build DepthSnapshot and fire callback on signal change / LV event.

        Parameters
        ----------
        msg : dict
            ``{'symbol', 'bids', 'asks', 'timestamp'}``
        """
        symbol    : str   = msg["symbol"]
        timestamp : float = msg["timestamp"]

        state = self._states.get(symbol)
        if state is None:
            return   # untracked symbol — fast exit

        raw_bids : Book = msg["bids"]
        raw_asks : Book = msg["asks"]
        N = self._depth_levels

        # ── 1. Trim to depth_levels ────────────────────────────────────
        top_bids = raw_bids[:N]
        top_asks = raw_asks[:N]

        # ── 2. Raw OBI ─────────────────────────────────────────────────
        # (bid_vol - ask_vol) / (bid_vol + ask_vol)  ∈ [-1, +1]
        bid_vol = 0.0
        ask_vol = 0.0
        for lvl in top_bids:
            bid_vol += lvl[1]
        for lvl in top_asks:
            ask_vol += lvl[1]

        total_vol = bid_vol + ask_vol
        if total_vol == 0.0:
            return   # degenerate book — skip without state mutation

        raw_obi: float = (bid_vol - ask_vol) / total_vol

        # ── 3. Weighted OBI ────────────────────────────────────────────
        # Each level i gets weight 1/(1+i): level 0→1.0, 1→0.5, 2→0.333…
        w_bid = 0.0
        w_ask = 0.0
        weights = self._weights
        for i in range(min(N, len(top_bids))):
            w_bid += top_bids[i][1] * weights[i]
        for i in range(min(N, len(top_asks))):
            w_ask += top_asks[i][1] * weights[i]
        w_total = w_bid + w_ask
        weighted_obi: float = (w_bid - w_ask) / w_total if w_total > 0.0 else 0.0

        # ── 4. Imbalance velocity ──────────────────────────────────────
        # Rate of change of OBI over the last velocity_window updates.
        # velocity = (OBI_now - OBI_oldest) / window_size
        # A large positive velocity from a negative OBI → wall being eaten (reversal signal).
        hist = state.imbalance_history
        hist.append(raw_obi)
        n_hist = len(hist)
        velocity: float = (hist[-1] - hist[0]) / n_hist if n_hist >= 2 else 0.0

        # ── 5. Dynamic threshold update ────────────────────────────────
        buf = state.abs_imbalance_buf
        idx = state.abs_buf_idx
        buf[idx] = abs(raw_obi)
        next_idx = idx + 1
        if next_idx >= self._dyn_buf_size:
            next_idx = 0
            state.abs_buf_filled = True
        state.abs_buf_idx = next_idx

        effective_threshold = self._obi_threshold
        if self._dynamic_threshold:
            filled_buf = buf if state.abs_buf_filled else buf[:idx + 1]
            if len(filled_buf) >= 10:
                dyn_t = float(np.percentile(filled_buf, 90))
                effective_threshold = max(dyn_t, 1e-6)   # guard zero

        # ── 6. Signal classification ───────────────────────────────────
        if raw_obi > effective_threshold:
            signal = SIGNAL_BUY
        elif raw_obi < -effective_threshold:
            signal = SIGNAL_SELL
        else:
            signal = SIGNAL_NEUTRAL

        # ── 7. LIQUIDITY_VANISH detection ──────────────────────────────
        # Condition: top-level depth drops > 50% across 3 consecutive
        # updates AND imbalance flips sign  →  wall was fake (spoof/iceberg).
        top_depth = 0.0
        if top_bids and top_asks:
            top_depth = top_bids[0][1] + top_asks[0][1]
        state.top_depth_history.append(top_depth)

        current_sign = 1 if raw_obi > 0.0 else (-1 if raw_obi < 0.0 else 0)
        liq_vanish = False

        if len(state.top_depth_history) == 3:
            d0, _d1, d2 = state.top_depth_history
            if (
                d0 > 0.0
                and d2 < 0.5 * d0                             # >50% depth drop
                and current_sign != 0
                and current_sign != state.prev_imbalance_sign # sign flipped
                and state.prev_imbalance_sign != 0
            ):
                liq_vanish = True

        state.prev_imbalance_sign = current_sign
        state.update_count += 1

        # ── 8. Emit on signal change or LIQUIDITY_VANISH ───────────────
        signal_changed = (signal != state.last_signal)
        should_emit    = liq_vanish or signal_changed

        if not should_emit:
            return   # nothing to report — fast exit

        # Build frozen snapshot only when emitting (avoids heap allocation on quiet ticks)
        snapshot = DepthSnapshot(
            symbol             = symbol,
            bids               = tuple(tuple(lvl) for lvl in top_bids),
            asks               = tuple(tuple(lvl) for lvl in top_asks),
            timestamp          = timestamp,
            imbalance          = raw_obi,
            weighted_imbalance = weighted_obi,
        )

        if liq_vanish:
            emit_signal  = SIGNAL_LIQUIDITY_VANISH
            lv_direction = SIGNAL_BUY if current_sign > 0 else SIGNAL_SELL
            logger.warning(
                "[OBI] LIQUIDITY_VANISH | %s | depth_drop=%.1f%% sign_flip=%s→%s "
                "OBI=%.4f wOBI=%.4f",
                symbol,
                (1.0 - (d2 / d0)) * 100.0 if d0 > 0 else 0.0,
                "+" if state.prev_imbalance_sign > 0 else "-",
                "+" if current_sign > 0 else "-",
                raw_obi, weighted_obi,
            )
            self._stats[symbol][SIGNAL_LIQUIDITY_VANISH] += 1
        else:
            emit_signal = signal
            logger.info(
                "[OBI] Signal: %s→%s | %s | OBI=%.4f wOBI=%.4f vel=%.4f thr=%.4f",
                state.last_signal, signal, symbol,
                raw_obi, weighted_obi, velocity, effective_threshold,
            )
            self._stats[symbol][signal] += 1
            state.last_signal = signal

        if self._signal_callback is not None:
            self._signal_callback(
                symbol,
                emit_signal,
                raw_obi,
                weighted_obi,
                velocity,
                timestamp,
                snapshot,
            )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

async def _run_demo(n_updates: int, threshold: float) -> None:
    """
    Full end-to-end demo coroutine.

    Parameters
    ----------
    n_updates : int    Number of order book updates to simulate.
    threshold : float  Bid/ask ratio threshold for OBI signals.
    """
    # ── Signal tracking ────────────────────────────────────────────────
    signals_received: List[Tuple[str, str, float, float, float]] = []

    def on_signal(
        symbol              : str,
        signal              : str,
        imbalance_value     : float,
        weighted_imbalance  : float,
        imbalance_velocity  : float,
        timestamp           : float,
        depth_snapshot      : DepthSnapshot,
    ) -> None:
        signals_received.append(
            (symbol, signal, imbalance_value, weighted_imbalance, imbalance_velocity)
        )
        direction_tag = ""
        if signal == SIGNAL_LIQUIDITY_VANISH:
            direction_tag = f" [!!] SPOOF/ICEBERG DETECTED on {symbol}"
        print(
            f"[SIGNAL] {signal:<20} | {symbol} | "
            f"OBI={imbalance_value:+.4f}  wOBI={weighted_imbalance:+.4f}  "
            f"vel={imbalance_velocity:+.4f}{direction_tag}"
        )

    # ── Set up feed and engine ─────────────────────────────────────────
    feed = MockWebSocketFeed(
        volatility = 0.0004,
        depth      = 10,
        update_hz  = 10_000,   # fast for test — will be capped by asyncio.sleep(0)
        spoof_prob = 0.03,
        seed       = 99,
    )

    obi = OrderBookImbalance(
        feed              = feed,
        depth_levels      = 3,
        threshold         = threshold,
        symbols           = ["BTCUSDT"],
        velocity_window   = 5,
        dynamic_threshold = False,
        dyn_buf_size      = 1000,
    )
    obi.register_signal_callback(on_signal)

    # ── Patch feed.run to stop after n_updates ─────────────────────────
    update_counter = {"n": 0}
    original_callback: Optional[Callable] = None

    def counting_callback(msg: dict) -> None:
        update_counter["n"] += 1
        if original_callback is not None:
            original_callback(msg)

    await feed.connect("wss://mock.exchange/ws", ["BTCUSDT"])
    await feed.subscribe(orderbook_depth=8)
    feed.on_message(obi._on_book_update)
    original_callback = obi._on_book_update

    # Override with counting wrapper
    feed.on_message(counting_callback)

    # ── Measure latency over n_updates ────────────────────────────────
    import time as _time

    latencies_ns: List[float] = []
    rng = random.Random(99)
    state_for_latency = obi._states["BTCUSDT"]

    print(f"\n{'='*70}")
    print(f"  Q-SonicFX  |  Order Book Imbalance Engine — Demo")
    print(f"  threshold={threshold} (ratio) | depth=3 | symbols=['BTCUSDT']")
    print(f"{'='*70}\n")

    for _ in range(n_updates):
        msg = feed._generate_book("BTCUSDT")
        t0 = _time.perf_counter_ns()
        obi._on_book_update(msg)
        latencies_ns.append(_time.perf_counter_ns() - t0)

    # ── Summary stats ──────────────────────────────────────────────────
    stats = obi.get_stats()["BTCUSDT"]
    lv_count = stats[SIGNAL_LIQUIDITY_VANISH]
    lat_arr  = np.array(latencies_ns, dtype=np.float64)

    print(f"\n{'='*70}")
    print(f"  SUMMARY  ({n_updates} updates)")
    print(f"{'='*70}")
    print(f"  Signal distribution:")
    for sig in (SIGNAL_BUY, SIGNAL_SELL, SIGNAL_NEUTRAL, SIGNAL_LIQUIDITY_VANISH):
        bar = "|" * min(stats[sig] // 5, 40)
        print(f"    {sig:<22} {stats[sig]:>5}  {bar}")
    print(f"\n  LIQUIDITY_VANISH events : {lv_count}")
    print(f"\n  Latency (per message handler):")
    print(f"    Mean    : {lat_arr.mean():.1f} ns  ({lat_arr.mean()/1000:.2f} us)")
    print(f"    Median  : {np.median(lat_arr):.1f} ns")
    print(f"    P99     : {np.percentile(lat_arr, 99):.1f} ns")
    print(f"    Max     : {lat_arr.max():.1f} ns")
    target_ns = 100_000   # 100 us
    pct_under = float(np.sum(lat_arr < target_ns) / len(lat_arr) * 100)
    print(f"    < 100us : {pct_under:.1f}% of updates")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level  = logging.WARNING,   # suppress INFO spam in demo; change to INFO to see all signals
        format = "[%(levelname)s] %(name)s: %(message)s",
    )

    # Override: show signal-change logs at WARNING level for readability
    logging.getLogger(__name__).setLevel(logging.WARNING)

    n_updates = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 1.2

    asyncio.run(_run_demo(n_updates=n_updates, threshold=threshold))
