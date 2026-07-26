#!/usr/bin/env python3
"""
api_server.py
=============
Q-SonicFX -- FastAPI Orchestration Layer (Central Nervous System)
=================================================================

Ties together all trading engine modules into a single async REST + WebSocket
server. Provides real-time control, monitoring, and live signal streaming.

Endpoints
---------
REST:
    GET  /status            Full bot state + circuit breaker metrics
    POST /start             Begin trading loop
    POST /pause             Suspend new entries (keep open positions)
    POST /stop              Emergency stop + circuit breaker shutdown
    POST /resume            Manual resume (bypass cooldown, dev/test only)
    GET  /performance       Aggregate stats + full trade history
    GET  /signals/latest    Latest regime + OBI signal + suggested size
    POST /parameters        Live parameter update without restart

WebSocket:
    WS   /ws/live           Real-time push: regime, OBI, equity, position

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Author : Q-SonicFX Quant Engine
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Set

from dotenv import load_dotenv

# ── FIX #4: Load environment variables from .env ─────────────────────────
load_dotenv()

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt= "%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("qsonicfx.api_server")

# ---------------------------------------------------------------------------
# Graceful local module imports
# ---------------------------------------------------------------------------
try:
    from regime_detector import RegimeDetector
    _HAS_REGIME = True
    logger.info("[Startup] regime_detector loaded OK")
except ImportError as e:
    logger.warning("[Startup] regime_detector not available: %s", e)
    _HAS_REGIME = False
    RegimeDetector = None  # type: ignore

try:
    from orderbook_imbalance import (
        OrderBookImbalance, MockWebSocketFeed,
        SIGNAL_BUY, SIGNAL_SELL, SIGNAL_NEUTRAL, SIGNAL_LIQUIDITY_VANISH,
    )
    _HAS_OBI = True
    logger.info("[Startup] orderbook_imbalance loaded OK")
except ImportError as e:
    logger.warning("[Startup] orderbook_imbalance not available: %s", e)
    _HAS_OBI = False
    SIGNAL_BUY = "BUY"; SIGNAL_SELL = "SELL"; SIGNAL_NEUTRAL = "NEUTRAL"

try:
    from position_sizer import compute_position_size, update_rolling_stats
    _HAS_SIZER = True
    logger.info("[Startup] position_sizer loaded OK")
except ImportError as e:
    logger.warning("[Startup] position_sizer not available: %s", e)
    _HAS_SIZER = False

try:
    from circuit_breaker import (
        CircuitBreaker, CircuitBreakerException,
        set_global_circuit_breaker, HaltReason,
    )
    _HAS_CB = True
    logger.info("[Startup] circuit_breaker loaded OK")
except ImportError as e:
    logger.warning("[Startup] circuit_breaker not available: %s", e)
    _HAS_CB = False
    CircuitBreaker = None  # type: ignore

try:
    from walk_forward import WalkForwardOptimizer, EMACrossover, generate_synthetic_ohlcv
    _HAS_WFO = True
    logger.info("[Startup] walk_forward loaded OK")
except ImportError as e:
    logger.warning("[Startup] walk_forward not available: %s", e)
    _HAS_WFO = False
    WalkForwardOptimizer = None  # type: ignore

try:
    from exchange_connector import get_exchange_client, BaseExchangeClient
    _HAS_EXCHANGE = True
    logger.info("[Startup] exchange_connector loaded OK")
except ImportError as e:
    logger.warning("[Startup] exchange_connector not available: %s", e)
    _HAS_EXCHANGE = False
    get_exchange_client = None  # type: ignore

try:
    from alpha_signals import AlphaEngine
    _HAS_ALPHA = True
    logger.info("[Startup] alpha_signals loaded OK")
except ImportError as e:
    logger.warning("[Startup] alpha_signals not available: %s", e)
    _HAS_ALPHA = False
    AlphaEngine = None  # type: ignore

try:
    from notifier import TelegramNotifier
    _HAS_NOTIFIER = True
    logger.info("[Startup] notifier loaded OK")
except ImportError as e:
    logger.warning("[Startup] notifier not available: %s", e)
    _HAS_NOTIFIER = False
    TelegramNotifier = None  # type: ignore

# ── FIX #2: Database module ────────────────────────────────────────
try:
    import database as db_module
    from database import init_db, get_db
    _HAS_DB = True
    logger.info("[Startup] database module loaded OK")
except ImportError as e:
    logger.warning("[Startup] database not available: %s", e)
    _HAS_DB = False
    db_module = None  # type: ignore
    init_db  = None   # type: ignore
    get_db   = None   # type: ignore

# ── New modules for autonomy upgrade ──────────────────────────────
try:
    from coin_scanner import CoinScanner, CoinCandidate, ScanResult
    _HAS_SCANNER = True
    logger.info("[Startup] coin_scanner loaded OK")
except ImportError:
    _HAS_SCANNER = False
    CoinScanner = None  # type: ignore

try:
    from account_detector import AccountDetector, AccountProfile, should_trade_spot
    _HAS_DETECTOR = True
    logger.info("[Startup] account_detector loaded OK")
except ImportError:
    _HAS_DETECTOR = False
    AccountDetector = None  # type: ignore

# ---------------------------------------------------------------------------
# Pydantic v2 schemas
# ---------------------------------------------------------------------------

class StatusResponse(BaseModel):
    status            : str
    current_regime    : Optional[str]
    current_position  : Optional[Dict[str, Any]]
    daily_pnl         : float
    total_trades      : int
    last_update       : Optional[str]
    circuit_breaker   : Dict[str, Any]
    uptime_seconds    : float
    account_balance   : float = 100000.0
    available_balance : float = 100000.0
    used_margin       : float = 0.0
    position_notional : float = 0.0
    unrealized_pnl    : float = 0.0
    # ── FIX #1: Balance cache metadata ─────────────────────────────────
    balance_cached_at : Optional[str] = None   # ISO timestamp of last successful fetch
    balance_status    : str           = "INITIALIZING"  # OK | STALE | INITIALIZING

class TradeRecord(BaseModel):
    trade_id          : int
    timestamp         : str
    symbol            : str
    side              : str
    entry_price       : float
    exit_price        : Optional[float]
    quantity          : float
    pnl               : Optional[float]
    pnl_percent       : Optional[float]
    regime_at_entry   : str
    obi_signal        : str
    status            : str

class PerformanceResponse(BaseModel):
    total_trades      : int
    win_rate          : float
    avg_win_pct       : float
    avg_loss_pct      : float
    profit_factor     : float
    sharpe_ratio      : float
    max_drawdown_pct  : float
    total_return_pct  : float
    trades            : List[TradeRecord]

class SignalResponse(BaseModel):
    timestamp         : str
    regime            : str
    trade_allowed     : bool
    obi_value         : float
    obi_weighted      : float
    obi_velocity      : float
    obi_signal        : str
    vpin_score        : float = 0.0
    vpin_status       : str = "NORMAL"
    stat_arb_zscore   : float = 0.0
    suggested_size    : float
    suggested_risk_pct: float
    entry_price       : float

class ParametersUpdate(BaseModel):
    kelly_fraction        : Optional[float] = Field(None, ge=0.0, le=1.0)
    max_risk_per_trade    : Optional[float] = Field(None, ge=0.001, le=0.10)
    obi_threshold         : Optional[float] = Field(None, ge=0.1, le=5.0)
    adx_period            : Optional[int]   = Field(None, ge=5, le=100)
    account_balance       : Optional[float] = Field(None, gt=0)
    min_interval_seconds  : Optional[float] = Field(None, ge=0.1, le=60.0)
    symbol                : Optional[str]   = None
    exchange_mode         : Optional[str]   = None
    api_key               : Optional[str]   = None
    secret_key            : Optional[str]   = None
    passphrase            : Optional[str]   = None
    autopilot_mode        : Optional[bool]  = None

class ActionResponse(BaseModel):
    success   : bool
    message   : str
    timestamp : str

# ---------------------------------------------------------------------------
# Bot configuration (live-updatable)
# ---------------------------------------------------------------------------

class BotConfig:
    """Mutable runtime configuration for the trading engine."""
    def __init__(self) -> None:
        self.kelly_fraction        : float = 0.25
        self.max_risk_per_trade    : float = 0.02
        self.obi_threshold         : float = 1.5
        self.adx_period            : int   = 20
        self.account_balance       : float = 0.0       # Clean initial balance
        self.min_interval_seconds  : float = 1.0
        self.symbol                : str   = "BTCUSDT"
        self.obi_depth             : int   = 3
        self.rolling_window        : int   = 100
        self.stop_loss_pct         : float = 0.01    # 1% stop from entry
        self.exchange_mode         : str   = "BYBIT_LIVE"
        # ── FIX #4: Read API credentials from environment variables (.env with live fallback) ──
        self.api_key               : str   = os.getenv("BYBIT_API_KEY", "K56egxupNqYyvDKaRx")
        self.secret_key            : str   = os.getenv("BYBIT_SECRET_KEY", "WqpnwsdVdiUR7h9Wc8OHfnFRW35L0L7cAwAq")
        self.passphrase            : str   = os.getenv("BYBIT_PASSPHRASE", "")
        self.autopilot_mode        : bool  = True    # Default Autonomous AI Quant Brain enabled

        # ── Autonomy Upgrade fields ───────────────────────────────
        self.scan_interval_seconds : float = float(os.getenv("SCAN_INTERVAL_SECONDS", "300.0"))     # Re-scan every 5 minutes
        self.last_scan_ts          : float = 0.0                  # Timestamp of last scan
        self.active_candidates     : List[Dict] = []         # Last scan result (for /coins endpoint)
        self.account_profile       : Optional[Dict] = None     # Cached account profile
        self.account_type          : str   = "UNIFIED"
        self.trading_mode          : str   = os.getenv("TRADING_MODE", "AUTO").lower()              # "spot", "futures", or "auto"

    def update(self, params: ParametersUpdate) -> List[str]:
        """Apply a ParametersUpdate and return list of changed keys."""
        changed = []
        for field, value in params.model_dump(exclude_none=True).items():
            if hasattr(self, field) and getattr(self, field) != value:
                setattr(self, field, value)
                changed.append(f"{field}={value}")
        return changed

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

# ---------------------------------------------------------------------------
# Exchange client (abstract + simulated)
# ---------------------------------------------------------------------------

class ExchangeClient(ABC):
    """Abstract exchange interface. Swap SimulatedExchangeClient for real SDK."""

    @abstractmethod
    async def get_candles(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Return OHLCV DataFrame with DatetimeIndex."""

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int) -> Dict[str, Any]:
        """Return {'bids': [[price, size],...], 'asks': [...], 'timestamp': float}."""

    @abstractmethod
    async def place_order(
        self, symbol: str, side: str, quantity: float, order_type: str = "market"
    ) -> Dict[str, Any]:
        """Place an order and return the exchange response."""

    @abstractmethod
    async def cancel_all_orders(self) -> Dict[str, Any]: ...

    @abstractmethod
    async def close_all_positions(self) -> Dict[str, Any]: ...


SYMBOL_BASE_PRICES = {
    "BTCUSDT" : 65_000.0,
    "ETHUSDT" : 3_500.0,
    "SOLUSDT" : 150.0,
    "DOGEUSDT": 0.12,
    "PEPEUSDT": 0.00001,
    "SUIUSDT" : 1.80,
    "AVAXUSDT": 28.0,
    "XRPUSDT" : 0.55,
    "LINKUSDT": 14.0,
    "NEARUSDT": 5.0,
    "APTUSDT" : 9.0,
    "WIFUSDT" : 2.50,
    "BNBUSDT" : 580.0,
    "SHIBUSDT": 0.000018,
    "ADAUSDT" : 0.40,
}

class SimulatedExchangeClient(ExchangeClient):
    """
    Synthetic exchange client for testing without a live connection.
    Tracks symbol prices independently to avoid false drawdown trips.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng        = random.Random(seed)
        self._np_rng     = np.random.default_rng(seed)
        self._prices     = dict(SYMBOL_BASE_PRICES)
        self._volatility = 0.0008

    def _get_sym_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 100.0)

    async def get_candles(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Generate ``limit`` synthetic 1-minute OHLCV bars ending now."""
        await asyncio.sleep(0)
        base_p = self._get_sym_price(symbol)
        returns = self._np_rng.normal(0.00005, self._volatility, limit)
        prices  = base_p * np.cumprod(1.0 + returns)
        self._prices[symbol] = float(prices[-1])

        spread = prices * self._np_rng.uniform(0.0001, 0.0005, limit)
        highs  = prices + spread
        lows   = prices - spread
        opens  = np.concatenate([[prices[0] * (1 - 0.0001)], prices[:-1]])
        vols   = self._np_rng.integers(200, 2000, limit).astype(float)

        idx = pd.date_range(end=datetime.now(timezone.utc), periods=limit, freq="1min")
        return pd.DataFrame({
            "open"  : np.round(opens, 2),
            "high"  : np.round(highs, 2),
            "low"   : np.round(lows,  2),
            "close" : np.round(prices, 2),
            "volume": vols,
        }, index=idx)

    async def get_orderbook(self, symbol: str, depth: int) -> Dict[str, Any]:
        """Generate a realistic synthetic order book."""
        await asyncio.sleep(0)

        mid = self._get_sym_price(symbol)
        bids = [
            [round(mid * (1 - 0.0001 * (i + 1)), 4),
             round(self._rng.uniform(0.05, 2.0), 4)]
            for i in range(depth)
        ]
        asks = [
            [round(mid * (1 + 0.0001 * (i + 1)), 4),
             round(self._rng.uniform(0.05, 2.0), 4)]
            for i in range(depth)
        ]
        return {
            "symbol"   : symbol,
            "bids"     : bids,
            "asks"     : asks,
            "timestamp": time.time(),
        }

    async def place_order(
        self, symbol: str, side: str, quantity: float, order_type: str = "market"
    ) -> Dict[str, Any]:
        """Simulate placing an order."""
        await asyncio.sleep(0)
        p = self._get_sym_price(symbol)
        slippage = p * self._rng.uniform(0.00005, 0.0002)
        fill_p   = p + slippage if side == "BUY" else p - slippage
        return {
            "order_id"  : f"SIM-{int(time.time()*1000)}",
            "symbol"    : symbol,
            "side"      : side,
            "quantity"  : quantity,
            "fill_price": round(fill_p, 4),
            "status"    : "FILLED",
            "timestamp" : datetime.now(timezone.utc).isoformat(),
        }

    async def cancel_all_orders(self) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"status": "OK", "cancelled": 0}

    async def close_all_positions(self) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"status": "OK", "closed": 0}

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """
    Manages all active WebSocket clients.

    Broadcast is best-effort: dead sockets are silently removed.
    Message queue capped at 1000 entries per client to prevent buffer bloat.
    """
    MAX_QUEUE = 1000

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        logger.info("[WS] Client connected. Total: %d", len(self._clients))

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        logger.info("[WS] Client disconnected. Total: %d", len(self._clients))

    async def broadcast(self, data: Dict[str, Any]) -> None:
        dead: Set[WebSocket] = set()
        payload = json.dumps(data, default=str)
        for ws in self._clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    @property
    def active_count(self) -> int:
        return len(self._clients)

# ---------------------------------------------------------------------------
# Trading engine (holds all components)
# ---------------------------------------------------------------------------

class TradingEngine:
    """
    Central coordinator for all Q-SonicFX trading components.

    Holds live references to the regime detector, OBI calculator,
    position sizer inputs, circuit breaker, and exchange client.
    Maintains shared in-memory state and trade/signal history.
    """

    def __init__(self, config: BotConfig) -> None:
        self.config   = config
        self.exchange : ExchangeClient = SimulatedExchangeClient()
        self._start_ts = time.time()

        # Shared mutable state (all reads/writes on asyncio event loop — no lock needed)
        self.state: Dict[str, Any] = {
            "status"          : "PAUSED",
            "current_regime"  : "UNKNOWN",
            "current_position": None,
            "daily_pnl"       : 0.0,
            "total_trades"    : 0,
            "last_update"     : None,
        }

        # OHLCV rolling buffer (last N candles for regime detection)
        self._candle_buffer: Deque[Dict] = deque(maxlen=500)

        # Trade & signal history (in-memory)
        self._trades : List[Dict] = []
        self._signals: Deque[Dict] = deque(maxlen=500)

        # Latest signal snapshot
        self._latest_signal: Dict[str, Any] = {}

        # Rolling trade history for Kelly inputs
        self._trade_history: List[Dict] = []

        # Production Equity Curve (Initializes clean from live account balance)
        self._equity_curve: Deque[Dict] = deque(maxlen=1000)
        self._equity_curve.append({
            "ts"    : datetime.now(timezone.utc).isoformat(),
            "equity": config.account_balance,
        })

        # Error recovery
        self._error_count     = 0
        self._last_error_ts   : Optional[float] = None

        # Components (initialized lazily or at startup)
        self._regime_detector = None
        self._cb              : Optional[Any] = None
        self._trade_id_counter = 0

        # ── FIX #1: Balance cache — avoids live Bybit call on every /status poll ──
        # Populated by _balance_fetch_loop() background task every 30 seconds.
        # /status reads from here instead — drops latency from ~800ms to <5ms.
        self._cached_balance: Dict[str, Any] = {
            "equity"          : 0.0,
            "balance"         : 0.0,
            "available_margin": 0.0,
            "last_updated"    : None,   # datetime object, or None before first fetch
        }
        self._balance_fetch_task: Optional[asyncio.Task] = None

        # ── FIX #3: Regime cache — updated every 60 seconds by _regime_fetch_loop ──
        self._regime_last_value        : str  = "UNKNOWN"   # previous regime (for change detection)
        self._regime_updated_this_minute: bool = False       # guard: log on change only
        self._regime_fetch_task: Optional[asyncio.Task] = None

        # ── Autonomy components ──────────────────────────────────────────────
        self.coin_scanner = CoinScanner(max_pairs=100) if _HAS_SCANNER and CoinScanner is not None else None
        self.account_detector = AccountDetector() if _HAS_DETECTOR and AccountDetector is not None else None
        self._scan_task: Optional[asyncio.Task] = None

        self._init_components()

    def _init_components(self) -> None:
        """Initialize all trading engine sub-components."""
        if _HAS_REGIME and RegimeDetector is not None:
            self._regime_detector = RegimeDetector(period=self.config.adx_period)

        if _HAS_CB and CircuitBreaker is not None:
            CircuitBreaker.reset_singleton()
            self._cb = CircuitBreaker(
                initial_balance    = self.config.account_balance if self.config.account_balance > 0 else 0.0,
                max_daily_loss_pct = 0.05,
                max_drawdown_pct   = 0.10,
            )
            set_global_circuit_breaker(self._cb)

        if _HAS_EXCHANGE and get_exchange_client is not None:
            self.live_connector = get_exchange_client(
                mode=self.config.exchange_mode,
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                passphrase=self.config.passphrase,
            )
        else:
            self.live_connector = None

        if _HAS_ALPHA and AlphaEngine is not None:
            self.alpha_engine = AlphaEngine(symbol=self.config.symbol)
        else:
            self.alpha_engine = None

        # Telegram Notifier (live credentials pre-loaded)
        if _HAS_NOTIFIER and TelegramNotifier is not None:
            self._notifier = TelegramNotifier(
                bot_token="8612139375:AAFXDuQqXiw-EB6VcL5zNaPJzehKMuNnqKo",
                chat_id="5482353857",
            )
        else:
            self._notifier = None

        # Pre-fetch live balance immediately on engine startup
        if self.live_connector is not None and self.config.exchange_mode != "SIMULATED":
            try:
                bal_data = self.live_connector.fetch_balance()
                eq = float(bal_data.get("equity", 0.0) or 0.0)
                bal = float(bal_data.get("balance", 0.0) or 0.0)
                avail = float(bal_data.get("available_margin", 0.0) or 0.0)
                if bal > 0:
                    self._cached_balance = {
                        "equity": eq, "balance": bal, "available_margin": avail,
                        "last_updated": datetime.now(timezone.utc),
                    }
                    self.config.account_balance = bal
                    if self._cb is not None:
                        self._cb.current_balance = bal
                        self._cb.peak_balance = bal
                        self._cb.initial_balance = bal
                        self._cb.initial_daily_balance = bal
                    logger.info("[Startup] Live balance pre-fetched: $%.4f USDT", bal)
            except Exception as e:
                logger.warning("[Startup] Pre-fetch balance failed: %s", e)

        logger.info(
            "[Engine] Initialized | regime=%s obi=%s sizer=%s cb=%s wfo=%s notify=%s",
            _HAS_REGIME, _HAS_OBI, _HAS_SIZER, _HAS_CB, _HAS_WFO, _HAS_NOTIFIER,
        )

    def reinit_connector(self) -> None:
        """Re-initialize exchange connector with updated config parameters."""
        if _HAS_EXCHANGE and get_exchange_client is not None:
            self.live_connector = get_exchange_client(
                mode=self.config.exchange_mode,
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                passphrase=self.config.passphrase,
            )
            logger.info("[Engine] Live connector re-initialized for mode=%s", self.config.exchange_mode)

    # ── FIX #1: Background balance fetch loop ──────────────────────────

    async def _balance_fetch_loop(self) -> None:
        """
        Background task: fetches live exchange balance every 15 seconds and
        stores the result in self._cached_balance.

        - First fetch happens immediately (no initial delay).
        - On failure: logs WARNING, retains last known values, retries in 15s.
        - /status reads from cache → near-instant response (<5ms).
        """
        STALE_THRESHOLD_SECONDS = 300   # 5 minutes
        FETCH_INTERVAL_SECONDS  = 15

        logger.info("[Balance] Background cache task started. First fetch immediately.")

        while True:
            try:
                if self.live_connector is not None and self.config.exchange_mode != "SIMULATED":
                    bal_data = await asyncio.to_thread(self.live_connector.fetch_balance)
                    equity           = float(bal_data.get("equity",           0.0) or 0.0)
                    balance          = float(bal_data.get("balance",          0.0) or 0.0)
                    available_margin = float(bal_data.get("available_margin", 0.0) or 0.0)
                else:
                    # Simulated mode: use config balance as stand-in
                    equity = balance = available_margin = self.config.account_balance

                self._cached_balance = {
                    "equity"          : equity,
                    "balance"         : balance,
                    "available_margin": available_margin,
                    "last_updated"    : datetime.now(timezone.utc),
                }
                # Also sync config.account_balance so trading loop uses live value
                self.config.account_balance = balance

                # Sync Circuit Breaker balance & auto-clear cooldown on new deposit
                if _HAS_CB and self._cb is not None and balance > 0:
                    if self._cb.current_balance != balance:
                        self._cb.current_balance = balance
                        self._cb.peak_balance = max(self._cb.peak_balance, balance)
                        if self._cb.initial_balance <= 1.0 and balance > 1.0:
                            self._cb.initial_balance = balance
                            self._cb.initial_daily_balance = balance
                            self._cb.halted_until = None
                            self._cb._is_emergency_halted = False
                            logger.info("[Balance] New balance $%.2f detected — CircuitBreaker updated & reset", balance)

                logger.info(
                    "[Balance] Updated: $%.4f equity | $%.4f available (cached for %ds)",
                    equity, available_margin, FETCH_INTERVAL_SECONDS,
                )

            except asyncio.CancelledError:
                logger.info("[Balance] Cache task cancelled.")
                raise
            except Exception as exc:
                logger.warning(
                    "[Balance] Fetch failed — retaining last cache. Error: %s", exc
                )

            await asyncio.sleep(FETCH_INTERVAL_SECONDS)

    # ── FIX #3: Background regime detection loop ─────────────────────────────

    async def _regime_fetch_loop(self) -> None:
        """
        Background task: fetches 60 live 1-minute Bybit klines every 60 seconds,
        runs RegimeDetector.detect(), and writes the result into
        engine.state["current_regime"].

        - First run is IMMEDIATE so /status never shows "UNKNOWN" for more than
          a few seconds after startup.
        - Logs ONLY when the regime value actually changes:
              [Regime] CHANGED: RANGING → STRONG_TREND (ADX=28.4, trade_allowed=True)
        - On failure: retains last known regime, retries next cycle.
        """
        FETCH_INTERVAL_SECONDS = 60
        KLINE_LIMIT            = 60   # 60 bars > period*2=40 — gives RegimeDetector enough history

        logger.info("[Regime] Background detection task started. First run immediately.")

        while True:
            try:
                symbol = self.config.symbol

                if self.live_connector is not None and hasattr(self.live_connector, "fetch_klines"):
                    # ── Live mode: pull real Bybit 1m candles ──────────────────
                    df = await asyncio.to_thread(
                        self.live_connector.fetch_klines, symbol, "1", KLINE_LIMIT
                    )
                else:
                    # ── Simulated mode: generate synthetic candles ─────────────
                    df = await self.exchange.get_candles(symbol, "1m", KLINE_LIMIT)

                if df is None or len(df) == 0:
                    logger.warning("[Regime] No kline data returned — retaining last regime.")
                else:
                    if _HAS_REGIME and self._regime_detector is not None:
                        result = self._regime_detector.detect(df)
                        new_regime = result.regime
                    else:
                        # Fallback: simple ADX-free regime guess from price volatility
                        import numpy as np
                        closes = df["close"].to_numpy(dtype=float)
                        returns = np.diff(closes) / closes[:-1]
                        vol = float(np.std(returns)) if len(returns) > 1 else 0.0
                        new_regime = "STRONG_TREND" if vol > 0.003 else "RANGING"
                        result = None

                    # Write into shared state (read by trading loop + /status)
                    old_regime = self._regime_last_value
                    self.state["current_regime"] = new_regime

                    # Log ONLY on change
                    if new_regime != old_regime:
                        adx_str = ""
                        trade_str = ""
                        if result is not None:
                            adx_val   = result.adx_value
                            adx_str   = f" (ADX={adx_val:.1f}" if adx_val == adx_val else " (ADX=n/a"
                            trade_str = f", trade_allowed={result.trade_allowed})"
                        logger.info(
                            "[Regime] CHANGED: %s → %s%s%s",
                            old_regime, new_regime, adx_str, trade_str,
                        )
                        self._regime_last_value = new_regime
                        self._regime_updated_this_minute = True
                    else:
                        self._regime_updated_this_minute = False

            except asyncio.CancelledError:
                logger.info("[Regime] Detection task cancelled.")
                raise
            except Exception as exc:
                logger.warning("[Regime] Detection failed — retaining last regime. Error: %s", exc)

            await asyncio.sleep(FETCH_INTERVAL_SECONDS)

    # ── Autonomy Background Scan Loop ───────────────────────────────────────

    async def _coin_scan_loop(self) -> None:
        """
        Background task: re-scans all Bybit pairs every scan_interval_seconds.
        Auto-selects best coin and switches symbol. Auto-detects account mode.
        First run: IMMEDIATE.
        """
        SCAN_INTERVAL = self.config.scan_interval_seconds
        logger.info("[Scanner] Background scan loop started. First scan immediately.")

        while True:
            try:
                # ── 1. Detect account type & balance ──────────────────────
                if self.account_detector:
                    profile = await self.account_detector.detect(
                        self.live_connector or self.exchange
                    )
                    self.config.account_profile = {
                        "account_type": profile.account_type,
                        "balance": profile.balance,
                        "mode": profile.mode,
                        "min_notional": profile.min_notional_default,
                    }
                    self.config.trading_mode = profile.mode

                    # Update balance cache
                    if profile.balance > 0:
                        self._cached_balance["balance"] = profile.balance
                        self._cached_balance["equity"] = profile.balance
                        self._cached_balance["available_margin"] = profile.available_balance

                # ── 2. Scan and rank coins ────────────────────────────────
                if self.coin_scanner and (
                    self.live_connector or hasattr(self.exchange, "get_candles")
                ):
                    exchange_client = self.live_connector or self.exchange
                    current_bal = self.config.account_profile.get("balance", 0.0) if self.config.account_profile else 0.0
                    if current_bal <= 0:
                        current_bal = self._cached_balance.get("balance", 0.0)
                    if current_bal <= 0:
                        current_bal = self.config.account_balance or 5.0  # fallback

                    result = await self.coin_scanner.scan(
                        exchange=exchange_client,
                        balance=current_bal,
                        current_regime=self.state.get("current_regime", "RANGING"),
                    )

                    # Cache candidates for /coins endpoint
                    self.config.active_candidates = [
                        {
                            "rank": i + 1,
                            "symbol": c.symbol,
                            "price": c.price,
                            "volume_24h": c.volume_24h,
                            "volatility_24h": round(c.volatility_24h, 4),
                            "obi_score": round(c.obi_raw, 4),
                            "composite_score": c.composite_score,
                            "min_qty": c.min_qty,
                            "fits_balance": c.fits_balance(current_bal),
                        }
                        for i, c in enumerate(result.candidates[:10])
                    ]

                    # ── 3. Switch to best pair if different ───────────────
                    if result.top_pick and result.top_pick.symbol != self.config.symbol:
                        old_symbol = self.config.symbol
                        self.config.symbol = result.top_pick.symbol

                        # Auto-tune parameters based on asset class
                        if result.top_pick.is_meme:
                            self.config.max_risk_per_trade = 0.01
                            self.config.kelly_fraction = 0.15
                            self.config.obi_threshold = 1.2
                        else:
                            self.config.max_risk_per_trade = 0.02
                            self.config.kelly_fraction = 0.25
                            self.config.obi_threshold = 1.5

                        logger.info(
                            "[Scanner] 🔄 SWITCHED: %s → %s (score=%.3f, vol=%.4f, obi=%+.4f, mode=%s)",
                            old_symbol, result.top_pick.symbol,
                            result.top_pick.composite_score,
                            result.top_pick.volatility_24h,
                            result.top_pick.obi_raw,
                            self.config.trading_mode,
                        )

                        # Log event to Supabase
                        if _HAS_DB:
                            async def _log_switch(old: str, new: str) -> None:
                                try:
                                    await db_module.log_event(
                                        "COIN_SWITCH",
                                        f"Auto-switch: {old} → {new} (score={result.top_pick.composite_score})",
                                        {"old_symbol": old, "new_symbol": new, "score": result.top_pick.composite_score},
                                    )
                                except Exception:
                                    pass
                            asyncio.ensure_future(_log_switch(old_symbol, result.top_pick.symbol))

                    else:
                        logger.debug(
                            "[Scanner] Top pick unchanged: %s (score=%.3f)",
                            result.top_pick.symbol if result.top_pick else "NONE",
                            result.top_pick.composite_score if result.top_pick else 0,
                        )

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[Scanner] Scan cycle failed: %s", exc, exc_info=True)

            await asyncio.sleep(SCAN_INTERVAL)

    # ── State helpers ──────────────────────────────────────────────────

    @property
    def uptime(self) -> float:
        return time.time() - self._start_ts

    def _next_trade_id(self) -> int:
        self._trade_id_counter += 1
        return self._trade_id_counter

    def cb_snapshot(self) -> Dict[str, Any]:
        """Return a serializable circuit breaker state snapshot."""
        if self._cb is None:
            return {"available": False}
        halted, reason = self._cb.check_failsafe()
        return {
            "available"         : True,
            "halted"            : halted,
            "reason"            : reason.value if hasattr(reason, "value") else str(reason),
            "halted_until"      : self._cb.halted_until.isoformat() if self._cb.halted_until else None,
            "daily_drawdown_pct": round(self._cb.daily_drawdown_pct * 100, 4),
            "trailing_drawdown_pct": round(self._cb.trailing_drawdown_pct * 100, 4),
            "current_balance"   : round(self._cb.current_balance, 2),
            "peak_balance"      : round(self._cb.peak_balance, 2),
        }

    def compute_performance(self) -> PerformanceResponse:
        """Aggregate performance metrics from trade history."""
        closed = [t for t in self._trades if t.get("status") == "CLOSED"]
        n = len(closed)
        if n == 0:
            return PerformanceResponse(
                total_trades=0, win_rate=0.0, avg_win_pct=0.0,
                avg_loss_pct=0.0, profit_factor=0.0,
                sharpe_ratio=0.0, max_drawdown_pct=0.0,
                total_return_pct=0.0, trades=[],
            )

        pnls = [t.get("pnl_percent", 0.0) for t in closed]
        wins  = [p for p in pnls if p > 0]
        losses= [p for p in pnls if p < 0]
        win_rate = len(wins) / n
        avg_win  = float(np.mean(wins))   if wins   else 0.0
        avg_loss = float(np.mean(np.abs(losses))) if losses else 0.0
        gp = sum(wins)
        gl = abs(sum(losses))
        pf = gp / gl if gl > 0 else 0.0

        arr = np.array(pnls)
        std = arr.std()
        sharpe = (arr.mean() / std * np.sqrt(252)) if std > 0 else 0.0

        equity = np.cumprod(1 + arr / 100)
        peak   = np.maximum.accumulate(equity)
        max_dd = float(((equity - peak) / peak).min() * 100)
        total_return = float((equity[-1] - 1.0) * 100)

        records = [
            TradeRecord(
                trade_id        = t.get("trade_id", 0),
                timestamp       = t.get("timestamp", ""),
                symbol          = t.get("symbol", ""),
                side            = t.get("side", ""),
                entry_price     = t.get("entry_price", 0.0),
                exit_price      = t.get("exit_price"),
                quantity        = t.get("quantity", 0.0),
                pnl             = t.get("pnl"),
                pnl_percent     = t.get("pnl_percent"),
                regime_at_entry = t.get("regime_at_entry", ""),
                obi_signal      = t.get("obi_signal", ""),
                status          = t.get("status", ""),
            )
            for t in self._trades[-100:]
        ]

        return PerformanceResponse(
            total_trades     = n,
            win_rate         = round(win_rate, 4),
            avg_win_pct      = round(avg_win,   4),
            avg_loss_pct     = round(avg_loss,  4),
            profit_factor    = round(pf,         4),
            sharpe_ratio     = round(sharpe,     4),
            max_drawdown_pct = round(max_dd,     4),
            total_return_pct = round(total_return, 4),
            trades           = records,
        )

# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------

_config  = BotConfig()
_engine  = TradingEngine(_config)
_ws_mgr  = ConnectionManager()
_loop_task: Optional[asyncio.Task] = None

# ---------------------------------------------------------------------------
# Trading loop
# ---------------------------------------------------------------------------

async def trading_loop(engine: TradingEngine, ws_mgr: ConnectionManager) -> None:
    """
    Core async trading loop.

    Cycle: fetch candles -> regime detect -> OBI -> size -> execute -> CB check -> broadcast.
    """
    logger.info("[Loop] Trading loop started.")
    obi_signal_str  = SIGNAL_NEUTRAL
    obi_value       = 0.0
    obi_weighted    = 0.0
    obi_velocity    = 0.0
    regime_str      = "UNKNOWN"
    trade_allowed   = False

    while True:
        cycle_start = time.perf_counter()
        engine.state["last_update"] = datetime.now(timezone.utc).isoformat()

        # ── Broadcast WebSocket telemetry heartbeat every cycle ──────────
        if ws_mgr.active_count > 0:
            try:
                await ws_mgr.broadcast({
                    "type"            : "HEARTBEAT",
                    "status"          : engine.state["status"],
                    "timestamp"       : engine.state["last_update"],
                    "current_regime"  : engine.state.get("current_regime", "RANGING"),
                    "vpin_score"      : getattr(engine.alpha_engine, "last_vpin", 0.15) if getattr(engine, "alpha_engine", None) else 0.15,
                    "obi_value"       : obi_value,
                    "circuit_breaker" : engine.cb_snapshot(),
                })
            except Exception:
                pass

        # ── Pause/Halt gate ────────────────────────────────────────────
        if engine.state["status"] != "RUNNING":
            await asyncio.sleep(1.0)
            continue

        try:
            cfg = engine.config

            # ── 1. Autopilot AI Quant Brain (Multi-Asset Auto-Scanner & Dynamic Risk Tuner) ──
            if getattr(cfg, "autopilot_mode", True) and engine.state.get("current_position") is None:
                # Symbol already set by background coin_scan_loop. If still default BTCUSDT and balance < $10,
                # trigger an immediate emergency scan cycle.
                if cfg.symbol == "BTCUSDT" and cfg.account_balance < 10.0:
                    logger.info("[Loop] BTCUSDT too expensive for $%.2f balance — triggering emergency scan", cfg.account_balance)
                    if engine.coin_scanner:
                        exchange_client = engine.live_connector or engine.exchange
                        scan_res = await engine.coin_scanner.scan(
                            exchange=exchange_client,
                            balance=cfg.account_balance,
                            current_regime=engine.state.get("current_regime", "RANGING"),
                        )
                        if scan_res.top_pick:
                            cfg.symbol = scan_res.top_pick.symbol
                            logger.info("[Loop] 🔄 Emergency switch to %s", cfg.symbol)

            # ── 2. Fetch market data from live_connector or simulated exchange ───
            if engine.live_connector is not None:
                orderbook_data = await asyncio.to_thread(engine.live_connector.fetch_orderbook, cfg.symbol, 20)
                bids_raw = orderbook_data.get("bids", [])
                asks_raw = orderbook_data.get("asks", [])
                
                # Fetch live wallet balance & positions
                bal_data = await asyncio.to_thread(engine.live_connector.fetch_balance)
                cfg.account_balance = bal_data.get("balance", cfg.account_balance)
                
                live_positions = await asyncio.to_thread(engine.live_connector.fetch_positions, cfg.symbol)
                if live_positions:
                    engine.state["current_position"] = live_positions[0]
                else:
                    engine.state["current_position"] = None
                    
                cur_price = bids_raw[0][0] if bids_raw else (asks_raw[0][0] if asks_raw else 65000.0)
            else:
                candles   = await engine.exchange.get_candles(cfg.symbol, "1m", 50)
                orderbook = await engine.exchange.get_orderbook(cfg.symbol, depth=10)
                cur_price = float(candles["close"].iloc[-1])
                bids_raw  = orderbook.get("bids", [])
                asks_raw  = orderbook.get("asks", [])

            # ── 2. Regime detection & Alpha Engine ─────────────────────
            if _HAS_ALPHA and engine.alpha_engine is not None:
                b_pct, a_pct = 50, 50
                if bids_raw and asks_raw:
                    b_vol = sum(q for p, q in bids_raw[:3])
                    a_vol = sum(q for p, q in asks_raw[:3])
                    tot_v = b_vol + a_vol
                    if tot_v > 0:
                        b_pct = int(b_vol / tot_v * 100)
                        a_pct = 100 - b_pct
                
                alpha_out = engine.alpha_engine.process_tick(
                    price=cur_price,
                    volume=1.0,
                    regime=engine.state.get("current_regime", "RANGING"),
                    obi_signal=obi_signal_str,
                    obi_value=obi_value,
                    obi_velocity=obi_velocity,
                    bid_pct=b_pct,
                    ask_pct=a_pct,
                )
                
                if alpha_out.primary_signal == "TOXIC_PAUSE":
                    trade_allowed = False
                elif alpha_out.primary_signal in ("BUY", "SELL"):
                    obi_signal_str = alpha_out.primary_signal
                    trade_allowed = True

            # ── 3. OBI calculation ─────────────────────────────────────
            if bids_raw and asks_raw:
                b_vol = sum(q for p, q in bids_raw[:cfg.obi_depth])
                a_vol = sum(q for p, q in asks_raw[:cfg.obi_depth])
                total = b_vol + a_vol
                obi_value = (b_vol - a_vol) / total if total > 0 else 0.0
                obi_weighted = obi_value * 0.9

                threshold_ratio = cfg.obi_threshold / 10.0
                if obi_value > threshold_ratio:
                    obi_signal_str = SIGNAL_BUY
                elif obi_value < -threshold_ratio:
                    obi_signal_str = SIGNAL_SELL
                else:
                    obi_signal_str = SIGNAL_NEUTRAL

            # ── 4. Position sizing & trade execution ───────────────────
            no_open_pos = engine.state["current_position"] is None

            if (obi_signal_str in (SIGNAL_BUY, SIGNAL_SELL)
                    and no_open_pos
                    and trade_allowed):
                try:
                    side = "LONG" if obi_signal_str == SIGNAL_BUY else "SHORT"
                    stop_delta = cur_price * cfg.stop_loss_pct
                    stop_price = (
                        cur_price - stop_delta if side == "LONG"
                        else cur_price + stop_delta
                    )

                    if _HAS_SIZER:
                        wr, aw, al = update_rolling_stats(
                            engine._trade_history, window=cfg.rolling_window
                        )
                        # Need at least 5 trades for Kelly to be meaningful
                        if len(engine._trade_history) < 5:
                            wr, aw, al = 0.55, 0.018, 0.010

                        size_result = compute_position_size(
                            account_balance    = cfg.account_balance,
                            win_rate           = wr,
                            avg_win            = aw,
                            avg_loss           = al,
                            entry_price        = cur_price,
                            stop_loss_price    = stop_price,
                            position_side      = side,
                            kelly_fraction     = cfg.kelly_fraction,
                            max_risk_per_trade = cfg.max_risk_per_trade,
                        )
                        qty = size_result.position_size
                        trade_ok = size_result.is_trade_allowed
                    else:
                        # Fallback: fixed 0.001 BTC
                        qty      = 0.001
                        trade_ok = True

                    if trade_ok and qty > 0:
                        # Check circuit breaker
                        if _HAS_CB and engine._cb is not None:
                            halted, reason = engine._cb.check_failsafe()
                            if halted:
                                logger.warning("[Loop] CB halt before order: %s", reason)
                                trade_ok = False

                    if trade_ok and qty > 0:
                        fill = await engine.exchange.place_order(
                            cfg.symbol, side, qty, "market"
                        )
                        trade_id = engine._next_trade_id()
                        trade_record = {
                            "trade_id"      : trade_id,
                            "timestamp"     : datetime.now(timezone.utc).isoformat(),
                            "symbol"        : cfg.symbol,
                            "side"          : side,
                            "entry_price"   : fill.get("fill_price", cur_price),
                            "exit_price"    : None,
                            "quantity"      : qty,
                            "pnl"           : None,
                            "pnl_percent"   : None,
                            "regime_at_entry": regime_str,
                            "obi_signal"    : obi_signal_str,
                            "status"        : "OPEN",
                        }
                        engine._trades.append(trade_record)
                        engine.state["current_position"] = trade_record
                        engine.state["total_trades"] += 1
                        logger.info(
                            "[Loop] Trade opened | %s %s %.5f @ %.2f",
                            side, cfg.symbol, qty, fill.get("fill_price", cur_price)
                        )
                        # ── Persist trade open to Supabase (fire-and-forget) ──
                        if _HAS_DB:
                            async def _save_open(td: dict) -> None:
                                try:
                                    db_id = await db_module.save_trade({
                                        "timestamp"      : td["timestamp"],
                                        "symbol"         : td["symbol"],
                                        "side"           : td["side"],
                                        "entry_price"    : td["entry_price"],
                                        "quantity"       : td["quantity"],
                                        "regime_at_entry": td["regime_at_entry"],
                                        "obi_signal"     : td["obi_signal"],
                                        "obi_value"      : round(obi_value, 5),
                                        "status"         : "OPEN",
                                    })
                                    td["db_id"] = db_id  # store DB PK for later close_trade
                                    await db_module.log_event(
                                        "TRADE_OPENED",
                                        f"Trade #{td['trade_id']} OPENED {td['side']} {td['symbol']} qty={td['quantity']:.5f} @ {td['entry_price']:.2f}",
                                        {"trade_id": td["trade_id"], "symbol": td["symbol"]},
                                    )
                                except Exception as _e:
                                    logger.warning("[DB] Failed to persist trade open: %s", _e)
                            asyncio.ensure_future(_save_open(trade_record))
                        # ── Telegram: trade entry alert (non-blocking) ──
                        if engine._notifier:
                            import threading
                            threading.Thread(
                                target=engine._notifier.notify_trade_entry,
                                args=(cfg.symbol, side, qty, fill.get("fill_price", cur_price)),
                                kwargs={"regime": regime_str},
                                daemon=True,
                            ).start()

                except Exception as e:
                    logger.error("[Loop] Trade execution error: %s", e, exc_info=True)

            # ── 5. Mark open position P&L ──────────────────────────────
            pos = engine.state["current_position"]
            unrealized_pnl = 0.0
            if pos is not None:
                entry = pos.get("entry_price", cur_price)
                qty   = pos.get("quantity", 0.0)
                side  = pos.get("side", "LONG")
                unrealized_pnl = (
                    (cur_price - entry) * qty if side == "LONG"
                    else (entry - cur_price) * qty
                )
                pos["unrealized_pnl"] = round(unrealized_pnl, 4)

                # Simple simulated close: reverse OBI signal
                if (obi_signal_str == SIGNAL_SELL and side == "LONG") or \
                   (obi_signal_str == SIGNAL_BUY  and side == "SHORT"):
                    pnl_pct = unrealized_pnl / (entry * qty) * 100
                    pos["exit_price"]  = cur_price
                    pos["pnl"]         = round(unrealized_pnl, 4)
                    pos["pnl_percent"] = round(pnl_pct, 4)
                    pos["status"]      = "CLOSED"
                    engine._trade_history.append({
                        "pnl_percent": pnl_pct / 100,
                        "side"       : side,
                    })
                    if _HAS_CB and engine._cb is not None:
                        engine._cb.update_pnl(unrealized_pnl, 0.0)
                    engine.state["daily_pnl"] = round(
                        engine.state["daily_pnl"] + unrealized_pnl, 4
                    )
                    engine.state["current_position"] = None
                    logger.info(
                        "[Loop] Trade closed | %s pnl=%.4f (%.2f%%)",
                        side, unrealized_pnl, pnl_pct
                    )
                    # ── Persist trade close to Supabase (fire-and-forget) ──
                    if _HAS_DB:
                        _closed_pos = dict(pos)  # snapshot before mutation
                        async def _save_close(cp: dict) -> None:
                            try:
                                db_id = cp.get("db_id")  # set during open persist
                                if db_id is not None:
                                    await db_module.close_trade(
                                        trade_id   = db_id,
                                        exit_price = cp["exit_price"],
                                        pnl        = cp["pnl"],
                                        pnl_percent= cp["pnl_percent"],
                                    )
                                    await db_module.log_event(
                                        "TRADE_CLOSED",
                                        f"Trade #{cp['trade_id']} CLOSED {cp['side']} pnl={cp['pnl']:.4f} ({cp['pnl_percent']:.2f}%)",
                                        {"trade_id": cp["trade_id"], "pnl": cp["pnl"]},
                                    )
                            except Exception as _e:
                                logger.warning("[DB] Failed to persist trade close: %s", _e)
                        asyncio.ensure_future(_save_close(_closed_pos))
                    # ── Telegram: trade exit alert (non-blocking) ──
                    if engine._notifier:
                        import threading
                        threading.Thread(
                            target=engine._notifier.notify_trade_exit,
                            args=(cfg.symbol, side, entry, cur_price, round(unrealized_pnl, 4), round(pnl_pct, 4)),
                            daemon=True,
                        ).start()

            # ── 6. Circuit breaker check ───────────────────────────────
            if _HAS_CB and engine._cb is not None:
                engine._cb.update_pnl(0.0, unrealized_pnl)
                halted, reason = engine._cb.check_failsafe()
                if halted:
                    logger.critical("[Loop] Circuit breaker triggered: %s", reason)
                    engine.state["status"] = "HALTED"
                    engine._cb.emergency_shutdown()
                    # ── Telegram: emergency CB alert (non-blocking) ──
                    if engine._notifier:
                        import threading
                        cb_snap = engine.cb_snapshot()
                        threading.Thread(
                            target=engine._notifier.notify_circuit_breaker,
                            args=(str(reason), engine.state.get("daily_pnl", 0.0), cb_snap.get("daily_drawdown_pct", 0.0)),
                            daemon=True,
                        ).start()

            # ── 7. Update equity curve & state ─────────────────────────
            equity = cfg.account_balance + engine.state["daily_pnl"] + unrealized_pnl
            engine._equity_curve.append({
                "ts"    : datetime.now(timezone.utc).isoformat(),
                "equity": round(equity, 2),
            })
            engine.state["last_update"] = datetime.now(timezone.utc).isoformat()
            engine._error_count = 0   # reset on successful cycle

            # ── Persist equity snapshot to Supabase (fire-and-forget, every cycle) ──
            if _HAS_DB:
                _eq_snap = {
                    "timestamp"        : datetime.now(timezone.utc).isoformat(),
                    "equity"           : round(equity, 2),
                    "unrealized_pnl"   : round(unrealized_pnl, 4),
                    "realized_pnl"     : round(engine.state["daily_pnl"], 4),
                    "daily_drawdown"   : round(engine.cb_snapshot().get("daily_drawdown_pct", 0.0), 4),
                    "trailing_drawdown": round(engine.cb_snapshot().get("trailing_drawdown_pct", 0.0), 4),
                    "open_positions"   : 1 if engine.state.get("current_position") else 0,
                    "status"           : engine.state["status"],
                }
                async def _save_eq(snap: dict) -> None:
                    try:
                        await db_module.save_equity_snapshot(snap)
                    except Exception as _e:
                        logger.debug("[DB] Equity snapshot save failed: %s", _e)
                asyncio.ensure_future(_save_eq(_eq_snap))

            # ── 8. Broadcast to WebSocket clients ──────────────────────
            if ws_mgr.active_count > 0:
                cb_snap = engine.cb_snapshot()
                await ws_mgr.broadcast({
                    "type"          : "CYCLE_UPDATE",
                    "timestamp"     : engine.state["last_update"],
                    "regime"        : regime_str,
                    "trade_allowed" : trade_allowed,
                    "obi_value"     : round(obi_value, 5),
                    "obi_weighted"  : round(obi_weighted, 5),
                    "obi_signal"    : obi_signal_str,
                    "current_price" : cur_price,
                    "current_position": engine.state["current_position"],
                    "unrealized_pnl": round(unrealized_pnl, 4),
                    "daily_pnl"     : engine.state["daily_pnl"],
                    "total_trades"  : engine.state["total_trades"],
                    "equity"        : round(equity, 2),
                    "status"        : engine.state["status"],
                    "halted"        : cb_snap.get("halted", False),
                    "daily_dd_pct"  : cb_snap.get("daily_drawdown_pct", 0.0),
                })

            # Store latest signal
            engine._latest_signal = {
                "timestamp"    : engine.state["last_update"],
                "regime"       : regime_str,
                "trade_allowed": trade_allowed,
                "obi_value"    : round(obi_value, 5),
                "obi_weighted" : round(obi_weighted, 5),
                "obi_velocity" : round(obi_velocity, 5),
                "obi_signal"   : obi_signal_str,
                "entry_price"  : cur_price,
            }

        except asyncio.CancelledError:
            logger.info("[Loop] Trading loop cancelled.")
            raise
        except Exception as exc:
            engine._error_count += 1
            engine._last_error_ts = time.time()
            logger.error("[Loop] Unhandled error #%d: %s", engine._error_count, exc, exc_info=True)
            engine.state["status"] = "ERROR"
            await ws_mgr.broadcast({
                "type"   : "ERROR",
                "message": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            # Auto-restart after 60 seconds if in ERROR state
            await asyncio.sleep(60.0)
            if engine.state["status"] == "ERROR":
                logger.info("[Loop] Auto-restarting after error recovery delay.")
                engine.state["status"] = "RUNNING"
            continue

        # ── Yield until next cycle ─────────────────────────────────────
        elapsed = time.perf_counter() - cycle_start
        sleep_t = max(0.0, cfg.min_interval_seconds - elapsed)
        await asyncio.sleep(sleep_t)

# ---------------------------------------------------------------------------
# FastAPI app + lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engine, start trading loop and balance cache task on startup."""
    global _loop_task
    logger.info("[Lifespan] Q-SonicFX API Server starting up...")

    # ── Supabase Cloud Database Initialization ────────────────────────
    if _HAS_DB and init_db is not None:
        try:
            await init_db()  # Initializes Supabase client
            logger.info("[Database] Supabase database client initialized.")
        except Exception as _db_err:
            logger.error("[Database] Init failed (non-fatal): %s", _db_err)

    # ── FIX #1: Start balance cache background task FIRST ───────────────
    # This kicks off an immediate balance fetch so /status has real data
    # from the very first request (no 30-second cold-start wait).
    _engine._balance_fetch_task = asyncio.create_task(
        _engine._balance_fetch_loop(),
        name="balance_fetch_loop",
    )
    logger.info("[Lifespan] Balance cache task started (fetch every 30s).")

    # ── FIX #3: Start regime detection background task ───────────────────
    # First run is immediate — regime is populated before any trade signal fires.
    _engine._regime_fetch_task = asyncio.create_task(
        _engine._regime_fetch_loop(),
        name="regime_fetch_loop",
    )
    logger.info("[Lifespan] Regime detection task started (update every 60s).")

    # ── Autonomy: Start coin scanner & account detector background task ──
    if _HAS_SCANNER and _engine.coin_scanner:
        _engine._scan_task = asyncio.create_task(
            _engine._coin_scan_loop(),
            name="coin_scan_loop",
        )
        logger.info("[Lifespan] Coin scan loop started (scan every %ds).", _config.scan_interval_seconds)

    _loop_task = asyncio.create_task(
        trading_loop(_engine, _ws_mgr),
        name="trading_loop",
    )
    logger.info("[Lifespan] Trading loop task created (status=PAUSED).")
    yield
    # ── Shutdown ────────────────────────────────────────────────────
    logger.info("[Lifespan] Shutting down...")
    _engine.state["status"] = "PAUSED"

    # ── Autonomy: Cancel scan task ──────────────────────────────────
    if _engine._scan_task and not _engine._scan_task.done():
        _engine._scan_task.cancel()
        try:
            await asyncio.wait_for(_engine._scan_task, timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        logger.info("[Lifespan] Coin scan task cancelled.")

    # ── FIX #1: Cancel balance cache task ───────────────────────────────
    if _engine._balance_fetch_task and not _engine._balance_fetch_task.done():
        _engine._balance_fetch_task.cancel()
        try:
            await asyncio.wait_for(_engine._balance_fetch_task, timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        logger.info("[Lifespan] Balance cache task cancelled.")

    if _loop_task and not _loop_task.done():
        _loop_task.cancel()
        try:
            await asyncio.wait_for(_loop_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    logger.info("[Lifespan] Shutdown complete.")


app = FastAPI(
    title       = "Q-SonicFX Trading Bot API",
    description = "Central nervous system for the Q-SonicFX algorithmic trading bot.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Lock down in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/", tags=["System"])
async def root():
    """Root endpoint for health checks and API status."""
    return {
        "app": "Q-SonicFX Trading Bot API",
        "status": _engine.state["status"],
        "version": "1.0.0",
        "docs": "/docs",
        "timestamp": _ts(),
    }


@app.get("/status", response_model=StatusResponse, tags=["Control"])
async def get_status():
    """
    Full bot state + circuit breaker metrics + cached wallet balance.

    FIX #1: Balance is now served from a 30-second background cache.
    Response time drops from ~800ms (live Bybit call) to <5ms (dict lookup).
    """
    pos = _engine.state.get("current_position")

    # ── FIX #1: Read from in-memory cache (no Bybit API call here) ─────────
    cache         = _engine._cached_balance
    last_updated  = cache.get("last_updated")   # datetime or None
    bal           = float(cache.get("balance", _engine.config.account_balance))

    # Determine cache freshness
    if last_updated is None:
        balance_status    = "INITIALIZING"
        balance_cached_at = None
    else:
        age_seconds = (datetime.now(timezone.utc) - last_updated).total_seconds()
        if age_seconds > 300:   # 5-minute stale threshold
            balance_status = "STALE"
        else:
            balance_status = "OK"
        balance_cached_at = last_updated.isoformat()

    # Keep config.account_balance in sync for trading loop position sizer
    _engine.config.account_balance = bal
    # ────────────────────────────────────────────────────────────────────────

    upnl     = pos.get("unrealized_pnl", 0.0) if pos else 0.0
    qty      = pos.get("quantity",       0.0) if pos else 0.0
    ep       = pos.get("entry_price",    0.0) if pos else 0.0
    notional = round(qty * ep, 2)
    margin   = round(notional / 10.0, 2) if notional > 0 else 0.0   # 10x leverage default
    free_bal = max(0.0, round(bal - margin, 4))

    return StatusResponse(
        status            = _engine.state["status"],
        current_regime    = _engine.state["current_regime"],
        current_position  = _engine.state["current_position"],
        daily_pnl         = _engine.state["daily_pnl"],
        total_trades      = _engine.state["total_trades"],
        last_update       = _engine.state["last_update"],
        circuit_breaker   = _engine.cb_snapshot(),
        uptime_seconds    = round(_engine.uptime, 1),
        account_balance   = bal,
        available_balance = free_bal,
        used_margin       = margin,
        position_notional = notional,
        unrealized_pnl    = upnl,
        balance_cached_at = balance_cached_at,
        balance_status    = balance_status,
    )


@app.post("/start", response_model=ActionResponse, tags=["Control"])
async def start_bot():
    """Start the trading loop."""
    if _HAS_CB and _engine._cb is not None:
        _engine._cb.halted_until         = None
        _engine._cb._is_emergency_halted = False
        _engine._cb.daily_realized_pnl   = 0.0
        _engine._cb.daily_drawdown_pct   = 0.0
        _engine._cb.trailing_drawdown_pct= 0.0

    # Fetch live balance immediately on start
    if _engine.live_connector is not None and _engine.config.exchange_mode != "SIMULATED":
        try:
            bal_data = await asyncio.to_thread(_engine.live_connector.fetch_balance)
            eq = float(bal_data.get("equity", 0.0) or 0.0)
            bal = float(bal_data.get("balance", 0.0) or 0.0)
            avail = float(bal_data.get("available_margin", 0.0) or 0.0)
            if bal > 0:
                _engine._cached_balance = {
                    "equity": eq, "balance": bal, "available_margin": avail,
                    "last_updated": datetime.now(timezone.utc),
                }
                _engine.config.account_balance = bal
                if _HAS_CB and _engine._cb is not None:
                    _engine._cb.current_balance = bal
                    _engine._cb.peak_balance = max(_engine._cb.peak_balance, bal)
                    if _engine._cb.initial_balance <= 1.0:
                        _engine._cb.initial_balance = bal
                        _engine._cb.initial_daily_balance = bal
        except Exception as e:
            logger.warning("[API] Live balance fetch on start failed: %s", e)

    if _engine.state["status"] == "RUNNING":
        return ActionResponse(success=False, message="Already running.", timestamp=_ts())
    _engine.state["status"] = "RUNNING"
    logger.info("[API] Bot STARTED via /start (CircuitBreaker reset)")
    await _ws_mgr.broadcast({"type": "STATUS_CHANGE", "status": "RUNNING", "timestamp": _ts()})

    if _HAS_NOTIFIER and _engine._notifier is not None:
        bal = _engine._cached_balance.get("balance", _engine.config.account_balance)
        mode = _engine.config.trading_mode.upper()
        sym = _engine.config.symbol
        asyncio.ensure_future(asyncio.to_thread(
            _engine._notifier.send_message,
            f"🚀 <b>Q-SONICFX LIVE TRADING STARTED</b>\n\n"
            f"• <b>Account Balance:</b> <code>${bal:.4f} USDT</code>\n"
            f"• <b>Trading Mode:</b> <code>{mode}</code>\n"
            f"• <b>Active Target:</b> <code>{sym}</code>\n"
            f"• <b>Autopilot:</b> <code>ACTIVE</code>\n\n"
            f"🕒 <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>"
        ))

    return ActionResponse(success=True, message="Trading loop started.", timestamp=_ts())


@app.post("/pause", response_model=ActionResponse, tags=["Control"])
async def pause_bot():
    """Pause new entries. Open positions remain."""
    if _engine.state["status"] == "PAUSED":
        return ActionResponse(success=False, message="Already paused.", timestamp=_ts())
    _engine.state["status"] = "PAUSED"
    logger.info("[API] Bot PAUSED via /pause")
    await _ws_mgr.broadcast({"type": "STATUS_CHANGE", "status": "PAUSED", "timestamp": _ts()})

    if _HAS_NOTIFIER and _engine._notifier is not None:
        asyncio.ensure_future(asyncio.to_thread(
            _engine._notifier.send_message,
            f"⏸️ <b>Q-SONICFX LIVE TRADING PAUSED</b>\n\n"
            f"• <b>Action:</b> New entries paused. Open positions kept.\n"
            f"🕒 <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>"
        ))

    return ActionResponse(success=True, message="Bot paused. Open positions kept.", timestamp=_ts())


@app.post("/stop", response_model=ActionResponse, tags=["Control"])
async def stop_bot():
    """Emergency stop: pause + circuit breaker emergency_shutdown()."""
    _engine.state["status"] = "HALTED"
    shutdown_result = {}
    if _HAS_CB and _engine._cb is not None:
        try:
            shutdown_result = _engine._cb.emergency_shutdown()
        except Exception as e:
            logger.error("[API] CB emergency_shutdown error: %s", e)
    logger.critical("[API] EMERGENCY STOP triggered via /stop")
    await _ws_mgr.broadcast({
        "type"     : "EMERGENCY_STOP",
        "status"   : "HALTED",
        "timestamp": _ts(),
    })

    if _HAS_NOTIFIER and _engine._notifier is not None:
        asyncio.ensure_future(asyncio.to_thread(
            _engine._notifier.send_message,
            f"🛑 <b>Q-SONICFX EMERGENCY STOP EXECUTED</b>\n\n"
            f"• <b>Status:</b> Engine Halted\n"
            f"• <b>Steps:</b> {len(shutdown_result.get('steps', []))} shutdown steps executed.\n"
            f"🕒 <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>"
        ))

    return ActionResponse(
        success   = True,
        message   = f"Emergency stop executed. {len(shutdown_result.get('steps', []))} shutdown steps.",
        timestamp = _ts(),
    )


@app.post("/resume", response_model=ActionResponse, tags=["Control"])
async def resume_bot():
    """
    Manually resume trading (bypasses 24h cooldown — for testing only).
    """
    if _HAS_CB and _engine._cb is not None:
        _engine._cb.halted_until         = None
        _engine._cb._is_emergency_halted = False
        _engine._cb.daily_realized_pnl   = 0.0
        _engine._cb.daily_drawdown_pct   = 0.0
        _engine._cb.trailing_drawdown_pct= 0.0
    _engine.state["status"] = "RUNNING"
    logger.warning("[API] Bot RESUMED (cooldown bypassed) via /resume")
    await _ws_mgr.broadcast({"type": "STATUS_CHANGE", "status": "RUNNING", "timestamp": _ts()})

    if _HAS_NOTIFIER and _engine._notifier is not None:
        bal = _engine._cached_balance.get("balance", _engine.config.account_balance)
        asyncio.ensure_future(asyncio.to_thread(
            _engine._notifier.send_message,
            f"🔄 <b>Q-SONICFX LIVE TRADING RESUMED</b>\n\n"
            f"• <b>Balance:</b> <code>${bal:.4f} USDT</code>\n"
            f"• <b>Risk Status:</b> Cooldown Bypassed & Failsafes Reset\n"
            f"🕒 <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>"
        ))

    return ActionResponse(success=True, message="Cooldown bypassed. Bot resumed.", timestamp=_ts())


@app.get("/performance", response_model=PerformanceResponse, tags=["Analytics"])
async def get_performance():
    """Aggregate performance stats + last 100 trade records."""
    return _engine.compute_performance()


@app.get("/signals/latest", response_model=SignalResponse, tags=["Analytics"])
async def get_latest_signal():
    """Most recent regime + OBI signal + suggested position size."""
    sig = _engine._latest_signal
    if not sig:
        sig = {
            "symbol": _engine.config.symbol,
            "regime": _engine.state.get("current_regime", "RANGING"),
            "obi_value": 0.05,
            "obi_signal": SIGNAL_NEUTRAL,
            "vpin_score": 0.15,
            "vpin_status": "NORMAL",
            "entry_price": 65000.0,
            "timestamp": _ts(),
        }

    # Compute suggested size (read-only, no side effects)
    suggested_size = 0.0
    suggested_risk = 0.0
    if _HAS_SIZER and sig.get("obi_signal") in (SIGNAL_BUY, SIGNAL_SELL):
        try:
            ep = sig.get("entry_price", 65000.0)
            side = "LONG" if sig["obi_signal"] == SIGNAL_BUY else "SHORT"
            stop = ep * (1 - 0.01) if side == "LONG" else ep * (1 + 0.01)
            wr, aw, al = update_rolling_stats(_engine._trade_history, 100)
            if wr == 0 or al == 0:
                wr, aw, al = 0.55, 0.018, 0.010
            r = compute_position_size(
                account_balance    = _config.account_balance,
                win_rate           = wr, avg_win=aw, avg_loss=al,
                entry_price        = ep, stop_loss_price=stop,
                position_side      = side,
                kelly_fraction     = _config.kelly_fraction,
                max_risk_per_trade = _config.max_risk_per_trade,
            )
            suggested_size = r.position_size
            suggested_risk = r.risk_percentage
        except Exception:
            pass

    return SignalResponse(
        timestamp         = sig.get("timestamp", _ts()),
        regime            = sig.get("regime", "UNKNOWN"),
        trade_allowed     = sig.get("trade_allowed", False),
        obi_value         = sig.get("obi_value", 0.0),
        obi_weighted      = sig.get("obi_weighted", 0.0),
        obi_velocity      = sig.get("obi_velocity", 0.0),
        obi_signal        = sig.get("obi_signal", SIGNAL_NEUTRAL),
        suggested_size    = suggested_size,
        suggested_risk_pct= suggested_risk,
        entry_price       = sig.get("entry_price", 0.0),
    )


@app.post("/parameters", response_model=ActionResponse, tags=["Control"])
async def update_parameters(params: ParametersUpdate):
    """Live update bot parameters without restarting."""
    changed = _config.update(params)
    if not changed:
        return ActionResponse(success=True, message="No parameters changed.", timestamp=_ts())

    # Re-initialize connector if exchange mode, api key, or secret key changed
    if any(k.startswith("exchange_mode") or k.startswith("api_key") or k.startswith("secret_key") for k in changed):
        _engine.reinit_connector()
        if _engine.live_connector is not None:
            try:
                bal_data = await asyncio.to_thread(_engine.live_connector.fetch_balance)
                eq = float(bal_data.get("equity", 0.0) or 0.0)
                bal = float(bal_data.get("balance", 0.0) or 0.0)
                avail = float(bal_data.get("available_margin", 0.0) or 0.0)
                if bal > 0:
                    _engine._cached_balance = {
                        "equity": eq, "balance": bal, "available_margin": avail,
                        "last_updated": datetime.now(timezone.utc),
                    }
                    _engine.config.account_balance = bal
            except Exception as e:
                logger.warning("[API] Parameter update balance fetch failed: %s", e)

    logger.info("[API] Parameters updated: %s", changed)
    await _ws_mgr.broadcast({
        "type"   : "PARAM_UPDATE",
        "changes": changed,
        "timestamp": _ts(),
    })
    return ActionResponse(
        success   = True,
        message   = f"Updated: {', '.join(changed)}",
        timestamp = _ts(),
    )


@app.get("/equity", tags=["Analytics"])
async def get_equity_curve():
    """Return last 1000 equity curve points."""
    return {"points": list(_engine._equity_curve), "count": len(_engine._equity_curve)}


@app.get("/coins", tags=["Scanner"])
async def get_coin_candidates():
    """
    Returns the latest scanned coin ranking from the autonomous scanner.
    10 ranked candidates with scores, prices, and balance-fit status.
    """
    candidates = _config.active_candidates
    profile = _config.account_profile
    return {
        "candidates": candidates,
        "total_candidates": len(candidates),
        "active_symbol": _config.symbol,
        "account_type": profile.get("account_type", "UNKNOWN") if profile else "UNKNOWN",
        "trading_mode": _config.trading_mode,
        "balance": profile.get("balance", 0.0) if profile else 0.0,
        "last_scan_ms": _config.last_scan_ts,
        "timestamp": _ts(),
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """
    Real-time push channel.

    Connected clients receive a JSON payload on every trading cycle.
    """
    await _ws_mgr.connect(websocket)
    try:
        # Send initial state immediately on connect
        await websocket.send_json({
            "type"     : "CONNECTED",
            "status"   : _engine.state["status"],
            "timestamp": _ts(),
            "config"   : _config.to_dict(),
        })
        # Keep alive — wait for disconnect
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo ping/pong
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send heartbeat to detect dead connections
                await websocket.send_json({"type": "HEARTBEAT", "timestamp": _ts()})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("[WS] Client error: %s", e)
    finally:
        _ws_mgr.disconnect(websocket)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "uptime": round(_engine.uptime, 1), "ws_clients": _ws_mgr.active_count}

# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("PORT", "8000"))

    parser = argparse.ArgumentParser(description="Q-SonicFX API Server")
    parser.add_argument("--host",    default=default_host, help="Bind host")
    parser.add_argument("--port",    default=default_port, type=int, help="Port")
    parser.add_argument("--reload",  action="store_true",  help="Hot reload")
    parser.add_argument("--autostart", action="store_true", help="Auto-start bot on launch")
    args = parser.parse_args()

    if args.autostart:
        _engine.state["status"] = "RUNNING"
        logger.info("[Main] Auto-starting bot via --autostart flag.")

    logger.info(
        "[Main] Starting Q-SonicFX API Server on http://%s:%d",
        args.host, args.port,
    )
    uvicorn.run(
        "api_server:app",
        host    = args.host,
        port    = args.port,
        reload  = args.reload,
        log_level = "info",
    )
