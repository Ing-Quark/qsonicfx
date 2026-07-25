#!/usr/bin/env python3
"""
circuit_breaker.py
==================
Q-SonicFX — Hard Risk Limits & Kill-Switch (Circuit Breaker)
=============================================================

Implements institutional-grade failsafe controls for live trading:

    - Daily P&L loss limit   (hard stop at configurable % of equity)
    - Trailing drawdown limit (peak-to-trough guard)
    - 24-hour cooldown after emergency halt
    - Automatic kill-switch: cancel all orders + close all positions
    - @circuit_breached decorator for any trading function
    - Auto-reset at new daily session

Architecture
------------
    CircuitBreaker  — singleton stateful risk monitor
    ExchangeAPI     — stub interface (swap in real implementation)
    circuit_breached — decorator factory for trading functions
    CircuitBreakerException — raised when trading is halted

Author : Q-SonicFX Quant Engine
"""
from __future__ import annotations

import functools
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, TypeVar

# ---------------------------------------------------------------------------
# Logging setup — structured ISO-8601 format
# ---------------------------------------------------------------------------

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%dT%H:%M:%S"

logging.basicConfig(level=logging.INFO, format=_FMT, datefmt=_DATE_FMT)
logger = logging.getLogger("qsonicfx.circuit_breaker")

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HaltReason(str, Enum):
    """Enumerated halt conditions returned by check_failsafe()."""
    DAILY_LOSS_LIMIT       = "DAILY_LOSS_LIMIT"
    TRAILING_DRAWDOWN_LIMIT = "TRAILING_DRAWDOWN_LIMIT"
    COOLDOWN_PERIOD        = "COOLDOWN_PERIOD"
    NORMAL                 = "NORMAL"


class ShutdownStep(str, Enum):
    """Steps executed during emergency_shutdown()."""
    CANCEL_ORDERS    = "CANCEL_ORDERS"
    CLOSE_POSITIONS  = "CLOSE_POSITIONS"
    SET_COOLDOWN     = "SET_COOLDOWN"
    WRITE_ALERT_LOG  = "WRITE_ALERT_LOG"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class CircuitBreakerException(Exception):
    """
    Raised by the @circuit_breached decorator when trading is halted.

    Attributes
    ----------
    reason         : HaltReason  Why trading is blocked.
    halted_until   : datetime    When trading may resume (UTC).
    state_snapshot : dict        Full CircuitBreaker state at time of raise.
    """

    def __init__(
        self,
        reason        : HaltReason,
        halted_until  : Optional[datetime],
        state_snapshot: Dict[str, Any],
    ) -> None:
        self.reason         = reason
        self.halted_until   = halted_until
        self.state_snapshot = state_snapshot
        resume = halted_until.isoformat() if halted_until else "N/A"
        super().__init__(
            f"CircuitBreaker HALTED | reason={reason.value} | resume_at={resume}"
        )


# ---------------------------------------------------------------------------
# Exchange API stub
# ---------------------------------------------------------------------------

class ExchangeAPI:
    """
    Stub exchange interface.

    Replace the body of each method with real exchange SDK calls
    (e.g. Binance, Bybit, Interactive Brokers) when going live.

    All methods return a result dict for audit logging.
    """

    def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Cancel every open order on the exchange.

        Returns
        -------
        dict
            ``{'action': 'CANCEL_ALL_ORDERS', 'status': 'OK', 'cancelled': int}``
        """
        logger.warning(
            "[ExchangeAPI] cancel_all_orders() called — cancelling all open orders."
        )
        # ── Real implementation: exchange.cancel_all_open_orders() ──────
        result = {
            "action"   : ShutdownStep.CANCEL_ORDERS.value,
            "status"   : "OK",
            "cancelled": 0,  # stub: 0 real orders
            "timestamp": datetime.now().isoformat(),
        }
        logger.info("[ExchangeAPI] All orders cancelled: %s", result)
        return result

    def close_all_positions(self, market: bool = True) -> Dict[str, Any]:
        """
        Close every open position using market orders.

        Parameters
        ----------
        market : bool  If True, use market orders (guarantees fill, not price).

        Returns
        -------
        dict
            ``{'action': 'CLOSE_ALL_POSITIONS', 'status': 'OK', 'closed': int}``
        """
        order_type = "MARKET" if market else "LIMIT"
        logger.warning(
            "[ExchangeAPI] close_all_positions(market=%s) called — "
            "flattening all positions via %s orders.",
            market, order_type,
        )
        # ── Real implementation: exchange.close_position_market() ───────
        result = {
            "action"    : ShutdownStep.CLOSE_POSITIONS.value,
            "status"    : "OK",
            "closed"    : 0,  # stub: 0 real positions
            "order_type": order_type,
            "timestamp" : datetime.now().isoformat(),
        }
        logger.info("[ExchangeAPI] All positions closed: %s", result)
        return result


# ---------------------------------------------------------------------------
# CircuitBreaker — stateful singleton
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Stateful singleton risk monitor with automatic kill-switch.

    Tracks daily P&L, trailing drawdown, and cooldown periods.
    Enforces hard limits via check_failsafe() and emergency_shutdown().

    Parameters
    ----------
    initial_balance    : float  Starting equity in quote currency.
    max_daily_loss_pct : float  Daily loss limit as fraction (default 0.05 = 5%).
    max_drawdown_pct   : float  Peak-to-trough limit as fraction (default 0.10 = 10%).
    exchange_api       : ExchangeAPI  Exchange interface (real or stub).
    alert_log_dir      : str    Directory for emergency alert log files.

    Notes
    -----
    Singleton: only one CircuitBreaker can be active at a time.
    Use ``CircuitBreaker.get_instance()`` after initial construction.
    Use ``CircuitBreaker.reset_singleton()`` in tests to get a fresh instance.
    """

    _instance  : ClassVar[Optional["CircuitBreaker"]] = None
    _initialized: bool = False

    # ── Singleton machinery ────────────────────────────────────────────

    def __new__(cls, *args: Any, **kwargs: Any) -> "CircuitBreaker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        initial_balance   : float,
        max_daily_loss_pct: float         = 0.05,
        max_drawdown_pct  : float         = 0.10,
        exchange_api      : Optional[ExchangeAPI] = None,
        alert_log_dir     : str           = ".",
    ) -> None:
        if self._initialized:
            return   # singleton: skip re-init on subsequent calls

        if initial_balance <= 0:
            raise ValueError(f"initial_balance must be > 0, got {initial_balance}")

        self.initial_balance      : float            = float(initial_balance)
        self.max_daily_loss_pct   : float            = float(max_daily_loss_pct)
        self.max_drawdown_pct     : float            = float(max_drawdown_pct)
        self.exchange_api         : ExchangeAPI      = exchange_api or ExchangeAPI()
        self.alert_log_dir        : str              = alert_log_dir

        # Session-reset fields
        self._session_date        : date             = date.today()
        self.initial_daily_balance: float            = float(initial_balance)

        # Live P&L tracking
        self.daily_realized_pnl   : float            = 0.0
        self.daily_unrealized_pnl : float            = 0.0
        self.current_balance      : float            = float(initial_balance)
        self.peak_balance         : float            = float(initial_balance)

        # Derived drawdown metrics (updated on every update_pnl call)
        self.daily_drawdown_pct   : float            = 0.0
        self.trailing_drawdown_pct: float            = 0.0

        # Halt state
        self.halted_until         : Optional[datetime] = None
        self._is_emergency_halted : bool             = False

        # Order / position tracking (populated by your execution layer)
        self.orders               : List[Dict]       = []
        self.positions            : List[Dict]       = []

        # P&L history for reporting
        self._pnl_log             : List[Dict]       = []

        self._initialized = True

        logger.info(
            "[CB] Initialized | balance=$%.2f | daily_limit=%.1f%% | "
            "drawdown_limit=%.1f%%",
            self.initial_balance,
            self.max_daily_loss_pct * 100,
            self.max_drawdown_pct * 100,
        )

    # ── Singleton access ───────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "CircuitBreaker":
        """Return the active singleton instance."""
        if cls._instance is None or not cls._instance._initialized:
            raise RuntimeError(
                "CircuitBreaker has not been initialized. "
                "Call CircuitBreaker(initial_balance=...) first."
            )
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        """Destroy the singleton (use in tests only)."""
        cls._instance    = None
        cls._initialized = False   # type: ignore[attr-defined]
        logger.debug("[CB] Singleton reset.")

    # ── Daily session reset ────────────────────────────────────────────

    def maybe_daily_reset(self) -> bool:
        """
        Reset daily P&L counters at the start of a new calendar day.

        Called automatically by ``update_pnl`` and ``check_failsafe``.
        Clears ``halted_until`` only if the cooldown has fully expired.

        Returns
        -------
        bool  True if a reset was performed, False otherwise.
        """
        today = date.today()
        if today == self._session_date:
            return False

        logger.info(
            "[CB] New session detected (%s). Resetting daily counters. "
            "Previous session: realized=$%.2f unrealized=$%.2f",
            today.isoformat(),
            self.daily_realized_pnl,
            self.daily_unrealized_pnl,
        )

        self._session_date         = today
        self.initial_daily_balance = self.current_balance
        self.daily_realized_pnl    = 0.0
        self.daily_unrealized_pnl  = 0.0
        self.peak_balance          = self.current_balance
        self.daily_drawdown_pct    = 0.0
        self.trailing_drawdown_pct = 0.0

        # Clear cooldown only if it has expired
        if self.halted_until is not None and datetime.now() >= self.halted_until:
            logger.info(
                "[CB] Cooldown expired (%s). Clearing halt. Trading may resume.",
                self.halted_until.isoformat(),
            )
            self.halted_until          = None
            self._is_emergency_halted  = False

        return True

    # ── P&L update ────────────────────────────────────────────────────

    def update_pnl(
        self,
        realized_pnl  : float,
        unrealized_pnl: float,
    ) -> None:
        """
        Update P&L state after every trade close or mark-to-market tick.

        Parameters
        ----------
        realized_pnl   : float  P&L of just-closed trade (signed, quote currency).
        unrealized_pnl : float  Current mark-to-market of all open positions.

        Side effects
        ------------
        Updates ``daily_realized_pnl``, ``daily_unrealized_pnl``,
        ``current_balance``, ``peak_balance``, ``daily_drawdown_pct``,
        and ``trailing_drawdown_pct``.
        """
        self.maybe_daily_reset()

        self.daily_realized_pnl   += realized_pnl
        self.daily_unrealized_pnl  = unrealized_pnl
        self.current_balance = (
            self.initial_balance
            + self.daily_realized_pnl
            + self.daily_unrealized_pnl
        )
        self.peak_balance = max(self.peak_balance, self.current_balance)

        # Daily drawdown from start-of-session balance
        if self.initial_daily_balance != 0.0:
            self.daily_drawdown_pct = (
                (self.current_balance - self.initial_daily_balance)
                / self.initial_daily_balance
            )

        # Trailing drawdown from intraday peak
        if self.peak_balance != 0.0:
            self.trailing_drawdown_pct = (
                (self.current_balance - self.peak_balance)
                / self.peak_balance
            )

        self._pnl_log.append({
            "ts"              : datetime.now().isoformat(),
            "realized"        : realized_pnl,
            "unrealized"      : unrealized_pnl,
            "current_balance" : self.current_balance,
            "peak_balance"    : self.peak_balance,
            "daily_dd_pct"    : round(self.daily_drawdown_pct * 100, 4),
            "trailing_dd_pct" : round(self.trailing_drawdown_pct * 100, 4),
        })

        logger.debug(
            "[CB] PnL update | realized=$%.2f unrealized=$%.2f balance=$%.2f "
            "daily_dd=%.3f%% trail_dd=%.3f%%",
            realized_pnl, unrealized_pnl, self.current_balance,
            self.daily_drawdown_pct * 100,
            self.trailing_drawdown_pct * 100,
        )

    # ── Failsafe check ─────────────────────────────────────────────────

    def check_failsafe(self) -> Tuple[bool, HaltReason]:
        """
        Evaluate all halt conditions and return the current status.

        Returns
        -------
        Tuple[bool, HaltReason]
            ``(halted, reason)`` where halted is True if any limit is breached.

        Priority order
        --------------
        1. COOLDOWN_PERIOD (explicit halt from emergency_shutdown)
        2. DAILY_LOSS_LIMIT
        3. TRAILING_DRAWDOWN_LIMIT
        """
        self.maybe_daily_reset()

        # ── Cooldown check ─────────────────────────────────────────────
        if self.halted_until is not None:
            if datetime.now() < self.halted_until:
                logger.debug(
                    "[CB] HALTED: cooldown active until %s",
                    self.halted_until.isoformat(),
                )
                return True, HaltReason.COOLDOWN_PERIOD
            else:
                # Cooldown expired — clear halt if on same day (daily reset handles cross-day)
                logger.info(
                    "[CB] Cooldown expired. Clearing halt. Trading may resume."
                )
                self.halted_until         = None
                self._is_emergency_halted = False

        # ── Daily loss limit ───────────────────────────────────────────
        if self.daily_drawdown_pct <= -self.max_daily_loss_pct:
            return True, HaltReason.DAILY_LOSS_LIMIT

        # ── Trailing drawdown limit ────────────────────────────────────
        if self.trailing_drawdown_pct <= -self.max_drawdown_pct:
            return True, HaltReason.TRAILING_DRAWDOWN_LIMIT

        return False, HaltReason.NORMAL

    # ── Emergency shutdown ─────────────────────────────────────────────

    def emergency_shutdown(self) -> Dict[str, Any]:
        """
        Execute the full kill-switch sequence.

        Steps
        -----
        1. Cancel all open orders via exchange_api.
        2. Close all positions via market orders.
        3. Set 24-hour cooldown (halted_until).
        4. Write emergency alert to ``emergency_alert_{date}.log``.

        Returns
        -------
        dict  Summary of actions taken, suitable for audit logs.
        """
        if self._is_emergency_halted:
            logger.warning(
                "[CB] emergency_shutdown() called but already halted. Skipping."
            )
            return {"status": "ALREADY_HALTED", "halted_until": self.halted_until.isoformat()}

        logger.critical(
            "[CB] *** EMERGENCY SHUTDOWN TRIGGERED *** | "
            "balance=$%.2f daily_dd=%.3f%% trail_dd=%.3f%%",
            self.current_balance,
            self.daily_drawdown_pct * 100,
            self.trailing_drawdown_pct * 100,
        )

        actions: Dict[str, Any] = {
            "timestamp" : datetime.now().isoformat(),
            "steps"     : [],
        }

        # ── Step 1: Cancel all orders ──────────────────────────────────
        try:
            cancel_result = self.exchange_api.cancel_all_orders()
            actions["steps"].append(cancel_result)
        except Exception as exc:
            logger.error("[CB] cancel_all_orders() failed: %s", exc)
            actions["steps"].append({
                "action": ShutdownStep.CANCEL_ORDERS.value,
                "status": "ERROR",
                "error" : str(exc),
            })

        # ── Step 2: Close all positions ────────────────────────────────
        try:
            close_result = self.exchange_api.close_all_positions(market=True)
            actions["steps"].append(close_result)
        except Exception as exc:
            logger.error("[CB] close_all_positions() failed: %s", exc)
            actions["steps"].append({
                "action": ShutdownStep.CLOSE_POSITIONS.value,
                "status": "ERROR",
                "error" : str(exc),
            })

        # ── Step 3: Set 24-hour cooldown ───────────────────────────────
        self.halted_until         = datetime.now() + timedelta(hours=24)
        self._is_emergency_halted = True
        actions["steps"].append({
            "action"      : ShutdownStep.SET_COOLDOWN.value,
            "status"      : "OK",
            "halted_until": self.halted_until.isoformat(),
        })

        logger.critical(
            "[CB] Trading HALTED for 24 hours until %s",
            self.halted_until.isoformat(),
        )

        # ── Step 4: Write emergency alert log ─────────────────────────
        state_snapshot = self._state_snapshot()
        actions["state_snapshot"] = state_snapshot

        alert_path = self._write_alert_log(state_snapshot, actions)
        actions["steps"].append({
            "action"    : ShutdownStep.WRITE_ALERT_LOG.value,
            "status"    : "OK",
            "alert_file": str(alert_path),
        })

        actions["status"] = "HALTED"
        return actions

    # ── State snapshot ─────────────────────────────────────────────────

    def _state_snapshot(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the current CB state."""
        return {
            "timestamp"            : datetime.now().isoformat(),
            "initial_balance"      : self.initial_balance,
            "initial_daily_balance": self.initial_daily_balance,
            "current_balance"      : self.current_balance,
            "peak_balance"         : self.peak_balance,
            "daily_realized_pnl"   : self.daily_realized_pnl,
            "daily_unrealized_pnl" : self.daily_unrealized_pnl,
            "daily_drawdown_pct"   : round(self.daily_drawdown_pct * 100, 4),
            "trailing_drawdown_pct": round(self.trailing_drawdown_pct * 100, 4),
            "max_daily_loss_pct"   : self.max_daily_loss_pct * 100,
            "max_drawdown_pct"     : self.max_drawdown_pct * 100,
            "halted_until"         : self.halted_until.isoformat() if self.halted_until else None,
            "session_date"         : self._session_date.isoformat(),
            "open_orders"          : len(self.orders),
            "open_positions"       : len(self.positions),
        }

    def _write_alert_log(
        self,
        state   : Dict[str, Any],
        actions : Dict[str, Any],
    ) -> Path:
        """Write emergency alert data to a dated log file."""
        filename  = f"emergency_alert_{date.today().isoformat()}.log"
        filepath  = Path(self.alert_log_dir) / filename

        payload = {
            "alert_type"    : "EMERGENCY_SHUTDOWN",
            "state_snapshot": state,
            "shutdown_actions": actions,
        }

        try:
            with filepath.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, indent=2))
                fh.write("\n" + "=" * 80 + "\n")
            logger.info("[CB] Emergency alert written to %s", filepath)
        except OSError as exc:
            logger.error("[CB] Failed to write alert log: %s", exc)

        return filepath

    # ── Reporting helpers ──────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable status string."""
        halted, reason = self.check_failsafe()
        status = f"HALTED ({reason.value})" if halted else "NORMAL"
        resume = self.halted_until.isoformat() if self.halted_until else "N/A"
        return (
            f"CircuitBreaker [{status}] | "
            f"balance=${self.current_balance:,.2f} | "
            f"daily_dd={self.daily_drawdown_pct*100:.3f}% | "
            f"trail_dd={self.trailing_drawdown_pct*100:.3f}% | "
            f"resume={resume}"
        )

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Global singleton accessor
# ---------------------------------------------------------------------------

_global_cb: Optional[CircuitBreaker] = None


def set_global_circuit_breaker(cb: CircuitBreaker) -> None:
    """
    Register *cb* as the process-wide circuit breaker used by the decorator.

    Parameters
    ----------
    cb : CircuitBreaker
    """
    global _global_cb
    _global_cb = cb


def get_global_circuit_breaker() -> CircuitBreaker:
    """
    Return the global CircuitBreaker, falling back to the singleton.

    Raises
    ------
    RuntimeError  If no circuit breaker has been initialised.
    """
    if _global_cb is not None:
        return _global_cb
    return CircuitBreaker.get_instance()


# ---------------------------------------------------------------------------
# @circuit_breached decorator
# ---------------------------------------------------------------------------

def circuit_breached(
    func: Optional[F] = None,
    *,
    cb                      : Optional[CircuitBreaker] = None,
    pnl_from_result         : bool = True,
    trigger_shutdown_on_error: bool = True,
) -> Any:
    """
    Decorator / decorator-factory that enforces circuit breaker rules.

    Usage
    -----
    ::

        @circuit_breached
        def execute_trade(symbol, qty, side):
            ...
            return {'realized_pnl': 150.0, 'unrealized_pnl': 0.0}

        @circuit_breached(cb=my_cb, trigger_shutdown_on_error=False)
        def place_order(...): ...

    Behaviour
    ---------
    **Before** function call:
      - Calls ``check_failsafe()``.
      - If halted, raises ``CircuitBreakerException`` immediately.

    **After** function returns:
      - If ``pnl_from_result=True`` and the return value is a ``dict``
        containing ``'realized_pnl'`` and/or ``'unrealized_pnl'``,
        calls ``update_pnl()`` with those values.

    **On exception inside wrapped function**:
      - If ``trigger_shutdown_on_error=True`` (default), calls
        ``emergency_shutdown()`` before re-raising the original exception.
      - ``CircuitBreakerException`` is always re-raised without shutdown.

    Parameters
    ----------
    func                     : Callable, optional  Function to wrap (None when called as factory).
    cb                       : CircuitBreaker, optional  Explicit CB instance.
    pnl_from_result          : bool  Auto-extract PnL from function return value (default True).
    trigger_shutdown_on_error: bool  Call emergency_shutdown() on unhandled exceptions (default True).
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            instance = cb or get_global_circuit_breaker()

            # ── Pre-call: check failsafe ───────────────────────────────
            halted, reason = instance.check_failsafe()
            if halted:
                snapshot = instance._state_snapshot()
                logger.warning(
                    "[CB] @circuit_breached BLOCKED call to '%s' | reason=%s",
                    f.__name__, reason.value,
                )
                raise CircuitBreakerException(
                    reason         = reason,
                    halted_until   = instance.halted_until,
                    state_snapshot = snapshot,
                )

            # ── Execute wrapped function ───────────────────────────────
            try:
                result = f(*args, **kwargs)
            except CircuitBreakerException:
                raise   # never swallow CB exceptions
            except Exception as exc:
                logger.error(
                    "[CB] Unhandled exception in '%s': %s — "
                    "trigger_shutdown=%s",
                    f.__name__, exc, trigger_shutdown_on_error,
                )
                if trigger_shutdown_on_error:
                    instance.emergency_shutdown()
                raise

            # ── Post-call: extract PnL if present ─────────────────────
            if pnl_from_result and isinstance(result, dict):
                realized   = float(result.get("realized_pnl",   0.0))
                unrealized = float(result.get("unrealized_pnl", 0.0))
                if realized != 0.0 or unrealized != 0.0:
                    instance.update_pnl(realized, unrealized)

            return result

        return wrapper  # type: ignore[return-value]

    # Support both @circuit_breached and @circuit_breached(...)
    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# __main__ — demo / test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import time

    # ── Clean console output (no duplicate handlers) ───────────────────
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format=_FMT, datefmt=_DATE_FMT)

    BALANCE      = 10_000.0
    N_TRADES     = 50
    LOSS_RATE    = 0.40     # 40% of trades are losses
    WIN_AMOUNT   = 120.0    # $120 avg win
    LOSS_AMOUNT  = 200.0    # $200 avg loss (simulates aggressive losses)
    SEED         = 7

    rng = random.Random(SEED)

    # ── 1. Initialize circuit breaker ──────────────────────────────────
    CircuitBreaker.reset_singleton()
    cb = CircuitBreaker(
        initial_balance    = BALANCE,
        max_daily_loss_pct = 0.05,    # 5% daily loss limit
        max_drawdown_pct   = 0.10,    # 10% trailing drawdown limit
        alert_log_dir      = ".",
    )
    set_global_circuit_breaker(cb)

    # ── 2. Define a guarded trade function ─────────────────────────────
    @circuit_breached
    def execute_trade(trade_id: int, pnl_dollars: float) -> Dict:
        """Simulated trade execution — returns PnL dict for CB to absorb."""
        return {
            "trade_id"      : trade_id,
            "realized_pnl"  : pnl_dollars,
            "unrealized_pnl": 0.0,
        }

    @circuit_breached
    def place_order(symbol: str, qty: float, side: str) -> Dict:
        """Simulated order placement."""
        return {"order_id": f"ORD-{rng.randint(1000, 9999)}", "status": "FILLED"}

    @circuit_breached(trigger_shutdown_on_error=False)
    def adjust_sl(trade_id: int, new_sl: float) -> Dict:
        """Adjust stop-loss — does not trigger shutdown on error."""
        return {"trade_id": trade_id, "new_sl": new_sl, "status": "OK"}

    # ── 3. Run simulated trade loop ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Q-SonicFX  |  Circuit Breaker Demo")
    print(f"  Balance: ${BALANCE:,.0f}  |  {N_TRADES} trades  |  Loss rate: {LOSS_RATE*100:.0f}%")
    print(f"  Daily limit: {cb.max_daily_loss_pct*100:.0f}%  |  "
          f"Drawdown limit: {cb.max_drawdown_pct*100:.0f}%")
    print(f"{'='*70}\n")

    blocked_count   = 0
    shutdown_triggered = False

    for i in range(1, N_TRADES + 1):
        is_loss   = rng.random() < LOSS_RATE
        pnl       = -(LOSS_AMOUNT * rng.uniform(0.7, 1.3)) if is_loss \
                    else (WIN_AMOUNT * rng.uniform(0.8, 1.2))
        pnl_round = round(pnl, 2)

        try:
            result = execute_trade(trade_id=i, pnl_dollars=pnl_round)
            sign   = "+" if pnl_round >= 0 else ""
            print(
                f"  Trade {i:>3} | {'LOSS' if is_loss else 'WIN ':4} | "
                f"PnL={sign}${pnl_round:>8.2f} | "
                f"Balance=${cb.current_balance:>9.2f} | "
                f"Daily_DD={cb.daily_drawdown_pct*100:>6.3f}% | "
                f"Peak=${cb.peak_balance:>9.2f}"
            )

            # Also test place_order and adjust_sl every 5 trades
            if i % 5 == 0:
                place_order("BTCUSDT", 0.01, "BUY")
                adjust_sl(i, 64_800.0)

        except CircuitBreakerException as e:
            blocked_count += 1
            if blocked_count == 1:
                # First block — show detailed state
                print(f"\n{'!'*70}")
                print(f"  CIRCUIT BREAKER TRIGGERED on trade {i}!")
                print(f"  Reason    : {e.reason.value}")
                print(f"  Balance   : ${e.state_snapshot['current_balance']:,.2f}")
                print(f"  Daily DD  : {e.state_snapshot['daily_drawdown_pct']:.3f}%")
                print(f"  Trail DD  : {e.state_snapshot['trailing_drawdown_pct']:.3f}%")
                resume = e.state_snapshot.get("halted_until") or "N/A"
                print(f"  Resume at : {resume}")
                print(f"{'!'*70}\n")
            else:
                # Subsequent blocks — short line
                print(
                    f"  Trade {i:>3} | BLOCKED [{e.reason.value}] — "
                    f"decorator prevented execution"
                )

    # ── 4. Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Trades attempted   : {N_TRADES}")
    print(f"  Trades executed    : {N_TRADES - blocked_count}")
    print(f"  Trades blocked     : {blocked_count}")
    print(f"  Final balance      : ${cb.current_balance:,.2f}")
    print(f"  Daily realized PnL : ${cb.daily_realized_pnl:,.2f}  "
          f"({cb.daily_drawdown_pct*100:.3f}%)")
    print(f"  Peak balance       : ${cb.peak_balance:,.2f}")
    print(f"  Trailing drawdown  : {cb.trailing_drawdown_pct*100:.3f}%")

    halted, reason = cb.check_failsafe()
    print(f"  CB Status          : {'HALTED' if halted else 'NORMAL'} ({reason.value})")
    if cb.halted_until:
        print(f"  Halted until       : {cb.halted_until.isoformat()}")

    # ── 5. Demonstrate check_failsafe states ───────────────────────────
    print(f"\n{'='*70}")
    print(f"  check_failsafe() states demonstration:")
    print(f"{'='*70}\n")

    # Force emergency shutdown for display
    if not cb._is_emergency_halted:
        print("  Triggering emergency_shutdown() manually...")
        shutdown_summary = cb.emergency_shutdown()
        print(f"  Shutdown steps: {len(shutdown_summary['steps'])}")
        for step in shutdown_summary["steps"]:
            print(f"    -> {step.get('action','?')}: {step.get('status','?')}"
                  + (f" | {step.get('alert_file','')}" if 'alert_file' in step else ""))

    halted, reason = cb.check_failsafe()
    print(f"\n  check_failsafe() now: halted={halted}, reason={reason.value}")
    print(f"  halted_until: {cb.halted_until.isoformat() if cb.halted_until else 'N/A'}")

    # Show decorator blocking after halt
    print(f"\n  Attempting 3 more trades after halt (all should be blocked):")
    for i in range(3):
        try:
            execute_trade(trade_id=999 + i, pnl_dollars=100.0)
            print(f"    Trade {999+i}: EXECUTED (should not happen)")
        except CircuitBreakerException as e:
            print(f"    Trade {999+i}: BLOCKED by @circuit_breached | {e.reason.value}")

    print(f"\n{'='*70}\n")
