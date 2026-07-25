#!/usr/bin/env python3
"""
regime_detector.py
==================
Q-SonicFX - Real-Time Market Regime Detector

Classifies the current market regime using three independent lenses:
  1. ADX (Wilder smoothing)       -- trend strength & direction
  2. ATR percentile bands         -- volatility regime
  3. Volume + spread profile      -- liquidity quality filter

Regimes returned:
    STRONG_TREND          - strong directional momentum, tradeable
    WEAK_TREND            - emerging trend, tradeable with caution
    RANGING               - sideways/compressed, skip (no trade)
    VOLATILITY_EXPANSION  - coiling breakout potential, tradeable
    LOW_LIQUIDITY_PAUSE   - thin/manipulated market, BLOCKED
    INSUFFICIENT_DATA     - warm-up period, BLOCKED

Usage
-----
    from regime_detector import RegimeDetector
    detector = RegimeDetector(period=20, confirm_timeframes=["5m", "15m"])
    result   = detector.detect(df)
    print(result)

Author : Q-SonicFX Quant Engine
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    """
    Immutable result of a single regime-detection call.

    Attributes
    ----------
    regime : str
        One of: STRONG_TREND, WEAK_TREND, RANGING,
        VOLATILITY_EXPANSION, LOW_LIQUIDITY_PAUSE, INSUFFICIENT_DATA.
    trade_allowed : bool
        False when regime is RANGING, LOW_LIQUIDITY_PAUSE, or INSUFFICIENT_DATA.
    adx_value : float
        ADX reading for the latest bar (0-100). NaN during warm-up.
    atr_percentile : float
        Percentile rank (0-100) of latest ATR within its 50-bar history.
    volume_percentile : float
        Percentile rank (0-100) of latest volume within its 50-bar history.
    primary_direction : str
        BULL when DI+ > DI-, BEAR when DI- > DI+, else NEUTRAL.
    """
    regime: str
    trade_allowed: bool
    adx_value: float
    atr_percentile: float
    volume_percentile: float
    primary_direction: str

    def __repr__(self) -> str:
        def _f(v: float) -> str:
            return f"{v:.2f}" if v == v else "nan"
        return (
            f"RegimeResult("
            f"regime={self.regime!r}, "
            f"trade_allowed={self.trade_allowed}, "
            f"adx={_f(self.adx_value)}, "
            f"atr_pct={_f(self.atr_percentile)}, "
            f"vol_pct={_f(self.volume_percentile)}, "
            f"direction={self.primary_direction!r})"
        )


# ---------------------------------------------------------------------------
# Low-level vectorised helpers
# ---------------------------------------------------------------------------

def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Apply Wilder Moving Average (RMA) to a 1-D float array.

    Seed  : simple mean of first `period` valid values.
    Update: result[i] = (result[i-1] * (period-1) + arr[i]) / period

    Parameters
    ----------
    arr    : np.ndarray  1-D float array, may have leading NaN.
    period : int         Smoothing period (>= 1).

    Returns
    -------
    np.ndarray  Same shape as arr; warm-up positions are NaN.
    """
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if period < 1 or n < period:
        return result

    valid_idx = np.where(~np.isnan(arr))[0]
    if len(valid_idx) < period:
        return result

    seed_end = int(valid_idx[period - 1])
    result[seed_end] = float(np.mean(arr[valid_idx[:period]]))

    k     = (period - 1) / period
    inv_k = 1.0 / period

    for i in range(seed_end + 1, n):
        v = arr[i]
        prev = result[i - 1]
        result[i] = prev if (v != v) else prev * k + v * inv_k

    return result


def _true_range(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """
    Compute True Range: max(H-L, |H-Cprev|, |L-Cprev|).
    Element 0 is NaN (no previous close).
    """
    n  = len(high)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = np.nan
    pc = close[:-1]
    h  = high[1:]
    l  = low[1:]
    tr[1:] = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return tr


def _directional_movement(
    high: np.ndarray, low: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute raw +DM and -DM (vectorised).
    +DM[i] = up_move  if up_move  > down_move and up_move  > 0, else 0
    -DM[i] = down_move if down_move > up_move  and down_move > 0, else 0
    Element 0 of both is NaN.
    """
    n         = len(high)
    plus_dm   = np.zeros(n, dtype=np.float64)
    minus_dm  = np.zeros(n, dtype=np.float64)
    up_move   = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_mask  = (up_move   > down_move) & (up_move   > 0.0)
    minus_mask = (down_move > up_move)   & (down_move > 0.0)
    plus_dm[1:]  = np.where(plus_mask,  up_move,   0.0)
    minus_dm[1:] = np.where(minus_mask, down_move, 0.0)
    plus_dm[0]   = np.nan
    minus_dm[0]  = np.nan
    return plus_dm, minus_dm


def _adx_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full ADX pipeline using Wilder smoothing.

    Steps
    -----
    1. TR, +DM, -DM  (vectorised)
    2. Wilder-smooth each -> sTR, s+DM, s-DM
    3. DI+ = 100 * s+DM / sTR
       DI- = 100 * s-DM / sTR
    4. DX  = 100 * |DI+ - DI-| / (DI+ + DI-)
    5. ADX = Wilder_smooth(DX, period)

    Returns
    -------
    Tuple[adx, di_plus, di_minus]  all same length as inputs.
    """
    tr       = _true_range(high, low, close)
    plus_dm, minus_dm = _directional_movement(high, low)

    s_tr       = _wilder_smooth(tr,       period)
    s_plus_dm  = _wilder_smooth(plus_dm,  period)
    s_minus_dm = _wilder_smooth(minus_dm, period)

    with np.errstate(invalid="ignore", divide="ignore"):
        di_plus  = np.where(s_tr > 0.0, 100.0 * s_plus_dm  / s_tr, 0.0)
        di_minus = np.where(s_tr > 0.0, 100.0 * s_minus_dm / s_tr, 0.0)
        di_sum   = di_plus + di_minus
        dx       = np.where(di_sum > 0.0,
                            100.0 * np.abs(di_plus - di_minus) / di_sum, 0.0)

    nan_mask = np.isnan(s_tr)
    di_plus[nan_mask]  = np.nan
    di_minus[nan_mask] = np.nan
    dx[nan_mask]       = np.nan

    adx = _wilder_smooth(dx, period)
    return adx, di_plus, di_minus


def _percentile_rank(value: float, history: np.ndarray) -> float:
    """
    Percentile rank (0-100) of `value` within `history`.
    Uses inclusive counting: fraction of history values <= value.
    NaN elements in history are ignored.
    """
    if value != value:
        return float("nan")
    valid = history[~np.isnan(history)]
    if len(valid) == 0:
        return float("nan")
    return float(np.sum(valid <= value) / len(valid) * 100.0)


def _normalise_tf(rule: str) -> str:
    """
    Normalise user timeframe strings to pandas resample aliases.
    '5m' -> '5min', '1h' stays '1h', '1D' stays '1D'.
    """
    rule = rule.strip()
    rule = re.sub(r"^(\d+)m$", r"\1min", rule)
    rule = re.sub(r"^(\d+)[Tt]$", r"\1min", rule)
    return rule


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV DataFrame to a higher timeframe using pandas .resample()."""
    return df.resample(rule).agg(
        open   = ("open",   "first"),
        high   = ("high",   "max"),
        low    = ("low",    "min"),
        close  = ("close",  "last"),
        volume = ("volume", "sum"),
    )


def _is_nan(v: float) -> bool:
    return v != v


def _insufficient() -> RegimeResult:
    return RegimeResult(
        regime            = "INSUFFICIENT_DATA",
        trade_allowed     = False,
        adx_value         = float("nan"),
        atr_percentile    = float("nan"),
        volume_percentile = float("nan"),
        primary_direction = "NEUTRAL",
    )


def _block(result: RegimeResult) -> RegimeResult:
    """Return a copy of result with trade_allowed forced False."""
    return RegimeResult(
        regime            = result.regime,
        trade_allowed     = False,
        adx_value         = result.adx_value,
        atr_percentile    = result.atr_percentile,
        volume_percentile = result.volume_percentile,
        primary_direction = result.primary_direction,
    )


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Real-time market regime detector for Q-SonicFX.

    Parameters
    ----------
    period : int
        Lookback for ADX and ATR calculations (default 20).
    history_period : int
        Rolling window for ATR/volume percentile ranking (default 50).
    confirm_timeframes : list of str, optional
        Higher timeframes for MTF confirmation e.g. ['5m', '15m'].
        trade_allowed is True only if ALL timeframes agree on a tradeable regime.
    """

    _BLOCKED: frozenset = frozenset(
        {"LOW_LIQUIDITY_PAUSE", "RANGING", "INSUFFICIENT_DATA"}
    )

    def __init__(
        self,
        period: int = 20,
        history_period: int = 50,
        confirm_timeframes: Optional[List[str]] = None,
    ) -> None:
        if period < 2:
            raise ValueError(f"period must be >= 2, got {period}.")
        if history_period < period:
            raise ValueError(
                f"history_period ({history_period}) must be >= period ({period})."
            )
        self.period = period
        self.history_period = history_period
        self.confirm_timeframes: List[str] = [
            _normalise_tf(tf) for tf in (confirm_timeframes or [])
        ]

    # ------------------------------------------------------------------
    def detect(
        self,
        df: pd.DataFrame,
        confirm_timeframes: Optional[List[str]] = None,
    ) -> RegimeResult:
        """
        Detect current market regime for the latest bar in df.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data. Required columns:
            ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
            timestamp may be a column or a DatetimeIndex. Any timeframe works.
        confirm_timeframes : list of str, optional
            Per-call override for MTF confirmation. Falls back to instance setting.

        Returns
        -------
        RegimeResult
        """
        ctf = (
            [_normalise_tf(tf) for tf in confirm_timeframes]
            if confirm_timeframes is not None
            else self.confirm_timeframes
        )

        df = self._prepare(df)

        min_rows = self.period * 2
        if len(df) < min_rows:
            logger.warning(
                "[RegimeDetector] Insufficient data: %d rows (need %d = period*2). "
                "Returning INSUFFICIENT_DATA.",
                len(df), min_rows,
            )
            return _insufficient()

        primary = self._detect_single(df)

        if not ctf or primary.regime in self._BLOCKED:
            return primary

        return self._apply_mtf(primary, df, ctf)

    # ------------------------------------------------------------------
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate, coerce dtypes, set DatetimeIndex, sort."""
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"[RegimeDetector] Missing required columns: {missing}"
            )
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "[RegimeDetector] DataFrame must have a DatetimeIndex "
                "or a 'timestamp' column."
            )
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_index()

    # ------------------------------------------------------------------
    def _detect_single(self, df: pd.DataFrame) -> RegimeResult:
        """
        Core regime classification on a validated, sorted DataFrame.

        Hot path - all numpy operations, minimal Python overhead.
        Typical runtime on 300 bars: << 1 ms.
        """
        n      = len(df)
        high   = df["high"].to_numpy(dtype=np.float64)
        low    = df["low"].to_numpy(dtype=np.float64)
        close  = df["close"].to_numpy(dtype=np.float64)
        volume = df["volume"].to_numpy(dtype=np.float64)

        # ── ADX / DI ──────────────────────────────────────────────────
        adx_arr, di_plus_arr, di_minus_arr = _adx_series(
            high, low, close, self.period
        )
        adx_val      = float(adx_arr[-1])
        di_plus_val  = float(di_plus_arr[-1])
        di_minus_val = float(di_minus_arr[-1])

        # ── ATR percentile ─────────────────────────────────────────────
        tr_arr      = _true_range(high, low, close)
        atr_arr     = _wilder_smooth(tr_arr, self.period)
        hist_start  = max(0, n - self.history_period)
        atr_history = atr_arr[hist_start: n - 1]      # exclude current bar
        atr_pct     = _percentile_rank(atr_arr[-1], atr_history)

        # ── Volume percentile ──────────────────────────────────────────
        vol_history = volume[hist_start: n - 1]
        vol_pct     = _percentile_rank(volume[-1], vol_history)

        # ── Spread % = (H-L)/C * 100 ───────────────────────────────────
        with np.errstate(invalid="ignore", divide="ignore"):
            spread = np.where(close > 0.0, (high - low) / close * 100.0, np.nan)
        spread_history = spread[hist_start: n - 1]
        spread_mean    = float(np.nanmean(spread_history)) if spread_history.size > 0 else float("nan")
        current_spread = float(spread[-1])

        # ── Guard: NaN ADX means warm-up incomplete ────────────────────
        if _is_nan(adx_val):
            return _insufficient()

        # ── Primary direction ──────────────────────────────────────────
        if di_plus_val > di_minus_val:
            direction = "BULL"
        elif di_minus_val > di_plus_val:
            direction = "BEAR"
        else:
            direction = "NEUTRAL"

        base = dict(
            adx_value         = adx_val,
            atr_percentile    = atr_pct,
            volume_percentile = vol_pct,
            primary_direction = direction,
        )

        # ══════════════════════════════════════════════════════════════
        # REGIME RULES  (in order of precedence)
        # ══════════════════════════════════════════════════════════════

        # Rule 0: LOW_LIQUIDITY_PAUSE - hard override, checked first
        # volume < 10th pct AND spread > 2x its own 50-period mean
        if (
            not _is_nan(vol_pct)
            and not _is_nan(current_spread)
            and not _is_nan(spread_mean)
            and vol_pct < 10.0
            and spread_mean > 0.0
            and current_spread > 2.0 * spread_mean
        ):
            return RegimeResult(regime="LOW_LIQUIDITY_PAUSE", trade_allowed=False, **base)

        di_spread = abs(di_plus_val - di_minus_val)

        # Rule 1: STRONG_TREND - ADX > 25 AND |DI+ - DI-| > 5
        if adx_val > 25.0 and di_spread > 5.0:
            return RegimeResult(regime="STRONG_TREND", trade_allowed=True, **base)

        # Rule 2: VOLATILITY_EXPANSION - ADX < 25 but ATR > 80th pct (coiling)
        if adx_val < 25.0 and not _is_nan(atr_pct) and atr_pct > 80.0:
            return RegimeResult(regime="VOLATILITY_EXPANSION", trade_allowed=True, **base)

        # Rule 3: WEAK_TREND - ADX between 20 and 25
        if 20.0 <= adx_val <= 25.0:
            return RegimeResult(regime="WEAK_TREND", trade_allowed=True, **base)

        # Rule 4: RANGING - ADX < 20, ATR normal/tight (default fallthrough)
        return RegimeResult(regime="RANGING", trade_allowed=False, **base)

    # ------------------------------------------------------------------
    def _apply_mtf(
        self,
        primary: RegimeResult,
        df: pd.DataFrame,
        ctf: List[str],
    ) -> RegimeResult:
        """
        Multi-timeframe confirmation pass.

        Returns primary unchanged if all higher TFs allow trading.
        Returns a blocked copy if any TF returns a blocked regime.
        """
        min_rows = self.period * 2

        for tf in ctf:
            try:
                tf_df = _resample_ohlcv(df, tf).dropna(subset=["close"])
            except Exception as exc:
                logger.error(
                    "[RegimeDetector] MTF resample to '%s' failed: %s. Blocking.", tf, exc
                )
                return _block(primary)

            if len(tf_df) < min_rows:
                logger.warning(
                    "[RegimeDetector] Only %d rows after resampling to '%s' "
                    "(need %d). Blocking trade as precaution.",
                    len(tf_df), tf, min_rows,
                )
                return _block(primary)

            tf_result = self._detect_single(tf_df)

            if tf_result.regime in self._BLOCKED:
                logger.info(
                    "[RegimeDetector] MTF block: '%s' -> regime '%s'. Trade blocked.",
                    tf, tf_result.regime,
                )
                return _block(primary)

        return primary


# ---------------------------------------------------------------------------
# Convenience one-shot function
# ---------------------------------------------------------------------------

def detect_regime(
    df: pd.DataFrame,
    period: int = 20,
    history_period: int = 50,
    confirm_timeframes: Optional[List[str]] = None,
) -> RegimeResult:
    """
    One-shot convenience wrapper around RegimeDetector.

    Parameters
    ----------
    df                 : OHLCV DataFrame with timestamp column or DatetimeIndex.
    period             : ADX / ATR period (default 20).
    history_period     : Window for percentile history (default 50).
    confirm_timeframes : Higher TFs for MTF confirmation e.g. ['5m', '15m'].

    Returns
    -------
    RegimeResult
    """
    return RegimeDetector(
        period=period,
        history_period=history_period,
        confirm_timeframes=confirm_timeframes,
    ).detect(df)


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------

def _generate_demo_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Synthesise a realistic 1-minute OHLCV DataFrame for demo / testing.

    Parameters
    ----------
    n    : int  Number of bars (default 300).
    seed : int  NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame  columns: [timestamp, open, high, low, close, volume]
    """
    rng = np.random.default_rng(seed)

    log_returns = rng.normal(0.0001, 0.002, n)
    close_arr   = 1800.0 * np.cumprod(np.exp(log_returns))
    body_half   = close_arr * rng.uniform(0.0005, 0.003, n)
    open_arr    = close_arr - body_half * rng.choice([-1, 1], n)
    wick_up     = close_arr * rng.uniform(0, 0.004, n)
    wick_down   = close_arr * rng.uniform(0, 0.004, n)
    high_arr    = np.maximum(open_arr, close_arr) + wick_up
    low_arr     = np.minimum(open_arr, close_arr) - wick_down
    base_vol    = rng.integers(100, 1000, n).astype(float)
    spike_mask  = rng.random(n) < 0.05
    if spike_mask.sum() > 0:
        base_vol[spike_mask] *= rng.uniform(5, 15, int(spike_mask.sum()))

    timestamps = pd.date_range(
        "2024-01-01 09:00", periods=n, freq="1min", tz="UTC"
    )
    return pd.DataFrame({
        "timestamp": timestamps,
        "open"     : np.round(open_arr,  2),
        "high"     : np.round(high_arr,  2),
        "low"      : np.round(low_arr,   2),
        "close"    : np.round(close_arr, 2),
        "volume"   : base_vol,
    })


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_banner(result: RegimeResult, elapsed_ms: float) -> None:
    ICONS = {
        "STRONG_TREND"        : "TRENDING UP",
        "WEAK_TREND"          : "WEAK TREND ",
        "RANGING"             : "RANGING    ",
        "VOLATILITY_EXPANSION": "VOL EXPAND ",
        "LOW_LIQUIDITY_PAUSE" : "LOW LIQ!   ",
        "INSUFFICIENT_DATA"   : "INSUFF DATA",
    }
    def _f(v: float, d: int = 2) -> str:
        return f"{v:.{d}f}" if v == v else "n/a"
    icon = ICONS.get(result.regime, "UNKNOWN    ")
    bar  = "=" * 54
    print(f"\n{bar}")
    print(f"  Q-SonicFX  |  Regime Detector")
    print(f"{bar}")
    print(f"  [{icon}]  Regime      : {result.regime}")
    print(f"  Trade Allowed        : {'YES' if result.trade_allowed else 'NO'}")
    print(f"  Primary Direction    : {result.primary_direction}")
    print(f"{bar}")
    print(f"  ADX Value            : {_f(result.adx_value)}")
    print(f"  ATR Percentile       : {_f(result.atr_percentile, 1)} / 100")
    print(f"  Volume Percentile    : {_f(result.volume_percentile, 1)} / 100")
    print(f"  Detection latency    : {elapsed_ms:.3f} ms")
    print(f"{bar}\n")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level  = logging.INFO,
        format = "[%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        print(f"[INFO] Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("[INFO] No CSV supplied - using 300-bar synthetic demo data.")
        df = _generate_demo_data(n=300)

    print(f"[INFO] {len(df)} bars loaded.  Columns: {list(df.columns)}")

    # -- Single TF --
    detector = RegimeDetector(period=20, history_period=50)
    t0 = time.perf_counter()
    result = detector.detect(df)
    elapsed = (time.perf_counter() - t0) * 1_000
    _print_banner(result, elapsed)

    # -- MTF confirmation demo --
    print("[INFO] MTF confirmation demo: primary=1m, confirm=[5min, 15min]")
    mtf_detector = RegimeDetector(
        period=20,
        history_period=50,
        confirm_timeframes=["5m", "15m"],
    )
    t0 = time.perf_counter()
    result_mtf = mtf_detector.detect(df)
    elapsed_mtf = (time.perf_counter() - t0) * 1_000
    print(f"  MTF Result  : {result_mtf}")
    print(f"  MTF Latency : {elapsed_mtf:.3f} ms\n")

    # -- Edge case: insufficient data --
    print("[INFO] Edge case - 10 rows (expect INSUFFICIENT_DATA):")
    small = detector.detect(df.head(10))
    print(f"  Result: {small}\n")
