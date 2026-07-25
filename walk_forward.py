#!/usr/bin/env python3
"""
walk_forward.py
===============
Q-SonicFX -- Walk-Forward Optimization (WFO) Framework
======================================================

Splits historical OHLCV data into sequential in-sample (training) and
out-of-sample (testing) windows, optimizes strategy parameters on each
training slice, validates on unseen test slices, and produces a full
overfitting assessment report.

Pipeline
--------
    Load CSV -> Slide Windows -> Grid/Random Search -> Vectorized Backtest
    -> Collect IS + OOS Results -> Overfitting Tests -> Report

Key design constraints
----------------------
- Pure pandas + numpy only (no Backtrader, no external backtesting lib)
- Vectorized backtester (no per-row Python loops)
- Chunked CSV reading for large files
- Random search fallback for large parameter grids (> 10,000 combos)
- Overfitting detected via IS/OOS Sharpe degradation and parameter instability

Author : Q-SonicFX Quant Engine
"""
from __future__ import annotations

import csv
import itertools
import logging
import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class InsufficientDataError(Exception):
    """Raised when the dataset has fewer rows than one full WFO window."""


class NoValidSignalsError(Exception):
    """Raised (and caught internally) when a param set produces zero trades."""


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Full result of a single vectorized backtest run.

    Attributes
    ----------
    sharpe_ratio  : float  Annualized Sharpe (risk-free = 0).
    sortino_ratio : float  Annualized Sortino (downside deviation only).
    max_drawdown  : float  Peak-to-trough drawdown as a negative %.
    win_rate      : float  Fraction of trades with positive return.
    profit_factor : float  Gross profit / gross loss.
    total_return  : float  Cumulative return over the window.
    num_trades    : int    Number of round-trip signal changes.
    equity_curve  : pd.Series  Normalized equity curve (starts at 1.0).
    drawdown_curve: pd.Series  Per-bar drawdown (negative values).
    """
    sharpe_ratio  : float
    sortino_ratio : float
    max_drawdown  : float
    win_rate      : float
    profit_factor : float
    total_return  : float
    num_trades    : int
    equity_curve  : pd.Series
    drawdown_curve: pd.Series

    def is_valid(self) -> bool:
        """Return True when the result has at least one trade."""
        return self.num_trades > 0

    def metric(self, name: str) -> float:
        """Look up a named scalar metric by string."""
        return float(getattr(self, name, 0.0))


# ---------------------------------------------------------------------------
# WalkForwardWindow
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardWindow:
    """
    Container for one complete walk-forward cycle (IS + OOS pair).

    Attributes
    ----------
    window_idx  : int              Cycle index (0-based).
    is_start    : pd.Timestamp     In-sample start date.
    is_end      : pd.Timestamp     In-sample end date.
    oos_start   : pd.Timestamp     Out-of-sample start date.
    oos_end     : pd.Timestamp     Out-of-sample end date.
    best_params : dict             Parameters selected by grid/random search.
    is_result   : BacktestResult   In-sample performance with best_params.
    oos_result  : BacktestResult   Out-of-sample performance with best_params.
    """
    window_idx  : int
    is_start    : pd.Timestamp
    is_end      : pd.Timestamp
    oos_start   : pd.Timestamp
    oos_end     : pd.Timestamp
    best_params : Dict[str, Any]
    is_result   : BacktestResult
    oos_result  : BacktestResult


# ---------------------------------------------------------------------------
# OverfitAssessment
# ---------------------------------------------------------------------------

@dataclass
class OverfitAssessment:
    """
    Overall overfitting assessment produced after all WFO windows are run.

    Verdict is one of: ``'PASS'``, ``'WARNING'``, ``'FAIL'``.
    """
    verdict           : str
    reasons           : List[str]
    avg_is_sharpe     : float
    avg_oos_sharpe    : float
    avg_oos_max_dd    : float
    sharpe_decay_ratio: float    # OOS / IS Sharpe ratio (ideal = 1.0)
    param_stability   : Dict[str, float]  # param_name -> std-dev / range


# ---------------------------------------------------------------------------
# TradingStrategy -- abstract base
# ---------------------------------------------------------------------------

class TradingStrategy(ABC):
    """
    Abstract base class for all walk-forward-compatible strategies.

    Subclasses must define:
      - ``parameters``: dict of {param_name: (min, max, step)} tuples.
      - ``should_maximize``: the metric name to optimise (e.g. 'sharpe_ratio').
      - ``generate_signals()``: vectorized signal generation.
      - (optional) ``validate_params()``: filter out invalid combos early.

    Notes
    -----
    ``generate_signals`` must be fully vectorized -- no ``apply`` or per-row
    loops allowed in the implementation.
    """

    #: ``{param_name: (min_val, max_val, step)}`` -- override in subclasses.
    parameters      : Dict[str, Tuple[float, float, float]] = {}
    should_maximize : str = "sharpe_ratio"

    @abstractmethod
    def generate_signals(
        self,
        data  : pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.Series:
        """
        Generate a signal series from OHLCV data and a parameter set.

        Parameters
        ----------
        data   : pd.DataFrame  Slice of OHLCV data (columns: open/high/low/close/volume).
        params : dict          Parameter values for this evaluation.

        Returns
        -------
        pd.Series of int
            +1 (long), -1 (short), 0 (flat) -- same index as ``data``.
        """

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Return True when a parameter combination is valid.

        Override this to reject nonsensical combos (e.g., fast_ema >= slow_ema)
        without running the backtest.

        Parameters
        ----------
        params : dict

        Returns
        -------
        bool
        """
        return True

    def parameter_grid(self) -> List[Dict[str, Any]]:
        """
        Enumerate all valid parameter combinations using the ranges in ``parameters``.

        Returns
        -------
        list of dict
        """
        names  = list(self.parameters.keys())
        ranges = [
            list(np.arange(mn, mx + step, step).astype(type(mn)))
            for mn, mx, step in self.parameters.values()
        ]
        combos = [
            dict(zip(names, combo))
            for combo in itertools.product(*ranges)
            if self.validate_params(dict(zip(names, combo)))
        ]
        return combos


# ---------------------------------------------------------------------------
# EMACrossover -- concrete strategy
# ---------------------------------------------------------------------------

class EMACrossover(TradingStrategy):
    """
    EMA Crossover strategy.

    Enters LONG when the fast EMA crosses above the slow EMA,
    SHORT when it crosses below. Uses exponential moving averages
    (Wilder-style ``adjust=False``).

    Parameters
    ----------
    ``ema_fast`` : (5, 50, 5)     Fast EMA period range.
    ``ema_slow`` : (20, 200, 20)  Slow EMA period range.

    Optimization target: ``sharpe_ratio``.
    """

    parameters: Dict[str, Tuple[float, float, float]] = {
        "ema_fast": (5,  50,  5),
        "ema_slow": (20, 200, 20),
    }
    should_maximize: str = "sharpe_ratio"

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Reject combos where fast period >= slow period."""
        return int(params["ema_fast"]) < int(params["ema_slow"])

    def generate_signals(
        self,
        data  : pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.Series:
        """
        Vectorized EMA crossover signal generation.

        Returns +1 when fast EMA > slow EMA, -1 when fast < slow, 0 when equal.
        """
        fast_span = int(params["ema_fast"])
        slow_span = int(params["ema_slow"])
        close     = data["close"]

        ema_fast = close.ewm(span=fast_span, adjust=False, min_periods=fast_span).mean()
        ema_slow = close.ewm(span=slow_span, adjust=False, min_periods=slow_span).mean()

        signal = np.where(ema_fast > ema_slow,  1,
                 np.where(ema_fast < ema_slow, -1, 0))

        return pd.Series(signal, index=data.index, dtype=np.int8)


# ---------------------------------------------------------------------------
# VectorizedBacktester
# ---------------------------------------------------------------------------

class VectorizedBacktester:
    """
    Pure numpy/pandas vectorized backtesting engine.

    No event loop, no per-row Python iteration.  All calculations are
    performed as array operations over the full signal/price series.

    Parameters
    ----------
    commission         : float  Round-trip cost per trade as fraction (default 0.001 = 0.1%).
    spread             : float  Additional spread cost per trade (default 0.0).
    annualization_factor: int   Bars per year for Sharpe/Sortino scaling (default 252).
    """

    def __init__(
        self,
        commission           : float = 0.001,
        spread               : float = 0.0,
        annualization_factor : int   = 252,
    ) -> None:
        self.commission            = commission
        self.spread                = spread
        self.annualization_factor  = annualization_factor

    def run(
        self,
        data   : pd.DataFrame,
        signals: pd.Series,
    ) -> BacktestResult:
        """
        Execute a full vectorized backtest.

        Parameters
        ----------
        data    : pd.DataFrame  OHLCV slice (must have 'close' column).
        signals : pd.Series     Signal series (+1/-1/0), same index as data.

        Returns
        -------
        BacktestResult
        """
        close = data["close"].reindex(signals.index)

        # -- Bar returns -------------------------------------------------
        price_returns = close.pct_change().fillna(0.0)

        # Signal applied on next bar (avoid look-ahead)
        position = signals.shift(1).fillna(0.0)
        raw_returns = position * price_returns

        # -- Transaction costs ------------------------------------------
        # Cost fires whenever signal changes (a new trade is entered/exited)
        trade_cost_per_side = self.commission + self.spread
        signal_change = signals.diff().abs().fillna(0.0) > 0
        # When signal goes 1->-1 or -1->1, we pay cost twice (close + open)
        signal_magnitude_change = signals.diff().abs().fillna(0.0)
        cost_multiplier = np.where(signal_magnitude_change == 2,
                                   2 * trade_cost_per_side,
                                   trade_cost_per_side)
        transaction_costs = pd.Series(
            np.where(signal_change, cost_multiplier, 0.0),
            index=signals.index,
        )

        net_returns = raw_returns - transaction_costs

        # -- Equity curve -----------------------------------------------
        equity_curve = (1.0 + net_returns).cumprod()

        # -- Drawdown curve ---------------------------------------------
        rolling_peak  = equity_curve.cummax()
        drawdown_curve = (equity_curve - rolling_peak) / rolling_peak
        max_drawdown  = float(drawdown_curve.min())

        # -- Trade count ------------------------------------------------
        # Each signal change = one trade (entry or exit/reverse)
        num_trades = int(signal_change.sum())

        # -- Win rate & profit factor -----------------------------------
        trade_returns = net_returns[signal_change]
        if len(trade_returns) > 0:
            wins   = trade_returns[trade_returns > 0]
            losses = trade_returns[trade_returns < 0]
            win_rate = float(len(wins) / len(trade_returns)) if len(trade_returns) > 0 else 0.0
            gross_profit = float(wins.sum())   if len(wins)   > 0 else 0.0
            gross_loss   = float(losses.abs().sum()) if len(losses) > 0 else 1e-9
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        else:
            win_rate = profit_factor = 0.0

        # -- Total return -----------------------------------------------
        total_return = float(equity_curve.iloc[-1] - 1.0)

        # -- Sharpe ratio (annualized) ----------------------------------
        std_ret = float(net_returns.std())
        mean_ret = float(net_returns.mean())
        ann = np.sqrt(self.annualization_factor)
        sharpe = (mean_ret / std_ret * ann) if std_ret > 1e-10 else 0.0

        # -- Sortino ratio (downside deviation) -------------------------
        downside = net_returns[net_returns < 0.0]
        downside_std = float(np.sqrt((downside ** 2).mean())) if len(downside) > 1 else 0.0
        sortino = (mean_ret / downside_std * ann) if downside_std > 1e-10 else 0.0

        if num_trades == 0:
            raise NoValidSignalsError("No trades generated -- skipping this parameter set.")

        return BacktestResult(
            sharpe_ratio   = round(sharpe, 6),
            sortino_ratio  = round(sortino, 6),
            max_drawdown   = round(max_drawdown * 100, 4),   # convert to %
            win_rate       = round(win_rate, 6),
            profit_factor  = round(profit_factor, 6),
            total_return   = round(total_return * 100, 4),   # convert to %
            num_trades     = num_trades,
            equity_curve   = equity_curve,
            drawdown_curve = drawdown_curve * 100,            # convert to %
        )


# ---------------------------------------------------------------------------
# WalkForwardOptimizer
# ---------------------------------------------------------------------------

class WalkForwardOptimizer:
    """
    Orchestrates the full walk-forward optimization loop.

    Parameters
    ----------
    strategy              : TradingStrategy  Strategy instance to optimize.
    window_size           : int              Training window size in bars.
    test_size             : int              Testing window size in bars.
    step_size             : int, optional    Slide step per WFO cycle (default = test_size).
    num_windows           : int              Maximum WFO cycles to run (default 6).
    commission            : float            Round-trip trade cost (default 0.001).
    spread                : float            Spread cost per trade (default 0.0).
    annualization_factor  : int              Bars/year for Sharpe scaling (default 252).
    max_grid_size         : int              Combos above this trigger random search (default 10_000).
    random_search_samples : int              Samples for random search (default 500).
    random_seed           : int              Seed for reproducibility (default 42).
    """

    def __init__(
        self,
        strategy             : TradingStrategy,
        window_size          : int,
        test_size            : int,
        step_size            : Optional[int] = None,
        num_windows          : int   = 6,
        commission           : float = 0.001,
        spread               : float = 0.0,
        annualization_factor : int   = 252,
        max_grid_size        : int   = 10_000,
        random_search_samples: int   = 500,
        random_seed          : int   = 42,
    ) -> None:
        self.strategy              = strategy
        self.window_size           = window_size
        self.test_size             = test_size
        self.step_size             = step_size if step_size is not None else test_size
        self.num_windows           = num_windows
        self.max_grid_size         = max_grid_size
        self.random_search_samples = random_search_samples
        self.random_seed           = random_seed

        self._backtester = VectorizedBacktester(
            commission           = commission,
            spread               = spread,
            annualization_factor = annualization_factor,
        )
        random.seed(random_seed)

    # -- Data loading ---------------------------------------------------

    @staticmethod
    def load_data(
        csv_path : str,
        chunksize: int = 100_000,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from a CSV file using chunked reading.

        Reads in chunks of ``chunksize`` rows to handle files with millions
        of rows without loading the entire dataset into memory at once.

        Parameters
        ----------
        csv_path  : str  Path to the CSV file.
        chunksize : int  Rows per read chunk (default 100,000).

        Returns
        -------
        pd.DataFrame
            Columns: open, high, low, close, volume.
            Index  : DatetimeIndex (parsed from 'timestamp' column).

        Raises
        ------
        FileNotFoundError  If the CSV path does not exist.
        KeyError           If required columns are missing.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        logger.info("[WFO] Loading data from %s (chunksize=%d)…", csv_path, chunksize)
        chunks = []
        total_rows = 0

        for chunk in pd.read_csv(
            path,
            chunksize   = chunksize,
            parse_dates = ["timestamp"],
        ):
            required = {"open", "high", "low", "close", "volume"}
            missing  = required - set(chunk.columns)
            if missing:
                raise KeyError(f"CSV is missing required columns: {missing}")

            for col in required:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            chunk = chunk.dropna(subset=list(required))

            if "timestamp" in chunk.columns:
                chunk = chunk.set_index("timestamp")
            chunk = chunk.sort_index()
            chunks.append(chunk)
            total_rows += len(chunk)

        if not chunks:
            raise InsufficientDataError("CSV produced zero valid rows after parsing.")

        df = pd.concat(chunks, axis=0)
        df = df[~df.index.duplicated(keep="first")].sort_index()

        logger.info("[WFO] Loaded %d rows from %s.", len(df), csv_path)
        return df

    # -- Main WFO orchestration -----------------------------------------

    def run(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """
        Execute the full walk-forward optimization loop.

        Parameters
        ----------
        data : pd.DataFrame
            Full OHLCV dataset (DatetimeIndex), loaded via ``load_data`` or
            provided directly.

        Returns
        -------
        list of WalkForwardWindow
            One entry per completed WFO cycle.

        Raises
        ------
        InsufficientDataError
            If the dataset is too small for even one full WFO window.
        """
        min_required = self.window_size + self.test_size
        if len(data) < min_required:
            raise InsufficientDataError(
                f"Dataset has {len(data)} rows; need at least "
                f"{min_required} (window_size={self.window_size} + "
                f"test_size={self.test_size})."
            )

        # Determine how many windows fit
        max_possible = (len(data) - min_required) // self.step_size + 1
        n_windows    = min(self.num_windows, max_possible)

        logger.info(
            "[WFO] Starting %d walk-forward windows | window=%d test=%d step=%d",
            n_windows, self.window_size, self.test_size, self.step_size,
        )

        # Pre-build parameter grid once (shared across all windows)
        grid = self.strategy.parameter_grid()
        use_random = len(grid) > self.max_grid_size
        if use_random:
            logger.info(
                "[WFO] Grid size %d > %d -- using random search (%d samples).",
                len(grid), self.max_grid_size, self.random_search_samples,
            )

        windows: List[WalkForwardWindow] = []

        for i in range(n_windows):
            start    = i * self.step_size
            is_slice = data.iloc[start : start + self.window_size]
            oos_slice = data.iloc[
                start + self.window_size :
                start + self.window_size + self.test_size
            ]

            if len(is_slice) < self.window_size // 2 or len(oos_slice) == 0:
                logger.warning("[WFO] Window %d: insufficient rows -- skipping.", i)
                continue

            logger.info(
                "[WFO] Window %d | IS: %s -> %s (%d rows) | OOS: %s -> %s (%d rows)",
                i,
                is_slice.index[0].date(), is_slice.index[-1].date(), len(is_slice),
                oos_slice.index[0].date(), oos_slice.index[-1].date(), len(oos_slice),
            )

            # -- Optimize on IS slice -----------------------------------
            best_params, is_result = self._optimize_window(
                is_slice, grid, use_random
            )
            if best_params is None:
                logger.warning("[WFO] Window %d: no valid params found -- skipping.", i)
                continue

            # -- Evaluate on OOS slice ----------------------------------
            try:
                oos_signals = self.strategy.generate_signals(oos_slice, best_params)
                oos_result  = self._backtester.run(oos_slice, oos_signals)
            except NoValidSignalsError:
                logger.warning("[WFO] Window %d: OOS produced no signals.", i)
                # Create a zero-return OOS result rather than skipping
                oos_result = self._zero_result(oos_slice)

            logger.info(
                "[WFO] Window %d | params=%s | IS_Sharpe=%.3f | OOS_Sharpe=%.3f | "
                "IS_DD=%.2f%% | OOS_DD=%.2f%%",
                i, best_params,
                is_result.sharpe_ratio, oos_result.sharpe_ratio,
                is_result.max_drawdown, oos_result.max_drawdown,
            )

            windows.append(WalkForwardWindow(
                window_idx  = i,
                is_start    = is_slice.index[0],
                is_end      = is_slice.index[-1],
                oos_start   = oos_slice.index[0],
                oos_end     = oos_slice.index[-1],
                best_params = best_params,
                is_result   = is_result,
                oos_result  = oos_result,
            ))

        return windows

    # -- Optimization helpers -------------------------------------------

    def _optimize_window(
        self,
        data      : pd.DataFrame,
        grid      : List[Dict],
        use_random: bool,
    ) -> Tuple[Optional[Dict], Optional[BacktestResult]]:
        """
        Find the best parameter set on a training slice.

        Uses full grid search when ``len(grid) <= max_grid_size``, otherwise
        falls back to random search.

        Parameters
        ----------
        data       : pd.DataFrame  Training slice.
        grid       : list of dict  All valid parameter combinations.
        use_random : bool          If True, sample ``random_search_samples`` combos.

        Returns
        -------
        Tuple[dict | None, BacktestResult | None]
            ``(best_params, best_result)`` -- both None if nothing valid found.
        """
        search_space = (
            random.sample(grid, min(self.random_search_samples, len(grid)))
            if use_random else grid
        )

        best_score  : float                    = -np.inf
        best_params : Optional[Dict]           = None
        best_result : Optional[BacktestResult] = None
        metric_name : str                      = self.strategy.should_maximize

        for params in search_space:
            try:
                signals = self.strategy.generate_signals(data, params)
                result  = self._backtester.run(data, signals)
                score   = result.metric(metric_name)
                if score > best_score:
                    best_score  = score
                    best_params = params
                    best_result = result
            except NoValidSignalsError:
                continue
            except Exception as exc:
                logger.debug("[WFO] Params %s raised: %s", params, exc)
                continue

        return best_params, best_result

    @staticmethod
    def _zero_result(data: pd.DataFrame) -> BacktestResult:
        """Return a flat (zero-return) BacktestResult for an empty OOS slice."""
        idx = data.index
        return BacktestResult(
            sharpe_ratio   = 0.0,
            sortino_ratio  = 0.0,
            max_drawdown   = 0.0,
            win_rate       = 0.0,
            profit_factor  = 0.0,
            total_return   = 0.0,
            num_trades     = 0,
            equity_curve   = pd.Series(1.0, index=idx),
            drawdown_curve = pd.Series(0.0, index=idx),
        )


# ---------------------------------------------------------------------------
# Overfitting assessment
# ---------------------------------------------------------------------------

def assess_overfitting(
    windows           : List[WalkForwardWindow],
    strategy          : TradingStrategy,
    max_oos_dd_limit  : float = -15.0,
    decay_threshold   : float = 0.50,
    stability_threshold: float = 0.30,
) -> OverfitAssessment:
    """
    Evaluate whether the walk-forward results pass overfitting rejection criteria.

    Criteria (in order of severity)
    --------------------------------
    1. FAIL  -- avg OOS Sharpe < 0  (no edge out-of-sample)
    2. FAIL  -- avg OOS max drawdown < max_oos_dd_limit (unacceptable risk)
    3. WARN  -- avg OOS Sharpe < avg IS Sharpe * decay_threshold (significant decay)
    4. WARN  -- any parameter std_dev / range > stability_threshold (instability)

    Parameters
    ----------
    windows            : list of WalkForwardWindow
    strategy           : TradingStrategy   Needed for parameter range info.
    max_oos_dd_limit   : float             OOS DD floor (default -15%).
    decay_threshold    : float             OOS/IS Sharpe ratio threshold (default 0.5).
    stability_threshold: float             Max allowed param std / range (default 0.3).

    Returns
    -------
    OverfitAssessment
    """
    if not windows:
        return OverfitAssessment(
            verdict="FAIL",
            reasons=["No walk-forward windows completed."],
            avg_is_sharpe=0.0,
            avg_oos_sharpe=0.0,
            avg_oos_max_dd=0.0,
            sharpe_decay_ratio=0.0,
            param_stability={},
        )

    is_sharpes  = [w.is_result.sharpe_ratio  for w in windows]
    oos_sharpes = [w.oos_result.sharpe_ratio for w in windows]
    oos_dds     = [w.oos_result.max_drawdown for w in windows]

    avg_is  = float(np.mean(is_sharpes))
    avg_oos = float(np.mean(oos_sharpes))
    avg_dd  = float(np.mean(oos_dds))
    decay   = avg_oos / avg_is if abs(avg_is) > 1e-9 else 0.0

    # -- Parameter stability --------------------------------------------
    param_stability: Dict[str, float] = {}
    for param_name, (mn, mx, step) in strategy.parameters.items():
        values = [w.best_params.get(param_name, mn) for w in windows]
        param_range = mx - mn
        std_dev = float(np.std(values)) if len(values) > 1 else 0.0
        param_stability[param_name] = round(std_dev / param_range, 4) if param_range > 0 else 0.0

    # -- Evaluate criteria ----------------------------------------------
    reasons: List[str] = []
    verdict = "PASS"

    if avg_oos < 0:
        reasons.append(
            f"FAIL: avg OOS Sharpe={avg_oos:.3f} < 0 -- no positive edge out-of-sample."
        )
        verdict = "FAIL"

    if avg_dd < max_oos_dd_limit:
        reasons.append(
            f"FAIL: avg OOS max drawdown={avg_dd:.2f}% < limit {max_oos_dd_limit:.1f}%."
        )
        verdict = "FAIL"

    if avg_is > 0 and decay < decay_threshold and verdict != "FAIL":
        reasons.append(
            f"WARNING: OOS/IS Sharpe ratio={decay:.3f} < {decay_threshold:.1f} -- "
            "significant in-sample overfitting detected."
        )
        verdict = "WARNING"

    for param_name, stability in param_stability.items():
        if stability > stability_threshold and verdict == "PASS":
            reasons.append(
                f"WARNING: '{param_name}' std/range={stability:.3f} > "
                f"{stability_threshold:.2f} -- parameter instability across windows."
            )
            verdict = "WARNING"

    if verdict == "PASS":
        reasons.append(
            f"PASS: OOS Sharpe={avg_oos:.3f} | IS Sharpe={avg_is:.3f} | "
            f"decay={decay:.3f} | OOS DD={avg_dd:.2f}%"
        )

    return OverfitAssessment(
        verdict            = verdict,
        reasons            = reasons,
        avg_is_sharpe      = round(avg_is,  4),
        avg_oos_sharpe     = round(avg_oos, 4),
        avg_oos_max_dd     = round(avg_dd,  4),
        sharpe_decay_ratio = round(decay,   4),
        param_stability    = param_stability,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    windows     : List[WalkForwardWindow],
    strategy    : TradingStrategy,
    output_dir  : str = ".",
    plot        : bool = True,
) -> str:
    """
    Generate a full WFO performance report as a formatted text string.

    Also writes:
    - ``wfo_report_{strategy}.txt``  -- full text report
    - ``wfo_results_{strategy}.csv`` -- per-window table

    Parameters
    ----------
    windows    : list of WalkForwardWindow  Completed WFO results.
    strategy   : TradingStrategy            Strategy instance (for name + params).
    output_dir : str                        Directory to write files (default ".").
    plot       : bool                       Attempt equity curve plot (default True).

    Returns
    -------
    str  The full text report.
    """
    strategy_name = type(strategy).__name__
    assessment    = assess_overfitting(windows, strategy)
    out_path      = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # -- Build lines ---------------------------------------------------
    sep   = "=" * 72
    lines = [
        sep,
        f"  Q-SonicFX  |  Walk-Forward Optimization Report",
        f"  Strategy   : {strategy_name}",
        f"  Params     : {list(strategy.parameters.keys())}",
        f"  Windows    : {len(windows)}",
        sep,
        "",
    ]

    if windows:
        lines += [
            f"  Data range : {windows[0].is_start.date()} -> {windows[-1].oos_end.date()}",
            f"  Total bars : IS bars={sum(w.is_result.num_trades and 1 for w in windows)} "
            f"windows × training",
            "",
        ]

    # -- Per-window table -----------------------------------------------
    col = (
        f"  {'Win':>3}  {'IS Start':>10}  {'IS End':>10}  "
        f"{'OOS Start':>10}  {'OOS End':>10}  "
        f"{'IS Sharpe':>10}  {'OOS Sharpe':>10}  "
        f"{'IS DD%':>7}  {'OOS DD%':>7}  "
        f"{'IS Trades':>9}  {'OOS Trades':>10}  Params"
    )
    lines.append(col)
    lines.append("  " + "-" * (len(col) - 2))

    for w in windows:
        param_str = "  ".join(f"{k}={v}" for k, v in w.best_params.items())
        lines.append(
            f"  {w.window_idx:>3}  "
            f"{str(w.is_start.date()):>10}  {str(w.is_end.date()):>10}  "
            f"{str(w.oos_start.date()):>10}  {str(w.oos_end.date()):>10}  "
            f"{w.is_result.sharpe_ratio:>10.4f}  {w.oos_result.sharpe_ratio:>10.4f}  "
            f"{w.is_result.max_drawdown:>7.2f}  {w.oos_result.max_drawdown:>7.2f}  "
            f"{w.is_result.num_trades:>9}  {w.oos_result.num_trades:>10}  "
            f"{param_str}"
        )

    lines += [
        "",
        sep,
        f"  SUMMARY",
        sep,
        f"  Avg IS  Sharpe      : {assessment.avg_is_sharpe:>8.4f}",
        f"  Avg OOS Sharpe      : {assessment.avg_oos_sharpe:>8.4f}",
        f"  OOS/IS Decay Ratio  : {assessment.sharpe_decay_ratio:>8.4f}  (ideal = 1.0)",
        f"  Avg OOS Max DD      : {assessment.avg_oos_max_dd:>8.2f}%",
        "",
        f"  Parameter Stability (std / range -- lower is better):",
    ]

    for param, stab in assessment.param_stability.items():
        bar = "|" * min(int(stab * 40), 40)
        lines.append(f"    {param:<20} {stab:.4f}  {bar}")

    lines += [
        "",
        sep,
        f"  OVERFITTING ASSESSMENT:  {assessment.verdict}",
        sep,
    ]
    for reason in assessment.reasons:
        lines.append(f"  -> {reason}")

    lines.append(sep)
    report_text = "\n".join(lines)

    # -- Write text report ----------------------------------------------
    txt_file = out_path / f"wfo_report_{strategy_name}.txt"
    txt_file.write_text(report_text, encoding="utf-8")
    logger.info("[WFO] Report written to %s", txt_file)

    # -- Write CSV ------------------------------------------------------
    csv_file = out_path / f"wfo_results_{strategy_name}.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        param_keys = list(windows[0].best_params.keys()) if windows else []
        header = (
            ["window", "is_start", "is_end", "oos_start", "oos_end"]
            + [f"param_{k}" for k in param_keys]
            + ["is_sharpe", "oos_sharpe", "is_sortino", "oos_sortino",
               "is_max_dd", "oos_max_dd", "is_win_rate", "oos_win_rate",
               "is_profit_factor", "oos_profit_factor",
               "is_total_return", "oos_total_return",
               "is_num_trades", "oos_num_trades"]
        )
        writer.writerow(header)
        for w in windows:
            param_vals = [w.best_params.get(k, "") for k in param_keys]
            writer.writerow(
                [w.window_idx,
                 w.is_start.isoformat(), w.is_end.isoformat(),
                 w.oos_start.isoformat(), w.oos_end.isoformat()]
                + param_vals
                + [w.is_result.sharpe_ratio,  w.oos_result.sharpe_ratio,
                   w.is_result.sortino_ratio,  w.oos_result.sortino_ratio,
                   w.is_result.max_drawdown,   w.oos_result.max_drawdown,
                   w.is_result.win_rate,        w.oos_result.win_rate,
                   w.is_result.profit_factor,  w.oos_result.profit_factor,
                   w.is_result.total_return,   w.oos_result.total_return,
                   w.is_result.num_trades,     w.oos_result.num_trades]
            )
    logger.info("[WFO] CSV written to %s", csv_file)

    # -- Optional matplotlib plot ---------------------------------------
    if plot and windows:
        _plot_equity_curves(windows, strategy_name, out_path)

    return report_text


def _plot_equity_curves(
    windows      : List[WalkForwardWindow],
    strategy_name: str,
    out_path     : Path,
) -> None:
    """Attempt to plot OOS equity curves overlaid. Silent no-op if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        cmap = plt.get_cmap("tab10")

        for i, w in enumerate(windows):
            color = cmap(i % 10)
            axes[0].plot(
                range(len(w.oos_result.equity_curve)),
                w.oos_result.equity_curve.values,
                color=color, label=f"W{w.window_idx} OOS", linewidth=1.2
            )
            axes[1].plot(
                range(len(w.oos_result.drawdown_curve)),
                w.oos_result.drawdown_curve.values,
                color=color, label=f"W{w.window_idx} DD", linewidth=1.0
            )

        axes[0].set_title(f"{strategy_name} -- OOS Equity Curves")
        axes[0].set_ylabel("Equity (normalized)")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("OOS Drawdown Curves")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color="black", linewidth=0.5)

        plt.tight_layout()
        fig_path = out_path / f"wfo_equity_{strategy_name}.png"
        plt.savefig(fig_path, dpi=120)
        plt.close(fig)
        logger.info("[WFO] Equity curve plot saved to %s", fig_path)

    except ImportError:
        logger.info("[WFO] matplotlib not available -- skipping equity curve plot.")
    except Exception as exc:
        logger.warning("[WFO] Plot failed: %s", exc)


# ---------------------------------------------------------------------------
# Synthetic data generator (for demo / tests)
# ---------------------------------------------------------------------------

def generate_synthetic_ohlcv(
    n     : int = 20_000,
    seed  : int = 42,
    trend : float = 0.00005,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic OHLCV DataFrame.

    Uses a geometric Brownian motion with slight upward trend, noise, and
    mean-reverting spread. Suitable for testing without real market data.

    Parameters
    ----------
    n     : int    Number of 1-minute bars (default 20,000 ≈ 14 trading days).
    seed  : int    NumPy random seed.
    trend : float  Per-bar drift (default 5 bps / bar).

    Returns
    -------
    pd.DataFrame  Columns: timestamp, open, high, low, close, volume.
    """
    rng     = np.random.default_rng(seed)
    returns = rng.normal(trend, 0.002, n)

    # Introduce regime shifts for realism
    regime_change = int(n * 0.4)
    returns[regime_change : regime_change + int(n * 0.2)] -= 0.0003   # drawdown regime

    close = 40_000.0 * np.cumprod(1.0 + returns)
    spread = close * rng.uniform(0.0001, 0.0008, n)

    open_  = np.concatenate([[close[0]], close[:-1]])
    high   = np.maximum(open_, close) + spread
    low    = np.minimum(open_, close) - spread
    volume = rng.integers(500, 8_000, n).astype(float)

    ts = pd.date_range("2024-01-01 00:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open"     : np.round(open_, 2),
        "high"     : np.round(high,  2),
        "low"      : np.round(low,   2),
        "close"    : np.round(close, 2),
        "volume"   : volume,
    })


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt= "%Y-%m-%dT%H:%M:%S",
    )

    N_ROWS      = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    N_WINDOWS   = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    WINDOW_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 4_000
    TEST_SIZE   = int(sys.argv[4]) if len(sys.argv) > 4 else 1_000

    print(f"\n{'='*72}")
    print(f"  Q-SonicFX  |  Walk-Forward Optimizer -- Demo")
    print(f"  Rows={N_ROWS}  Windows={N_WINDOWS}  "
          f"TrainSize={WINDOW_SIZE}  TestSize={TEST_SIZE}")
    print(f"{'='*72}\n")

    # -- 1. Generate synthetic data -------------------------------------
    print("[1/4] Generating synthetic OHLCV data…")
    t0 = time.perf_counter()
    df = generate_synthetic_ohlcv(n=N_ROWS, seed=42)
    df_indexed = df.set_index("timestamp")
    print(f"      {len(df)} bars | {df['timestamp'].iloc[0].date()} -> "
          f"{df['timestamp'].iloc[-1].date()} | "
          f"elapsed: {(time.perf_counter()-t0)*1000:.0f}ms")

    # -- 2. Run WFO ----------------------------------------------------
    print(f"\n[2/4] Running Walk-Forward Optimization ({N_WINDOWS} windows)…")
    strategy  = EMACrossover()
    optimizer = WalkForwardOptimizer(
        strategy             = strategy,
        window_size          = WINDOW_SIZE,
        test_size            = TEST_SIZE,
        step_size            = TEST_SIZE,
        num_windows          = N_WINDOWS,
        commission           = 0.001,
        annualization_factor = 252 * 24 * 60,  # 1-min data -> per-minute bars
        max_grid_size        = 10_000,
    )

    t0 = time.perf_counter()
    windows = optimizer.run(df_indexed)
    elapsed = time.perf_counter() - t0
    print(f"      {len(windows)} windows completed in {elapsed:.2f}s")

    # -- 3. Per-window quick stats --------------------------------------
    print(f"\n[3/4] Window results:")
    print(f"  {'Win':>3}  {'Best Params':<28}  "
          f"{'IS Sharpe':>10}  {'OOS Sharpe':>10}  "
          f"{'IS DD%':>7}  {'OOS DD%':>7}")
    print("  " + "-" * 70)
    for w in windows:
        p = "  ".join(f"{k}={int(v)}" for k, v in w.best_params.items())
        print(
            f"  {w.window_idx:>3}  {p:<28}  "
            f"{w.is_result.sharpe_ratio:>10.4f}  {w.oos_result.sharpe_ratio:>10.4f}  "
            f"{w.is_result.max_drawdown:>7.2f}  {w.oos_result.max_drawdown:>7.2f}"
        )

    # -- 4. Full report -------------------------------------------------
    print(f"\n[4/4] Generating report…")
    report = generate_report(
        windows    = windows,
        strategy   = strategy,
        output_dir = ".",
        plot       = True,
    )
    print("\n" + report)
