#!/usr/bin/env python3
"""
position_sizer.py
=================
Q-SonicFX — Dynamic Position Sizer (Fractional Kelly Criterion)
===============================================================

Implements institutional-grade position sizing using the Fractional Kelly
Criterion with layered hard risk controls:

    1. Full Kelly formula  (f* = W - (1-W)/R)
    2. Fractional reduction (f_used = f* * kelly_fraction, default 0.25)
    3. Dollar-risk cap     (min of Kelly size vs. 2% of equity / risk-per-unit)
    4. Wide-stop penalty   (>20% away from entry -> halve the risk cap to 1%)
    5. Minimum lot size    (configurable, default 0.001 for BTC)
    6. Zero-edge guard     (f* <= 0 -> no trade)

Kelly formula reference
-----------------------
    R         = avg_win / avg_loss        (risk-reward ratio)
    f*        = win_rate - (1 - win_rate) / R   (full Kelly fraction)
    f_used    = f* * kelly_fraction        (fractional / conservative)

Position sizing
---------------
    risk_per_unit      = |entry_price - stop_loss_price|
    max_loss_dollars   = account_balance * max_risk_per_trade
    kelly_size         = (account_balance * f_used) / entry_price
    risk_capped_size   = max_loss_dollars / risk_per_unit
    final_size         = min(kelly_size, risk_capped_size)

Author : Q-SonicFX Quant Engine
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class PositionSizeResult:
    """
    Output of a single position-sizing computation.

    Attributes
    ----------
    position_size   : float  Units of the asset to buy (LONG) or sell (SHORT).
                             0.0 when no trade is allowed.
    risk_percentage : float  Actual % of equity at risk for this trade (0-100).
    kelly_percentage: float  Full Kelly fraction before fractional multiplier (0-100).
    fraction_used   : float  The fractional Kelly multiplier actually applied.
    max_loss_dollars: float  Maximum dollar loss if stop is hit.
    is_trade_allowed: bool   False when any condition produces size == 0.
    notes           : str    Human-readable explanation of any adjustments made.
    """
    position_size   : float
    risk_percentage : float
    kelly_percentage: float
    fraction_used   : float
    max_loss_dollars: float
    is_trade_allowed: bool
    notes           : str = ""

    def __repr__(self) -> str:
        status = "ALLOWED" if self.is_trade_allowed else "BLOCKED"
        return (
            f"PositionSizeResult({status} | "
            f"size={self.position_size:.6f} units | "
            f"risk={self.risk_percentage:.3f}% | "
            f"kelly={self.kelly_percentage:.3f}% | "
            f"max_loss=${self.max_loss_dollars:.2f} | "
            f"notes={self.notes!r})"
        )


# ---------------------------------------------------------------------------
# Rolling statistics helper
# ---------------------------------------------------------------------------

def update_rolling_stats(
    trades_history: List[Dict],
    window: int = 100,
) -> Tuple[float, float, float]:
    """
    Compute adaptive Kelly inputs from a rolling window of trade results.

    Processes the last ``window`` trades in ``trades_history`` and returns
    ``(win_rate, avg_win, avg_loss)`` ready to feed directly into
    :class:`PositionSizer`.

    Parameters
    ----------
    trades_history : list of dict
        Each entry must contain ``'pnl_percent'`` (signed float, e.g. +0.02
        means +2% gain, -0.015 means -1.5% loss) and ``'side'`` (str,
        ``'LONG'`` or ``'SHORT'`` -- stored for reference, not used in math).
    window : int
        Number of most-recent trades to include (default 100).

    Returns
    -------
    Tuple[float, float, float]
        ``(win_rate, avg_win, avg_loss)`` where:
        - ``win_rate``  ∈ [0, 1]
        - ``avg_win``   > 0  (mean return of winning trades as a decimal)
        - ``avg_loss``  > 0  (mean absolute return of losing trades as a decimal)

    Notes
    -----
    If fewer than 2 trades exist in the window, returns (0.0, 0.0, 0.0) so
    the caller's zero-edge guard fires and no trade is taken.

    Examples
    --------
    >>> history = [{'pnl_percent': 0.02, 'side': 'LONG'},
    ...            {'pnl_percent': -0.01, 'side': 'LONG'}]
    >>> win_rate, avg_win, avg_loss = update_rolling_stats(history)
    >>> win_rate
    0.5
    """
    if not trades_history:
        return 0.0, 0.0, 0.0

    recent = trades_history[-window:]

    if len(recent) < 2:
        logger.warning(
            "[PositionSizer] Rolling window has only %d trade(s); need >= 2. "
            "Returning zero stats.", len(recent)
        )
        return 0.0, 0.0, 0.0

    pnl_arr = np.array([t["pnl_percent"] for t in recent], dtype=np.float64)

    wins  = pnl_arr[pnl_arr > 0.0]
    losses = pnl_arr[pnl_arr < 0.0]

    win_rate = float(len(wins) / len(pnl_arr))
    avg_win  = float(np.mean(wins))   if len(wins)   > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0

    logger.debug(
        "[PositionSizer] Rolling stats (last %d): win_rate=%.3f avg_win=%.4f avg_loss=%.4f",
        len(recent), win_rate, avg_win, avg_loss,
    )
    return win_rate, avg_win, avg_loss


# ---------------------------------------------------------------------------
# Main position sizer
# ---------------------------------------------------------------------------

class PositionSizer:
    """
    Dynamic position sizer using the Fractional Kelly Criterion.

    Parameters
    ----------
    account_balance   : float  Total equity in quote currency (must be > 0).
    win_rate          : float  Historical win rate in [0, 1].
    avg_win           : float  Average winning trade return (e.g. 0.02 = 2%).
    avg_loss          : float  Average losing trade return as positive decimal
                               (e.g. 0.01 = 1% loss magnitude).
    kelly_fraction    : float  Conservative multiplier on full Kelly (default 0.25).
    max_risk_per_trade: float  Hard cap as fraction of equity (default 0.02 = 2%).
    entry_price       : float  Intended entry price for this specific trade.
    stop_loss_price   : float  Stop-loss level for this trade.
    position_side     : str    ``"LONG"`` or ``"SHORT"``.
    min_position_size : float  Minimum lot size; below this returns 0.
                               (default 0.001 for BTC).

    Raises
    ------
    ValueError
        If ``account_balance <= 0``, or ``entry_price == stop_loss_price``,
        or ``position_side`` is not ``"LONG"`` / ``"SHORT"``.
    """

    def __init__(
        self,
        account_balance   : float,
        win_rate          : float,
        avg_win           : float,
        avg_loss          : float,
        entry_price       : float,
        stop_loss_price   : float,
        position_side     : str,
        kelly_fraction    : float = 0.25,
        max_risk_per_trade: float = 0.02,
        min_position_size : float = 0.001,
    ) -> None:
        # ── Validation ─────────────────────────────────────────────────
        if account_balance <= 0.0:
            raise ValueError(
                f"account_balance must be > 0, got {account_balance}."
            )
        if entry_price == stop_loss_price:
            raise ValueError(
                "entry_price and stop_loss_price must not be equal — "
                "would cause division by zero in risk-per-unit calculation."
            )
        if position_side not in ("LONG", "SHORT"):
            raise ValueError(
                f"position_side must be 'LONG' or 'SHORT', got {position_side!r}."
            )

        self.account_balance    = float(account_balance)
        self.win_rate           = float(win_rate)
        self.avg_win            = float(avg_win)
        self.avg_loss           = float(avg_loss)
        self.entry_price        = float(entry_price)
        self.stop_loss_price    = float(stop_loss_price)
        self.position_side      = position_side.upper()
        self.kelly_fraction     = float(kelly_fraction)
        self.max_risk_per_trade = float(max_risk_per_trade)
        self.min_position_size  = float(min_position_size)

    # ------------------------------------------------------------------
    def compute(self) -> PositionSizeResult:
        """
        Execute the full position-sizing pipeline and return a result.

        Pipeline
        --------
        1. Guard: zero-edge inputs  -> return 0.
        2. Compute full Kelly f*.
        3. Guard: f* <= 0           -> return 0 (no positive edge).
        4. Apply fractional Kelly.
        5. Calculate risk_per_unit.
        6. Apply wide-stop penalty if stop > 20% from entry.
        7. Compute kelly_size and risk_capped_size.
        8. Final size = min(kelly_size, risk_capped_size).
        9. Guard: below min_position_size -> return 0.

        Returns
        -------
        PositionSizeResult
        """
        notes_parts: List[str] = []

        # ── 1. Zero-edge guard ─────────────────────────────────────────
        if self.win_rate == 0.0 or self.avg_loss == 0.0:
            logger.warning(
                "[PositionSizer] win_rate=%.3f avg_loss=%.4f — insufficient data. "
                "Returning size=0.", self.win_rate, self.avg_loss
            )
            return self._zero_result(
                "Insufficient data: win_rate or avg_loss is zero."
            )

        # ── 2. Full Kelly f* ───────────────────────────────────────────
        R = self.avg_win / self.avg_loss           # risk-reward ratio
        f_star = self.win_rate - (1.0 - self.win_rate) / R

        kelly_pct = f_star * 100.0                 # for output reporting

        logger.debug(
            "[PositionSizer] R=%.4f  f*=%.4f  kelly_pct=%.3f%%",
            R, f_star, kelly_pct,
        )

        # ── 3. No-edge guard ──────────────────────────────────────────
        if f_star <= 0.0:
            logger.info(
                "[PositionSizer] Full Kelly f*=%.4f <= 0 — no positive edge. "
                "Skipping trade.", f_star
            )
            return self._zero_result(
                f"No positive edge: f*={f_star:.4f} (R={R:.3f} win_rate={self.win_rate:.3f}).",
                kelly_pct=kelly_pct,
            )

        # ── 4. Fractional Kelly ────────────────────────────────────────
        f_used = f_star * self.kelly_fraction

        # ── 5. Risk per unit ──────────────────────────────────────────
        # For LONG:  risk is downside move (entry - stop). Must have stop < entry.
        # For SHORT: risk is upside move   (stop - entry). Must have stop > entry.
        raw_delta = self.entry_price - self.stop_loss_price   # positive for LONG

        if self.position_side == "LONG" and raw_delta <= 0.0:
            logger.warning(
                "[PositionSizer] LONG trade but stop_loss (%.4f) >= entry (%.4f). "
                "Invalid setup — returning 0.", self.stop_loss_price, self.entry_price
            )
            return self._zero_result(
                "LONG trade requires stop_loss_price < entry_price.",
                kelly_pct=kelly_pct, fraction_used=f_used,
            )

        if self.position_side == "SHORT" and raw_delta >= 0.0:
            logger.warning(
                "[PositionSizer] SHORT trade but stop_loss (%.4f) <= entry (%.4f). "
                "Invalid setup — returning 0.", self.stop_loss_price, self.entry_price
            )
            return self._zero_result(
                "SHORT trade requires stop_loss_price > entry_price.",
                kelly_pct=kelly_pct, fraction_used=f_used,
            )

        risk_per_unit = abs(raw_delta)             # always positive

        # ── 6. Wide-stop penalty ───────────────────────────────────────
        effective_max_risk = self.max_risk_per_trade
        wide_stop_threshold = self.entry_price * 0.20    # 20% away

        if risk_per_unit > wide_stop_threshold:
            effective_max_risk = self.max_risk_per_trade * 0.5   # halve to 1%
            note = (
                f"Wide stop detected: risk_per_unit={risk_per_unit:.4f} > "
                f"20% of entry ({wide_stop_threshold:.4f}). "
                f"Max risk halved to {effective_max_risk*100:.1f}%."
            )
            notes_parts.append(note)
            logger.info("[PositionSizer] %s", note)

        # ── 7. Sizing computations ─────────────────────────────────────
        max_loss_dollars    = self.account_balance * effective_max_risk
        kelly_position_size = (self.account_balance * f_used) / self.entry_price
        risk_capped_size    = max_loss_dollars / risk_per_unit

        # ── 8. Conservative final size ─────────────────────────────────
        final_size = min(kelly_position_size, risk_capped_size)

        # Determine which constraint was binding
        if risk_capped_size < kelly_position_size:
            notes_parts.append(
                f"Risk cap binding: capped at {risk_capped_size:.6f} units "
                f"(Kelly would have been {kelly_position_size:.6f})."
            )
        else:
            notes_parts.append(
                f"Kelly binding: {kelly_position_size:.6f} units "
                f"(risk cap allowed {risk_capped_size:.6f})."
            )

        # ── 9. Minimum size guard ──────────────────────────────────────
        if final_size < self.min_position_size:
            note = (
                f"Calculated size {final_size:.6f} below minimum lot "
                f"{self.min_position_size:.6f} — returning 0."
            )
            notes_parts.append(note)
            logger.info("[PositionSizer] %s", note)
            return self._zero_result(
                " | ".join(notes_parts),
                kelly_pct=kelly_pct,
                fraction_used=f_used,
                max_loss_dollars=max_loss_dollars,
            )

        # ── Actual risk % ──────────────────────────────────────────────
        actual_risk_dollars = final_size * risk_per_unit
        risk_pct = (actual_risk_dollars / self.account_balance) * 100.0

        logger.info(
            "[PositionSizer] %s %s | size=%.6f | risk=%.3f%% | "
            "kelly_f*=%.4f | f_used=%.4f | max_loss=$%.2f",
            self.position_side, "entry",
            final_size, risk_pct, f_star, f_used, actual_risk_dollars,
        )

        return PositionSizeResult(
            position_size    = round(final_size, 8),
            risk_percentage  = round(risk_pct, 6),
            kelly_percentage = round(kelly_pct, 6),
            fraction_used    = round(f_used, 6),
            max_loss_dollars = round(actual_risk_dollars, 4),
            is_trade_allowed = True,
            notes            = " | ".join(notes_parts),
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_rolling_history(
        cls,
        trades_history    : List[Dict],
        account_balance   : float,
        entry_price       : float,
        stop_loss_price   : float,
        position_side     : str,
        kelly_fraction    : float = 0.25,
        max_risk_per_trade: float = 0.02,
        min_position_size : float = 0.001,
        window            : int   = 100,
    ) -> "PositionSizer":
        """
        Construct a PositionSizer by computing rolling stats from trade history.

        Parameters
        ----------
        trades_history : list of dict
            Trade log entries; each must have ``'pnl_percent'`` and ``'side'``.
        account_balance : float
        entry_price     : float
        stop_loss_price : float
        position_side   : str
        kelly_fraction  : float
        max_risk_per_trade : float
        min_position_size  : float
        window          : int    Rolling window size (default 100).

        Returns
        -------
        PositionSizer
        """
        win_rate, avg_win, avg_loss = update_rolling_stats(trades_history, window)
        return cls(
            account_balance    = account_balance,
            win_rate           = win_rate,
            avg_win            = avg_win,
            avg_loss           = avg_loss,
            entry_price        = entry_price,
            stop_loss_price    = stop_loss_price,
            position_side      = position_side,
            kelly_fraction     = kelly_fraction,
            max_risk_per_trade = max_risk_per_trade,
            min_position_size  = min_position_size,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zero_result(
        self,
        notes            : str   = "",
        kelly_pct        : float = 0.0,
        fraction_used    : float = 0.0,
        max_loss_dollars : float = 0.0,
    ) -> PositionSizeResult:
        """Return a blocked/zero PositionSizeResult."""
        return PositionSizeResult(
            position_size    = 0.0,
            risk_percentage  = 0.0,
            kelly_percentage = kelly_pct,
            fraction_used    = fraction_used,
            max_loss_dollars = max_loss_dollars,
            is_trade_allowed = False,
            notes            = notes,
        )


# ---------------------------------------------------------------------------
# Convenience one-shot function
# ---------------------------------------------------------------------------

def compute_position_size(
    account_balance   : float,
    win_rate          : float,
    avg_win           : float,
    avg_loss          : float,
    entry_price       : float,
    stop_loss_price   : float,
    position_side     : str,
    kelly_fraction    : float = 0.25,
    max_risk_per_trade: float = 0.02,
    min_position_size : float = 0.001,
) -> PositionSizeResult:
    """
    One-shot convenience wrapper around :class:`PositionSizer`.

    Parameters
    ----------
    See :class:`PositionSizer` for full parameter descriptions.

    Returns
    -------
    PositionSizeResult
    """
    return PositionSizer(
        account_balance    = account_balance,
        win_rate           = win_rate,
        avg_win            = avg_win,
        avg_loss           = avg_loss,
        entry_price        = entry_price,
        stop_loss_price    = stop_loss_price,
        position_side      = position_side,
        kelly_fraction     = kelly_fraction,
        max_risk_per_trade = max_risk_per_trade,
        min_position_size  = min_position_size,
    ).compute()


# ---------------------------------------------------------------------------
# Demo / __main__
# ---------------------------------------------------------------------------

def _simulate_trades(
    n_trades   : int   = 500,
    base_wr    : float = 0.55,
    base_rr    : float = 1.8,
    wr_noise   : float = 0.08,
    rr_noise   : float = 0.3,
    seed       : int   = 42,
) -> List[Dict]:
    """
    Simulate a sequence of trade results with evolving win rates and R:R.

    Parameters
    ----------
    n_trades  : int    Number of trades to simulate.
    base_wr   : float  Base win rate (e.g. 0.55).
    base_rr   : float  Base risk-reward (e.g. 1.8 → avg_win = 1.8 * avg_loss).
    wr_noise  : float  Gaussian noise on win rate per trade.
    rr_noise  : float  Gaussian noise on risk-reward per trade.
    seed      : int    NumPy random seed.

    Returns
    -------
    list of dict  Each entry: {'pnl_percent': float, 'side': str}
    """
    rng    = np.random.default_rng(seed)
    trades = []

    # Regime: deteriorate win rate in the middle third to test safety caps
    for i in range(n_trades):
        # Regime shift: lower win rate from trade 150-300 (simulate drawdown phase)
        if 150 <= i < 300:
            regime_wr = base_wr - 0.18
        else:
            regime_wr = base_wr

        wr   = float(np.clip(regime_wr + rng.normal(0, wr_noise), 0.05, 0.95))
        rr   = float(max(0.1, base_rr + rng.normal(0, rr_noise)))
        side = rng.choice(["LONG", "SHORT"])
        won  = rng.random() < wr

        avg_l = 0.01    # 1% base loss magnitude
        avg_w = avg_l * rr

        if won:
            pnl = float(rng.uniform(avg_w * 0.5, avg_w * 1.5))
        else:
            pnl = -float(rng.uniform(avg_l * 0.5, avg_l * 1.5))

        trades.append({"pnl_percent": round(pnl, 5), "side": side})

    return trades


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level  = logging.WARNING,
        format = "[%(levelname)s] %(name)s: %(message)s",
    )

    N        = int(sys.argv[1])  if len(sys.argv) > 1 else 500
    BALANCE  = float(sys.argv[2]) if len(sys.argv) > 2 else 100_000.0
    WINDOW   = int(sys.argv[3])  if len(sys.argv) > 3 else 100

    print(f"\n{'='*72}")
    print(f"  Q-SonicFX  |  Position Sizer — Fractional Kelly Demo")
    print(f"  Account: ${BALANCE:,.0f}  |  {N} simulated trades  |  Window: {WINDOW}")
    print(f"{'='*72}\n")

    # ── Generate trade history ─────────────────────────────────────────
    all_trades = _simulate_trades(n_trades=N)

    # ── Demo scenarios ─────────────────────────────────────────────────
    SCENARIOS = [
        # (label,           entry,   stop,    side,    snap_after_trade_idx)
        ("Early (50 trades)",  65_000, 64_500,  "LONG",   50),
        ("Mid-dip (175 trades)", 65_000, 64_500, "LONG",  175),
        ("Recovery (350 trades)", 3_500, 3_450,  "LONG",  350),
        ("Wide stop (20%+)",  3_500,  2_750,   "LONG",  400),
        ("SHORT trade",       3_500,  3_560,   "SHORT", 450),
        ("Full history (500)", 65_000, 64_500, "LONG",  499),
    ]

    header = (
        f"{'Scenario':<26} {'WR%':>5} {'AvgW%':>6} {'AvgL%':>6} "
        f"{'f*%':>6} {'Used%':>6} {'Size':>10} {'Risk%':>7} "
        f"{'MaxLoss$':>9} {'OK?':>5}"
    )
    print(header)
    print("-" * len(header))

    for label, entry, stop, side, snap_idx in SCENARIOS:
        snapshot = all_trades[: snap_idx + 1]
        wr, aw, al = update_rolling_stats(snapshot, window=WINDOW)

        try:
            sizer = PositionSizer(
                account_balance    = BALANCE,
                win_rate           = wr,
                avg_win            = aw,
                avg_loss           = al,
                entry_price        = entry,
                stop_loss_price    = stop,
                position_side      = side,
                kelly_fraction     = 0.25,
                max_risk_per_trade = 0.02,
                min_position_size  = 0.001,
            )
            r = sizer.compute()
        except ValueError as e:
            print(f"  {label:<26} ERROR: {e}")
            continue

        print(
            f"  {label:<26} "
            f"{wr*100:>5.1f} "
            f"{aw*100:>6.3f} "
            f"{al*100:>6.3f} "
            f"{r.kelly_percentage:>6.2f} "
            f"{r.fraction_used*100:>6.3f} "
            f"{r.position_size:>10.5f} "
            f"{r.risk_percentage:>7.3f} "
            f"${r.max_loss_dollars:>8.2f} "
            f"{'YES' if r.is_trade_allowed else 'NO':>5}"
        )
        if r.notes:
            for note in r.notes.split(" | "):
                if note:
                    print(f"    -> {note}")

    # ── Edge-case demos ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Edge-case demonstrations:")
    print(f"{'='*72}\n")

    edge_cases = [
        ("Zero win_rate",       dict(win_rate=0.0,  avg_win=0.02,  avg_loss=0.01)),
        ("Zero avg_loss",       dict(win_rate=0.55, avg_win=0.02,  avg_loss=0.0)),
        ("Negative edge f*",    dict(win_rate=0.3,  avg_win=0.005, avg_loss=0.02)),
        ("Below min lot",       dict(win_rate=0.52, avg_win=0.001, avg_loss=0.0009)),
    ]

    for label, kwargs in edge_cases:
        try:
            base = dict(
                account_balance=BALANCE, entry_price=65_000,
                stop_loss_price=64_500, position_side="LONG",
            )
            base.update(kwargs)
            r = PositionSizer(**base).compute()
            status = "ALLOWED" if r.is_trade_allowed else "BLOCKED"
            print(f"  [{status}] {label:<25} size={r.position_size:.6f}")
            if r.notes:
                print(f"           Notes: {r.notes}")
        except ValueError as e:
            print(f"  [RAISED ] {label:<25} ValueError: {e}")

    # ValueError triggers
    print()
    for label, kw in [
        ("entry == stop",   dict(entry_price=65_000, stop_loss_price=65_000)),
        ("balance <= 0",    dict(account_balance=-1.0)),
    ]:
        try:
            base = dict(
                account_balance=BALANCE, win_rate=0.55, avg_win=0.02, avg_loss=0.01,
                entry_price=65_000, stop_loss_price=64_500, position_side="LONG",
            )
            base.update(kw)
            PositionSizer(**base).compute()
        except ValueError as e:
            print(f"  [RAISED ] {label:<25} ValueError: {e}")

    # ── Safety cap demonstration ───────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Safety cap: evolving Kelly vs. 2% hard cap across trade history")
    print(f"{'='*72}\n")

    print(f"  {'Trades':>7} {'WinRate':>8} {'f*%':>7} {'KellySize':>11} "
          f"{'CapSize':>10} {'Final':>10} {'Binding':>10}")
    print("  " + "-" * 68)

    ENTRY = 65_000.0
    STOP  = 64_500.0    # $500 / unit risk

    for n in range(20, N + 1, 40):
        wr, aw, al = update_rolling_stats(all_trades[:n], window=WINDOW)
        if wr == 0.0 or al == 0.0:
            continue
        R = aw / al
        f_star = wr - (1.0 - wr) / R
        if f_star <= 0:
            continue
        f_used = f_star * 0.25
        kelly_sz = (BALANCE * f_used) / ENTRY
        cap_sz   = (BALANCE * 0.02) / (ENTRY - STOP)
        final    = min(kelly_sz, cap_sz)
        binding  = "RiskCap" if cap_sz < kelly_sz else "Kelly"
        print(
            f"  {n:>7} {wr*100:>8.1f}% {f_star*100:>7.2f}  "
            f"{kelly_sz:>11.5f} {cap_sz:>10.5f} {final:>10.5f} {binding:>10}"
        )

    print(f"\n{'='*72}\n")
