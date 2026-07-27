#!/usr/bin/env python3
"""
account_detector.py
===================
Q-SonicFX Auto Account Type & Mode Detector
===========================================

Probes the exchange connector to determine:
 - Account type (UNIFIED, FUND, SPOT, DERIVATIVES)
 - Current balance & available margin
 - Recommended trading mode (spot vs futures)

For Bybit accounts with < $10 USDT, defaults to spot-compatible sizing
to avoid margin/liquidation requirements for very small capital.

Author : Q-SonicFX Quant Engine
"""

from __future__ import annotations
import logging
import asyncio
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("qsonicfx.account_detector")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AccountProfile:
    """Detected account characteristics."""
    account_type         : str    # "UNIFIED", "SPOT", "FUND", "DERIVATIVES"
    balance              : float  # Total equity / wallet balance (USDT)
    available_balance    : float  # Available for new orders
    mode                 : str    # "futures" or "spot"
    min_notional_default : float = 5.0  # Minimum order notional in USD


# ---------------------------------------------------------------------------
# AccountDetector
# ---------------------------------------------------------------------------

class AccountDetector:
    """
    Detects account type and recommends trading mode based on balance.

    Logic
    -----
    1. Call exchange.fetch_balance() to retrieve live balance.
    2. If balance < $10 → mode="spot"  (avoid margin/liquidation risks).
    3. Otherwise        → mode="futures" (full perpetual access).
    """

    async def detect(self, exchange: Any) -> AccountProfile:
        """
        Run account detection against the provided exchange client.

        Parameters
        ----------
        exchange : BaseExchangeClient  Live or simulated connector.

        Returns
        -------
        AccountProfile
        """
        balance_data: dict = {}

        try:
            if hasattr(exchange, "fetch_balance"):
                balance_data = await asyncio.to_thread(exchange.fetch_balance)
        except Exception as exc:
            logger.warning("[AccountDetector] fetch_balance failed: %s", exc)

        bal   = float(balance_data.get("balance",          0.0) or 0.0)
        eq    = float(balance_data.get("equity",           bal) or bal)
        avail = float(balance_data.get("available_margin", bal) or bal)

        # Determine account type label (Bybit always returns UNIFIED via connector)
        acct_type = "UNIFIED"

        # Recommend mode based on balance size
        if eq < 10.0:
            mode = "spot"
            min_notional = 1.0
            logger.info(
                "[AccountDetector] Small balance $%.4f — recommending SPOT mode "
                "(avoids margin/liquidation requirements)",
                eq,
            )
        else:
            mode = "futures"
            min_notional = 5.0
            logger.info(
                "[AccountDetector] Balance $%.4f — FUTURES mode",
                eq,
            )

        return AccountProfile(
            account_type         = acct_type,
            balance              = eq  if eq  > 0 else bal,
            available_balance    = avail,
            mode                 = mode,
            min_notional_default = min_notional,
        )


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def should_trade_spot(balance: float) -> bool:
    """
    Return True if the balance is too small for futures margin requirements.

    Parameters
    ----------
    balance : float  Current USDT balance.

    Returns
    -------
    bool
    """
    return balance < 10.0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio, os

    async def _test() -> None:
        from exchange_connector import get_exchange_client
        client  = get_exchange_client(
            mode       = "BYBIT_LIVE",
            api_key    = os.getenv("BYBIT_API_KEY", ""),
            secret_key = os.getenv("BYBIT_SECRET_KEY", ""),
        )
        detector = AccountDetector()
        profile  = await detector.detect(client)
        print(f"Account type      : {profile.account_type}")
        print(f"Balance           : ${profile.balance:.4f} USDT")
        print(f"Available         : ${profile.available_balance:.4f} USDT")
        print(f"Trading mode      : {profile.mode}")
        print(f"Min notional      : ${profile.min_notional_default:.2f}")
        print(f"Should spot?      : {should_trade_spot(profile.balance)}")

    asyncio.run(_test())
