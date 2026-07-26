"""
account_detector.py
===================
Q-SonicFX — Autonomous Account Type & Balance Detector
=======================================================

Probes the exchange to determine:
  1. Account type: Spot vs Unified (Futures+Spot)
  2. Minimum tradeable balance
  3. Available mode: Cross margin, isolated, or spot only

Radically simplifies: if balance < $10 → force SPOT mode
Because Bybit spot has lower minimum notionals (~$1) vs futures ($5+ for altcoins).

Author : Q-SonicFX Quant Engine
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("QSonicFX.AccountDetector")

SPOT_MODE_MIN_BALANCE = 10.0   # $10 threshold: below → force spot


@dataclass
class AccountProfile:
    account_type: str        # "SPOT" | "UNIFIED" | "CONTRACT"
    balance: float           # total USDT wallet balance
    available_balance: float # free margin / available to trade
    mode: str                # "spot" | "futures"
    suggested_pair_category: str  # "spot" | "linear"
    min_notional_default: float   # $1 for spot, $5 for futures


class AccountDetector:
    """
    Detects account configuration and sets optimal trading mode.
    Single probe: fetch unified account coin balance.
    - If Unified balance > 0 → Unified account (futures + spot)
    - If spot balance > 0 but Unified = 0 → Spot-only account
    - If both = 0 → new account, default to spot (lower barriers)
    """

    async def detect(self, exchange: Any) -> AccountProfile:
        """
        Run detection probes against the exchange.
        exchange.fetch_balance() already returns unified + fund balances.
        We extend it with a simple heuristic.
        """
        bal_data = await asyncio.to_thread(exchange.fetch_balance)
        total_bal = bal_data.get("balance", 0.0)
        available = bal_data.get("available_margin", 0.0)

        # If we can't get real balance, use the cached value
        if total_bal <= 0:
            total_bal = available

        # Try to detect account type from Bybit's V5 account info endpoint
        account_type = "UNIFIED"  # default for Bybit V5
        try:
            from exchange_connector import BybitLinearConnector
            if isinstance(exchange, BybitLinearConnector):
                # Check wallet balance structure
                res = await asyncio.to_thread(
                    exchange._request,
                    "GET",
                    "/v5/account/wallet-balance",
                    {"accountType": "UNIFIED", "coin": "USDT"},
                )
                if res.get("retCode") == 0:
                    account_type = "UNIFIED"
                else:
                    account_type = "SPOT"
        except Exception:
            account_type = "SPOT"

        # Balance-based mode selection
        if total_bal < SPOT_MODE_MIN_BALANCE:
            mode = "spot"
            category = "spot"
            min_notional = 1.0
            logger.info(
                "[Account] Balance=$%.2f < $%d → FORCE SPOT mode (min_notional=$%.1f)",
                total_bal, SPOT_MODE_MIN_BALANCE, min_notional,
            )
        else:
            mode = "futures"
            category = "linear"
            min_notional = 5.0
            logger.info(
                "[Account] Balance=$%.2f ≥ $%d → FUTURES mode (min_notional=$%.1f)",
                total_bal, SPOT_MODE_MIN_BALANCE, min_notional,
            )

        return AccountProfile(
            account_type=account_type,
            balance=total_bal,
            available_balance=available,
            mode=mode,
            suggested_pair_category=category,
            min_notional_default=min_notional,
        )


def should_trade_spot(balance: float) -> bool:
    """Quick static check: true if balance < $10."""
    return balance < SPOT_MODE_MIN_BALANCE
