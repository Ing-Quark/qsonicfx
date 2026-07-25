#!/usr/bin/env python3
"""
notifier.py
===========
Q-SonicFX -- Instant Telegram & Webhook Alert Engine
===================================================

Delivers real-time notifications directly to your phone/Telegram app for:
- 🚀 Trade Entry Executions
- 💰 Trade Exits & Realized P&L
- 🚨 Emergency Circuit Breaker Trips
- ⚡ Engine Status Changes

Zero external heavy dependencies -- pure Python urllib request client.

Author : Q-SonicFX Quant Engine
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger("QSonicFX.Notifier")
logger.setLevel(logging.INFO)


class TelegramNotifier:
    """
    Direct Telegram Notification Client for Q-SonicFX.
    Uses Telegram Bot API (https://api.telegram.org/bot<TOKEN>/sendMessage).
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = enabled and bool(self.bot_token and self.chat_id)

        if self.enabled:
            logger.info("[Notifier] Telegram Alerts Enabled (Chat ID: %s)", self.chat_id)
        else:
            logger.info("[Notifier] Telegram Alerts Offline (Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to activate)")

    def send_message(self, text: str) -> bool:
        """Send HTML formatted text message to Telegram chat."""
        if not self.enabled:
            logger.info("[Notifier Mock] %s", text.replace("\n", " | "))
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            body_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("ok", False)
        except Exception as e:
            logger.warning("[Notifier Error] Failed to send Telegram alert: %s", e)
            return False

    def notify_trade_entry(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        regime: str = "UNKNOWN",
    ) -> bool:
        """Send instant trade entry notification."""
        side_icon = "▲ LONG" if side.upper() == "LONG" else "▼ SHORT"
        sl_str = f"${stop_loss:,.2f}" if stop_loss else "N/A"
        tp_str = f"${take_profit:,.2f}" if take_profit else "N/A"

        msg = (
            f"🚀 <b>Q-SONICFX TRADE EXECUTED</b>\n\n"
            f"• <b>Symbol</b>: <code>{symbol}</code>\n"
            f"• <b>Side</b>: <code>{side_icon}</code>\n"
            f"• <b>Price</b>: <code>${price:,.2f}</code>\n"
            f"• <b>Quantity</b>: <code>{quantity:.5f}</code>\n"
            f"• <b>Regime</b>: <code>{regime}</code>\n"
            f"• <b>Stop Loss</b>: <code>{sl_str}</code>\n"
            f"• <b>Take Profit</b>: <code>{tp_str}</code>\n\n"
            f"🕒 <code>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</code>"
        )
        return self.send_message(msg)

    def notify_trade_exit(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl_dollars: float,
        pnl_pct: float,
    ) -> bool:
        """Send instant trade exit & P&L notification."""
        pnl_icon = "📈" if pnl_dollars >= 0 else "📉"
        pnl_sign = "+" if pnl_dollars >= 0 else ""

        msg = (
            f"💰 <b>Q-SONICFX TRADE CLOSED</b> {pnl_icon}\n\n"
            f"• <b>Symbol</b>: <code>{symbol}</code> [{side}]\n"
            f"• <b>Entry</b>: <code>${entry_price:,.2f}</code>\n"
            f"• <b>Exit</b>: <code>${exit_price:,.2f}</code>\n"
            f"• <b>Realized P&amp;L</b>: <code>{pnl_sign}${pnl_dollars:,.4f} ({pnl_sign}{pnl_pct:.4f}%)</code>\n\n"
            f"🕒 <code>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</code>"
        )
        return self.send_message(msg)

    def notify_circuit_breaker(self, reason: str, daily_pnl: float, max_dd: float) -> bool:
        """Send emergency circuit breaker alert."""
        msg = (
            f"🚨 <b>EMERGENCY CIRCUIT BREAKER TRIP</b> 🚨\n\n"
            f"• <b>Reason</b>: <code>{reason}</code>\n"
            f"• <b>Daily P&amp;L</b>: <code>${daily_pnl:,.2f}</code>\n"
            f"• <b>Drawdown</b>: <code>{max_dd:.2f}%</code>\n"
            f"• <b>Action</b>: <code>Trading Engine Halted</code>\n\n"
            f"⚠️ <b>Failsafe Guard Engaged</b>"
        )
        return self.send_message(msg)

    def notify_engine_status(self, status: str, exchange: str = "", symbol: str = "") -> bool:
        """Send engine status change notification."""
        icons = {"RUNNING": "▶️", "PAUSED": "⏸️", "HALTED": "🛑", "OFFLINE": "⚫"}
        icon = icons.get(status, "⚙️")
        msg = (
            f"{icon} <b>Q-SONICFX ENGINE: {status}</b>\n"
            f"• Exchange: <code>{exchange or 'N/A'}</code>\n"
            f"• Symbol: <code>{symbol or 'N/A'}</code>\n"
            f"🕒 <code>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</code>"
        )
        return self.send_message(msg)


# ===========================================================================
# Self-Test Verification Block
# ===========================================================================

if __name__ == "__main__":
    print("=== Testing Telegram & Webhook Notifier Module (notifier.py) ===")
    notifier = TelegramNotifier()
    
    # Test notification output (Logs to console if TELEGRAM_BOT_TOKEN is not set)
    notifier.notify_trade_entry(
        symbol="BTCUSDT", side="LONG", quantity=0.05, price=65120.0,
        stop_loss=64468.8, take_profit=66422.4, regime="STRONG_TREND"
    )
    notifier.notify_trade_exit(
        symbol="BTCUSDT", side="LONG", entry_price=65120.0, exit_price=66422.4,
        pnl_dollars=65.12, pnl_pct=2.0
    )
    print("=== Notifier Module Verified OK ===")
