#!/usr/bin/env python3
"""
database.py
===========
Q-SonicFX -- Cloud Supabase Persistence Layer
=============================================
Replaces local SQLite with Supabase PostgreSQL client.

Tables:
    trades            -- Every trade open/close with full metadata
    signals           -- Every regime + OBI signal generated
    equity_snapshots  -- Equity curve snapshots
    bot_events        -- State change audit log
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

logger = logging.getLogger("qsonicfx.database")

supabase_url: str = os.getenv("SUPABASE_URL", "")
supabase_key: str = os.getenv("SUPABASE_ANON_KEY", "")
supabase: Optional[Client] = None


async def init_db() -> Optional[Client]:
    """Initialize the Supabase client connection."""
    global supabase
    if not supabase_url or not supabase_key:
        logger.warning("[Database] SUPABASE_URL or SUPABASE_ANON_KEY missing in .env — cloud persistence disabled")
        return None
    try:
        supabase = create_client(supabase_url, supabase_key)
        logger.info("[Database] Connected to Supabase project at %s", supabase_url)
        print(f"[Database] Connected to Supabase project")
        return supabase
    except Exception as e:
        logger.error("[Database] Supabase connection failed: %s", e)
        return None


async def get_db() -> Client:
    """Return the Supabase client instance."""
    global supabase
    if supabase is None:
        if supabase_url and supabase_key:
            supabase = create_client(supabase_url, supabase_key)
        else:
            raise RuntimeError("Database not initialized. Call init_db() first.")
    return supabase


async def save_trade(trade_data: dict) -> int:
    """Insert a trade record. Returns the inserted ID."""
    try:
        client = await get_db()
        result = client.table("trades").insert(trade_data).execute()
        return result.data[0]["id"] if result.data else 0
    except Exception as e:
        logger.warning("[Database] save_trade failed: %s", e)
        return 0


async def close_trade(trade_id: int, exit_price: float, pnl: float, pnl_percent: float):
    """Update an open trade with exit data."""
    try:
        client = await get_db()
        client.table("trades").update({
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "status": "CLOSED"
        }).eq("id", trade_id).execute()
    except Exception as e:
        logger.warning("[Database] close_trade failed: %s", e)


async def save_equity_snapshot(snapshot: dict):
    """Insert an equity snapshot row."""
    try:
        client = await get_db()
        client.table("equity_snapshots").insert(snapshot).execute()
    except Exception as e:
        logger.warning("[Database] save_equity_snapshot failed: %s", e)


async def save_signal(signal_data: dict) -> int:
    """Insert a signal record. Returns the inserted ID."""
    try:
        client = await get_db()
        result = client.table("signals").insert(signal_data).execute()
        return result.data[0]["id"] if result.data else 0
    except Exception as e:
        logger.warning("[Database] save_signal failed: %s", e)
        return 0


async def log_event(event_type: str, message: str, details: dict = None):
    """Insert a bot event log entry."""
    try:
        client = await get_db()
        client.table("bot_events").insert({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "message": message,
            "details": details or {}
        }).execute()
    except Exception as e:
        logger.warning("[Database] log_event failed: %s", e)


async def get_trades(limit: int = 100, offset: int = 0, status: str = None) -> list:
    """Fetch recent trades."""
    try:
        client = await get_db()
        query = client.table("trades").select("*").order("id", desc=True).limit(limit).offset(offset)
        if status:
            query = query.eq("status", status)
        result = query.execute()
        return result.data or []
    except Exception as e:
        logger.warning("[Database] get_trades failed: %s", e)
        return []


async def get_equity_curve(start_date: str = None, end_date: str = None, limit: int = 500) -> list:
    """Fetch equity curve data points."""
    try:
        client = await get_db()
        query = client.table("equity_snapshots").select("*").order("id", desc=True).limit(limit)
        result = query.execute()
        data = result.data or []
        return list(reversed(data))  # Return in chronological order
    except Exception as e:
        logger.warning("[Database] get_equity_curve failed: %s", e)
        return []


async def get_recent_signals(limit: int = 50) -> list:
    """Fetch recent trading signals."""
    try:
        client = await get_db()
        result = client.table("signals").select("*").order("id", desc=True).limit(limit).execute()
        return result.data or []
    except Exception as e:
        logger.warning("[Database] get_recent_signals failed: %s", e)
        return []


async def get_performance_summary() -> dict:
    """Fetch aggregate performance stats."""
    try:
        client = await get_db()
        trades = client.table("trades").select("*").eq("status", "CLOSED").execute()
        data = trades.data or []
        if not data:
            return {"total_trades": 0, "win_rate": 0, "profit_factor": 0, "total_pnl": 0, "max_drawdown": 0}

        wins = [t for t in data if (t.get("pnl") or 0) > 0]
        losses = [t for t in data if (t.get("pnl") or 0) < 0]
        total_pnl = sum(t.get("pnl") or 0 for t in data)
        gross_profit = sum(t.get("pnl") or 0 for t in wins)
        gross_loss = abs(sum(t.get("pnl") or 0 for t in losses)) or 1

        return {
            "total_trades": len(data),
            "win_rate": round(len(wins) / len(data), 4) if data else 0,
            "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss else 0,
            "total_pnl": round(total_pnl, 4),
            "max_drawdown": 0  # Calculated from equity curve separately
        }
    except Exception as e:
        logger.warning("[Database] get_performance_summary failed: %s", e)
        return {"total_trades": 0, "win_rate": 0, "profit_factor": 0, "total_pnl": 0, "max_drawdown": 0}


async def get_recent_events(limit: int = 50, event_type: str = None) -> list:
    """Fetch recent bot events."""
    try:
        client = await get_db()
        query = client.table("bot_events").select("*").order("id", desc=True).limit(limit)
        if event_type:
            query = query.eq("event_type", event_type)
        result = query.execute()
        return result.data or []
    except Exception as e:
        logger.warning("[Database] get_recent_events failed: %s", e)
        return []
