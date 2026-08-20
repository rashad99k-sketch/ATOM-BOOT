"""Centralized environment configuration for RF Liquidity Pro."""
from __future__ import annotations
import os


def as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_settings():
    return {
        "paper_mode": as_bool(os.getenv("PAPER_MODE"), True),
        "max_open_positions": max(1, int(os.getenv("MAX_OPEN_POSITIONS", "6"))),
        "deep_scan_enabled": as_bool(os.getenv("DEEP_SCAN_ENABLED"), True),
        "deep_scan_interval_sec": int(os.getenv("GLOBAL_SCAN_INTERVAL_SEC", os.getenv("DEEP_SCAN_INTERVAL_SEC", "1200"))),
        "deep_scan_max_symbols": int(os.getenv("DEEP_WATCHLIST_SIZE", os.getenv("DEEP_SCAN_MAX_SYMBOLS", "60"))),
        "deep_scan_radar_symbols": int(os.getenv("DEEP_SCAN_RADAR_SYMBOLS", "240")),
        "radar_symbols": [x.strip() for x in os.getenv("RADAR_SYMBOLS", "").split(",") if x.strip()],
        "deep_scan_batch_size": int(os.getenv("WATCHLIST_DEEP_BATCH_SIZE", os.getenv("DEEP_SCAN_BATCH_SIZE", "10"))),
        "portfolio_margin_cap_pct": float(os.getenv("PORTFOLIO_MARGIN_CAP_PCT", "0.60")),
        "position_margin_pct": float(os.getenv("POSITION_MARGIN_PCT", "0.10")),
        "news_enabled": as_bool(os.getenv("NEWS_ENABLED"), True),
        "news_risk_block": float(os.getenv("NEWS_RISK_BLOCK", "80")),
        "watchlist_deep_interval_sec": int(os.getenv("WATCHLIST_DEEP_INTERVAL_SEC", "10")),
        "watchlist_queue_min_score": float(os.getenv("WATCHLIST_QUEUE_MIN_SCORE", "8.0")),
        "port": int(os.getenv("PORT", "8000")),
        "dashboard_control_token_configured": bool(os.getenv("DASHBOARD_CONTROL_TOKEN", "").strip()),
    }
