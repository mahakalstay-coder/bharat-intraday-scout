"""Market-regime context (Nifty trend + India VIX) logged with every signal.

This is a *feature for the learning journal*, not a trading gate: it lets the
journal later measure whether setups behave differently in UP/DOWN/FLAT or
low/high-volatility sessions. Any failure returns UNKNOWN and never blocks a scan.
"""
from __future__ import annotations
import os
import time
from typing import Any, Dict, Optional

import pandas as pd

TREND_PCT = float(os.getenv("REGIME_TREND_PCT", "0.40"))
VIX_LOW = float(os.getenv("REGIME_VIX_LOW", "13"))
VIX_HIGH = float(os.getenv("REGIME_VIX_HIGH", "18"))
_CACHE: Dict[str, Any] = {"at": 0.0, "day": None, "value": None}
CACHE_SECONDS = 300


def classify(nifty_change_pct: Optional[float], vix: Optional[float]) -> Dict[str, Any]:
    if nifty_change_pct is None:
        trend = "UNKNOWN"
    elif nifty_change_pct >= TREND_PCT:
        trend = "UP"
    elif nifty_change_pct <= -TREND_PCT:
        trend = "DOWN"
    else:
        trend = "FLAT"
    if vix is None:
        vix_bucket = "UNKNOWN"
    elif vix < VIX_LOW:
        vix_bucket = "LOW"
    elif vix > VIX_HIGH:
        vix_bucket = "HIGH"
    else:
        vix_bucket = "NORMAL"
    return {"nifty_trend": trend, "vix_bucket": vix_bucket,
            "nifty_change_pct": None if nifty_change_pct is None else round(float(nifty_change_pct), 2),
            "vix": None if vix is None else round(float(vix), 2)}


def _last_two(df: pd.DataFrame, today: str):
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) < 2 or str(close.index[-1].date()) != today:
        return None, None  # stale (holiday / feed lag): do not guess
    return float(close.iloc[-1]), float(close.iloc[-2])


def current_regime(now, download=None) -> Dict[str, Any]:
    """Cached, exception-safe regime snapshot for the current IST session."""
    today = now.date().isoformat()
    if _CACHE["day"] == today and time.time() - _CACHE["at"] < CACHE_SECONDS and _CACHE["value"]:
        return _CACHE["value"]
    try:
        if download is None:
            import yfinance as yf
            download = lambda sym: yf.download(sym, period="10d", interval="1d",
                                               progress=False, auto_adjust=False, threads=False)
        n_last, n_prev = _last_two(download("^NSEI"), today)
        change = None if n_last is None else (n_last / n_prev - 1.0) * 100.0
        try:
            v_last, _ = _last_two(download("^INDIAVIX"), today)
        except Exception:
            v_last = None
        value = classify(change, v_last)
    except Exception:
        value = classify(None, None)
    _CACHE.update(at=time.time(), day=today, value=value)
    return value
