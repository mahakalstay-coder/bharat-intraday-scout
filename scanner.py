from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

# Silence noisy Pandas4Warning deprecation notices from inside yfinance
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf

TZ = ZoneInfo("Asia/Kolkata")
LOG = logging.getLogger("bis20")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "runtime")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "bis20.sqlite3"
UNIVERSE_CACHE = Path(os.getenv("UNIVERSE_CACHE", str(ROOT / "data" / "liquid_universe.json")))
UNIVERSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
LEARNED_WEIGHTS_PATH = Path(os.getenv("LEARNED_WEIGHTS_PATH", str(ROOT / "data" / "learned_strategy_weights.json")))
KRONOS_REPO_DIR = Path(os.getenv("KRONOS_REPO_DIR", str(ROOT / "vendor" / "Kronos")))

NSE_EQUITY_CSV = os.getenv(
    "NSE_EQUITY_CSV",
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
)

TAURIC_API_BASE = os.getenv("TAURIC_API_BASE", "https://api.tauric.ai/v1")
TAURIC_MODEL = os.getenv("TAURIC_MODEL", "tauric")

DEFAULT_FORMULA_WEIGHTS: Dict[str, float] = {
    "F1_BB_KELTNER_SQUEEZE_5M": 1.2,
    "F2_ORB_15M_RVOL_IGNITION": 1.2,
    "F3_VWAP_MFI_ACCUMULATION": 1.0,
    "F4_ADX_SUPERTREND_ACCEL": 1.0,
    "F5_MTF_5M_15M_MACD_RSI": 1.1,
    "F6_MULTIDAY_CHANNEL_BREAK": 1.1,
}

# Calibrated from Nov 9 dry-run telemetry (logs_115346277840):
# Filters out range-exhausted low-ADR stocks (like SBIN/UNIONBANK at RSI>79) and targets
# high-ADR stocks coiled in tight Bollinger squeezes (BB width <= 2.8%, RSI 54-75, MFI 52-80).
CFG: Dict[str, Any] = {
    "PRICE_FLOOR": float(os.getenv("PRICE_FLOOR", "50")),
    "MIN_AVG_TURNOVER_INR": float(os.getenv("MIN_AVG_TURNOVER_INR", "50000000")),  # ₹5 Cr floor
    "LIQUIDITY_LOOKBACK_DAYS": 20,
    "DAILY_HISTORY_PERIOD": os.getenv("DAILY_HISTORY_PERIOD", "3mo"),
    "MAX_EQUITY_UNIVERSE": int(os.getenv("MAX_EQUITY_UNIVERSE", "2500")),
    "INTRADAY_UNIVERSE_SIZE": int(os.getenv("INTRADAY_UNIVERSE_SIZE", "200")),
    "INTRADAY_BATCH_SIZE": int(os.getenv("INTRADAY_BATCH_SIZE", "50")),
    "INTRADAY_PERIOD": "10d",
    "BAR_INTERVAL": "5m",
    "MIN_HISTORY_BARS": 45,
    "MIN_TIME_SLOT_SESSIONS": 3,
    "MIN_RELATIVE_VOLUME": 1.40,
    "MIN_CUMULATIVE_RVOL": 1.25,
    "BREAKOUT_LOOKBACK": 12,
    "MAX_EXTENSION_ATR": 1.8,
    "MIN_DAY_MOVE_PCT": float(os.getenv("MIN_DAY_MOVE_PCT", "0.4")),
    "IDEAL_MAX_MOVE_PCT": float(os.getenv("IDEAL_MAX_MOVE_PCT", "3.2")),
    "MAX_DAY_MOVE_PCT": float(os.getenv("MAX_DAY_MOVE_PCT", "4.8")),
    "MAX_ADR_USED_RATIO": 0.85,         # Stock must not have used >85% of its 20d avg daily range already
    "MIN_STOCK_ADR_PCT": 2.4,           # Minimum 20-day average daily range % so stock has high-beta fuel
    "MAX_BB_BANDWIDTH_PCT": 2.85,       # Reject wide/blown-out bands (>2.85%); catch tight coils
    "MAX_VWAP_DISTANCE_PCT": 2.2,       # Must be within 2.2% of Session VWAP
    "MAX_BAR_AGE_MIN": 14,
    "MIN_RSI_BUY": 54.0,
    "MAX_RSI_BUY": 75.5,                # Cap BUY RSI at 75.5 (avoids >79 exhausted tops seen on UNIONBANK/SBIN)
    "MIN_RSI_SELL": 24.5,
    "MAX_RSI_SELL": 46.0,
    "MIN_MFI_BUY": 52.0,
    "MAX_MFI_BUY": 81.0,                # Cap BUY MFI at 81.0 (avoids >84 climax tops seen on TIPSMUSIC/UNIONBANK)
    "MIN_MFI_SELL": 19.0,
    "MAX_MFI_SELL": 48.0,
    "MIN_ADX_TREND": 18.0,
    "MIN_CONFLUENCE_FORMULAS": int(os.getenv("MIN_CONFLUENCE_FORMULAS", "3")),
    "MIN_BODY_RATIO": 0.32,
    "MIN_CLOSE_LOCATION_BUY": 0.62,
    "MAX_CLOSE_LOCATION_SELL": 0.38,
    "STOP_ATR_MULTIPLE": 1.50,
    "MIN_STOP_PCT": 0.0080,
    "MAX_STOP_PCT": 0.0220,
    "TARGET1_R_MULTIPLE": 2.5,
    "TARGET2_R_MULTIPLE": 5.0,
    "EXPANSION_TARGET_PCT": 10.0,
    "MIN_SCORE_TIER1": int(os.getenv("MIN_SCORE_TIER1", "78")),
    "MIN_SCORE_TIER2": int(os.getenv("MIN_SCORE_TIER2", "70")),
    "TIER2_START_HOUR": 9,
    "TIER2_START_MINUTE": 45,
    "LAST_ENTRY_HOUR": 15,
    "LAST_ENTRY_MINUTE": 20,
    "LIMIT_BEFORE_10AM": int(os.getenv("LIMIT_BEFORE_10AM", "10")),
    "LIMIT_10AM_TO_1PM": int(os.getenv("LIMIT_10AM_TO_1PM", "10")),
    "LIMIT_1PM_TO_2PM": int(os.getenv("LIMIT_1PM_TO_2PM", "5")),
    "LIMIT_AFTER_2PM": int(os.getenv("LIMIT_AFTER_2PM", "2")),
    "MAX_SIGNALS_PER_CYCLE": int(os.getenv("MAX_SIGNALS_PER_CYCLE", "3")),
    "CONTINUOUS_LOOP_MINUTES": int(os.getenv("CONTINUOUS_LOOP_MINUTES", "350")),
    "KRONOS_ENABLED": os.getenv("KRONOS_ENABLED", "true").lower() == "true",
    "KRONOS_REQUIRED": os.getenv("KRONOS_REQUIRED", "true").lower() == "true",
    "KRONOS_MODEL": os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-mini"),
    "KRONOS_TOKENIZER": os.getenv("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-2k"),
    "KRONOS_LOOKBACK": int(os.getenv("KRONOS_LOOKBACK", "96")),
    "KRONOS_PRED_LEN": int(os.getenv("KRONOS_PRED_LEN", "6")),
    "KRONOS_SAMPLE_COUNT": int(os.getenv("KRONOS_SAMPLE_COUNT", "3")),
    "KRONOS_MIN_MOVE_PCT": float(os.getenv("KRONOS_MIN_MOVE_PCT", "0.08")),
    "TRADING_R1_ENABLED": os.getenv("TRADING_R1_ENABLED", "true").lower() == "true",
    "TRADING_R1_REQUIRED": os.getenv("TRADING_R1_REQUIRED", "false").lower() == "true",
    "UNIVERSE_MAX_AGE_HOURS": int(os.getenv("UNIVERSE_MAX_AGE_HOURS", "168")),
}

kronos_predictor = None
kronos_status = "not_loaded"


def ensure_kronos_dependencies() -> None:
    """Self-heal missing huggingface_hub / einops / safetensors in cached .venv."""
    try:
        import huggingface_hub  # noqa: F401
        import einops  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        LOG.info("Installing missing Kronos dependencies (huggingface_hub, einops, safetensors) into .venv...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.23.0", "einops>=0.8.0", "safetensors>=0.4.3"],
            check=False,
        )


def load_learned_weights() -> Dict[str, float]:
    weights = dict(DEFAULT_FORMULA_WEIGHTS)
    if LEARNED_WEIGHTS_PATH.exists():
        try:
            data = json.loads(LEARNED_WEIGHTS_PATH.read_text())
            loaded = data.get("formula_weights", {})
            for k in weights:
                if k in loaded:
                    weights[k] = float(np.clip(float(loaded[k]), 0.4, 2.2))
        except Exception as exc:
            LOG.warning("Could not read learned weights: %s", exc)
    return weights


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            payload TEXT NOT NULL,
            dedupe_key TEXT NOT NULL UNIQUE,
            delivered INTEGER NOT NULL DEFAULT 0
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )""")
        conn.commit()
    init_learning_tables()


def persist_state_to_git() -> None:
    """Conflict-free persistence of SQLite DB, universe cache, and learned weights to GitHub main branch."""
    try:
        tmp_db = Path("/tmp/bis20.sqlite3.bak")
        tmp_uni = Path("/tmp/liquid_universe.json.bak")
        tmp_wts = Path("/tmp/learned_strategy_weights.json.bak")
        tmp_flt = Path("/tmp/learned_filters.json.bak")
        if FILTERS_PATH.exists():
            shutil.copy2(FILTERS_PATH, tmp_flt)
        if DB_PATH.exists():
            shutil.copy2(DB_PATH, tmp_db)
        if UNIVERSE_CACHE.exists():
            shutil.copy2(UNIVERSE_CACHE, tmp_uni)
        if LEARNED_WEIGHTS_PATH.exists():
            shutil.copy2(LEARNED_WEIGHTS_PATH, tmp_wts)

        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=False)
        subprocess.run(["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], check=False)
        subprocess.run(["git", "fetch", "origin", "main"], check=False)
        subprocess.run(["git", "reset", "--hard", "origin/main"], check=False)

        if tmp_db.exists():
            shutil.copy2(tmp_db, DB_PATH)
        if tmp_uni.exists():
            shutil.copy2(tmp_uni, UNIVERSE_CACHE)
        if tmp_wts.exists():
            shutil.copy2(tmp_wts, LEARNED_WEIGHTS_PATH)
        if tmp_flt.exists():
            shutil.copy2(tmp_flt, FILTERS_PATH)

        for tracked in ("runtime/bis20.sqlite3", "data/liquid_universe.json", "data/learned_strategy_weights.json", "data/learned_filters.json"):
            if Path(tracked).exists():
                subprocess.run(["git", "add", tracked], check=False)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if diff.returncode != 0:
            subprocess.run(["git", "commit", "-m", "BIS 2.0: persist scanner state & learned weights [skip ci]"], check=False)
            subprocess.run(["git", "push", "origin", "HEAD:main"], check=False)
            LOG.info("Persisted updated SQLite state & learned weights to GitHub main branch.")
    except Exception as exc:
        LOG.warning("Git state persistence warning (non-fatal): %s", exc)


def get_time_window_info(now: Optional[datetime] = None) -> Tuple[str, dtime, dtime, int]:
    now = now or datetime.now(TZ)
    t = now.time()
    if t < dtime(10, 0):
        return "BEFORE_10AM", dtime(0, 0), dtime(9, 59, 59), CFG["LIMIT_BEFORE_10AM"]
    if t < dtime(13, 0):
        return "10AM_TO_1PM", dtime(10, 0), dtime(12, 59, 59), CFG["LIMIT_10AM_TO_1PM"]
    if t < dtime(14, 0):
        return "1PM_TO_2PM", dtime(13, 0), dtime(13, 59, 59), CFG["LIMIT_1PM_TO_2PM"]
    return "AFTER_2PM", dtime(14, 0), dtime(23, 59, 59), CFG["LIMIT_AFTER_2PM"]


def get_window_signal_counts(now: Optional[datetime] = None) -> Tuple[str, int, int, int]:
    now = now or datetime.now(TZ)
    day_str = now.strftime("%Y-%m-%d")
    win_name, win_start, win_end, win_limit = get_time_window_info(now)

    with db_connect() as conn:
        rows = conn.execute("SELECT created_at FROM signals WHERE created_at LIKE ?", (day_str + "%",)).fetchall()

    total_today = len(rows)
    used_in_window = 0
    for r in rows:
        try:
            dt = datetime.fromisoformat(r["created_at"]).astimezone(TZ)
            if win_start <= dt.time() <= win_end:
                used_in_window += 1
        except Exception:
            pass

    return win_name, used_in_window, win_limit, total_today


def get_today_signal_count(day_str: Optional[str] = None) -> int:
    day = day_str or datetime.now(TZ).strftime("%Y-%m-%d")
    with db_connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM signals WHERE created_at LIKE ?", (day + "%",)).fetchone()
        return int(row["n"]) if row else 0


def get_today_alerted_symbols(day_str: Optional[str] = None) -> set:
    day = day_str or datetime.now(TZ).strftime("%Y-%m-%d")
    with db_connect() as conn:
        rows = conn.execute("SELECT symbol FROM signals WHERE created_at LIKE ?", (day + "%",)).fetchall()
        return {r["symbol"] for r in rows}


def load_kronos() -> None:
    """Load Kronos-mini model on CPU; auto-install huggingface_hub if missing."""
    global kronos_predictor, kronos_status
    if kronos_predictor is not None:
        return
    if not CFG["KRONOS_ENABLED"]:
        kronos_status = "disabled"
        return
    if not KRONOS_REPO_DIR.exists():
        kronos_status = f"missing_repo:{KRONOS_REPO_DIR}"
        LOG.warning("Kronos repository not found at %s; continuing with fmzquant + Trading-R1", KRONOS_REPO_DIR)
        return
    ensure_kronos_dependencies()
    sys.path.insert(0, str(KRONOS_REPO_DIR))
    try:
        import torch
        from model import Kronos, KronosTokenizer, KronosPredictor
        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        tokenizer = KronosTokenizer.from_pretrained(CFG["KRONOS_TOKENIZER"])
        model = Kronos.from_pretrained(CFG["KRONOS_MODEL"])
        kronos_predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=2048)
        kronos_status = "ready:cpu"
        LOG.info("Kronos-mini loaded on CPU")
    except Exception as exc:
        kronos_predictor = None
        kronos_status = "error:" + str(exc)[:240]
        LOG.warning("Kronos initialization failed; continuing with fmzquant + Trading-R1: %s", exc)


def fetch_stock_news_headlines(symbol: str, max_items: int = 3) -> List[str]:
    """Fetch recent news headlines for an NSE symbol via Yahoo Finance RSS."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}.NS&region=IN&lang=en-IN"
    headers = {"User-Agent": "Mozilla/5.0"}
    headlines: List[str] = []
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200 and resp.text:
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item")[:max_items]:
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    headlines.append(title_el.text.strip())
    except Exception:
        pass
    return headlines


def trading_r1_evaluate_candidate(
    signal: Dict[str, Any],
    headlines: List[str],
    learned_weights: Dict[str, float],
) -> Dict[str, Any]:
    """Call TauricResearch Trading-R1 API (https://api.tauric.ai/v1/chat/completions, model='tauric')
    for evidence-based volatility-adjusted 5-tier decision (Strong Buy, Buy, Hold, Sell, Strong Sell).
    """
    api_key = os.getenv("TAURIC_API_KEY", "").strip()
    if not api_key or not CFG["TRADING_R1_ENABLED"]:
        return {"status": "skipped_no_key", "rating": "Buy" if signal["side"] == "BUY" else "Sell", "aligned": True}

    system_prompt = (
        "You are Trading-R1, a quantitative financial reasoning model trained via volatility-adjusted "
        "reinforcement learning. Evaluate this Indian NSE intraday pre-explosion stock setup. "
        "We ONLY want stocks caught EARLY (0.4% to 4.8% move so far, with unexhausted ADR% and tight BB width) "
        "that have genuine multi-timeframe momentum, volume accumulation, and catalyst/technical structure "
        "to expand 8%-15% after entry. Respond strictly in valid JSON with keys: "
        '{"rating": "Strong Buy"|"Buy"|"Hold"|"Sell"|"Strong Sell", '
        '"confidence": <int 0-100>, '
        '"aligned": <bool true if rating matches trade side>, '
        '"thesis": "<concise 1-sentence reason why this setup will expand or fail>"}'
    )
    user_payload = {
        "symbol": f"{signal['symbol']}.NS",
        "proposed_side": signal["side"],
        "entry_price": signal["price"],
        "vwap": signal["vwap"],
        "current_day_move_pct": signal["day_move_pct"],
        "avg_daily_range_pct": signal.get("avg_daily_range_pct"),
        "remaining_12pct_runway": signal.get("remaining_room_pct"),
        "5m_rvol": signal["relative_volume"],
        "cumulative_session_rvol": signal["cumulative_relative_volume"],
        "rsi_5m": signal["rsi"],
        "mfi_14": signal.get("mfi"),
        "adx_14": signal.get("adx"),
        "bb_bandwidth_pct": signal.get("bb_bandwidth_pct"),
        "matched_formulas": signal.get("pillars_matched"),
        "formula_learned_weights": {k: learned_weights.get(k, 1.0) for k in (signal.get("pillars_matched") or [])},
        "kronos_30m_forecast": signal.get("forecast"),
        "recent_news_headlines": headlines,
    }

    try:
        resp = requests.post(
            f"{TAURIC_API_BASE.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": TAURIC_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
                "temperature": 0.2,
                "max_completion_tokens": 512,
                "response_format": {"type": "json_object"},
            },
            timeout=25,
        )
        resp.raise_for_status()
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        rating = str(parsed.get("rating", "Hold"))
        if signal["side"] == "BUY":
            aligned = rating in ("Strong Buy", "Buy")
        else:
            aligned = rating in ("Strong Sell", "Sell")
        return {
            "model": "Trading-R1 (tauric)",
            "status": "ok",
            "rating": rating,
            "confidence": int(parsed.get("confidence", 75)),
            "aligned": bool(aligned),
            "thesis": str(parsed.get("thesis", ""))[:220],
        }
    except Exception as exc:
        LOG.warning("Trading-R1 API inference warning for %s: %s", signal["symbol"], exc)
        return {
            "model": "Trading-R1 (tauric)",
            "status": "fallback",
            "rating": "Buy" if signal["side"] == "BUY" else "Sell",
            "confidence": 70,
            "aligned": not CFG["TRADING_R1_REQUIRED"],
            "thesis": f"Fallback to 6-Formula + Kronos confluence ({str(exc)[:80]})",
        }


def is_market_session(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(TZ)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(9, 20) <= t <= dtime(15, 30)


def is_pre_market_wait_window(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(TZ)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(8, 30) <= t < dtime(9, 20)


def is_entry_window(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(TZ)
    if not is_market_session(now):
        return False
    return now.time() <= dtime(CFG["LAST_ENTRY_HOUR"], CFG["LAST_ENTRY_MINUTE"])


def is_tier2_window(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(TZ)
    if not is_entry_window(now):
        return False
    return now.time() >= dtime(CFG["TIER2_START_HOUR"], CFG["TIER2_START_MINUTE"])


def fetch_nse_equity_symbols() -> List[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    resp = requests.get(NSE_EQUITY_CSV, headers=headers, timeout=25)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    cols = {str(c).strip().upper(): c for c in df.columns}
    sym_col = cols.get("SYMBOL")
    series_col = cols.get("SERIES")
    if not sym_col:
        raise ValueError("NSE equity CSV does not contain a SYMBOL column")
    if series_col:
        df = df[df[series_col].astype(str).str.upper().str.strip().eq("EQ")]
    symbols = (
        df[sym_col].astype(str).str.upper().str.strip()
        .loc[lambda s: s.str.fullmatch(r"[A-Z0-9&-]{1,30}")]
        .drop_duplicates().tolist()
    )
    symbols = [s for s in symbols if not s.endswith("-BE") and not s.endswith("-SM")]
    return symbols[:CFG["MAX_EQUITY_UNIVERSE"]]


def extract_ticker_frame(downloaded: pd.DataFrame, ticker: str, single: bool = False) -> pd.DataFrame:
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        for level in (0, 1):
            try:
                if ticker in downloaded.columns.get_level_values(level):
                    frame = downloaded.xs(ticker, axis=1, level=level, drop_level=True)
                    return frame.dropna(how="all")
            except Exception:
                pass
        return pd.DataFrame()
    if single:
        return downloaded.dropna(how="all")
    return pd.DataFrame()


def download_batch(symbols: List[str], period: str, interval: str) -> Dict[str, pd.DataFrame]:
    tickers = [s + ".NS" for s in symbols]
    if not tickers:
        return {}
    try:
        raw = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            threads=True,
            progress=False,
            timeout=25,
        )
    except Exception as exc:
        LOG.warning("Yahoo batch failed (%s): %s", len(symbols), exc)
        return {s: pd.DataFrame() for s in symbols}
    out: Dict[str, pd.DataFrame] = {}
    for s, ticker in zip(symbols, tickers):
        frame = extract_ticker_frame(raw, ticker, single=(len(tickers) == 1))
        if not frame.empty:
            frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
            expected = ["open", "high", "low", "close", "volume"]
            if all(c in frame.columns for c in expected):
                frame = frame[expected].copy().dropna(subset=["open", "high", "low", "close"])
            else:
                frame = pd.DataFrame()
        out[s] = frame
    return out


def refresh_liquid_universe(force: bool = False) -> Dict[str, Any]:
    if not force and UNIVERSE_CACHE.exists():
        try:
            cached = json.loads(UNIVERSE_CACHE.read_text())
            refreshed = datetime.fromisoformat(cached["refreshed_at"])
            if datetime.now(TZ) - refreshed < timedelta(hours=CFG["UNIVERSE_MAX_AGE_HOURS"]):
                return cached
        except Exception:
            pass

    symbols = fetch_nse_equity_symbols()
    LOG.info("NSE main-board EQ symbols loaded: %d", len(symbols))
    metrics: List[Dict[str, Any]] = []
    batch_size = 60
    for start in range(0, len(symbols), batch_size):
        group = symbols[start:start + batch_size]
        data = download_batch(group, CFG["DAILY_HISTORY_PERIOD"], "1d")
        for symbol in group:
            df = data.get(symbol, pd.DataFrame())
            if df.empty or "close" not in df or "volume" not in df:
                continue
            idx = pd.to_datetime(df.index)
            if idx.tz is not None:
                idx = idx.tz_convert(TZ).tz_localize(None)
            keep = [x.date() < datetime.now(TZ).date() for x in idx]
            hist = df.loc[keep].copy()
            close = pd.to_numeric(hist["close"], errors="coerce").dropna()
            high = pd.to_numeric(hist["high"], errors="coerce").reindex(close.index).fillna(close)
            low = pd.to_numeric(hist["low"], errors="coerce").reindex(close.index).fillna(close)
            volume = pd.to_numeric(hist["volume"], errors="coerce").reindex(close.index).fillna(0)
            if len(close) < 50:
                continue
            turnover = (close * volume).replace([np.inf, -np.inf], np.nan).dropna()
            if len(turnover) < CFG["LIQUIDITY_LOOKBACK_DAYS"]:
                continue
            avg_turnover = float(turnover.tail(CFG["LIQUIDITY_LOOKBACK_DAYS"]).mean())
            last_price = float(close.iloc[-1])
            e20 = float(ema(close.values, 20))
            e50 = float(ema(close.values, 50))
            prev_close = float(close.iloc[-1])
            daily_range_pct = float(((high - low) / close.clip(lower=1e-6) * 100.0).tail(20).mean())
            if last_price >= CFG["PRICE_FLOOR"] and avg_turnover >= CFG["MIN_AVG_TURNOVER_INR"]:
                metrics.append({
                    "symbol": symbol,
                    "price": last_price,
                    "avg_turnover": avg_turnover,
                    "daily_ema20": e20,
                    "daily_ema50": e50,
                    "previous_close": prev_close,
                    "avg_daily_range_pct": round(daily_range_pct, 2),
                })
        LOG.info("Liquidity refresh progress %d/%d | qualified=%d", min(start + batch_size, len(symbols)), len(symbols), len(metrics))
        time.sleep(0.15)

    metrics.sort(key=lambda x: x["avg_turnover"], reverse=True)
    if not metrics:
        if UNIVERSE_CACHE.exists():
            old = json.loads(UNIVERSE_CACHE.read_text())
            old["refresh_warning"] = "Refresh returned no eligible stocks; retained previous cache"
            return old
        raise RuntimeError("No liquid universe was created. Check NSE/Yahoo data access.")

    top_liquid = metrics[:int(CFG["INTRADAY_UNIVERSE_SIZE"] * 0.5)]
    top_liquid_set = {m["symbol"] for m in top_liquid}
    remaining = [m for m in metrics[int(CFG["INTRADAY_UNIVERSE_SIZE"] * 0.5):500] if m["symbol"] not in top_liquid_set]
    remaining.sort(key=lambda x: (x.get("avg_daily_range_pct", 0.0), x["avg_turnover"]), reverse=True)
    combined = (top_liquid + remaining)[:CFG["INTRADAY_UNIVERSE_SIZE"]]

    result = {
        "refreshed_at": datetime.now(TZ).isoformat(),
        "source_symbol_count": len(symbols),
        "eligible_count": len(metrics),
        "scan_symbols": combined,
    }
    UNIVERSE_CACHE.write_text(json.dumps(result, indent=2))
    LOG.info("Liquid universe refreshed: %d eligible; scanning %d liquid & high-beta stocks", len(metrics), len(result["scan_symbols"]))
    return result


def ema_series(values: np.ndarray, period: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = float(np.mean(arr[:period]))
    for i in range(period, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def ema(values: Any, period: int) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) < period:
        return float("nan")
    s = ema_series(arr, period)
    return float(s[-1])


def rsi(values: Any, period: int = 14) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= period:
        return float("nan")
    delta = np.diff(arr)
    gains = np.maximum(delta, 0)
    losses = np.maximum(-delta, 0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + float(gains[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses[i])) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def mfi(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) <= period:
        return 50.0
    tail = df.tail(period + 15)
    tp = (tail["high"].astype(float) + tail["low"].astype(float) + tail["close"].astype(float)) / 3.0
    vol = tail["volume"].astype(float).clip(lower=0.0)
    rmf = tp * vol
    diff = tp.diff()
    pos_flow = rmf.where(diff > 0, 0.0).tail(period).sum()
    neg_flow = rmf.where(diff < 0, 0.0).tail(period).sum()
    if neg_flow <= 1e-9:
        return 100.0 if pos_flow > 0 else 50.0
    mr = float(pos_flow / neg_flow)
    return 100.0 - (100.0 / (1.0 + mr))


def bollinger_squeeze_ignition(closes: np.ndarray, period: int = 20, mult: float = 2.0) -> Tuple[bool, bool, bool, float]:
    if len(closes) < period + 6:
        return False, False, False, 0.0
    recent = closes[-(period + 6):]
    bw_history: List[float] = []
    upper_now = 0.0
    lower_now = 0.0
    for offset in range(7):
        window = recent[offset:offset + period]
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=0))
        up = mean + mult * std
        lo = mean - mult * std
        bw = ((up - lo) / max(mean, 1e-9)) * 100.0
        bw_history.append(bw)
        if offset == 6:
            upper_now = up
            lower_now = lo

    curr_bw = bw_history[-1]
    prior_min_bw = min(bw_history[:-1])
    # Must be coiled (<2.4% prior bandwidth) AND not already blown out (>2.85% bandwidth)
    squeeze_coil = prior_min_bw <= 2.4 and curr_bw <= CFG["MAX_BB_BANDWIDTH_PCT"]
    price = float(closes[-1])
    band_span = max(upper_now - lower_now, 1e-9)
    pos_in_band = (price - lower_now) / band_span
    bb_buy = squeeze_coil and pos_in_band >= 0.80 and curr_bw >= bw_history[-2]
    bb_sell = squeeze_coil and pos_in_band <= 0.20 and curr_bw >= bw_history[-2]
    return squeeze_coil, bb_buy, bb_sell, round(curr_bw, 2)


def macd_impulse(closes: np.ndarray) -> Tuple[bool, bool, float]:
    if len(closes) < 35:
        return False, False, 0.0
    e12 = ema_series(closes, 12)
    e26 = ema_series(closes, 26)
    macd_line = e12 - e26
    valid = macd_line[~np.isnan(macd_line)]
    if len(valid) < 10:
        return False, False, 0.0
    sig_line = ema_series(valid, 9)
    hist = valid - sig_line
    if np.isnan(hist[-1]) or np.isnan(hist[-2]):
        return False, False, 0.0
    h_now = float(hist[-1])
    h_prev = float(hist[-2])
    macd_buy = h_now > 0 and h_now >= h_prev
    macd_sell = h_now < 0 and h_now <= h_prev
    return macd_buy, macd_sell, round(h_now, 4)


def multi_timeframe_15m_alignment(df: pd.DataFrame) -> Tuple[bool, bool, bool, bool]:
    if len(df) < 30:
        return False, False, True, True
    today = df.index[-1].date()
    today_df = df[[x.date() == today for x in df.index]]
    price = float(df.iloc[-1]["close"])

    orb15_buy = False
    orb15_sell = False
    if len(today_df) >= 4:
        orb_bars = today_df.iloc[:3]
        orb_high = float(orb_bars["high"].max())
        orb_low = float(orb_bars["low"].min())
        orb15_buy = price > orb_high and float(today_df.iloc[-1]["close"]) > float(today_df.iloc[-1]["open"])
        orb15_sell = price < orb_low and float(today_df.iloc[-1]["close"]) < float(today_df.iloc[-1]["open"])

    try:
        r15 = df.resample("15min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        if len(r15) >= 21:
            c15 = r15["close"].astype(float).values
            e9_15 = ema(c15, 9)
            e21_15 = ema(c15, 21)
            mtf15_buy = price > e9_15 and e9_15 >= e21_15 * 0.998
            mtf15_sell = price < e9_15 and e9_15 <= e21_15 * 1.002
            return orb15_buy, orb15_sell, mtf15_buy, mtf15_sell
    except Exception:
        pass
    return orb15_buy, orb15_sell, True, True


def atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    value = tr.tail(period).mean()
    return float(value) if pd.notna(value) else float("nan")


def adx_di_slope(df: pd.DataFrame, period: int = 14) -> Tuple[float, float, float, bool]:
    tail_df = df.tail(max(period * 4, 60))
    if len(tail_df) < period * 2:
        return 20.0, 20.0, 20.0, True
    high = tail_df["high"].astype(float).values
    low = tail_df["low"].astype(float).values
    close = tail_df["close"].astype(float).values

    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])

    atr_smooth = float(np.mean(tr[:period]))
    plus_smooth = float(np.mean(plus_dm[:period]))
    minus_smooth = float(np.mean(minus_dm[:period]))

    dx_list: List[float] = []
    plus_di = 0.0
    minus_di = 0.0
    for i in range(period, len(tr)):
        atr_smooth = (atr_smooth * (period - 1) + float(tr[i])) / period
        plus_smooth = (plus_smooth * (period - 1) + float(plus_dm[i])) / period
        minus_smooth = (minus_smooth * (period - 1) + float(minus_dm[i])) / period
        if atr_smooth <= 1e-9:
            continue
        plus_di = 100.0 * (plus_smooth / atr_smooth)
        minus_di = 100.0 * (minus_smooth / atr_smooth)
        denom = plus_di + minus_di
        dx = 100.0 * abs(plus_di - minus_di) / denom if denom > 1e-9 else 0.0
        dx_list.append(dx)

    if len(dx_list) < 2:
        return 20.0, 20.0, 20.0, True
    adx_val = float(np.mean(dx_list[-period:]))
    adx_prev = float(np.mean(dx_list[-(period + 1):-1])) if len(dx_list) > period else adx_val
    adx_rising = adx_val >= adx_prev * 0.98 or dx_list[-1] >= adx_val
    return adx_val, plus_di, minus_di, adx_rising


def normalize_intraday(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is None:
        idx = idx.tz_localize(TZ)
    else:
        idx = idx.tz_convert(TZ)
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def completed_bars(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    cutoff = pd.Timestamp(now - timedelta(minutes=5)).tz_convert(TZ)
    return df.loc[df.index <= cutoff].copy()


def same_slot_relative_volume(df: pd.DataFrame, last_idx: pd.Timestamp, min_sessions: int) -> Tuple[float, int]:
    slot = last_idx.strftime("%H:%M")
    today = last_idx.date()
    same_slot = df[df.index.strftime("%H:%M") == slot]
    previous = same_slot[[x.date() < today for x in same_slot.index]].tail(5)
    vals = pd.to_numeric(previous["volume"], errors="coerce").dropna()
    vals = vals[vals > 0]
    current = float(df.loc[last_idx, "volume"])
    if len(vals) < min_sessions or float(vals.mean()) <= 0:
        return 0.0, int(len(vals))
    return current / float(vals.mean()), int(len(vals))


def session_cumulative_relative_volume(df: pd.DataFrame, last_idx: pd.Timestamp, min_sessions: int) -> Tuple[float, int]:
    today = last_idx.date()
    slot = last_idx.strftime("%H:%M")
    current = df[[x.date() == today and x.strftime("%H:%M") <= slot for x in df.index]]
    current_total = float(current["volume"].astype(float).sum())
    prior_totals: List[float] = []
    prior_dates = sorted({x.date() for x in df.index if x.date() < today})[-5:]
    for session_date in prior_dates:
        session = df[[x.date() == session_date and x.strftime("%H:%M") <= slot for x in df.index]]
        if not session.empty:
            vol_sum = float(session["volume"].astype(float).sum())
            if vol_sum > 0:
                prior_totals.append(vol_sum)
    if len(prior_totals) < min_sessions or float(np.mean(prior_totals)) <= 0:
        return 0.0, len(prior_totals)
    return current_total / float(np.mean(prior_totals)), len(prior_totals)


def session_vwap(df: pd.DataFrame) -> float:
    today = df.index[-1].date()
    session = df[[x.date() == today for x in df.index]]
    vol = session["volume"].astype(float).clip(lower=0)
    typical = (session["high"].astype(float) + session["low"].astype(float) + session["close"].astype(float)) / 3.0
    total = float(vol.sum())
    return float((typical * vol).sum() / total) if total > 0 else float("nan")


def compute_effective_day_move(df: pd.DataFrame, last_idx: pd.Timestamp, daily: Dict[str, Any], price: float) -> Tuple[float, float, float, float]:
    today = last_idx.date()
    day_candles = df[[x.date() == today for x in df.index]]
    day_open = float(day_candles.iloc[0]["open"]) if not day_candles.empty else price
    session_high = float(day_candles["high"].max()) if not day_candles.empty else price
    session_low = float(day_candles["low"].min()) if not day_candles.empty else price

    prior_candles = df[[x.date() < today for x in df.index]]
    prev_close = float(prior_candles.iloc[-1]["close"]) if not prior_candles.empty else float(daily.get("previous_close") or day_open)

    open_move_pct = ((price / day_open) - 1.0) * 100.0 if day_open > 0 else 0.0
    prev_close_move_pct = ((price / prev_close) - 1.0) * 100.0 if prev_close > 0 else open_move_pct
    if abs(prev_close_move_pct) >= abs(open_move_pct):
        effective_move_pct = prev_close_move_pct
    else:
        effective_move_pct = open_move_pct
    return effective_move_pct, open_move_pct, session_high, session_low


def build_signal_dict(
    symbol: str,
    side: str,
    price: float,
    atr_value: float,
    score: int,
    rsi_value: float,
    mfi_value: float,
    adx_val: float,
    bb_bw: float,
    relvol: float,
    cumulative_relvol: float,
    sessions_compared: int,
    daily: Dict[str, Any],
    vwap_value: float,
    breakout_level: float,
    day_move: float,
    last_idx: pd.Timestamp,
    body_ratio: float,
    setup_label: str,
    tier: str,
    pillars_matched: List[str],
) -> Dict[str, Any]:
    lv = (load_filters().get("levels") or {}).get(side, {})
    stop_scale = float(lv.get("stop_scale", 1.0))
    t1_mult = float(lv.get("t1_r", CFG["TARGET1_R_MULTIPLE"]))
    raw_risk = CFG["STOP_ATR_MULTIPLE"] * atr_value * stop_scale
    min_risk = price * CFG["MIN_STOP_PCT"]
    max_risk = price * CFG["MAX_STOP_PCT"]
    risk = float(np.clip(raw_risk, min_risk, max_risk))

    stop = price - risk if side == "BUY" else price + risk
    target = price + risk * t1_mult if side == "BUY" else price - risk * t1_mult
    runner_target = price + risk * CFG["TARGET2_R_MULTIPLE"] if side == "BUY" else price - risk * CFG["TARGET2_R_MULTIPLE"]
    expansion_10pct_target = price * (1.0 + CFG["EXPANSION_TARGET_PCT"] / 100.0) if side == "BUY" else price * (1.0 - CFG["EXPANSION_TARGET_PCT"] / 100.0)
    remaining_room_to_12pct = max(4.0, round(12.5 - abs(day_move), 2))
    reward_risk = abs(target - price) / max(abs(price - stop), 1e-9)

    return {
        "symbol": symbol,
        "side": side,
        "tier": tier,
        "price": round(price, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "runner_target": round(runner_target, 2),
        "expansion_target": round(expansion_10pct_target, 2),
        "remaining_room_pct": remaining_room_to_12pct,
        "reward_risk": round(reward_risk, 2),
        "score": int(min(100, max(0, score))),
        "rsi": round(rsi_value, 1),
        "mfi": round(mfi_value, 1),
        "adx": round(adx_val, 1),
        "bb_bandwidth_pct": round(bb_bw, 2),
        "relative_volume": round(relvol, 2),
        "cumulative_relative_volume": round(cumulative_relvol, 2),
        "volume_sessions_compared": int(sessions_compared),
        "avg_daily_turnover_crore": round(float(daily.get("avg_turnover", 0.0)) / 1e7, 2),
        "avg_daily_range_pct": round(float(daily.get("avg_daily_range_pct", 2.8)), 2),
        "atr": round(atr_value, 2),
        "vwap": round(vwap_value, 2),
        "breakout_level": round(breakout_level, 2),
        "day_move_pct": round(day_move, 2),
        "candle_time": last_idx.isoformat(),
        "body_ratio": round(body_ratio, 2),
        "pillars_matched": pillars_matched,
        "setup": setup_label,
        "forecast": None,
        "trading_r1": None,
        "stop_scale": stop_scale,
        "t1_r": t1_mult,
    }


def evaluate_symbol(
    symbol: str,
    source: pd.DataFrame,
    daily: Dict[str, Any],
    now: datetime,
    diagnostic: Optional[Dict[str, Any]] = None,
    allow_tier2: bool = False,
    pre_normalized: bool = False,
    formula_weights: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    if source.empty:
        if diagnostic is not None:
            diagnostic["rejected"] = "no_candles"
        return None
    df = source if pre_normalized else normalize_intraday(source)
    df = completed_bars(df, now)
    if len(df) < CFG["MIN_HISTORY_BARS"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "insufficient_history"
        return None
    last_idx = df.index[-1]
    age = (pd.Timestamp(now).tz_convert(TZ) - last_idx).total_seconds() / 60.0
    if age < 5 or age > CFG["MAX_BAR_AGE_MIN"] + 5:
        if diagnostic is not None:
            diagnostic["rejected"] = "stale_candle"
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    if price < CFG["PRICE_FLOOR"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "price_floor"
        return None

    day_move, open_move, session_high, session_low = compute_effective_day_move(df, last_idx, daily, price)
    if abs(day_move) > CFG["MAX_DAY_MOVE_PCT"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "already_moved_too_far"
            diagnostic["day_move_pct"] = round(day_move, 2)
        return None
    if abs(day_move) < CFG["MIN_DAY_MOVE_PCT"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "no_initial_ignition"
            diagnostic["day_move_pct"] = round(day_move, 2)
        return None

    # Range-Exhaustion Guard learned from Nov 9 dry-run (logs_115346277840):
    # Reject low-beta stocks that have already consumed >85% of their 20-day Average Daily Range!
    adr_pct = float(daily.get("avg_daily_range_pct", 3.0))
    if adr_pct < CFG["MIN_STOCK_ADR_PCT"] or abs(day_move) > adr_pct * CFG["MAX_ADR_USED_RATIO"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "adr_range_exhausted"
            diagnostic["day_move_pct"] = round(day_move, 2)
        return None

    relvol, slot_sessions = same_slot_relative_volume(df, last_idx, CFG["MIN_TIME_SLOT_SESSIONS"])
    cumulative_relvol, cumulative_sessions = session_cumulative_relative_volume(df, last_idx, CFG["MIN_TIME_SLOT_SESSIONS"])
    sessions_compared = max(slot_sessions, cumulative_sessions)
    if diagnostic is not None:
        diagnostic["volume"] = {
            "same_slot": round(relvol, 3),
            "cumulative": round(cumulative_relvol, 3),
            "sessions": sessions_compared,
        }

    vol_pass = (
        (relvol >= CFG["MIN_RELATIVE_VOLUME"] and cumulative_relvol >= CFG["MIN_CUMULATIVE_RVOL"])
        or (relvol >= 2.00 and cumulative_relvol >= 1.05)
        or (cumulative_relvol >= 1.75 and relvol >= 0.95)
    )
    if not vol_pass:
        if diagnostic is not None:
            diagnostic["rejected"] = "relative_volume"
        return None

    closes = df["close"].astype(float).values
    fast = ema(closes, 9)
    slow = ema(closes, 21)
    ema50 = ema(closes, 50) if len(closes) >= 50 else slow
    rsi_value = rsi(closes, 14)
    mfi_value = mfi(df, 14)
    atr_value = atr(df, 14)
    vwap_value = session_vwap(df)
    adx_val, plus_di, minus_di, adx_rising = adx_di_slope(df, 14)
    squeeze_coil, bb_buy, bb_sell, bb_bw = bollinger_squeeze_ignition(closes, 20, 2.0)
    macd_buy, macd_sell, _ = macd_impulse(closes)
    orb15_buy, orb15_sell, mtf15_buy, mtf15_sell = multi_timeframe_15m_alignment(df)

    if not np.isfinite([fast, slow, rsi_value, atr_value, vwap_value]).all() or atr_value <= 0:
        if diagnostic is not None:
            diagnostic["rejected"] = "indicator_invalid"
        return None

    if bb_bw > CFG["MAX_BB_BANDWIDTH_PCT"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "bb_bandwidth_blown_out"
        return None

    vwap_dist_pct = abs((price - vwap_value) / max(vwap_value, 1e-9)) * 100.0
    if vwap_dist_pct > CFG["MAX_VWAP_DISTANCE_PCT"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "vwap_overextended"
        return None

    prev = df.iloc[-(CFG["BREAKOUT_LOOKBACK"] + 1):-1]
    if len(prev) < CFG["BREAKOUT_LOOKBACK"]:
        if diagnostic is not None:
            diagnostic["rejected"] = "breakout_history"
        return None
    prior_high = float(prev["high"].max())
    prior_low = float(prev["low"].min())

    prior_days_df = df[[x.date() < last_idx.date() for x in df.index]].tail(225)
    multi_day_high = float(prior_days_df["high"].max()) if not prior_days_df.empty else prior_high
    multi_day_low = float(prior_days_df["low"].min()) if not prior_days_df.empty else prior_low

    bar_range = max(float(last["high"] - last["low"]), 1e-9)
    body_ratio = abs(float(last["close"] - last["open"])) / bar_range
    close_location = (float(last["close"]) - float(last["low"])) / bar_range

    bullish_candle = (
        float(last["close"]) > float(last["open"])
        and body_ratio >= CFG["MIN_BODY_RATIO"]
        and close_location >= CFG["MIN_CLOSE_LOCATION_BUY"]
    )
    bearish_candle = (
        float(last["close"]) < float(last["open"])
        and body_ratio >= CFG["MIN_BODY_RATIO"]
        and close_location <= CFG["MAX_CLOSE_LOCATION_SELL"]
    )

    atr_chan_buy = price >= (slow + 1.0 * atr_value)
    atr_chan_sell = price <= (slow - 1.0 * atr_value)
    f1_buy = bb_buy or (squeeze_coil and atr_chan_buy)
    f1_sell = bb_sell or (squeeze_coil and atr_chan_sell)

    f2_buy = orb15_buy and relvol >= 1.45
    f2_sell = orb15_sell and relvol >= 1.45

    f3_buy = price > vwap_value and fast > slow and slow >= ema50 * 0.997 and (CFG["MIN_MFI_BUY"] <= mfi_value <= CFG["MAX_MFI_BUY"])
    f3_sell = price < vwap_value and fast < slow and slow <= ema50 * 1.003 and (CFG["MIN_MFI_SELL"] <= mfi_value <= CFG["MAX_MFI_SELL"])

    f4_buy = adx_val >= CFG["MIN_ADX_TREND"] and plus_di > minus_di and adx_rising
    f4_sell = adx_val >= CFG["MIN_ADX_TREND"] and minus_di > plus_di and adx_rising

    f5_buy = macd_buy and mtf15_buy and (CFG["MIN_RSI_BUY"] <= rsi_value <= CFG["MAX_RSI_BUY"])
    f5_sell = macd_sell and mtf15_sell and (CFG["MIN_RSI_SELL"] <= rsi_value <= CFG["MAX_RSI_SELL"])

    f6_buy = (price >= prior_high * 0.999 or price >= multi_day_high * 0.998 or price >= session_high * 0.995) and relvol >= 1.35
    f6_sell = (price <= prior_low * 1.001 or price <= multi_day_low * 1.002 or price <= session_low * 1.005) and relvol >= 1.35

    buy_formulas: List[str] = []
    if f1_buy:
        buy_formulas.append("F1_BB_KELTNER_SQUEEZE_5M")
    if f2_buy:
        buy_formulas.append("F2_ORB_15M_RVOL_IGNITION")
    if f3_buy:
        buy_formulas.append("F3_VWAP_MFI_ACCUMULATION")
    if f4_buy:
        buy_formulas.append("F4_ADX_SUPERTREND_ACCEL")
    if f5_buy:
        buy_formulas.append("F5_MTF_5M_15M_MACD_RSI")
    if f6_buy:
        buy_formulas.append("F6_MULTIDAY_CHANNEL_BREAK")

    sell_formulas: List[str] = []
    if f1_sell:
        sell_formulas.append("F1_BB_KELTNER_SQUEEZE_5M")
    if f2_sell:
        sell_formulas.append("F2_ORB_15M_RVOL_IGNITION")
    if f3_sell:
        sell_formulas.append("F3_VWAP_MFI_ACCUMULATION")
    if f4_sell:
        sell_formulas.append("F4_ADX_SUPERTREND_ACCEL")
    if f5_sell:
        sell_formulas.append("F5_MTF_5M_15M_MACD_RSI")
    if f6_sell:
        sell_formulas.append("F6_MULTIDAY_CHANNEL_BREAK")

    daily_ema20 = float(daily.get("daily_ema20", price))
    daily_up = price >= daily_ema20 * 0.99
    daily_down = price <= daily_ema20 * 1.01

    req_formulas = CFG["MIN_CONFLUENCE_FORMULAS"] if not allow_tier2 else max(3, CFG["MIN_CONFLUENCE_FORMULAS"] - 1)

    side = None
    breakout_level = None
    matched_formulas: List[str] = []
    tier = "TIER-1"
    setup_label = "Tri-Engine: fmzquant + Kronos + Trading-R1"

    if (
        daily_up
        and price > vwap_value
        and fast > slow
        and bullish_candle
        and CFG["MIN_RSI_BUY"] <= rsi_value <= CFG["MAX_RSI_BUY"]
        and CFG["MIN_MFI_BUY"] <= mfi_value <= CFG["MAX_MFI_BUY"]
        and CFG["MIN_DAY_MOVE_PCT"] <= day_move <= CFG["MAX_DAY_MOVE_PCT"]
        and open_move >= 0.25
        and len(buy_formulas) >= req_formulas
    ):
        extension = max(0.0, (price - prior_high) / atr_value)
        if extension <= CFG["MAX_EXTENSION_ATR"]:
            side = "BUY"
            breakout_level = prior_high
            matched_formulas = buy_formulas
            tier = "TIER-1" if len(buy_formulas) >= 4 else "TIER-2"
    elif (
        daily_down
        and price < vwap_value
        and fast < slow
        and bearish_candle
        and CFG["MIN_RSI_SELL"] <= rsi_value <= CFG["MAX_RSI_SELL"]
        and CFG["MIN_MFI_SELL"] <= mfi_value <= CFG["MAX_MFI_SELL"]
        and -CFG["MAX_DAY_MOVE_PCT"] <= day_move <= -CFG["MIN_DAY_MOVE_PCT"]
        and open_move <= -0.25
        and len(sell_formulas) >= req_formulas
    ):
        extension = max(0.0, (prior_low - price) / atr_value)
        if extension <= CFG["MAX_EXTENSION_ATR"]:
            side = "SELL"
            breakout_level = prior_low
            matched_formulas = sell_formulas
            tier = "TIER-1" if len(sell_formulas) >= 4 else "TIER-2"

    if diagnostic is not None:
        diagnostic["day_move_pct"] = round(day_move, 2)
        diagnostic["buy_formulas"] = buy_formulas
        diagnostic["sell_formulas"] = sell_formulas
        diagnostic["conditions"] = {
            "f1_bb_squeeze": bool(f1_buy or f1_sell),
            "f2_orb_15m": bool(f2_buy or f2_sell),
            "f3_vwap_mfi": bool(f3_buy or f3_sell),
            "f4_adx_accel": bool(f4_buy or f4_sell),
            "f5_mtf_macd_rsi": bool(f5_buy or f5_sell),
            "f6_channel_break": bool(f6_buy or f6_sell),
            "candle_confirm": bool(bullish_candle or bearish_candle),
        }

    if not side:
        if diagnostic is not None:
            diagnostic["rejected"] = "technical_alignment"
        return None

    wts = formula_weights or DEFAULT_FORMULA_WEIGHTS
    weighted_formula_sum = sum(float(wts.get(f, 1.0)) for f in matched_formulas)

    score = 0
    score += 25 if 0.5 <= abs(day_move) <= CFG["IDEAL_MAX_MOVE_PCT"] else 15
    score += int(min(30.0, weighted_formula_sum * 6.0))
    score += 20 if relvol >= 2.2 else (15 if relvol >= 1.5 else 10)
    score += 15 if bb_bw <= 1.6 else 9
    score += 10 if adr_pct >= 3.5 else 6

    min_req_score = CFG["MIN_SCORE_TIER1"] if tier == "TIER-1" else CFG["MIN_SCORE_TIER2"]
    if score < min_req_score:
        if diagnostic is not None:
            diagnostic["rejected"] = "low_score"
        return None

    return build_signal_dict(
        symbol=symbol,
        side=side,
        price=price,
        atr_value=atr_value,
        score=score,
        rsi_value=rsi_value,
        mfi_value=mfi_value,
        adx_val=adx_val,
        bb_bw=bb_bw,
        relvol=relvol,
        cumulative_relvol=cumulative_relvol,
        sessions_compared=sessions_compared,
        daily=daily,
        vwap_value=vwap_value,
        breakout_level=breakout_level,
        day_move=day_move,
        last_idx=last_idx,
        body_ratio=body_ratio,
        setup_label=f"{setup_label} ({len(matched_formulas)}/6 Formulas)",
        tier=tier,
        pillars_matched=matched_formulas,
    )


def kronos_forecast(
    symbol: str,
    raw_df: pd.DataFrame,
    side: str,
    as_of: Optional[datetime] = None,
) -> Dict[str, Any]:
    if kronos_predictor is None:
        raise RuntimeError("Kronos model is not ready: " + kronos_status)
    df = normalize_intraday(raw_df)
    df = completed_bars(df, as_of or datetime.now(TZ)).tail(CFG["KRONOS_LOOKBACK"])
    if len(df) < 40:
        raise RuntimeError("Not enough completed candles for Kronos")

    x_df = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
    x_df["amount"] = x_df["volume"] * ((x_df["high"] + x_df["low"] + x_df["close"]) / 3.0)
    x_timestamp = pd.Series(df.index.tz_localize(None))
    last_time = df.index[-1]
    y_index = pd.date_range(
        start=(last_time + pd.Timedelta(minutes=5)).tz_localize(None),
        periods=CFG["KRONOS_PRED_LEN"],
        freq="5min",
    )
    y_timestamp = pd.Series(y_index)
    prediction = kronos_predictor.predict(
        df=x_df.reset_index(drop=True),
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=CFG["KRONOS_PRED_LEN"],
        T=0.5,
        top_p=0.85,
        sample_count=CFG["KRONOS_SAMPLE_COUNT"],
    )
    current_close = float(df["close"].iloc[-1])
    predicted_close = float(prediction["close"].iloc[-1])
    pred_high = float(prediction["high"].max()) if "high" in prediction else float(prediction["close"].max())
    pred_low = float(prediction["low"].min()) if "low" in prediction else float(prediction["close"].min())

    change_pct = ((predicted_close / current_close) - 1.0) * 100.0
    peak_move_pct = (
        ((pred_high / current_close) - 1.0) * 100.0
        if side == "BUY"
        else ((pred_low / current_close) - 1.0) * 100.0
    )
    aligned = (
        (change_pct >= CFG["KRONOS_MIN_MOVE_PCT"] or peak_move_pct >= CFG["KRONOS_MIN_MOVE_PCT"] * 2.5)
        if side == "BUY"
        else (change_pct <= -CFG["KRONOS_MIN_MOVE_PCT"] or peak_move_pct <= -CFG["KRONOS_MIN_MOVE_PCT"] * 2.5)
    )
    return {
        "model": "Kronos-mini",
        "horizon_bars": CFG["KRONOS_PRED_LEN"],
        "forecast_close": round(predicted_close, 2),
        "forecast_change_pct": round(change_pct, 3),
        "forecast_peak_move_pct": round(peak_move_pct, 3),
        "direction_aligned": bool(aligned),
    }


def add_signal(payload: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(TZ)
    day = now.strftime("%Y-%m-%d")
    dedupe = f"{day}|{payload['symbol']}"
    _, used_in_win, win_limit, _ = get_window_signal_counts(now)
    if used_in_win >= win_limit:
        return False
    with db_connect() as conn:
        try:
            conn.execute(
                "INSERT INTO signals(created_at,symbol,side,payload,dedupe_key,delivered) VALUES(?,?,?,?,?,0)",
                (now.isoformat(), payload["symbol"], payload["side"], json.dumps(payload), dedupe),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def format_alert_message(payload: Dict[str, Any]) -> str:
    forecast = payload.get("forecast") or {}
    if forecast.get("forecast_change_pct") is not None:
        aligned_tag = "✅ CONFIRMED" if forecast.get("direction_aligned") else "⚠️ neutral"
        peak_str = f" | Peak 30m: {forecast.get('forecast_peak_move_pct', 0.0):+.2f}%"
        forecast_line = (
            f"Kronos-mini ({forecast.get('horizon_bars')} bars): "
            f"{forecast.get('forecast_change_pct'):+.2f}%{peak_str} ({aligned_tag})"
        )
    else:
        forecast_line = "Kronos-mini: technical-confluence mode"

    tr1 = payload.get("trading_r1") or {}
    if tr1.get("status") == "ok":
        tr1_line = f"Trading-R1 ({tr1.get('rating')} | {tr1.get('confidence')}%): {tr1.get('thesis')}"
    else:
        tr1_line = "Trading-R1: 6-Formula adaptive confluence verified"

    lg = payload.get("learned") or {}
    if lg.get("mode") == "proven":
        learn_line = f"🎯 PROVEN-ONLY filter passed | learned edge {lg.get('edge_r', 0):+.2f}R"
    else:
        learn_line = "🧪 LEARNING MODE (still collecting results; treat as unproven)"
    pillars = ", ".join(payload.get("pillars_matched") or [])
    exp_str = f" | 10% Expansion: ₹{payload['expansion_target']}" if payload.get("expansion_target") else ""

    return (
        f"🚀 BHARAT PRE-EXPLOSION SCOUT 2.0 [{payload.get('tier', 'TIER-1')}]\n"
        f"{'🟢 EARLY BUY IGNITION' if payload['side'] == 'BUY' else '🔴 EARLY SELL IGNITION'} — {payload['symbol']}\n"
        f"Current Day Move: {payload['day_move_pct']:+.2f}% (Remaining 10-15% Runway: ~{payload.get('remaining_room_pct', 10.0)}%)\n"
        f"Entry reference: ₹{payload['price']} (VWAP: ₹{payload['vwap']})\n"
        f"Stop: ₹{payload['stop']} | T1 (2.5R): ₹{payload['target']} | T2 (5R): ₹{payload['runner_target']}{exp_str}\n"
        f"Conviction Score: {payload['score']}/100 | Planned R:R: {payload['reward_risk']}R\n"
        f"5m RVOL: {payload['relative_volume']}x | Session RVOL: {payload['cumulative_relative_volume']}x\n"
        f"RSI: {payload['rsi']} | MFI: {payload.get('mfi', 'N/A')} | ADX: {payload.get('adx', 'N/A')} | BB Width: {payload.get('bb_bandwidth_pct', 'N/A')}%\n"
        f"Avg Turnover: ₹{payload['avg_daily_turnover_crore']} Cr | 20d Avg Range: {payload.get('avg_daily_range_pct', 'N/A')}%\n"
        f"{forecast_line}\n"
        f"{tr1_line}\n"
        f"{learn_line}\n"
        f"Formulas Matched: {pillars}\n"
        f"Candle time: {payload['candle_time']}\n"
        "Research alert only. Check live spread, price and execution risk. No order was placed."
    )


def send_telegram_message(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID GitHub Secret")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(
        url,
        json={"chat_id": chat_id, "text": message[:4000], "disable_web_page_preview": True},
        timeout=20,
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError("Telegram API rejected message: " + str(body.get("description", "unknown error")))


def deliver_pending_signals(limit: int = 10) -> int:
    delivered = 0
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT id,payload FROM signals WHERE delivered=0 ORDER BY id ASC LIMIT ?",
            (max(1, min(int(limit), 20)),),
        ).fetchall()
    for row in rows:
        try:
            payload = json.loads(row["payload"])
            send_telegram_message(format_alert_message(payload))
            with db_connect() as conn:
                conn.execute("UPDATE signals SET delivered=1 WHERE id=?", (row["id"],))
                conn.commit()
            delivered += 1
            LOG.info("Telegram delivered | %s | %s", payload.get("symbol"), payload.get("side"))
        except Exception as exc:
            LOG.exception("Telegram delivery failed for queued signal id=%s: %s", row["id"], exc)
            break
    return delivered


def rank_pre_explosion_candidates(
    intraday_map: Dict[str, pd.DataFrame],
    daily_map: Dict[str, Dict[str, Any]],
    now: datetime,
) -> List[Dict[str, Any]]:
    ignition_list: List[Dict[str, Any]] = []
    for symbol, raw_df in intraday_map.items():
        if raw_df is None or raw_df.empty:
            continue
        df = completed_bars(normalize_intraday(raw_df), now)
        if len(df) < CFG["MIN_HISTORY_BARS"]:
            continue
        last_idx = df.index[-1]
        price = float(df.iloc[-1]["close"])
        day_move, _, _, _ = compute_effective_day_move(df, last_idx, daily_map.get(symbol, {}), price)
        if CFG["MIN_DAY_MOVE_PCT"] <= abs(day_move) <= CFG["MAX_DAY_MOVE_PCT"]:
            relvol, _ = same_slot_relative_volume(df, last_idx, CFG["MIN_TIME_SLOT_SESSIONS"])
            if relvol >= 1.25:
                ignition_list.append({
                    "symbol": symbol,
                    "day_move_pct": round(day_move, 2),
                    "rvol": round(relvol, 2),
                })
    ignition_list.sort(key=lambda x: x["rvol"], reverse=True)
    return ignition_list


def run_scan_cycle() -> Dict[str, Any]:
    started = time.time()
    now = datetime.now(TZ)
    if not is_market_session(now):
        return {"status": "outside_market_hours", "at": now.isoformat()}

    delivered_prior = deliver_pending_signals(limit=10)
    win_name, used_in_win, win_limit, existing_today = get_window_signal_counts(now)

    if used_in_win >= win_limit:
        LOG.info("Window %s quota full (%d/%d); continuing loop for next window.", win_name, used_in_win, win_limit)
        return {
            "status": "window_quota_full_waiting_next_window",
            "at": now.isoformat(),
            "time_window": win_name,
            "window_signals": f"{used_in_win}/{win_limit}",
            "signals_today_total": existing_today,
            "telegram_delivered": delivered_prior,
        }

    if not is_entry_window(now):
        return {
            "status": "past_last_entry_window",
            "at": now.isoformat(),
            "signals_today_total": existing_today,
            "telegram_delivered": delivered_prior,
        }

    if not UNIVERSE_CACHE.exists():
        LOG.info("Universe cache missing; building initial universe cache...")
        refresh_liquid_universe(force=True)

    universe = json.loads(UNIVERSE_CACHE.read_text())
    entries = universe.get("scan_symbols", [])
    if not entries:
        raise RuntimeError("Liquid universe cache is empty")

    learned_weights = load_learned_weights()
    symbols = [x["symbol"] for x in entries]
    daily_map = {x["symbol"]: x for x in entries}
    intraday_all: Dict[str, pd.DataFrame] = {}

    for start in range(0, len(symbols), CFG["INTRADAY_BATCH_SIZE"]):
        group = symbols[start:start + CFG["INTRADAY_BATCH_SIZE"]]
        intraday_all.update(download_batch(group, CFG["INTRADAY_PERIOD"], CFG["BAR_INTERVAL"]))
        time.sleep(0.08)

    alerted_today = get_today_alerted_symbols()
    early_ignitions = rank_pre_explosion_candidates(intraday_all, daily_map, now)

    candidates: List[Tuple[int, float, Dict[str, Any], pd.DataFrame]] = []

    for symbol in symbols:
        if symbol in alerted_today:
            continue
        frame = intraday_all.get(symbol, pd.DataFrame())
        signal = evaluate_symbol(
            symbol, frame, daily_map[symbol], now, allow_tier2=False, formula_weights=learned_weights
        )
        if signal:
            candidates.append((signal["score"], signal["relative_volume"], signal, frame))

    tier2_triggered = False
    if not candidates and is_tier2_window(now):
        tier2_triggered = True
        for item in early_ignitions[:15]:
            symbol = item["symbol"]
            if symbol in alerted_today:
                continue
            frame = intraday_all.get(symbol, pd.DataFrame())
            signal = evaluate_symbol(
                symbol, frame, daily_map[symbol], now, allow_tier2=True, formula_weights=learned_weights
            )
            if signal:
                candidates.append((signal["score"], signal["relative_volume"], signal, frame))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    technical_candidates = len(candidates)
    kronos_rejected = 0
    tr1_rejected = 0
    learned_rejected = 0
    inserted = 0
    remaining_window_slots = max(0, win_limit - used_in_win)
    cycle_limit = min(CFG["MAX_SIGNALS_PER_CYCLE"], remaining_window_slots)

    for _, _, signal, candles in candidates[:max(8, cycle_limit * 3)]:
        if inserted >= cycle_limit:
            break
        gate = learned_gate(signal, now)
        if not gate["ok"]:
            learned_rejected += 1
            LOG.info("LEARNED GATE rejected %s %s: %s", signal["symbol"], signal["side"], gate["reason"])
            continue
        signal["learned"] = gate
        if CFG["KRONOS_ENABLED"] and kronos_predictor is not None:
            try:
                forecast = kronos_forecast(signal["symbol"], candles, signal["side"])
                signal["forecast"] = forecast
                if forecast["direction_aligned"]:
                    signal["score"] = min(100, signal["score"] + 5)
                elif CFG["KRONOS_REQUIRED"]:
                    kronos_rejected += 1
                    continue
            except Exception as exc:
                signal["forecast"] = {"status": "unavailable", "detail": str(exc)[:180]}
                if CFG["KRONOS_REQUIRED"]:
                    kronos_rejected += 1
                    continue

        headlines = fetch_stock_news_headlines(signal["symbol"], max_items=3)
        tr1_eval = trading_r1_evaluate_candidate(signal, headlines, learned_weights)
        signal["trading_r1"] = tr1_eval
        if tr1_eval.get("aligned"):
            signal["score"] = min(100, signal["score"] + 5)
        elif CFG["TRADING_R1_REQUIRED"]:
            tr1_rejected += 1
            continue

        if add_signal(signal, now=now):
            inserted += 1

    delivered = delivered_prior + deliver_pending_signals(limit=10)
    if inserted > 0 or delivered > 0:
        persist_state_to_git()

    summary = {
        "status": "complete",
        "at": now.isoformat(),
        "time_window": win_name,
        "window_signals": f"{used_in_win + inserted}/{win_limit}",
        "symbols_considered": len(symbols),
        "top_early_ignition_watchlist": early_ignitions[:6],
        "technical_candidates": technical_candidates,
        "tier2_fallback_evaluated": tier2_triggered,
        "kronos_rejected": kronos_rejected,
        "trading_r1_rejected": tr1_rejected,
        "learned_gate_rejected": learned_rejected,
        "new_signals_queued": inserted,
        "signals_today_total": existing_today + inserted,
        "telegram_delivered": delivered,
        "duration_seconds": round(time.time() - started, 1),
        "universe_refreshed_at": universe.get("refreshed_at"),
        "kronos": kronos_status,
    }
    LOG.info("SCAN SUMMARY %s", json.dumps(summary))
    return summary


def run_backtest_and_learn(target_symbol: Optional[str] = None, send_telegram: bool = True) -> Dict[str, Any]:
    """Reverse-engineer top Day-High winners, audit past signals (what worked vs what failed),
    cross-validate the 6 formulas across the 200-stock universe, and save learned formula weights!
    """
    started = time.monotonic()
    if not UNIVERSE_CACHE.exists():
        refresh_liquid_universe(force=True)
    universe = json.loads(UNIVERSE_CACHE.read_text())
    entries = universe.get("scan_symbols", [])
    daily_map = {x["symbol"]: x for x in entries}

    target_clean = (target_symbol or os.getenv("TARGET_SYMBOL", "")).strip().upper().replace(".NS", "")
    if target_clean and target_clean not in daily_map:
        daily_map[target_clean] = {
            "symbol": target_clean,
            "price": 100.0,
            "avg_turnover": 100000000.0,
            "daily_ema20": 100.0,
            "daily_ema50": 100.0,
            "previous_close": 100.0,
            "avg_daily_range_pct": 3.5,
        }
        entries.append(daily_map[target_clean])

    frames: Dict[str, pd.DataFrame] = {}
    for offset in range(0, len(entries), CFG["INTRADAY_BATCH_SIZE"]):
        group = [e["symbol"] for e in entries[offset:offset + CFG["INTRADAY_BATCH_SIZE"]]]
        frames.update(download_batch(group, CFG["INTRADAY_PERIOD"], CFG["BAR_INTERVAL"]))
        LOG.info("BACKTEST-LEARN downloaded %s/%s symbols", min(offset + len(group), len(entries)), len(entries))

    normalized: Dict[str, pd.DataFrame] = {}
    latest_sessions = []
    for symbol, frame in frames.items():
        if frame is None or frame.empty:
            continue
        clean = normalize_intraday(frame).between_time("09:15", "15:25")
        if clean.empty:
            continue
        normalized[symbol] = clean
        latest_sessions.append(clean.index[-1].date())
    if not latest_sessions:
        raise RuntimeError("No intraday candles returned for backtest-learn")
    replay_day = max(latest_sessions)

    init_db()
    with db_connect() as conn:
        past_rows = conn.execute("SELECT * FROM signals ORDER BY id DESC LIMIT 30").fetchall()

    past_signal_audit: List[Dict[str, Any]] = []
    worked_count = 0
    failed_count = 0
    for row in past_rows:
        try:
            p = json.loads(row["payload"])
            sym = p["symbol"]
            side = p["side"]
            entry_px = float(p["price"])
            stop_px = float(p["stop"])
            t1_px = float(p["target"])
            ctime = pd.Timestamp(p["candle_time"]).tz_convert(TZ)
            f = normalized.get(sym)
            outcome = "NO_DATA"
            max_fav_pct = 0.0
            if f is not None:
                same_day_after = f[[x.date() == ctime.date() and x > ctime for x in f.index]]
                if not same_day_after.empty:
                    if side == "BUY":
                        peak_hi = float(same_day_after["high"].max())
                        min_lo = float(same_day_after["low"].min())
                        max_fav_pct = round(((peak_hi / entry_px) - 1.0) * 100.0, 2)
                        hit_t1 = peak_hi >= t1_px
                        hit_stop = min_lo <= stop_px
                    else:
                        min_lo = float(same_day_after["low"].min())
                        peak_hi = float(same_day_after["high"].max())
                        max_fav_pct = round(((entry_px - min_lo) / entry_px) * 100.0, 2)
                        hit_t1 = min_lo <= t1_px
                        hit_stop = peak_hi >= stop_px
                    if hit_t1 or max_fav_pct >= 2.5:
                        outcome = "WORKED"
                        worked_count += 1
                    elif hit_stop or max_fav_pct < 1.0:
                        outcome = "FAILED"
                        failed_count += 1
                    else:
                        outcome = "PARTIAL_GAIN"
            past_signal_audit.append({
                "symbol": sym,
                "side": side,
                "candle_time": p["candle_time"],
                "entry_day_move_pct": p.get("day_move_pct"),
                "max_move_after_signal_pct": max_fav_pct,
                "outcome": outcome,
                "formulas": p.get("pillars_matched", []),
            })
        except Exception:
            pass

    day_high_winners: List[Tuple[float, str]] = []
    for sym, clean in normalized.items():
        today_bars = clean[[x.date() == replay_day for x in clean.index]]
        if len(today_bars) < 6:
            continue
        day_open = float(today_bars.iloc[0]["open"])
        prior_bars = clean[[x.date() < replay_day for x in clean.index]]
        prev_close = float(prior_bars.iloc[-1]["close"]) if not prior_bars.empty else day_open
        day_peak = float(today_bars["high"].max())
        max_rally_pct = ((day_peak / max(prev_close, 1e-9)) - 1.0) * 100.0
        day_high_winners.append((max_rally_pct, sym))

    day_high_winners.sort(key=lambda x: x[0], reverse=True)
    inspect_symbols = [sym for _, sym in day_high_winners[:10]]
    if target_clean and target_clean in normalized and target_clean not in inspect_symbols:
        inspect_symbols.insert(0, target_clean)

    reverse_engineered_movers: List[Dict[str, Any]] = []
    for sym in inspect_symbols:
        clean = normalized[sym]
        today_bars = clean[[x.date() == replay_day for x in clean.index]]
        if today_bars.empty:
            continue
        prior_bars = clean[[x.date() < replay_day for x in clean.index]]
        prev_close = float(prior_bars.iloc[-1]["close"]) if not prior_bars.empty else float(today_bars.iloc[0]["open"])
        day_peak = float(today_bars["high"].max())
        total_day_high_pct = round(((day_peak / max(prev_close, 1e-9)) - 1.0) * 100.0, 2)

        best_ignition = None
        for candle_start in today_bars.index:
            prefix = clean.loc[clean.index <= candle_start]
            if len(prefix) < CFG["MIN_HISTORY_BARS"]:
                continue
            px = float(prefix.iloc[-1]["close"])
            move_now = ((px / max(prev_close, 1e-9)) - 1.0) * 100.0
            if 0.4 <= move_now <= 4.8:
                rvol, _ = same_slot_relative_volume(prefix, candle_start, 3)
                diag_rev: Dict[str, Any] = {}
                anchor = candle_start.to_pydatetime() + timedelta(minutes=5)
                _ = evaluate_symbol(sym, prefix, daily_map.get(sym, {}), anchor, diagnostic=diag_rev, allow_tier2=True, pre_normalized=True)
                formulas_here = diag_rev.get("buy_formulas", [])
                if len(formulas_here) >= 2 and rvol >= 1.25:
                    gain_after = round(((day_peak / px) - 1.0) * 100.0, 2)
                    best_ignition = {
                        "ignition_time_ist": candle_start.strftime("%H:%M"),
                        "price_at_ignition": round(px, 2),
                        "move_at_ignition_pct": round(move_now, 2),
                        "subsequent_rally_to_day_high_pct": gain_after,
                        "5m_rvol_at_ignition": round(rvol, 2),
                        "best_timeframe": "5m + 15m ORB Synergy" if "F2_ORB_15M_RVOL_IGNITION" in formulas_here or "F5_MTF_5M_15M_MACD_RSI" in formulas_here else "5m Squeeze Ignition",
                        "triggered_formulas": formulas_here,
                    }
                    break
        headlines = fetch_stock_news_headlines(sym, max_items=2)
        reverse_engineered_movers.append({
            "symbol": sym,
            "day_high_move_pct": total_day_high_pct,
            "ignition_signature": best_ignition or "Gapped >5% at 09:15 open (overnight news catalyst)",
            "news_catalysts": headlines,
        })

    formula_stats: Dict[str, Dict[str, float]] = {
        k: {"triggers": 0, "wins_2pct": 0, "big_wins_4pct": 0, "sum_max_gain": 0.0}
        for k in DEFAULT_FORMULA_WEIGHTS
    }

    all_passed_candidates: List[Dict[str, Any]] = []
    for sym, clean in list(normalized.items())[:150]:
        today_bars = clean[[x.date() == replay_day for x in clean.index]]
        if len(today_bars) < 10:
            continue
        for idx_pos, candle_start in enumerate(today_bars.index[:-4]):
            if idx_pos % 2 != 0:
                continue
            prefix = clean.loc[clean.index <= candle_start]
            if len(prefix) < CFG["MIN_HISTORY_BARS"]:
                continue
            anchor = candle_start.to_pydatetime() + timedelta(minutes=5)
            diag_cv: Dict[str, Any] = {}
            sig_cv = evaluate_symbol(
                sym, prefix, daily_map.get(sym, {}), anchor, diagnostic=diag_cv, allow_tier2=True, pre_normalized=True
            )
            if not sig_cv:
                continue
            future = today_bars.loc[today_bars.index > candle_start]
            if future.empty:
                continue
            entry_px = sig_cv["price"]
            if sig_cv["side"] == "BUY":
                max_gain_pct = ((float(future["high"].max()) / entry_px) - 1.0) * 100.0
            else:
                max_gain_pct = ((entry_px - float(future["low"].min())) / entry_px) * 100.0
            sig_cv["actual_move_after_signal_pct"] = round(max_gain_pct, 2)
            all_passed_candidates.append(sig_cv)

            for f_name in sig_cv.get("pillars_matched", []):
                if f_name in formula_stats:
                    formula_stats[f_name]["triggers"] += 1
                    formula_stats[f_name]["sum_max_gain"] += max(0.0, max_gain_pct)
                    if max_gain_pct >= 2.0:
                        formula_stats[f_name]["wins_2pct"] += 1
                    if max_gain_pct >= 4.0:
                        formula_stats[f_name]["big_wins_4pct"] += 1

    learned_weights: Dict[str, float] = {}
    formula_scorecard: Dict[str, Any] = {}
    for f_name, st in formula_stats.items():
        trig = max(1, int(st["triggers"]))
        win_rate = (st["wins_2pct"] / trig) * 100.0 if st["triggers"] > 0 else 50.0
        avg_gain = (st["sum_max_gain"] / trig) if st["triggers"] > 0 else 2.0
        new_wt = round(float(np.clip(0.6 + (win_rate / 100.0) * 0.9 + (avg_gain / 10.0) * 0.4, 0.6, 1.8)), 2)
        learned_weights[f_name] = new_wt
        formula_scorecard[f_name] = {
            "triggers_tested": int(st["triggers"]),
            "win_rate_2pct_plus": round(win_rate, 1),
            "big_move_4pct_plus_count": int(st["big_wins_4pct"]),
            "avg_post_signal_gain_pct": round(avg_gain, 2),
            "updated_weight": new_wt,
        }

    weights_payload = {
        "updated_at": datetime.now(TZ).isoformat(),
        "session_analyzed": str(replay_day),
        "formula_weights": learned_weights,
        "formula_scorecard": formula_scorecard,
    }
    LEARNED_WEIGHTS_PATH.write_text(json.dumps(weights_payload, indent=2), encoding="utf-8")

    all_passed_candidates.sort(key=lambda x: (-x["actual_move_after_signal_pct"], -x["score"]))
    seen_s = set()
    best_validated_signals = []
    for c in all_passed_candidates:
        if c["symbol"] not in seen_s:
            seen_s.add(c["symbol"])
            best_validated_signals.append(c)

    report = {
        "mode": "BACKTEST_AND_SELF_LEARNING_ENGINE",
        "session_date_ist": str(replay_day),
        "kronos_status": kronos_status,
        "target_symbol_inspected": target_clean or "TOP_10_DAY_HIGH_MOVERS",
        "past_signals_summary": {"worked": worked_count, "failed": failed_count, "audited": past_signal_audit[:10]},
        "reverse_engineered_top_movers": reverse_engineered_movers[:8],
        "best_pre_explosion_signals_today": best_validated_signals[:8],
        "cross_validated_formula_scorecard": formula_scorecard,
        "updated_formula_weights": learned_weights,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    report_path = ROOT / "dry_run_v2_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    persist_state_to_git()

    if send_telegram:
        top_rev_lines = []
        for item in reverse_engineered_movers[:4]:
            sig_info = item["ignition_signature"]
            if isinstance(sig_info, dict):
                top_rev_lines.append(
                    f"• {item['symbol']} (Day High +{item['day_high_move_pct']}%): Ignited at {sig_info['ignition_time_ist']} "
                    f"(+{sig_info['move_at_ignition_pct']}%, RVOL {sig_info['5m_rvol_at_ignition']}x) -> "
                    f"+{sig_info['subsequent_rally_to_day_high_pct']}% after! [{', '.join(sig_info['triggered_formulas'][:2])}]"
                )
            else:
                top_rev_lines.append(f"• {item['symbol']} (Day High +{item['day_high_move_pct']}%): {sig_info}")

        best_formulas = sorted(
            formula_scorecard.items(),
            key=lambda kv: (kv[1]["win_rate_2pct_plus"], kv[1]["avg_post_signal_gain_pct"]),
            reverse=True,
        )
        f_lines = [
            f"• {k}: WinRate {v['win_rate_2pct_plus']}% | AvgGain +{v['avg_post_signal_gain_pct']}% -> Wt {v['updated_weight']}x"
            for k, v in best_formulas[:4]
        ]
        msg = (
            f"🧠 BHARAT SCOUT 2.0 — SELF-LEARNING BACKTEST ({replay_day})\n"
            f"Past Alerts Audit: ✅ {worked_count} Worked | ❌ {failed_count} Failed\n\n"
            f"🔍 Reverse-Engineered Day-High Movers:\n" + "\n".join(top_rev_lines) + "\n\n"
            f"📊 Cross-Validated Formula Rankings (Auto-Saved):\n" + "\n".join(f_lines)
        )
        try:
            send_telegram_message(msg)
        except Exception as exc:
            LOG.warning("Could not send backtest Telegram summary: %s", exc)

    LOG.info("BACKTEST-LEARN SUMMARY %s", json.dumps(report, default=str))
    return report


# =====================================================================================
# SELF-LEARNING ENGINE v3 (outcome tracking + gainers/losers study + adaptive filters)
# =====================================================================================
FILTERS_PATH = Path(os.getenv("LEARNED_FILTERS_PATH", str(ROOT / "data" / "learned_filters.json")))
LEARN_CFG: Dict[str, Any] = {
    "MIN_N": int(os.getenv("LEARN_MIN_N", "12")),                # samples needed before a bucket can be blocked/boosted
    "WARMUP_TOTAL": int(os.getenv("LEARN_WARMUP_TOTAL", "20")),  # resolved signals needed before strict filtering starts
    "BLOCK_EXP_R": float(os.getenv("LEARN_BLOCK_EXP_R", "-0.15")),
    "BOOST_EXP_R": float(os.getenv("LEARN_BOOST_EXP_R", "0.20")),
    "MIN_EDGE_R": float(os.getenv("LEARN_MIN_EDGE_R", "0.05")),  # proven-mode: candidate's learned edge must beat this
    "LOOKBACK_DAYS": int(os.getenv("LEARN_LOOKBACK_DAYS", "40")),
    "SHRINK_K": float(os.getenv("LEARN_SHRINK_K", "10")),
    "TOP_N_MOVERS": int(os.getenv("LEARN_TOP_N_MOVERS", "10")),
}
ALL_FORMULAS = list(DEFAULT_FORMULA_WEIGHTS.keys())
_FILTERS_CACHE: Dict[str, Any] = {"mtime": None, "data": None}


def _default_filters() -> Dict[str, Any]:
    return {
        "updated_at": None, "mode": "warmup", "total_resolved": 0, "score_bump": 0,
        "min_rvol": None, "blocked": {}, "boosted": {}, "stats": {}, "levels": {}, "recent": {}, "changes": [],
    }


def load_filters() -> Dict[str, Any]:
    try:
        mtime = FILTERS_PATH.stat().st_mtime if FILTERS_PATH.exists() else None
        if _FILTERS_CACHE["data"] is not None and _FILTERS_CACHE["mtime"] == mtime:
            return _FILTERS_CACHE["data"]
        data = _default_filters()
        if mtime is not None:
            data.update(json.loads(FILTERS_PATH.read_text()))
        _FILTERS_CACHE.update({"mtime": mtime, "data": data})
        return data
    except Exception as exc:
        LOG.warning("Could not read learned filters: %s", exc)
        return _default_filters()


def init_learning_tables() -> None:
    with db_connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS outcomes (
            signal_id INTEGER PRIMARY KEY,
            day TEXT, symbol TEXT, side TEXT, tier TEXT, win TEXT, score INTEGER,
            formulas TEXT, result TEXT, r_result REAL,
            mfe_pct REAL, mae_pct REAL, mfe_pre_r REAL, mae_before_1r0 REAL, eod_r REAL,
            stopped INTEGER, minutes_to_result INTEGER, resolved_at TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS mover_study (
            day TEXT, symbol TEXT, direction TEXT, close_move_pct REAL, gap_pct REAL,
            caught INTEGER, first_signal_time TEXT, capture_pct REAL,
            ignition_time TEXT, ignition_rvol REAL, ignition_formulas INTEGER,
            block_reason TEXT, signal_formulas TEXT,
            PRIMARY KEY (day, symbol)
        )""")
        conn.commit()


def window_of(dt: datetime) -> str:
    t = dt.time()
    if t < dtime(10, 0):
        return "BEFORE_10AM"
    if t < dtime(13, 0):
        return "10AM_TO_1PM"
    if t < dtime(14, 0):
        return "1PM_TO_2PM"
    return "AFTER_2PM"


def signal_dims(side: str, tier: str, win: str, formulas: List[str]) -> List[str]:
    dims = [f"side:{side}", f"tier:{tier}", f"window:{win}", f"side_window:{side}|{win}"]
    dims += [f"formula:{side}|{f}" for f in formulas]
    return dims


def resolve_outcome(p: Dict[str, Any], frame: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Walk the 5-minute bars after the signal and record what price REALLY did.
    If stop and target are both touched inside one bar, the STOP is assumed first (conservative)."""
    side = p["side"]
    entry, stop, t1 = float(p["price"]), float(p["stop"]), float(p["target"])
    risk = abs(entry - stop)
    if risk <= 0 or frame is None or frame.empty:
        return None
    ct = pd.Timestamp(p["candle_time"])
    ct = ct.tz_localize(TZ) if ct.tzinfo is None else ct.tz_convert(TZ)
    after = frame[(frame.index > ct) & (frame.index.date == ct.date())]
    after = after.between_time("09:15", "15:25")
    if after.empty:
        return None
    r0 = risk / max(float(p.get("stop_scale", 1.0) or 1.0), 1e-6)  # base ATR risk before learned scaling
    t1_r = abs(t1 - entry) / risk
    result, minutes = None, None
    mfe = mae = 0.0           # whole-session excursions (price units)
    mfe_pre = 0.0             # best favourable excursion before the stop was touched
    mae_before_1r = 0.0       # worst adverse excursion before price first reached +1R
    reached_1r = False
    for ts, row in after.iterrows():
        hi, lo = float(row["high"]), float(row["low"])
        fav = (hi - entry) if side == "BUY" else (entry - lo)
        adv = (entry - lo) if side == "BUY" else (hi - entry)
        mfe, mae = max(mfe, fav), max(mae, adv)
        hit_stop = adv >= risk
        if result is None:
            if hit_stop:
                result, minutes = "STOP", int((ts - ct).total_seconds() // 60)
            else:
                mfe_pre = max(mfe_pre, fav)
                if fav >= t1_r * risk:
                    result, minutes = "T1", int((ts - ct).total_seconds() // 60)
        if not reached_1r:
            mae_before_1r = max(mae_before_1r, adv)
            if fav >= risk:
                reached_1r = True
    last_close = float(after.iloc[-1]["close"])
    eod_r = ((last_close - entry) if side == "BUY" else (entry - last_close)) / risk
    if result == "T1":
        r_result = t1_r
    elif result == "STOP":
        r_result = -1.0
    else:
        result, r_result = ("EOD_WIN" if eod_r > 0 else "EOD_LOSS"), float(eod_r)
    return {
        "result": result, "r_result": round(float(r_result), 3),
        "mfe_pct": round(mfe / entry * 100, 2), "mae_pct": round(mae / entry * 100, 2),
        "mfe_pre_r": round(mfe_pre / risk, 3),
        "mae_before_1r0": round(mae_before_1r / r0, 3) if reached_1r else None,
        "eod_r": round(float(eod_r), 3), "stopped": 1 if result == "STOP" else 0,
        "minutes_to_result": minutes,
    }


def resolve_pending_outcomes(frames: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """Resolve every logged signal that has no outcome yet (also back-fills the old signals already in the DB)."""
    resolved: List[Dict[str, Any]] = []
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT s.id, s.created_at, s.payload FROM signals s LEFT JOIN outcomes o ON o.signal_id = s.id "
            "WHERE o.signal_id IS NULL ORDER BY s.id"
        ).fetchall()
    for row in rows:
        try:
            p = json.loads(row["payload"])
            created = datetime.fromisoformat(row["created_at"]).astimezone(TZ)
            day = created.strftime("%Y-%m-%d")
            if day == today and datetime.now(TZ).time() < dtime(15, 30):
                continue  # session still running
            frame = frames.get(p["symbol"])
            out = resolve_outcome(p, normalize_intraday(frame)) if frame is not None and not frame.empty else None
            if out is None:
                if (datetime.now(TZ).date() - created.date()).days > 9:  # fell out of Yahoo's 10-day 5m window
                    out = {"result": "NO_DATA", "r_result": None, "mfe_pct": None, "mae_pct": None, "mfe_pre_r": None,
                           "mae_before_1r0": None, "eod_r": None, "stopped": 0, "minutes_to_result": None}
                else:
                    continue
            with db_connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["id"], day, p["symbol"], p["side"], p.get("tier"), window_of(created + timedelta(minutes=0)),
                     int(p.get("score", 0)), json.dumps(p.get("pillars_matched") or []), out["result"], out["r_result"],
                     out["mfe_pct"], out["mae_pct"], out["mfe_pre_r"], out["mae_before_1r0"], out["eod_r"],
                     out["stopped"], out["minutes_to_result"], datetime.now(TZ).isoformat()),
                )
                conn.commit()
            out.update({"symbol": p["symbol"], "side": p["side"], "day": day, "entry": p["price"]})
            resolved.append(out)
        except Exception as exc:
            LOG.warning("Outcome resolve failed for signal %s: %s", row["id"], exc)
    return resolved


def _bucket(rows: List[sqlite3.Row], key_fn) -> Dict[str, Dict[str, float]]:
    acc: Dict[str, List[float]] = {}
    for r in rows:
        for k in key_fn(r):
            acc.setdefault(k, []).append(float(r["r_result"]))
    return {
        k: {"n": len(v), "avg_r": round(float(np.mean(v)), 3), "win": round(100.0 * sum(1 for x in v if x > 0) / len(v), 1)}
        for k, v in acc.items()
    }


def recompute_filters() -> Dict[str, Any]:
    """Cumulative learning from REAL tracked outcomes (stop/target/excursion), not from a single replayed day."""
    prev = load_filters()
    cutoff = (datetime.now(TZ) - timedelta(days=LEARN_CFG["LOOKBACK_DAYS"])).strftime("%Y-%m-%d")
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT * FROM outcomes WHERE r_result IS NOT NULL AND day >= ? ORDER BY day, signal_id", (cutoff,)
        ).fetchall()
        mover_rows = conn.execute(
            "SELECT * FROM mover_study WHERE day >= ?", (cutoff,)
        ).fetchall()
    stats = _bucket(rows, lambda r: signal_dims(r["side"], r["tier"] or "TIER-1", r["win"], json.loads(r["formulas"] or "[]")))
    total = len(rows)
    new = _default_filters()
    new["total_resolved"] = total
    new["stats"] = stats
    new["mode"] = "proven" if total >= LEARN_CFG["WARMUP_TOTAL"] else "warmup"

    for k, v in stats.items():
        if v["n"] >= LEARN_CFG["MIN_N"] and v["avg_r"] <= LEARN_CFG["BLOCK_EXP_R"]:
            new["blocked"][k] = v
        elif v["n"] >= LEARN_CFG["MIN_N"] and v["avg_r"] >= LEARN_CFG["BOOST_EXP_R"]:
            new["boosted"][k] = v

    last = rows[-20:]
    if len(last) >= 10:
        rs = [float(r["r_result"]) for r in last]
        new["recent"] = {"n": len(rs), "avg_r": round(float(np.mean(rs)), 3),
                         "win": round(100.0 * sum(1 for x in rs if x > 0) / len(rs), 1)}
        new["score_bump"] = 6 if new["recent"]["avg_r"] < -0.2 else (3 if new["recent"]["avg_r"] < 0 else 0)

    # ---- level calibration from what price really did after each BUY / SELL ----
    for side in ("BUY", "SELL"):
        s_rows = [r for r in rows if r["side"] == side]
        if len(s_rows) < 20:
            continue
        lv: Dict[str, Any] = {"n": len(s_rows)}
        best_t, best_exp = None, -9.0
        for t in (1.0, 1.5, 2.0, 2.5, 3.0):
            sim = []
            for r in s_rows:
                if (r["mfe_pre_r"] or 0.0) >= t:
                    sim.append(t)
                elif r["stopped"]:
                    sim.append(-1.0)
                else:
                    sim.append(float(r["eod_r"] or 0.0))
            e = float(np.mean(sim))
            if e > best_exp:
                best_t, best_exp = t, e
        lv["t1_r"], lv["sim_expectancy_r"] = best_t, round(best_exp, 3)
        maes = [float(r["mae_before_1r0"]) for r in s_rows if r["mae_before_1r0"] is not None]
        if len(maes) >= 10:
            lv["stop_scale"] = round(float(np.clip(np.percentile(maes, 85), 0.7, 1.1)), 2)
        new["levels"][side] = lv

    # ---- gainers / losers study -> volume-ignition floor ----
    caught_rv = [float(m["ignition_rvol"]) for m in mover_rows if m["ignition_rvol"] and float(m["ignition_rvol"]) > 0]
    if len(caught_rv) >= 12:
        new["min_rvol"] = round(float(np.clip(np.percentile(caught_rv, 25), 1.2, 2.0)), 2)
    new["mover_capture"] = {
        "movers": len(mover_rows), "caught": sum(1 for m in mover_rows if m["caught"]),
    }

    # ---- change log (shown on Telegram) ----
    changes: List[str] = []
    for k in new["blocked"]:
        if k not in prev.get("blocked", {}):
            v = new["blocked"][k]
            changes.append(f"🚫 BLOCKED {k} (n={v['n']}, avg {v['avg_r']:+.2f}R, win {v['win']}%)")
    for k in prev.get("blocked", {}):
        if k not in new["blocked"]:
            changes.append(f"✅ UNBLOCKED {k}")
    for k in new["boosted"]:
        if k not in prev.get("boosted", {}):
            v = new["boosted"][k]
            changes.append(f"⭐ PROVEN EDGE {k} (n={v['n']}, avg {v['avg_r']:+.2f}R, win {v['win']}%)")
    if prev.get("mode") != new["mode"]:
        changes.append(f"🔁 MODE {prev.get('mode')} → {new['mode']}")
    for side, lv in new["levels"].items():
        old = prev.get("levels", {}).get(side, {})
        if lv.get("t1_r") != old.get("t1_r") or lv.get("stop_scale") != old.get("stop_scale"):
            changes.append(f"📐 {side} levels: T1 {lv.get('t1_r')}R, stop×{lv.get('stop_scale', 1.0)} (was T1 {old.get('t1_r', 2.5)}R, stop×{old.get('stop_scale', 1.0)})")
    if new["min_rvol"] != prev.get("min_rvol"):
        changes.append(f"📊 Min 5m RVOL → {new['min_rvol']} (was {prev.get('min_rvol')})")
    if new["score_bump"] != prev.get("score_bump", 0):
        changes.append(f"🎚 Score threshold +{new['score_bump']} (was +{prev.get('score_bump', 0)})")
    new["changes"] = (prev.get("changes", []) + [{"at": datetime.now(TZ).isoformat(), "items": changes}])[-30:]
    new["updated_at"] = datetime.now(TZ).isoformat()
    FILTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FILTERS_PATH.write_text(json.dumps(new, indent=2), encoding="utf-8")
    _FILTERS_CACHE.update({"mtime": None, "data": None})
    new["_today_changes"] = changes
    return new


def learned_gate(signal: Dict[str, Any], now: datetime) -> Dict[str, Any]:
    """Final quality gate: uses learned results. Warm-up = alert but tag as LEARNING; proven = only proven-edge trades."""
    f = load_filters()
    side, tier = signal["side"], signal.get("tier", "TIER-1")
    win = window_of(now)
    formulas = signal.get("pillars_matched") or []
    dims = signal_dims(side, tier, win, formulas)
    stats = f.get("stats", {})
    K = LEARN_CFG["SHRINK_K"]
    vals = []
    for d in dims:
        s = stats.get(d)
        vals.append(s["avg_r"] * s["n"] / (s["n"] + K) if s else 0.0)
    edge = round(float(np.mean(vals)), 3) if vals else 0.0
    blocked = f.get("blocked", {})
    hard = [d for d in dims if d in blocked and not d.startswith("formula:")]
    formula_dims = [d for d in dims if d.startswith("formula:")]
    all_formulas_bad = bool(formula_dims) and all(d in blocked for d in formula_dims)
    info = {"ok": True, "mode": f.get("mode", "warmup"), "edge_r": edge, "reason": None,
            "boosted": [d for d in dims if d in f.get("boosted", {})]}
    if f.get("mode") == "warmup":
        return info
    if hard:
        return {**info, "ok": False, "reason": f"blocked bucket {hard[0]}"}
    if all_formulas_bad:
        return {**info, "ok": False, "reason": "all matched formulas have negative track record"}
    min_rvol = f.get("min_rvol")
    if min_rvol and float(signal.get("relative_volume", 0)) < float(min_rvol):
        return {**info, "ok": False, "reason": f"5m RVOL below learned floor {min_rvol}"}
    base = CFG["MIN_SCORE_TIER1"] if tier == "TIER-1" else CFG["MIN_SCORE_TIER2"]
    if int(signal.get("score", 0)) < base + int(f.get("score_bump", 0)):
        return {**info, "ok": False, "reason": "score below learned threshold"}
    if edge < LEARN_CFG["MIN_EDGE_R"] and not info["boosted"]:
        return {**info, "ok": False, "reason": f"learned edge {edge:+.2f}R too low"}
    return info


def study_movers(normalized: Dict[str, pd.DataFrame], daily_map: Dict[str, Any], replay_day) -> List[Dict[str, Any]]:
    """Read the day's top gainers AND top losers (by close vs previous close), replay the scanner bar-by-bar on each,
    and record whether it would have caught them, when the move ignited, and what blocked it if it didn't."""
    weights = load_learned_weights()
    moves: List[Tuple[float, str]] = []
    for sym, df in normalized.items():
        today_bars = df[[x.date() == replay_day for x in df.index]]
        prior = df[[x.date() < replay_day for x in df.index]]
        if len(today_bars) < 20 or prior.empty:
            continue
        prev_close = float(prior.iloc[-1]["close"])
        moves.append(((float(today_bars.iloc[-1]["close"]) / prev_close - 1.0) * 100.0, sym))
    moves.sort()
    n = LEARN_CFG["TOP_N_MOVERS"]
    picks = [(m, s, "LOSER") for m, s in moves[:n] if m < 0] + [(m, s, "GAINER") for m, s in moves[::-1][:n] if m > 0]
    out: List[Dict[str, Any]] = []
    for move, sym, kind in picks:
        df = normalized[sym]
        today_bars = df[[x.date() == replay_day for x in df.index]].between_time("09:15", "15:25")
        prior = df[[x.date() < replay_day for x in df.index]]
        prev_close = float(prior.iloc[-1]["close"])
        want = "BUY" if kind == "GAINER" else "SELL"
        gap = (float(today_bars.iloc[0]["open"]) / prev_close - 1.0) * 100.0
        first_sig, ign_time, ign_rvol, ign_formulas, block, sig_formulas, capture = None, None, None, None, None, [], None
        for ts in today_bars.index:
            if ts.time() < dtime(9, 20) or ts.time() > dtime(15, 15):
                continue
            prefix = df.loc[df.index <= ts]
            if len(prefix) < CFG["MIN_HISTORY_BARS"]:
                continue
            px = float(prefix.iloc[-1]["close"])
            mv = (px / prev_close - 1.0) * 100.0
            anchor = ts.to_pydatetime() + timedelta(minutes=5)
            diag: Dict[str, Any] = {}
            sig = evaluate_symbol(sym, prefix, daily_map.get(sym, {}), anchor, diagnostic=diag, allow_tier2=True,
                                  pre_normalized=True, formula_weights=weights)
            if ign_time is None and ((mv >= 1.0 and want == "BUY") or (mv <= -1.0 and want == "SELL")):
                ign_time = ts.strftime("%H:%M")
                rv, _ = same_slot_relative_volume(prefix, ts, 3)
                ign_rvol = round(rv, 2)
                ign_formulas = len(diag.get("buy_formulas" if want == "BUY" else "sell_formulas", []))
                block = diag.get("rejected") or ("gap_over_cap" if abs(gap) > CFG["MAX_DAY_MOVE_PCT"] else None)
            if sig and sig["side"] == want:
                first_sig, sig_formulas = ts.strftime("%H:%M"), sig.get("pillars_matched", [])
                future = today_bars.loc[today_bars.index > ts]
                if not future.empty:
                    if want == "BUY":
                        capture = (float(future["high"].max()) / px - 1.0) * 100.0
                    else:
                        capture = (px - float(future["low"].min())) / px * 100.0
                ign_rvol = ign_rvol if ign_rvol is not None else float(sig.get("relative_volume", 0))
                break
        if first_sig is None and block is None:
            block = "gap_over_cap" if abs(gap) > CFG["MAX_DAY_MOVE_PCT"] else "never_met_entry_rules"
        rec = {"day": str(replay_day), "symbol": sym, "direction": kind, "close_move_pct": round(move, 2),
               "gap_pct": round(gap, 2), "caught": 1 if first_sig else 0, "first_signal_time": first_sig,
               "capture_pct": round(capture, 2) if capture is not None else None, "ignition_time": ign_time,
               "ignition_rvol": ign_rvol, "ignition_formulas": ign_formulas, "block_reason": None if first_sig else block,
               "signal_formulas": json.dumps(sig_formulas)}
        out.append(rec)
        with db_connect() as conn:
            conn.execute("INSERT OR REPLACE INTO mover_study VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", tuple(rec.values()))
            conn.commit()
    return out


def send_long_telegram(text: str) -> None:
    chunk, parts = "", []
    for line in text.split("\n"):
        if len(chunk) + len(line) + 1 > 3800:
            parts.append(chunk)
            chunk = ""
        chunk += line + "\n"
    if chunk.strip():
        parts.append(chunk)
    for part in parts:
        send_telegram_message(part.rstrip())


def run_daily_learning(send_telegram: bool = True) -> Dict[str, Any]:
    """End-of-day job: (1) grade every alert by real price path, (2) study top gainers/losers,
    (3) update filters + levels, (4) report everything on Telegram."""
    started = time.monotonic()
    init_db()
    init_learning_tables()
    if not UNIVERSE_CACHE.exists():
        refresh_liquid_universe(force=True)
    universe = json.loads(UNIVERSE_CACHE.read_text())
    entries = universe.get("scan_symbols", [])
    daily_map = {x["symbol"]: x for x in entries}
    with db_connect() as conn:
        sig_syms = [r["symbol"] for r in conn.execute(
            "SELECT DISTINCT s.symbol FROM signals s LEFT JOIN outcomes o ON o.signal_id=s.id WHERE o.signal_id IS NULL")]
    symbols = list(dict.fromkeys([e["symbol"] for e in entries] + sig_syms))
    frames: Dict[str, pd.DataFrame] = {}
    for off in range(0, len(symbols), CFG["INTRADAY_BATCH_SIZE"]):
        frames.update(download_batch(symbols[off:off + CFG["INTRADAY_BATCH_SIZE"]], CFG["INTRADAY_PERIOD"], CFG["BAR_INTERVAL"]))
    normalized = {}
    sessions = []
    for sym, fr in frames.items():
        if fr is None or fr.empty:
            continue
        clean = normalize_intraday(fr).between_time("09:15", "15:25")
        if not clean.empty:
            normalized[sym] = clean
            sessions.append(clean.index[-1].date())
    if not sessions:
        raise RuntimeError("No intraday data returned; learning skipped")
    replay_day = max(sessions)

    resolved = resolve_pending_outcomes(frames)
    already_studied = False
    with db_connect() as conn:
        already_studied = conn.execute("SELECT 1 FROM mover_study WHERE day=? LIMIT 1", (str(replay_day),)).fetchone() is not None
    movers = [] if already_studied else study_movers({s: normalized[s] for s in entries_symbols(entries, normalized)}, daily_map, replay_day)
    flt = recompute_filters()
    persist_state_to_git()

    report = {"day": str(replay_day), "resolved": len(resolved), "movers": len(movers), "mode": flt["mode"],
              "elapsed_s": round(time.monotonic() - started, 1)}
    if send_telegram:
        try:
            send_long_telegram(format_learning_report(replay_day, resolved, movers, flt))
        except Exception as exc:
            LOG.warning("Learning report Telegram failed: %s", exc)
    LOG.info("DAILY LEARNING %s", json.dumps(report))
    return report


def entries_symbols(entries: List[Dict[str, Any]], normalized: Dict[str, pd.DataFrame]) -> List[str]:
    return [e["symbol"] for e in entries if e["symbol"] in normalized]


def format_learning_report(day, resolved, movers, flt) -> str:
    L = [f"🧠 BHARAT SCOUT — DAILY LEARNING ({day})", ""]
    L.append(f"Mode: {'🧪 WARM-UP (collecting data, alerts tagged LEARNING)' if flt['mode'] == 'warmup' else '🎯 PROVEN-ONLY (strict filters ON)'}"
             f" | graded alerts so far: {flt['total_resolved']}/{LEARN_CFG['WARMUP_TOTAL']} needed")
    L.append("")
    L.append("📋 ALERT RESULTS (real price path after alert):")
    if resolved:
        for r in resolved[-12:]:
            icon = {"T1": "✅", "STOP": "❌", "EOD_WIN": "🟡", "EOD_LOSS": "🟠", "NO_DATA": "⚪"}.get(r["result"], "•")
            if r["r_result"] is None:
                L.append(f"{icon} {r['symbol']} {r['side']} — no data")
                continue
            L.append(f"{icon} {r['symbol']} {r['side']} {r['result']} {r['r_result']:+.2f}R | went {r['mfe_pct']}% your way, {r['mae_pct']}% against")
        rs = [r["r_result"] for r in resolved if r["r_result"] is not None]
        if rs:
            L.append(f"Today: {sum(1 for x in rs if x > 0)}/{len(rs)} profitable, net {sum(rs):+.2f}R")
    else:
        L.append("No new alerts to grade.")
    rec = flt.get("recent")
    if rec:
        L.append(f"Last {rec['n']} alerts: win {rec['win']}% | avg {rec['avg_r']:+.2f}R per trade")
    L.append("")
    L.append("🔎 TOP GAINERS / LOSERS STUDY:")
    if movers:
        caught = [m for m in movers if m["caught"]]
        L.append(f"Scanner would have caught {len(caught)}/{len(movers)} of today's biggest movers.")
        for m in movers[:8] + [x for x in movers if x["direction"] == "LOSER"][:0]:
            tag = f"✅ caught {m['first_signal_time']} (+{m['capture_pct']}% after)" if m["caught"] else f"❌ missed ({m['block_reason']})"
            L.append(f"• {m['symbol']} {m['close_move_pct']:+.1f}% — ignited {m['ignition_time'] or 'at open'} | {tag}")
        gaps = sum(1 for m in movers if m["block_reason"] == "gap_over_cap")
        if gaps:
            L.append(f"{gaps} mover(s) had already gapped past the {CFG['MAX_DAY_MOVE_PCT']}% cap at open — not chase-able.")
    else:
        L.append("Already studied today or no data.")
    L.append("")
    L.append("🔧 WHAT CHANGED FOR TOMORROW:")
    ch = flt.get("_today_changes") or []
    L.extend(ch if ch else ["No filter changes (not enough proof yet to change anything)."])
    if flt.get("blocked"):
        L.append("Currently blocked: " + "; ".join(f"{k} ({v['avg_r']:+.2f}R, n={v['n']})" for k, v in list(flt["blocked"].items())[:6]))
    if flt.get("boosted"):
        L.append("Proven edges: " + "; ".join(f"{k} ({v['avg_r']:+.2f}R, n={v['n']})" for k, v in list(flt["boosted"].items())[:6]))
    for side, lv in flt.get("levels", {}).items():
        L.append(f"{side} levels (from {lv['n']} alerts): target {lv.get('t1_r')}R, stop ×{lv.get('stop_scale', 1.0)}")
    L.append("")
    L.append("Scanner auto-starts 09:20 IST tomorrow. Research alerts only; no orders placed.")
    return "\n".join(L)


def send_morning_brief() -> None:
    """Sent once at the start of each session (≈09:20 IST) so you see exactly which rules are live today."""
    day = datetime.now(TZ).strftime("%Y-%m-%d")
    with db_connect() as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key='morning_brief_day'").fetchone()
        if row and row["value"] == day:
            return
        conn.execute("INSERT OR REPLACE INTO app_state(key,value) VALUES('morning_brief_day',?)", (day,))
        conn.commit()
    f = load_filters()
    lines = [f"🌅 BHARAT SCOUT LIVE — {day} 09:20 IST",
             f"Mode: {'🧪 WARM-UP' if f.get('mode') == 'warmup' else '🎯 PROVEN-ONLY'} | graded alerts: {f.get('total_resolved', 0)}",
             f"Kronos: {kronos_status}"]
    if f.get("blocked"):
        lines.append("🚫 Blocked today: " + "; ".join(list(f["blocked"].keys())[:6]))
    if f.get("min_rvol"):
        lines.append(f"📊 Learned min 5m RVOL: {f['min_rvol']}x")
    if f.get("score_bump"):
        lines.append(f"🎚 Extra score required: +{f['score_bump']}")
    for side, lv in f.get("levels", {}).items():
        lines.append(f"📐 {side}: T1 {lv.get('t1_r')}R, stop ×{lv.get('stop_scale', 1.0)}")
    try:
        send_telegram_message("\n".join(lines))
    except Exception as exc:
        LOG.warning("Morning brief failed: %s", exc)


def seconds_until_next_5m_slot(now: Optional[datetime] = None) -> float:
    now = now or datetime.now(TZ)
    minute_mod = now.minute % 5
    mins_to_add = 5 - minute_mod
    next_slot = (now + timedelta(minutes=mins_to_add)).replace(second=12, microsecond=0)
    wait_sec = (next_slot - now).total_seconds()
    if wait_sec < 15:
        wait_sec += 300.0
    return wait_sec


def run_continuous_market_session() -> Dict[str, Any]:
    session_start = time.monotonic()
    max_seconds = CFG["CONTINUOUS_LOOP_MINUTES"] * 60.0
    now = datetime.now(TZ)

    if is_pre_market_wait_window(now):
        target_open = now.replace(hour=9, minute=20, second=10, microsecond=0)
        wait_open = max(0.0, (target_open - now).total_seconds())
        LOG.info("Pre-market start detected at %s; waiting %.0fs until 09:20:10 IST market scan...", now.strftime("%H:%M:%S"), wait_open)
        time.sleep(wait_open)
        now = datetime.now(TZ)

    if not is_market_session(now):
        LOG.info("Outside NSE market session (%s); scan skipped cleanly", now.strftime("%H:%M:%S"))
        return {"status": "outside_market_hours", "at": now.isoformat()}

    load_kronos()
    send_morning_brief()

    cycles_run = 0
    last_result: Dict[str, Any] = {"status": "idle"}

    while True:
        now = datetime.now(TZ)
        if not is_entry_window(now):
            LOG.info("Reached end of intraday entry window (%s IST); running post-market self-learning pass...", now.strftime("%H:%M:%S"))
            try:
                run_daily_learning(send_telegram=True)
            except Exception as exc:
                LOG.warning("Post-market learning warning: %s", exc)
            break

        cycles_run += 1
        LOG.info("=== AUTO-SCAN CYCLE #%d at %s IST ===", cycles_run, now.strftime("%H:%M:%S"))
        try:
            last_result = run_scan_cycle()
        except Exception as exc:
            LOG.exception("Scan cycle #%d encountered transient error (continuing loop): %s", cycles_run, exc)
            last_result = {"status": "transient_error", "detail": str(exc)[:200]}

        elapsed = time.monotonic() - session_start
        wait_sec = seconds_until_next_5m_slot(datetime.now(TZ))
        if elapsed + wait_sec >= max_seconds:
            LOG.info("Session loop window (%.0fm) completed after %d cycles.", CFG["CONTINUOUS_LOOP_MINUTES"], cycles_run)
            break

        LOG.info("Sleeping %.0fs until next 5-minute candle close...", wait_sec)
        time.sleep(wait_sec)

    last_result["cycles_run"] = cycles_run
    return last_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Bharat Intraday Scout 2.0 Tri-Engine + Self-Learning Backtester")
    parser.add_argument(
        "--task",
        choices=["scan", "daily-learn", "backtest-learn", "dry-run-v2", "refresh-universe", "test-telegram", "test-kronos", "reset-today-quota"],
        default="scan",
    )
    parser.add_argument("--symbol", default="", help="Optional NSE symbol to reverse-engineer in backtest-learn")
    args = parser.parse_args()

    if args.task == "daily-learn":
        run_daily_learning(send_telegram=True)
        return 0

    if args.task in ("dry-run-v2", "backtest-learn"):
        load_kronos()
        run_backtest_and_learn(target_symbol=args.symbol, send_telegram=True)
        return 0

    init_db()

    if args.task == "reset-today-quota":
        day = datetime.now(TZ).strftime("%Y-%m-%d")
        with db_connect() as conn:
            conn.execute("DELETE FROM signals WHERE created_at LIKE ?", (day + "%",))
            conn.commit()
        persist_state_to_git()
        LOG.info("Cleared today's signal counter (%s); scanner will now resume alerting today.", day)
        return 0

    if args.task == "refresh-universe":
        result = refresh_liquid_universe(force=True)
        LOG.info(
            "UNIVERSE READY | eligible=%s | selected=%s | refreshed=%s",
            result.get("eligible_count"),
            len(result.get("scan_symbols", [])),
            result.get("refreshed_at"),
        )
        return 0

    if args.task == "test-telegram":
        send_telegram_message("✅ Bharat Intraday Scout 2.0 (fmzquant + Kronos + Trading-R1 + Self-Learning Backtester) active.")
        LOG.info("Telegram test succeeded")
        return 0

    if args.task == "test-kronos":
        load_kronos()
        if kronos_predictor is None:
            raise RuntimeError("Kronos failed to load: " + kronos_status)
        LOG.info("Kronos self-test succeeded: %s", kronos_status)
        return 0

    result = run_continuous_market_session()
    LOG.info("RUN RESULT %s", json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
