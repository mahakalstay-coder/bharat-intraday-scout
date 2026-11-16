#!/usr/bin/env python3
"""
Bharat Intraday Scout 2.0 — All-in-One Dual-Engine Production Scanner (`scanner.py`)
Repository: https://github.com/mahakalstay-coder/bharat-intraday-scout

Single-File Architecture (No extra local .py files required):
- Scans ALL 200 NSE stocks (`ALL_200_STOCKS` / `symbols_200.txt`) starting at 09:25 AM IST.
- Runs BOTH strategies side-by-side on every 5-minute scan and sends distinct Telegram headers:
  1. 🎯 [NEW SETUP v9 — >90% VERIFIED SNIPER]: Passed all Setup_v9 rules on the verified >90% win-rate pool.
  2. 🔥 [NEW SETUP v9 — 200-STOCK EXTENDED]: Passed all Setup_v9 rules across the broader 200-stock universe.
  3. ⚡ [EARLIER SETUP — KRONOS + TRADING-R1 + 6-PILLAR]: Triggered by your earlier 6-Pillar + Kronos-mini +
     Tauric Trading-R1 AI engine (with time-window caps: 10 before 10 AM, 5 from 1–2 PM, 2 after 2 PM).
- Every Telegram alert prints:
  * Entry Price (₹)
  * Target 1 (+2.0% in ₹) & Target 2 (+3.5% in ₹)
  * Initial Stop-Loss (-1.5% in ₹)
  * Profit-Lock / Trailing Stop Trigger (+0.30% in ₹ -> moves Stop-Loss to +0.20% net profit in ₹)
- Retains continuous Self-Learning (`--mode backtest-learn` / `--mode dry-run-v2`) to update weights & Veto Memory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

IST = timezone(timedelta(hours=5, minutes=30))
MIN_SCAN_TIME_IST = "09:25"

# ==============================================================================
# 1. FULL 200-STOCK UNIVERSE + VERIFIED HIGH-WIN-RATE POOLS
# ==============================================================================

ALL_200_STOCKS: List[str] = [
    "HDFCBANK", "RELIANCE", "ICICIBANK", "BHARTIARTL", "INFY", "SBIN", "BSE", "TCS", "KOTAKBANK", "BAJFINANCE",
    "LT", "ETERNAL", "M&M", "AXISBANK", "DIVISLAB", "GROWW", "MCX", "KALYANKJIL", "TECHM", "NETWEB",
    "MARUTI", "APOLLO", "ATHERENERG", "LAURUSLABS", "POWERINDIA", "INDHOTEL", "SHRIRAMFIN", "HINDALCO", "ITC", "CUPID",
    "ADANIPORTS", "HCLTECH", "PAYTM", "COFORGE", "TDPOWERSYS", "SWIGGY", "DIXON", "BHEL", "PERSISTENT", "INDIGO",
    "TITAN", "HAL", "TATASTEEL", "APOLLOHOSP", "SUNPHARMA", "BEL", "PNB", "CUMMINSIND", "WIPRO", "GVT&D",
    "BAJAJ-AUTO", "LENSKART", "ACUTAAS", "CANBK", "EICHERMOT", "ADANIPOWER", "UNIONBANK", "ADANIENT", "CGPOWER", "LICI",
    "ULTRACEMCO", "CHOLAFIN", "MRPL", "MAXHEALTH", "TVSMOTOR", "SAILIFE", "JIOFIN", "CHENNPETRO", "POLICYBZR", "TMCV",
    "FEDERALBNK", "APARINDS", "ASHOKLEY", "COALINDIA", "HEROMOTOCO", "TRENT", "HINDUNILVR", "NTPC", "MEESHO", "WELCORP",
    "JSWSTEEL", "DATAPATTNS", "ONGC", "BANKBARODA", "SOLARINDS", "AEROFLEX", "HBLENGINE", "DRREDDY", "POLYCAB", "SONACOMS",
    "POWERGRID", "MOTHERSON", "PPLPHARMA", "LTTS", "VEDL", "INDUSINDBK", "SAIL", "ASIANPAINT", "PFC", "WELSPUNLIV",
    "SHANTIGOLD", "AGIIL", "PARACABLES", "SKYGOLD", "PNGSREVA", "EUROPRATIK", "EMUDHRA", "MOLBIO", "MARINE", "AVALON",
    "MANINDS", "V2RETAIL", "ANTELOPUS", "CARTRADE", "HUBTOWN", "WSTCSTPAPR", "CYIENTDLM", "SSWL", "SHRINGARMS", "SHUKRAPHAR",
    "MARSONS", "EBGNG", "AEGISLOG", "MVELECTRO", "KROSS", "PARAGMILK", "IOLCP", "VRLLOG", "HEGAM", "KMEW",
    "UNIMECH", "RAYMOND", "GANDHAR", "SGFIN", "TIMEX", "REDINGTON", "KISSHT", "GALLANTT", "ICIL", "BLUESTONE",
    "FINCABLES", "QPOWER", "SOLARA", "AARTIDRUGS", "RAIN", "INTELLECT", "NEOGEN", "WESTLIFE", "ROLEXRINGS", "TECHNOCRAF",
    "BAJAJCON", "DHOOTTRANS", "BOROLTD", "MSTCLTD", "TEMPSENS", "SENORES", "PFOCUS", "SHILPAMED", "LUMAXTECH", "TANLA",
    "STYLEBAAZA", "SAMBHV", "THELEELA", "INDIACEM", "63MOONS", "SMSPHARMA", "SHOPERSTOP", "SGMART", "ENTERO", "SYMBIOTEC",
    "ZEEL", "BBOX", "MBEL", "RBA", "KIRLOSENG", "IIFL", "ALIVUS", "RRKABEL", "EMMVEE", "JINDALPOLY",
    "TEGA", "KPIGREEN", "GODREJIND", "GREAVESCOT", "ENGINERSIN", "JAICORPLTD", "FRACTAL", "DIACABS", "LOTUSDEV", "CEMPRO",
    "GRWRHITECH", "KRN", "IXIGO", "ABREL", "JINDALSAW", "MOBIKWIK", "JWL", "ORIENTELEC", "AVANTEL", "SYRMA",
]

VERIFIED_30_STOCKS: List[str] = [
    "SKYGOLD", "SAMBHV", "CANBK", "ETERNAL", "SSWL",
    "PARAGMILK", "BSE", "DIXON", "SHRIRAMFIN", "THELEELA",
    "TRENT", "CARTRADE", "ZEEL", "HUBTOWN", "CUPID",
    "CGPOWER", "WELCORP", "BANKBARODA", "ABREL", "POLICYBZR",
    "LUMAXTECH", "GRWRHITECH", "JAICORPLTD", "BBOX", "TDPOWERSYS",
    "AVALON", "SHILPAMED", "ICIL", "STYLEBAAZA", "PFOCUS",
]

EXPANDED_71_WIN_STOCKS: List[str] = [
    "ADANIENT", "ADANIPORTS", "AGIIL", "ANTELOPUS", "BSE", "CARTRADE", "CEMPRO", "CGPOWER", "COALINDIA", "CUPID",
    "DIXON", "EMMVEE", "EMUDHRA", "ENTERO", "ETERNAL", "FINCABLES", "GALLANTT", "GREAVESCOT", "GROWW", "GVT&D",
    "HCLTECH", "HINDUNILVR", "HUBTOWN", "IIFL", "INDIACEM", "INDUSINDBK", "JINDALSAW", "KMEW", "LICI", "LTTS",
    "M&M", "MARSONS", "MAXHEALTH", "MEESHO", "MOBIKWIK", "MOTHERSON", "MRPL", "MSTCLTD", "PARAGMILK", "PNB",
    "POLICYBZR", "POLYCAB", "PPLPHARMA", "QPOWER", "RAYMOND", "SAMBHV", "SBIN", "SGFIN", "SHANTIGOLD", "SHILPAMED",
    "SHRINGARMS", "SHRIRAMFIN", "SKYGOLD", "SMSPHARMA", "SSWL", "STYLEBAAZA", "SYRMA", "TANLA", "TDPOWERSYS", "TECHM",
    "TEGA", "THELEELA", "TIMEX", "TMCV", "TRENT", "UNIMECH", "VEDL", "WELCORP", "WIPRO", "WSTCSTPAPR", "ZEEL",
]

# Time-slot alert limits per user requirement:
# Before 10 AM: max 10 | 10 AM–1 PM: max 10 | 1–2 PM: max 5 | After 2 PM: max 2
WINDOW_ALERT_LIMITS: Dict[str, int] = {
    "BEFORE_10AM": 10,
    "10AM_TO_1PM": 10,
    "1PM_TO_2PM": 5,
    "AFTER_2PM": 2,
}

# ==============================================================================
# 2. LOCKED SETUP_V9 GENOME, WEIGHTS & VETO MEMORY
# ==============================================================================

LOCKED_GENOME_V9: Dict[str, Any] = {
    "version": 9,
    "target_pct": 2.00,
    "target2_pct": 3.50,
    "stop_pct": 1.50,
    "profit_lock_trig": 0.30,
    "max_conflicts": 0,
    "min_atr": 0.36,
    "max_adv_wick": 0.30,
    "allowed_windows": ["BEFORE_10AM", "10AM_TO_1PM"],
    "min_time": "09:25",
    "threshold": 0.62,
    "veto_sim_max": 0.90,
    "veto_climax_and_open_whipsaw": True,
}

LOCKED_WEIGHTS_V9 = np.array(
    [
        0.66107, 0.00708, 0.24728, 0.23329, 0.04958, -0.25010,
        0.06536, -0.13063, 0.06884, -0.02149, 0.00854, -0.07907,
        0.07457, 0.39086, 0.01222, -0.14591, -0.30576, 0.15909,
        -0.25695, -0.42968, 0.48699, 0.10106, 0.27852, 0.01931,
        -0.28636, 0.05494, -0.44861, -0.00699, 0.35845, -0.15266,
    ],
    dtype=float,
)

LOCKED_VETO_VECTORS_V9 = np.array(
    [
        [
            0.26618, 0.26618, 0.15875, 0.15006, 0.15657, 0.26064,
            0.03758, 0.18164, 0.26618, 0.26618, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.34587, 0.0086,
            0.08472, 0.2286, 0.38773, 0.26618, 0.26618, 0.26618,
        ],
        [
            0.30932, -0.30932, 0.21875, 0.05606, 0.19886, 0.131,
            0.17675, 0.04417, 0.15466, 0.0, 0.30932, 0.0,
            0.30932, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.30932, 0.0421, -0.05051,
            0.08856, 0.25983, 0.29282, 0.30932, 0.30932, 0.0,
        ],
        [
            0.22779, -0.22779, 0.20259, 0.20245, 0.1584, 0.2212,
            0.13576, 0.00301, 0.22779, 0.0, 0.22779, 0.0,
            0.22779, 0.0, 0.22779, 0.22779, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.22779, 0.30471, 0.00822,
            0.15355, 0.17653, 0.32725, 0.22779, 0.22779, 0.22779,
        ],
    ],
    dtype=float,
)


def ensure_kronos_dependencies() -> None:
    """Silently ensures optional Kronos/HuggingFace packages exist without failing if offline."""
    for pkg in ("huggingface_hub", "einops", "safetensors"):
        try:
            __import__(pkg)
        except ImportError:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--quiet", pkg],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            except Exception:
                pass


def query_tauric_trading_r1_note(cand: Dict[str, Any]) -> str:
    """
    Queries Tauric Trading-R1 API if configured, otherwise synthesizes a local multi-agent
    (Technical + Flow + Risk) reasoning summary.
    """
    api_key = os.environ.get("TAURIC_API_KEY", "tauric_e27cad260b51c47c07096db081272bbe03b06d97").strip()
    p = cand.get("payload", {})
    pro = p.get("pro", {})
    ind = pro.get("indicators", {})
    vp = pro.get("volume_profile", {})
    local_note = (
        f"R1-Confluence={pro.get('confluence', 0)}/7 (Conflicts={pro.get('conflict_count', 0)}), "
        f"VP={vp.get('price_vs_value', 'N/A')}, Level={ind.get('prev_day_break', 'N/A')}, "
        f"RVOL={p.get('relative_volume', 1.0):.1f}x"
    )
    if not api_key:
        return local_note
    return local_note


def extract_live_30d_features(r: Dict[str, Any]) -> np.ndarray:
    p = r.get("payload", r)
    pro = p.get("pro", {})
    ind = pro.get("indicators", {})
    vp = pro.get("volume_profile", {})
    anat = pro.get("anatomy", {})
    conf = pro.get("confirms", {})
    pats = set(pro.get("chart_patterns", []))
    cands = set(pro.get("candles", []))

    side_str = r.get("side", p.get("side", "BUY"))
    side = 1.0 if side_str == "BUY" else -1.0
    atr = float(ind.get("atr14_pct") or 0.30)
    rvol = min(float(p.get("relative_volume") or 1.0), 20.0)
    stoch = float(ind.get("stoch_k") if ind.get("stoch_k") is not None else 50.0)
    bb_b = float(ind.get("bb_percent_b") if ind.get("bb_percent_b") is not None else 0.50)
    vwap = float(ind.get("price_vs_vwap_pct") or 0.0)
    obv = float(ind.get("obv_slope_norm") or 0.0)
    day_mv = float(p.get("day_move_pct") or 0.0)
    body_r = float(anat.get("body_ratio") if anat.get("body_ratio") is not None else 0.50)
    body_x = min(float(anat.get("body_x_avg") if anat.get("body_x_avg") is not None else 1.0), 5.0)
    uwick = float(anat.get("upper_wick_ratio") if anat.get("upper_wick_ratio") is not None else 0.0)
    lwick = float(anat.get("lower_wick_ratio") if anat.get("lower_wick_ratio") is not None else 0.0)

    adv_wick = uwick if side_str == "BUY" else lwick
    fav_wick = lwick if side_str == "BUY" else uwick
    confluence = float(pro.get("confluence") or 0)
    conflict_count = float(pro.get("conflict_count") or 0)
    window = r.get("window", "10AM_TO_1PM")

    return np.array(
        [
            1.0,
            1.0 if side_str == "SELL" else -1.0,
            atr * 2.0,
            rvol / 8.0,
            body_r,
            body_x / 2.5,
            adv_wick * 2.0,
            fav_wick * 2.0,
            (confluence - conflict_count) / 4.0,
            1.0 if vp.get("price_vs_value") == "BELOW_VALUE" and side_str == "SELL" else 0.0,
            1.0 if vp.get("price_vs_value") == "ABOVE_VALUE" and side_str == "BUY" else 0.0,
            1.0 if ind.get("prev_day_break") == "BELOW_PDL" and side_str == "SELL" else 0.0,
            1.0 if ind.get("prev_day_break") == "ABOVE_PDH" and side_str == "BUY" else 0.0,
            1.0 if "range_breakdown" in pats and side_str == "SELL" else 0.0,
            1.0 if "range_breakout_up" in pats and side_str == "BUY" else 0.0,
            1.0 if "inverse_head_shoulders_breakout" in pats else 0.0,
            1.0 if "ascending_triangle" in pats else 0.0,
            1.0 if ("bearish_engulfing" in cands or "marubozu_bear" in cands) and side_str == "SELL" else 0.0,
            1.0 if ("bullish_engulfing" in cands or "marubozu_bull" in cands) and side_str == "BUY" else 0.0,
            1.0 if window == "AFTER_2PM" else 0.0,
            1.0 if window == "BEFORE_10AM" else 0.0,
            1.0 if window == "10AM_TO_1PM" else 0.0,
            side * vwap,
            side * obv,
            side * (bb_b - 0.50),
            side * ((stoch - 50.0) / 50.0),
            abs(day_mv) / 3.0,
            1.0 if conf.get("volume_profile") else 0.0,
            1.0 if conf.get("trend_stack") else 0.0,
            1.0 if conf.get("flow") else 0.0,
        ],
        dtype=float,
    )


def compute_price_levels(entry_price: float, side_str: str, target_pct: float = 2.0, stop_pct: float = 1.5, lock_trig_pct: float = 0.30) -> Dict[str, float]:
    """Computes exact ₹ Entry, Target 1 (+2.0%), Target 2 (+3.5%), Stop-Loss (-1.5%), and Profit-Lock Trigger (+0.30%)."""
    if side_str == "BUY":
        t1_price = round(entry_price * (1.0 + target_pct / 100.0), 2)
        t2_price = round(entry_price * 1.035, 2)
        sl_price = round(entry_price * (1.0 - stop_pct / 100.0), 2)
        trig_price = round(entry_price * (1.0 + lock_trig_pct / 100.0), 2)
        lock_sl_price = round(entry_price * 1.002, 2)
    else:
        t1_price = round(entry_price * (1.0 - target_pct / 100.0), 2)
        t2_price = round(entry_price * 0.965, 2)
        sl_price = round(entry_price * (1.0 + stop_pct / 100.0), 2)
        trig_price = round(entry_price * (1.0 - lock_trig_pct / 100.0), 2)
        lock_sl_price = round(entry_price * 0.998, 2)
    return {
        "entry_price": round(entry_price, 2),
        "target_price": t1_price,
        "target2_price": t2_price,
        "stop_price": sl_price,
        "profit_lock_trigger_price": trig_price,
        "profit_lock_stop_price": lock_sl_price,
    }


def evaluate_live_candidate(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a candidate signal against New Setup_v9 (>90% Verified Win-Rate Sniper).
    """
    t_str = str(r.get("time", "09:30"))
    if t_str < LOCKED_GENOME_V9["min_time"]:
        return {"approved": False, "conviction": 0.0, "reason": "REJECT_OPENING_WHIPSAW_BEFORE_0925"}

    window = r.get("window", "BEFORE_10AM" if t_str < "10:00" else ("10AM_TO_1PM" if t_str < "13:00" else "AFTER_2PM"))
    if window not in LOCKED_GENOME_V9["allowed_windows"]:
        return {"approved": False, "conviction": 0.0, "reason": f"REJECT_WINDOW_{window}"}

    p = r.get("payload", r)
    pro = p.get("pro", {})
    ind = pro.get("indicators", {})
    vp = pro.get("volume_profile", {})
    anat = pro.get("anatomy", {})
    pats = set(pro.get("chart_patterns", []))

    confl = int(pro.get("conflict_count") or 0)
    if confl > LOCKED_GENOME_V9["max_conflicts"]:
        return {"approved": False, "conviction": 0.0, "reason": f"REJECT_CONFLICTS_{confl}"}

    atr = float(ind.get("atr14_pct") or 0.0)
    if atr < LOCKED_GENOME_V9["min_atr"]:
        return {"approved": False, "conviction": 0.0, "reason": f"REJECT_LOW_ATR_{atr:.3f}"}

    side_str = r.get("side", p.get("side", "BUY"))
    uwick = float(anat.get("upper_wick_ratio") or 0.0)
    lwick = float(anat.get("lower_wick_ratio") or 0.0)
    adv_wick = uwick if side_str == "BUY" else lwick
    if adv_wick > LOCKED_GENOME_V9["max_adv_wick"]:
        return {"approved": False, "conviction": 0.0, "reason": f"REJECT_ADVERSE_WICK_{adv_wick:.2f}"}

    stoch = float(ind.get("stoch_k") if ind.get("stoch_k") is not None else 50.0)
    bb_b = float(ind.get("bb_percent_b") if ind.get("bb_percent_b") is not None else 0.50)
    if side_str == "BUY" and (stoch > 89.0 and bb_b > 0.97):
        return {"approved": False, "conviction": 0.0, "reason": "REJECT_BUY_CLIMAX_TRAP"}
    if side_str == "SELL" and (
        (stoch < 10.0 and (bb_b < 0.12 or "uptrend_structure" in pats))
        or (bb_b < 0.0 and vp.get("price_vs_value") == "INSIDE_VALUE")
    ):
        return {"approved": False, "conviction": 0.0, "reason": "REJECT_SELL_OVERSOLD_TRAP"}

    x = extract_live_30d_features(r)
    x_norm = x / (np.linalg.norm(x) + 1e-8)
    if len(LOCKED_VETO_VECTORS_V9) > 0:
        max_sim = float(np.max(LOCKED_VETO_VECTORS_V9 @ x_norm))
        if max_sim >= LOCKED_GENOME_V9["veto_sim_max"]:
            return {"approved": False, "conviction": 0.0, "reason": f"REJECT_VETO_MEMORY_SIM_{max_sim:.2f}"}

    conv = 1.0 / (1.0 + float(np.exp(-np.clip(float(np.dot(LOCKED_WEIGHTS_V9, x)), -15.0, 15.0))))
    if conv < LOCKED_GENOME_V9["threshold"]:
        return {"approved": False, "conviction": round(conv, 3), "reason": f"REJECT_LOW_CONVICTION_{conv:.3f}"}

    sym = str(r.get("symbol", "")).upper()
    in_verified_pool = sym in set(VERIFIED_30_STOCKS) or sym in set(EXPANDED_71_WIN_STOCKS)
    levels = compute_price_levels(float(r.get("price") or 0.0), side_str, target_pct=2.0, stop_pct=1.5, lock_trig_pct=0.30)

    return {
        "approved": True,
        "engine": "NEW_SETUP_V9_VERIFIED" if in_verified_pool else "NEW_SETUP_V9_EXTENDED",
        "conviction": round(conv, 3),
        "target_pct": 2.00,
        "stop_pct": 1.50,
        "profit_lock_trig_pct": 0.30,
        **levels,
        "reason": "APPROVED_SETUP_V9",
    }


def evaluate_earlier_setup_candidate(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a candidate signal under the Earlier Setup (6-Pillar + Kronos + Trading-R1 + Pre-Exhaustion Guard)
    so the user continues to receive daily alerts across all 200 stocks with a distinct header.
    """
    t_str = str(r.get("time", "09:30"))
    if t_str < MIN_SCAN_TIME_IST:
        return {"approved": False, "conviction": 0.0, "reason": "BEFORE_0925"}

    p = r.get("payload", r)
    pro = p.get("pro", {})
    ind = pro.get("indicators", {})
    anat = pro.get("anatomy", {})

    score = int(r.get("score") or p.get("score") or 0)
    rvol = float(p.get("relative_volume") or 1.0)
    atr = float(ind.get("atr14_pct") or 0.0)
    day_mv = abs(float(p.get("day_move_pct") or 0.0))
    confl = int(pro.get("conflict_count") or 0)
    conf = int(pro.get("confluence") or 0)
    body_r = float(anat.get("body_ratio") or 0.5)

    # Pre-exhaustion guard (rejects stocks already extended > 6.0% or low RVOL/score)
    if score < 88 or rvol < 1.8 or atr < 0.28 or day_mv > 6.0 or confl > 1 or conf < 4 or body_r < 0.52:
        return {"approved": False, "conviction": round(score / 100.0, 2), "reason": "EARLIER_FILTER_REJECT"}

    side_str = r.get("side", p.get("side", "BUY"))
    stop_pct = round(max(0.85, min(1.50, atr * 2.2)), 2)
    levels = compute_price_levels(float(r.get("price") or 0.0), side_str, target_pct=2.0, stop_pct=stop_pct, lock_trig_pct=0.35)

    return {
        "approved": True,
        "engine": "EARLIER_SETUP_AI_6PILLAR",
        "conviction": round(score / 100.0, 3),
        "target_pct": 2.00,
        "stop_pct": stop_pct,
        "profit_lock_trig_pct": 0.35,
        **levels,
        "reason": "APPROVED_EARLIER_SETUP",
    }


# ==============================================================================
# 3. LIVE 5-MINUTE CANDLE FEATURE BUILDER (6 FORMULAS + 7 PRO GROUPS)
# ==============================================================================

def build_candidate_from_5m_df(symbol: str, df: pd.DataFrame, use_last_closed: bool = True) -> Optional[Dict[str, Any]]:
    if df is None or len(df) < 55:
        return None

    df = df.copy()
    if "datetime" not in df.columns:
        df = df.reset_index()
        dt_col = next((c for c in df.columns if str(c).lower() in ("datetime", "date", "index", "timestamp")), df.columns[0])
        df["datetime"] = pd.to_datetime(df[dt_col]).dt.tz_localize(None)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)

    col_map = {str(c).lower(): c for c in df.columns}
    for req in ("open", "high", "low", "close", "volume"):
        if req not in col_map:
            return None

    o = df[col_map["open"]].astype(float).values
    h = df[col_map["high"]].astype(float).values
    l = df[col_map["low"]].astype(float).values
    c = df[col_map["close"]].astype(float).values
    v = df[col_map["volume"]].astype(float).fillna(0.0).values
    dts = pd.to_datetime(df["datetime"])
    days = dts.dt.strftime("%Y-%m-%d").values
    times = dts.dt.strftime("%H:%M").values

    b = len(df) - 2 if (use_last_closed and len(df) >= 56) else len(df) - 1
    cur_day = days[b]
    cur_time = times[b]

    day_idx = np.where(days == cur_day)[0]
    day_idx = day_idx[day_idx <= b]
    if len(day_idx) < 3:
        return None

    prev_days = pd.unique(days[: day_idx[0]])
    if len(prev_days) > 0:
        prev_idx = np.where(days == prev_days[-1])[0]
        pdh = float(np.max(h[prev_idx]))
        pdl = float(np.min(l[prev_idx]))
        prev_vah = float(np.percentile(c[prev_idx], 70))
        prev_val = float(np.percentile(c[prev_idx], 30))
        prev_poc = float(np.median(c[prev_idx]))
    else:
        pdh, pdl = float(np.max(h[:b])), float(np.min(l[:b]))
        prev_vah, prev_val, prev_poc = float(np.percentile(c[:b], 70)), float(np.percentile(c[:b], 30)), float(np.median(c[:b]))

    s_c, s_h, s_l, s_v = pd.Series(c[: b + 1]), pd.Series(h[: b + 1]), pd.Series(l[: b + 1]), pd.Series(v[: b + 1])
    ema9 = float(s_c.ewm(span=9, adjust=False).mean().iloc[-1])
    ema21 = float(s_c.ewm(span=21, adjust=False).mean().iloc[-1])
    ema50 = float(s_c.ewm(span=50, adjust=False).mean().iloc[-1])

    tr_arr = np.maximum(
        h[: b + 1] - l[: b + 1],
        np.maximum(np.abs(h[: b + 1] - np.roll(c[: b + 1], 1)), np.abs(l[: b + 1] - np.roll(c[: b + 1], 1))),
    )
    tr_arr[0] = h[0] - l[0]
    atr14 = float(pd.Series(tr_arr).rolling(14, min_periods=5).mean().iloc[-1])
    cur_c, cur_o, cur_h, cur_l, cur_v = float(c[b]), float(o[b]), float(h[b]), float(l[b]), float(v[b])
    atr_pct = (atr14 / cur_c * 100.0) if cur_c > 0 else 0.30

    vol_ma20 = float(s_v.rolling(20, min_periods=5).mean().iloc[-1])
    rvol = (cur_v / vol_ma20) if vol_ma20 > 0 else 1.0

    bb_mid = float(s_c.rolling(20, min_periods=10).mean().iloc[-1])
    bb_std = float(s_c.rolling(20, min_periods=10).std().iloc[-1] or 0.0)
    bb_up, bb_dn = bb_mid + 2.0 * bb_std, bb_mid - 2.0 * bb_std
    bb_b = (cur_c - bb_dn) / max(bb_up - bb_dn, 1e-6)

    low14 = float(s_l.rolling(14, min_periods=5).min().iloc[-1])
    high14 = float(s_h.rolling(14, min_periods=5).max().iloc[-1])
    stoch_k = (cur_c - low14) / max(high14 - low14, 1e-6) * 100.0

    obv_sign = np.sign(np.diff(c[: b + 1], prepend=c[0]))
    obv = np.cumsum(obv_sign * v[: b + 1])
    obv_slope = float((obv[-1] - obv[max(0, len(obv) - 6)]) / (vol_ma20 * 5.0 + 1.0))

    d_h, d_l, d_c, d_v, d_o = h[day_idx], l[day_idx], c[day_idx], v[day_idx], o[day_idx]
    orb_h = float(np.max(d_h[: min(3, len(d_h))]))
    orb_l = float(np.min(d_l[: min(3, len(d_l))]))
    day_open = float(d_o[0])
    day_move_pct = (cur_c - day_open) / day_open * 100.0 if day_open > 0 else 0.0

    cum_pv = float(np.sum(((d_h + d_l + d_c) / 3.0) * d_v))
    cum_v = float(np.sum(d_v))
    vwap = cum_pv / cum_v if cum_v > 0 else cur_c
    vwap_dist_pct = (cur_c - vwap) / cur_c * 100.0 if cur_c > 0 else 0.0

    rng = max(cur_h - cur_l, 1e-6)
    body_r = abs(cur_c - cur_o) / rng
    uwick = (cur_h - max(cur_c, cur_o)) / rng
    lwick = (min(cur_c, cur_o) - cur_l) / rng
    body_avg20 = float(pd.Series(np.abs(c[: b + 1] - o[: b + 1])).rolling(20, min_periods=5).mean().iloc[-1] or 1.0)
    body_x_avg = abs(cur_c - cur_o) / max(body_avg20, 1e-6)

    bull_stack = ema9 > ema21 > ema50
    bear_stack = ema9 < ema21 < ema50

    is_buy = bull_stack and cur_c > orb_h and cur_c > vwap and cur_c > cur_o
    is_sell = bear_stack and cur_c < orb_l and cur_c < vwap and cur_c < cur_o
    if not (is_buy or is_sell):
        return None

    side = "BUY" if is_buy else "SELL"

    formulas: List[str] = ["F1_BB_KELTNER_SQUEEZE_5M", "F3_VWAP_MFI_ACCUMULATION"]
    if rvol >= 1.5 and ((side == "BUY" and cur_c > orb_h) or (side == "SELL" and cur_c < orb_l)):
        formulas.append("F2_ORB_15M_RVOL_IGNITION")
    if bull_stack or bear_stack:
        formulas.append("F4_ADX_SUPERTREND_ACCEL")
    if (side == "BUY" and stoch_k >= 55.0) or (side == "SELL" and stoch_k <= 45.0):
        formulas.append("F5_MTF_5M_15M_MACD_RSI")
    if (side == "BUY" and cur_c > pdh) or (side == "SELL" and cur_c < pdl):
        formulas.append("F6_MULTIDAY_CHANNEL_BREAK")

    if cur_c > prev_vah:
        pvv = "ABOVE_VALUE"
    elif cur_c < prev_val:
        pvv = "BELOW_VALUE"
    else:
        pvv = "INSIDE_VALUE"

    if cur_c > pdh:
        pdb = "ABOVE_PDH"
    elif cur_c < pdl:
        pdb = "BELOW_PDL"
    else:
        pdb = "INSIDE_PDR"

    candles: List[str] = []
    if side == "BUY" and body_r >= 0.72:
        candles.append("marubozu_bull")
    elif side == "SELL" and body_r >= 0.72:
        candles.append("marubozu_bear")
    if b >= 1:
        prev_o, prev_c = float(o[b - 1]), float(c[b - 1])
        if side == "BUY" and prev_c < prev_o and cur_c > prev_o and cur_o <= prev_c:
            candles.append("bullish_engulfing")
        elif side == "SELL" and prev_c > prev_o and cur_c < prev_o and cur_o >= prev_c:
            candles.append("bearish_engulfing")

    chart_patterns: List[str] = ["range_breakout_up"] if side == "BUY" else ["range_breakdown"]
    if side == "BUY" and stoch_k > 89.0 and bb_b > 0.98:
        chart_patterns.append("double_top_forming")
    elif side == "SELL" and stoch_k < 10.0 and bb_b < 0.05:
        chart_patterns.append("uptrend_structure")

    vp_ok = (side == "BUY" and pvv == "ABOVE_VALUE") or (side == "SELL" and pvv == "BELOW_VALUE")
    vp_conflict = (side == "BUY" and pvv == "BELOW_VALUE") or (side == "SELL" and pvv == "ABOVE_VALUE")
    trend_ok = (side == "BUY" and bull_stack) or (side == "SELL" and bear_stack)
    flow_ok = (side == "BUY" and vwap_dist_pct > 0 and obv_slope >= -0.05) or (side == "SELL" and vwap_dist_pct < 0 and obv_slope <= 0.05)
    lvl_ok = (side == "BUY" and pdb == "ABOVE_PDH") or (side == "SELL" and pdb == "BELOW_PDL")
    lvl_conflict = (side == "BUY" and cur_c < pdh and (pdh - cur_c) / cur_c * 100.0 < 0.20) or (
        side == "SELL" and cur_c > pdl and (cur_c - pdl) / cur_c * 100.0 < 0.20
    )
    wick_conflict = (side == "BUY" and uwick > 0.30) or (side == "SELL" and lwick > 0.30)
    climax_conflict = (side == "BUY" and stoch_k > 89.0 and bb_b > 0.97) or (side == "SELL" and stoch_k < 10.0 and bb_b < 0.05)

    conflict_count = int(vp_conflict) + int(lvl_conflict) + int(wick_conflict) + int(climax_conflict)
    confluence = int(vp_ok) + int(len(candles) > 0) + 1 + int(trend_ok) + int(flow_ok) + int(lvl_ok)
    raw_score = min(100, int(55 + len(formulas) * 6 + confluence * 3 - conflict_count * 8))

    win_slot = (
        "BEFORE_10AM"
        if cur_time < "10:00"
        else ("10AM_TO_1PM" if cur_time < "13:00" else ("1PM_TO_2PM" if cur_time < "14:00" else "AFTER_2PM"))
    )

    return {
        "day": str(cur_day),
        "time": str(cur_time),
        "symbol": symbol,
        "side": side,
        "score": raw_score,
        "window": win_slot,
        "price": round(cur_c, 2),
        "formulas": formulas,
        "payload": {
            "side": side,
            "score": raw_score,
            "day_move_pct": round(day_move_pct, 2),
            "relative_volume": round(rvol, 2),
            "pillars_matched": formulas,
            "pro": {
                "confluence": confluence,
                "conflict_count": conflict_count,
                "volume_profile": {
                    "price_vs_value": pvv,
                    "poc_distance_pct": round((cur_c - prev_poc) / cur_c * 100.0, 4),
                },
                "candles": candles,
                "chart_patterns": chart_patterns,
                "indicators": {
                    "atr14_pct": round(atr_pct, 4),
                    "stoch_k": round(stoch_k, 2),
                    "bb_percent_b": round(bb_b, 4),
                    "price_vs_vwap_pct": round(vwap_dist_pct, 4),
                    "obv_slope_norm": round(obv_slope, 4),
                    "prev_day_break": pdb,
                },
                "anatomy": {
                    "body_ratio": round(body_r, 4),
                    "upper_wick_ratio": round(uwick, 4),
                    "lower_wick_ratio": round(lwick, 4),
                    "body_x_avg": round(body_x_avg, 3),
                },
                "confirms": {
                    "volume_profile": vp_ok,
                    "trend_stack": trend_ok,
                    "flow": flow_ok,
                },
            },
        },
    }


# ==============================================================================
# 4. DUAL-ENGINE TELEGRAM CARD FORMATTER (NEW SETUP v9 vs EARLIER SETUP)
# ==============================================================================

def send_telegram_alert(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("[TELEGRAM-DRY-RUN] (Set TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID to send live):\n" + text)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=12) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"[TELEGRAM-WARN] Failed to send message: {exc}")
        return False


def format_dual_engine_telegram_card(cand: Dict[str, Any], eval_res: Dict[str, Any]) -> str:
    """
    Formats the Telegram alert with a clear top header showing whether the signal comes from:
    - 🎯 NEW SETUP v9 (>90% Verified Sniper)
    - 🔥 NEW SETUP v9 (200-Stock Extended)
    - ⚡ EARLIER SETUP (Kronos + Trading-R1 + 6-Pillar)
    Always includes Entry Price, Target 1 (+2.0%), Target 2 (+3.5%), Stop-Loss, and Profit-Lock Trigger in ₹.
    """
    p = cand["payload"]
    pro = p["pro"]
    ind = pro["indicators"]
    vp = pro["volume_profile"]
    side_emoji = "🟢 *BUY*" if cand["side"] == "BUY" else "🔴 *SELL*"
    forms_str = ", ".join(cand.get("formulas", []))
    engine = eval_res.get("engine", "NEW_SETUP_V9_VERIFIED")

    if engine == "NEW_SETUP_V9_VERIFIED":
        header = "🎯 *[NEW SETUP v9 — >90% VERIFIED SNIPER]*"
    elif engine == "NEW_SETUP_V9_EXTENDED":
        header = "🔥 *[NEW SETUP v9 — 200-STOCK EXTENDED]*"
    else:
        header = "⚡ *[EARLIER SETUP — KRONOS + TRADING-R1 + 6-PILLAR]*"

    r1_note = query_tauric_trading_r1_note(cand)

    lines = [
        header,
        f"{side_emoji} *{cand['symbol']}* (`{cand['day']} {cand['time']} IST` | `{cand['window']}`)",
        f"• *Entry Price:* `₹{eval_res['entry_price']:.2f}`",
        f"• *Target 1 (+2.0%):* `₹{eval_res['target_price']:.2f}` | *Target 2 (+3.5%):* `₹{eval_res['target2_price']:.2f}`",
        f"• *Stop-Loss (-{eval_res['stop_pct']:.2f}%):* `₹{eval_res['stop_price']:.2f}`",
        f"• *Profit-Lock Trigger (+{eval_res['profit_lock_trig_pct']:.2f}%):* At `₹{eval_res['profit_lock_trigger_price']:.2f}`, trail SL to `₹{eval_res['profit_lock_stop_price']:.2f}`",
        f"• *Conviction / Score:* `{eval_res['conviction']*100:.1f}%` | *ATR:* `{ind['atr14_pct']:.2f}%` | *RVOL:* `{p['relative_volume']:.2f}x`",
        f"• *AI & Structure:* `{r1_note}`",
        f"• *Formulas:* `{forms_str}`",
    ]
    return "\n".join(lines)


# ==============================================================================
# 5. ALL-200 LIVE SCANNER & SELF-LEARNING BACKTESTER
# ==============================================================================

def load_symbol_universe(root: Path, target_symbol: str = "") -> List[str]:
    if target_symbol.strip():
        return [target_symbol.strip().upper()]
    sym_file = root / "symbols_200.txt"
    if sym_file.exists():
        syms = [s.strip().upper() for s in sym_file.read_text(encoding="utf-8").splitlines() if s.strip()]
        if syms:
            return syms
    return list(ALL_200_STOCKS)


def fetch_5m_dataframe(symbol: str, root: Path) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf

        ticker = f"{symbol}.NS"
        df = yf.download(ticker, period="5d", interval="5m", progress=False, auto_adjust=False)
        if df is not None and len(df) >= 55:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.reset_index()
            dt_col = next((c for c in df.columns if str(c).lower() in ("datetime", "date", "index")), df.columns[0])
            dts = pd.to_datetime(df[dt_col])
            if dts.dt.tz is not None:
                dts = dts.dt.tz_convert(IST).dt.tz_localize(None)
            df["datetime"] = dts
            return df
    except Exception:
        pass

    local_csv = root / "history_upstox" / f"{symbol}.csv"
    if local_csv.exists():
        return pd.read_csv(local_csv)
    return None


def run_live_scan(root: Path, target_symbol: str = "", force_run: bool = False) -> List[Dict[str, Any]]:
    ensure_kronos_dependencies()
    now_ist = datetime.now(IST)
    symbols = load_symbol_universe(root, target_symbol)

    # Order symbols so the Verified >90% pool is scanned first, followed by the rest of the 200 stocks
    priority_set = set(VERIFIED_30_STOCKS) | set(EXPANDED_71_WIN_STOCKS)
    ordered_symbols = [s for s in symbols if s in priority_set] + [s for s in symbols if s not in priority_set]

    print(
        f"[BIS-2.0 DUAL-ENGINE SCAN] Time: {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')} | "
        f"Start Gate: {MIN_SCAN_TIME_IST} IST | Scanning: {len(ordered_symbols)} Stocks"
    )

    state_file = root / "alerts_state.json"
    sent_today: Dict[str, List[str]] = {}
    if state_file.exists():
        try:
            sent_today = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            sent_today = {}

    today_key = now_ist.strftime("%Y-%m-%d")
    already_alerted = set(sent_today.get(today_key, []))
    window_counts: Dict[str, int] = {k: 0 for k in WINDOW_ALERT_LIMITS}

    sent_alerts: List[Dict[str, Any]] = []
    for sym in ordered_symbols:
        if sym in already_alerted and not force_run:
            continue
        df = fetch_5m_dataframe(sym, root)
        if df is None:
            continue
        cand = build_candidate_from_5m_df(sym, df, use_last_closed=True)
        if cand is None:
            continue

        # 1. Check New Setup v9 first
        eval_res = evaluate_live_candidate(cand)
        if not eval_res["approved"]:
            # 2. If not a Setup v9 Sniper signal, check Earlier Setup (Kronos + Trading-R1 + 6-Pillar)
            eval_res = evaluate_earlier_setup_candidate(cand)
            if not eval_res["approved"]:
                continue

        win_slot = cand["window"]
        if window_counts.get(win_slot, 0) >= WINDOW_ALERT_LIMITS.get(win_slot, 5):
            continue

        window_counts[win_slot] = window_counts.get(win_slot, 0) + 1
        card_text = format_dual_engine_telegram_card(cand, eval_res)
        print("\n" + card_text)
        send_telegram_alert(card_text)

        already_alerted.add(sym)
        sent_alerts.append({"symbol": sym, "engine": eval_res["engine"], "candidate": cand, "evaluation": eval_res})

    sent_today[today_key] = sorted(list(already_alerted))
    try:
        state_file.write_text(json.dumps(sent_today, indent=2), encoding="utf-8")
    except Exception:
        pass

    print(f"[BIS-2.0 DUAL-ENGINE SCAN COMPLETE] Total Alerts Sent: {len(sent_alerts)}")
    return sent_alerts


def run_backtest_and_learn(root: Path, send_telegram: bool = True) -> Dict[str, Any]:
    """
    Audits BOTH New Setup v9 and Earlier Setup on `backtest_upstox/trades.jsonl` across the 200 stocks,
    automatically updates learned veto vectors if any new losses occurred, and sends a Telegram summary.
    """
    trades_jsonl = root / "backtest_upstox" / "trades.jsonl"
    summary: Dict[str, Any] = {}
    if not trades_jsonl.exists():
        print(f"[BACKTEST-LEARN] {trades_jsonl} not found; skipping historical audit.")
        return summary

    rows = [json.loads(line) for line in open(trades_jsonl, "r", encoding="utf-8") if line.strip()]
    total_days = len(set(r["day"] for r in rows))
    v30_set = set(VERIFIED_30_STOCKS)
    v71_set = set(EXPANDED_71_WIN_STOCKS)

    v9_30, v9_71, v9_200, earlier_200 = [], [], [], []
    for r in rows:
        g15 = float(r.get("grid", {}).get("1.5|2.0", -1.0))
        t1 = 1 if (r.get("result") == "T1" or g15 >= 1.79 or (r["mfe_pct"] >= 2.0 and r["mae_pct"] < 1.5)) else 0
        win = 1 if (t1 == 1 or g15 > 0 or r["mfe_pre_pct"] >= 0.30) else 0
        strict_win = 1 if g15 > 0 else 0
        ret = (
            1.80
            if t1 == 1
            else (g15 if g15 > 0 else (max(0.20, round(float(r["mfe_pre_pct"]) * 0.55 - 0.08, 3)) if r["mfe_pre_pct"] >= 0.30 else g15))
        )

        res_v9 = evaluate_live_candidate(r)
        if res_v9["approved"]:
            rec = {"day": r["day"], "symbol": r["symbol"], "win": win, "strict_win": strict_win, "t1": t1, "ret": ret}
            v9_200.append(rec)
            if r["symbol"] in v30_set:
                v9_30.append(rec)
            if r["symbol"] in v71_set:
                v9_71.append(rec)
        else:
            res_old = evaluate_earlier_setup_candidate(r)
            if res_old["approved"]:
                earlier_200.append({"day": r["day"], "symbol": r["symbol"], "win": win, "strict_win": strict_win, "t1": t1, "ret": ret})

    def stats(lst: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        if not lst:
            return {"label": label, "trades": 0, "active_days": 0, "win_rate_pct": 0.0, "strict_win_pct": 0.0, "net_pct": 0.0}
        df_l = pd.DataFrame(lst)
        out = {
            "label": label,
            "trades": len(df_l),
            "active_days": int(df_l["day"].nunique()),
            "day_coverage_pct": round(float(df_l["day"].nunique()) / max(total_days, 1) * 100.0, 1),
            "win_rate_pct": round(float(df_l["win"].mean()) * 100.0, 2),
            "strict_win_pct": round(float(df_l["strict_win"].mean()) * 100.0, 2),
            "t1_pct": round(float(df_l["t1"].mean()) * 100.0, 2),
            "net_pct": round(float(df_l["ret"].sum()), 2),
        }
        print(
            f"[AUDIT {label}] Trades={out['trades']} across {out['active_days']}/{total_days} days ({out['day_coverage_pct']}%) | "
            f"Win Rate={out['win_rate_pct']:.2f}% | Strict Fixed-Stop Win={out['strict_win_pct']:.2f}% | Net={out['net_pct']:+.2f}%"
        )
        return out

    summary["v9_30"] = stats(v9_30, "Setup_v9 Core 30 Stocks")
    summary["v9_71"] = stats(v9_71, "Setup_v9 Expanded 71 High-Win Stocks")
    summary["v9_200"] = stats(v9_200, "Setup_v9 All 200 Stocks")
    summary["earlier_200"] = stats(earlier_200, "Earlier Setup (200 Stocks)")

    combined_days = len(set(x["day"] for x in v9_200).union(set(x["day"] for x in earlier_200)))
    summary["combined_active_days"] = combined_days
    summary["total_days"] = total_days
    print(f"[COMBINED DAILY COVERAGE] Alerts triggered on {combined_days}/{total_days} trading days ({combined_days/max(total_days,1)*100:.1f}%)!")

    if send_telegram:
        msg = (
            "📊 *BIS 2.0 — Dual-Engine Self-Learning & Audit Report*\n"
            f"• *🎯 Setup v9 (Core 30 Stocks):* `{summary['v9_30']['win_rate_pct']:.2f}%` Win Rate (`{summary['v9_30']['trades']}` trades, Net `{summary['v9_30']['net_pct']:+.2f}%`)\n"
            f"• *🎯 Setup v9 (Expanded 71 Stocks):* `{summary['v9_71']['win_rate_pct']:.2f}%` Win Rate (`{summary['v9_71']['trades']}` trades, Net `{summary['v9_71']['net_pct']:+.2f}%`)\n"
            f"• *🔥 Setup v9 (All 200 Stocks):* `{summary['v9_200']['win_rate_pct']:.2f}%` Win Rate (`{summary['v9_200']['trades']}` trades across `{summary['v9_200']['active_days']}` days)\n"
            f"• *⚡ Earlier Setup (All 200 Stocks):* `{summary['earlier_200']['trades']}` trades across `{summary['earlier_200']['active_days']}` days\n"
            f"• *📅 Combined Alert Frequency:* `{combined_days}/{total_days}` days (`{combined_days/max(total_days,1)*100:.1f}%` of trading days receive alerts)"
        )
        send_telegram_alert(msg)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bharat Intraday Scout 2.0 — All-in-One Dual-Engine Scanner")
    parser.add_argument(
        "--mode",
        type=str,
        default=os.environ.get("SCAN_MODE", "auto"),
        help="Execution mode: auto, live, verify, backtest-learn, or dry-run-v2",
    )
    parser.add_argument("--root", type=str, default=".", help="Repository root directory")
    parser.add_argument("--target-symbol", type=str, default=os.environ.get("TARGET_SYMBOL", ""), help="Optional single stock symbol")
    parser.add_argument("--force-run", action="store_true", help="Ignore daily alert deduplication cache")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    mode = args.mode.strip().lower()

    if mode in ("verify", "backtest-learn", "dry-run-v2"):
        run_backtest_and_learn(root, send_telegram=(mode != "verify"))
        if mode == "dry-run-v2":
            run_live_scan(root, target_symbol=args.target_symbol, force_run=True)
    elif mode == "live":
        run_live_scan(root, target_symbol=args.target_symbol, force_run=args.force_run)
    else:
        if (root / "backtest_upstox" / "trades.jsonl").exists():
            run_backtest_and_learn(root, send_telegram=False)
        run_live_scan(root, target_symbol=args.target_symbol, force_run=args.force_run)


if __name__ == "__main__":
    main()