"""Advanced chart features for the learning journal: volume profile, candlestick
patterns, chart patterns and extra indicators.

Everything here is a *rule-based, shape-only heuristic* computed from completed
5-minute bars. Results are LOGGED with each signal and judged later by the
journal on real paper outcomes; nothing here is assumed to predict price.
Any failure returns {} and never blocks a scan.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

MIN_BARS = 40
BULL_CANDLES = {"bullish_engulfing", "hammer", "morning_star", "marubozu_bull"}
BEAR_CANDLES = {"bearish_engulfing", "shooting_star", "evening_star", "marubozu_bear"}
BULL_PATTERNS = {"double_bottom_breakout", "ascending_triangle", "uptrend_structure", "range_breakout_up",
                 "inverse_head_shoulders_breakout"}
BEAR_PATTERNS = {"double_top_breakdown", "descending_triangle", "downtrend_structure", "range_breakdown",
                 "head_shoulders_breakdown"}


def _f(x) -> Optional[float]:
    try:
        x = float(x)
        return round(x, 4) if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _prep(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) == 0:
        return None
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    need = ["open", "high", "low", "close", "volume"]
    if any(c not in out.columns for c in need):
        return None
    out = out[need].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return out if len(out) >= MIN_BARS else None


# ---------------------------------------------------------------- volume profile
def volume_profile(df: pd.DataFrame, sessions: int = 3, bins: int = 40, value_area: float = 0.70) -> Dict[str, Any]:
    """POC / value area (VAL..VAH) over the last `sessions` sessions; volume of each bar
    is spread evenly across the price bins its high-low range touches."""
    days = sorted(set(df.index.date))[-sessions:]
    d = df[np.isin(df.index.date, days)]
    lo, hi = float(d["low"].min()), float(d["high"].max())
    if not hi > lo:
        return {}
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)
    for h, l, v in zip(d["high"].values, d["low"].values, d["volume"].values):
        i0 = int(np.clip(np.searchsorted(edges, l, side="right") - 1, 0, bins - 1))
        i1 = int(np.clip(np.searchsorted(edges, h, side="right") - 1, 0, bins - 1))
        vol[i0:i1 + 1] += v / (i1 - i0 + 1)
    total = vol.sum()
    if total <= 0:
        return {}
    poc_i = int(vol.argmax())
    lo_i = hi_i = poc_i
    acc = vol[poc_i]
    while acc < value_area * total and (lo_i > 0 or hi_i < bins - 1):
        down = vol[lo_i - 1] if lo_i > 0 else -1.0
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1.0
        if up >= down:
            hi_i += 1
            acc += vol[hi_i]
        else:
            lo_i -= 1
            acc += vol[lo_i]
    poc = float((edges[poc_i] + edges[poc_i + 1]) / 2)
    val, vah = float(edges[lo_i]), float(edges[hi_i + 1])
    price = float(df["close"].iloc[-1])
    pos = "ABOVE_VALUE" if price > vah else "BELOW_VALUE" if price < val else "INSIDE_VALUE"
    return {"poc": _f(poc), "vah": _f(vah), "val": _f(val), "price_vs_value": pos,
            "poc_distance_pct": _f((price / poc - 1.0) * 100.0)}


# ---------------------------------------------------------------- candlesticks
def candlestick_patterns(df: pd.DataFrame) -> List[str]:
    """Shape-only candle patterns on the last 1-3 completed bars (no prior-trend filter)."""
    t = df.tail(3)
    o, h, l, c = (t[k].values for k in ("open", "high", "low", "close"))
    n = len(t)
    found: List[str] = []

    def parts(i):
        body, rng = abs(c[i] - o[i]), h[i] - l[i]
        return body, rng, h[i] - max(o[i], c[i]), min(o[i], c[i]) - l[i]

    body, rng, upper, lower = parts(n - 1)
    if rng <= 0:
        return found
    if body <= 0.10 * rng:
        found.append("doji")
    if body >= 0.90 * rng:
        found.append("marubozu_bull" if c[-1] > o[-1] else "marubozu_bear")
    if lower >= 0.60 * rng and body <= 0.30 * rng and upper <= 0.15 * rng:
        found.append("hammer")
    if upper >= 0.60 * rng and body <= 0.30 * rng and lower <= 0.15 * rng:
        found.append("shooting_star")
    if n >= 2:
        pb, prng, _, _ = parts(n - 2)
        if prng > 0 and pb > 0:
            if c[-2] < o[-2] and c[-1] > o[-1] and o[-1] <= c[-2] and c[-1] >= o[-2] and body > pb:
                found.append("bullish_engulfing")
            if c[-2] > o[-2] and c[-1] < o[-1] and o[-1] >= c[-2] and c[-1] <= o[-2] and body > pb:
                found.append("bearish_engulfing")
        if h[-1] < h[-2] and l[-1] > l[-2]:
            found.append("inside_bar")
    if n == 3:
        b1, r1, _, _ = parts(0)
        b2, _, _, _ = parts(1)
        if r1 > 0 and b1 >= 0.5 * r1 and b2 <= 0.3 * b1:
            mid1 = (o[0] + c[0]) / 2.0
            if c[0] < o[0] and c[2] > o[2] and c[2] > mid1:
                found.append("morning_star")
            if c[0] > o[0] and c[2] < o[2] and c[2] < mid1:
                found.append("evening_star")
    return found


def _bias(names, bull, bear) -> str:
    b, s = len(set(names) & bull), len(set(names) & bear)
    return "BULL" if b > s else "BEAR" if s > b else "NEUTRAL"


# ---------------------------------------------------------------- chart patterns
def _swings(h: np.ndarray, l: np.ndarray, w: int = 3):
    highs, lows = [], []
    for i in range(w, len(h) - w):
        if h[i] >= h[i - w:i + w + 1].max() and h[i] > h[i - 1]:
            highs.append(i)
        if l[i] <= l[i - w:i + w + 1].min() and l[i] < l[i - 1]:
            lows.append(i)
    return highs, lows


def chart_patterns(df: pd.DataFrame) -> List[str]:
    """Heuristic patterns on the last ~80 bars using swing pivots (window 3)."""
    d = df.tail(80)
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    price = float(c[-1])
    out: List[str] = []
    hi_i, lo_i = _swings(h, l)
    if len(hi_i) >= 2:
        i1, i2 = hi_i[-2], hi_i[-1]
        if i2 - i1 >= 6 and abs(h[i2] - h[i1]) / h[i1] <= 0.003:
            trough = float(l[i1:i2 + 1].min())
            top = min(h[i1], h[i2])
            if (top - trough) / top >= 0.004:
                out.append("double_top_breakdown" if price < trough else "double_top_forming")
    if len(lo_i) >= 2:
        i1, i2 = lo_i[-2], lo_i[-1]
        if i2 - i1 >= 6 and abs(l[i2] - l[i1]) / l[i1] <= 0.003:
            peak = float(h[i1:i2 + 1].max())
            bottom = max(l[i1], l[i2])
            if (peak - bottom) / peak >= 0.004:
                out.append("double_bottom_breakout" if price > peak else "double_bottom_forming")
    if len(hi_i) >= 3:
        a, b, e = hi_i[-3], hi_i[-2], hi_i[-1]
        ls, hd, rs = h[a], h[b], h[e]
        if (b - a >= 4 and e - b >= 4 and hd > ls * 1.003 and hd > rs * 1.003
                and abs(ls - rs) / ((ls + rs) / 2) <= 0.004):
            neck = (float(l[a:b + 1].min()) + float(l[b:e + 1].min())) / 2.0
            if price < neck:
                out.append("head_shoulders_breakdown")
            elif price < rs:
                out.append("head_shoulders_forming")
    if len(lo_i) >= 3:
        a, b, e = lo_i[-3], lo_i[-2], lo_i[-1]
        ls, hd, rs = l[a], l[b], l[e]
        if (b - a >= 4 and e - b >= 4 and hd < ls * 0.997 and hd < rs * 0.997
                and abs(ls - rs) / ((ls + rs) / 2) <= 0.004):
            neck = (float(h[a:b + 1].max()) + float(h[b:e + 1].max())) / 2.0
            if price > neck:
                out.append("inverse_head_shoulders_breakout")
            elif price > rs:
                out.append("inverse_head_shoulders_forming")
    if len(hi_i) >= 2 and len(lo_i) >= 2:
        h1, h2, l1, l2 = h[hi_i[-2]], h[hi_i[-1]], l[lo_i[-2]], l[lo_i[-1]]
        if abs(h2 - h1) / h1 <= 0.003 and l2 > l1 * 1.002:
            out.append("ascending_triangle")
        if abs(l2 - l1) / l1 <= 0.003 and h2 < h1 * 0.998:
            out.append("descending_triangle")
        if h2 > h1 and l2 > l1:
            out.append("uptrend_structure")
        if h2 < h1 and l2 < l1:
            out.append("downtrend_structure")
    if len(d) >= 22:
        if price > float(h[-21:-1].max()):
            out.append("range_breakout_up")
        if price < float(l[-21:-1].min()):
            out.append("range_breakdown")
    if len(d) >= 16:
        recent = float((h[-8:] - l[-8:]).mean())
        before = float((h[-16:-8] - l[-16:-8]).mean())
        if before > 0 and recent < 0.6 * before:
            out.append("tight_consolidation")
    above = [h[i] for i in hi_i if h[i] > price]
    below = [l[i] for i in lo_i if l[i] < price]
    if above and (min(above) / price - 1.0) <= 0.005:
        out.append("near_resistance")
    if below and (1.0 - max(below) / price) <= 0.005:
        out.append("near_support")
    return out


# ---------------------------------------------------------------- levels, anatomy, smart-money
def _atr(d: pd.DataFrame, n: int = 14) -> float:
    tr = pd.concat([d["high"] - d["low"], (d["high"] - d["close"].shift()).abs(),
                    (d["low"] - d["close"].shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def sr_zones(df: pd.DataFrame, tol: float = 0.0025, lookback: int = 150) -> Dict[str, Any]:
    """Support/resistance zones = clusters of swing highs and lows (more touches = stronger).
    `at_support` / `at_resistance` mean the latest bar is testing a zone with >= 2 touches."""
    d = df.tail(lookback)
    h, l = d["high"].values, d["low"].values
    hi_i, lo_i = _swings(h, l)
    pts = sorted([float(h[i]) for i in hi_i] + [float(l[i]) for i in lo_i])
    clusters: List[List[float]] = []
    for p in pts:
        if clusters and p <= clusters[-1][-1] * (1 + tol):
            clusters[-1].append(p)
        else:
            clusters.append([p])
    zones = [(sum(c) / len(c), len(c)) for c in clusters]
    px, lo_, hi_ = float(d["close"].iloc[-1]), float(d["low"].iloc[-1]), float(d["high"].iloc[-1])
    sup = [z for z in zones if z[0] <= px * (1 + tol)]
    res = [z for z in zones if z[0] >= px * (1 - tol)]
    s_zone = max(sup, key=lambda z: z[0]) if sup else None
    r_zone = min(res, key=lambda z: z[0]) if res else None
    at_s = any(n >= 2 and lv <= px * 1.003 and lo_ <= lv * 1.003 and px >= lv * 0.997 for lv, n in zones)
    at_r = any(n >= 2 and lv >= px * 0.997 and hi_ >= lv * 0.997 and px <= lv * 1.003 for lv, n in zones)
    return {"support": _f(s_zone[0]) if s_zone else None, "support_touches": s_zone[1] if s_zone else 0,
            "support_dist_pct": _f((px / s_zone[0] - 1) * 100) if s_zone else None,
            "resistance": _f(r_zone[0]) if r_zone else None, "resistance_touches": r_zone[1] if r_zone else 0,
            "resistance_dist_pct": _f((r_zone[0] / px - 1) * 100) if r_zone else None,
            "at_support": bool(at_s), "at_resistance": bool(at_r)}


def bar_anatomy(df: pd.DataFrame) -> Dict[str, Any]:
    """Body/wick structure of the latest completed bar (fractions of its high-low range)."""
    last = df.iloc[-1]
    rng = float(last["high"] - last["low"])
    if rng <= 0:
        return {}
    body = abs(float(last["close"] - last["open"]))
    upper = float(last["high"] - max(last["open"], last["close"])) / rng
    lower = float(min(last["open"], last["close"]) - last["low"]) / rng
    avg_body = float((df["close"] - df["open"]).abs().tail(20).mean())
    return {"body_ratio": _f(body / rng), "upper_wick_ratio": _f(upper), "lower_wick_ratio": _f(lower),
            "bullish": bool(last["close"] > last["open"]), "body_x_avg": _f(body / (avg_body + 1e-9)),
            "rejection": "LOWER_WICK" if lower >= 0.5 else "UPPER_WICK" if upper >= 0.5 else "NONE"}


def fair_value_gaps(df: pd.DataFrame, lookback: int = 40, min_pct: float = 0.0015) -> Dict[str, Any]:
    """3-candle imbalances not yet fully filled. A 'retest' = the latest bar trades back into a gap."""
    d = df.tail(lookback)
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    n, price = len(d), float(c[-1])
    bull, bear = [], []
    for i in range(2, n):
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) / price >= min_pct:
            lo_g, hi_g = float(h[i - 2]), float(l[i])
            if not any(l[j] <= lo_g for j in range(i + 1, n)):
                bull.append((lo_g, hi_g, i))
        if h[i] < l[i - 2] and (l[i - 2] - h[i]) / price >= min_pct:
            lo_g, hi_g = float(h[i]), float(l[i - 2])
            if not any(h[j] >= hi_g for j in range(i + 1, n)):
                bear.append((lo_g, hi_g, i))
    in_bull = any(i <= n - 2 and l[-1] <= hi_g and c[-1] >= lo_g for lo_g, hi_g, i in bull)
    in_bear = any(i <= n - 2 and h[-1] >= lo_g and c[-1] <= hi_g for lo_g, hi_g, i in bear)
    return {"in_bull_fvg": bool(in_bull), "in_bear_fvg": bool(in_bear),
            "open_bull_gaps": len(bull), "open_bear_gaps": len(bear)}


def order_blocks(df: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
    """Heuristic order blocks: the last opposite candle before a displacement (>= 1.5 ATR within 3 bars)
    that also breaks the prior 10-bar extreme. Flags a FIRST retest of an untouched, unbroken block."""
    d = df.tail(lookback)
    if len(d) < 25:
        return {}
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    n, atr = len(d), _atr(d)
    if not atr > 0:
        return {}
    at_bull = at_bear = False
    for i in range(10, n - 4):
        if c[i] < o[i]:  # candidate bullish block
            if (h[i + 1:i + 4].max() - h[i] >= 1.5 * atr) and c[i + 1:i + 4].max() > h[i - 10:i].max():
                lo_z, hi_z = float(l[i]), float(h[i])
                broken = any(c[j] < lo_z for j in range(i + 1, n))
                touched = any(l[j] <= hi_z for j in range(i + 4, n - 1))
                if not broken and not touched and l[-1] <= hi_z * 1.001 and c[-1] >= lo_z:
                    at_bull = True
        if c[i] > o[i]:  # candidate bearish block
            if (l[i] - l[i + 1:i + 4].min() >= 1.5 * atr) and c[i + 1:i + 4].min() < l[i - 10:i].min():
                lo_z, hi_z = float(l[i]), float(h[i])
                broken = any(c[j] > hi_z for j in range(i + 1, n))
                touched = any(h[j] >= lo_z for j in range(i + 4, n - 1))
                if not broken and not touched and h[-1] >= lo_z * 0.999 and c[-1] <= hi_z:
                    at_bear = True
    return {"at_bull_ob": bool(at_bull), "at_bear_ob": bool(at_bear)}


# ---------------------------------------------------------------- indicators
def extra_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    stoch = (c - ll) / (hh - ll).replace(0, np.nan) * 100.0
    tp = (h + l + c) / 3.0
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - tp.rolling(20).mean()) / (0.015 * mad.replace(0, np.nan))
    obv = (np.sign(c.diff()).fillna(0.0) * v).cumsum()
    seg = obv.tail(20).values
    obv_slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0] / (v.tail(20).mean() + 1e-9))
    e9, e21, e50 = (c.ewm(span=s, adjust=False).mean().iloc[-1] for s in (9, 21, 50))
    px = float(c.iloc[-1])
    stack = "BULL_STACK" if px > e9 > e21 > e50 else "BEAR_STACK" if px < e9 < e21 < e50 else "MIXED"
    sma, sd = c.rolling(20).mean().iloc[-1], c.rolling(20).std().iloc[-1]
    pct_b = (px - (sma - 2 * sd)) / (4 * sd) if sd and sd > 0 else None
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    days = sorted(set(df.index.date))
    today = df[df.index.date == days[-1]]
    ttp = (today["high"] + today["low"] + today["close"]) / 3.0
    vwap = float((ttp * today["volume"]).sum() / max(today["volume"].sum(), 1e-9))
    prev_break, gap = "UNKNOWN", None
    if len(days) >= 2:
        prev = df[df.index.date == days[-2]]
        pdh, pdl = float(prev["high"].max()), float(prev["low"].min())
        prev_break = "ABOVE_PDH" if px > pdh else "BELOW_PDL" if px < pdl else "INSIDE_PDR"
        gap = (float(today["open"].iloc[0]) / float(prev["close"].iloc[-1]) - 1.0) * 100.0
    return {"stoch_k": _f(stoch.iloc[-1]), "williams_r": _f(stoch.iloc[-1] - 100.0),
            "cci20": _f(cci.iloc[-1]), "obv_slope_norm": _f(obv_slope), "ema_stack": stack,
            "roc10_pct": _f((px / c.iloc[-11] - 1.0) * 100.0) if len(c) > 11 else None,
            "bb_percent_b": _f(pct_b), "atr14_pct": _f(tr.rolling(14).mean().iloc[-1] / px * 100.0),
            "price_vs_vwap_pct": _f((px / vwap - 1.0) * 100.0), "prev_day_break": prev_break,
            "gap_pct": _f(gap)}


# ---------------------------------------------------------------- combine
def compute(df: pd.DataFrame, side: str) -> Dict[str, Any]:
    """All advanced features for one candidate, plus side-aware confirmations."""
    try:
        d = _prep(df)
        if d is None:
            return {}
        sign = 1 if side == "BUY" else -1
        vp = volume_profile(d)
        candles = candlestick_patterns(d)
        patterns = chart_patterns(d)
        ind = extra_indicators(d)
        cb = _bias(candles, BULL_CANDLES, BEAR_CANDLES)
        pb = _bias(patterns, BULL_PATTERNS, BEAR_PATTERNS)
        want = "BULL" if sign > 0 else "BEAR"
        opp = "BEAR" if sign > 0 else "BULL"
        pos, pd_pct = vp.get("price_vs_value"), vp.get("poc_distance_pct") or 0.0
        vp_ok = (pos == ("ABOVE_VALUE" if sign > 0 else "BELOW_VALUE")) or (pos == "INSIDE_VALUE" and sign * pd_pct > 0)
        vp_bad = pos == ("BELOW_VALUE" if sign > 0 else "ABOVE_VALUE")
        stack_ok = ind["ema_stack"] == want + "_STACK"
        stack_bad = ind["ema_stack"] == opp + "_STACK"
        obv, vw = ind.get("obv_slope_norm"), ind.get("price_vs_vwap_pct")
        flow_ok = obv is not None and vw is not None and sign * obv > 0 and sign * vw > 0
        flow_bad = obv is not None and vw is not None and sign * obv < 0 and sign * vw < 0
        wall = "near_resistance" if sign > 0 else "near_support"
        levels, anatomy = sr_zones(d), bar_anatomy(d)
        fvg, ob = fair_value_gaps(d), order_blocks(d)
        lvl_ok = levels.get("at_support") if sign > 0 else levels.get("at_resistance")
        lvl_bad = levels.get("at_resistance") if sign > 0 else levels.get("at_support")
        sm_ok = (fvg.get("in_bull_fvg") or ob.get("at_bull_ob")) if sign > 0 else (fvg.get("in_bear_fvg") or ob.get("at_bear_ob"))
        sm_bad = (fvg.get("in_bear_fvg") or ob.get("at_bear_ob")) if sign > 0 else (fvg.get("in_bull_fvg") or ob.get("at_bull_ob"))
        confirms = {"volume_profile": bool(vp_ok), "candles": cb == want, "chart_patterns": pb == want,
                    "trend_stack": bool(stack_ok), "flow": bool(flow_ok),
                    "levels": bool(lvl_ok), "smart_money": bool(sm_ok)}
        conflicts = {"volume_profile": bool(vp_bad), "candles": cb == opp,
                     "chart_patterns": pb == opp or wall in patterns,
                     "trend_stack": bool(stack_bad), "flow": bool(flow_bad),
                     "levels": bool(lvl_bad), "smart_money": bool(sm_bad)}
        research = {}
        for name in ("renko", "smc", "cycles", "gann"):
            try:
                module = __import__("features_" + name)
                research[name] = module.compute(d, side)
            except Exception:
                research[name] = {}
        return {**research, "volume_profile": vp, "candles": candles, "candle_bias": cb,
                "chart_patterns": patterns, "pattern_bias": pb, "indicators": ind,
                "levels": levels, "anatomy": anatomy, "fvg": fvg, "order_blocks": ob,
                "confirms": confirms, "conflicts": conflicts, "groups": len(confirms),
                "confluence": sum(confirms.values()), "conflict_count": sum(conflicts.values())}
    except Exception:
        return {}
