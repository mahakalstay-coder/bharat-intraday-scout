#!/usr/bin/env python3
"""Replay the LIVE signal engine over months of 5-minute history and report what works.

    python backtest_history.py --data ./history --out ./backtest_out [--workers 4] [--max-symbols 50]

Input : a folder with one CSV/Parquet per symbol (file name = symbol), or one CSV with a `symbol` column.
        Columns (any case): datetime|date|timestamp, open, high, low, close, volume. 1-minute data is resampled to 5m.
        Naive timestamps are assumed to be IST.
Output: trades.jsonl, trades.csv, report.json, report.md

Fidelity: calls the real `scanner.evaluate_symbol` with the same 5-minute anchor, a trailing ~8-session window, tier-2 rules after
09:45 and one signal per symbol per day; grades with the real `scanner.resolve_outcome` (stop first if both touched) and the real
cost model. Baseline stop/target levels are used (learned filters are switched off) so results describe the original strategy.
NOT replayed: Kronos, news, the sector/peer alert gate and Nifty/VIX regime (no historical inputs). Daily metadata is computed from
sessions BEFORE each day only (no look-ahead).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TZ = ZoneInfo("Asia/Kolkata")
WARMUP_SESSIONS = 25
STOPS_PCT = (0.8, 1.0, 1.25, 1.5)
TARGETS_PCT = (1.5, 2.0, 2.5, 3.0, 4.0)
DT_COLS = ("datetime", "date_time", "timestamp", "date", "time")


# ------------------------------------------------------------------ loading
def standardise(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    cols = {str(c).lower().strip(): c for c in raw.columns}
    tcol = next((cols[c] for c in DT_COLS if c in cols), None)
    if tcol is None or any(k not in cols for k in ("open", "high", "low", "close", "volume")):
        return None
    t = raw[tcol]
    if pd.api.types.is_numeric_dtype(t):
        mag = abs(float(t.dropna().iloc[0]))
        unit = "s" if mag < 1e11 else "ms" if mag < 1e14 else "us" if mag < 1e17 else "ns"
        idx = pd.to_datetime(t, unit=unit, utc=True, errors="coerce").dt.tz_convert(TZ)
    else:
        idx = pd.to_datetime(t, errors="coerce")
        idx = idx.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward") if idx.dt.tz is None else idx.dt.tz_convert(TZ)
    df = pd.DataFrame({k: pd.to_numeric(raw[cols[k]], errors="coerce") for k in ("open", "high", "low", "close", "volume")})
    df.index = pd.DatetimeIndex(idx)
    df = df[df.index.notna()].dropna(subset=["open", "high", "low", "close"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if len(df) > 3:
        step = df.index.to_series().diff().dropna().median()
        if step < pd.Timedelta(minutes=5):
            df = df.resample("5min", label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
    df["volume"] = df["volume"].fillna(0.0)
    return df.between_time("09:15", "15:25")


def _name(stem: str) -> str:
    s = stem.upper().replace(".NS", "")
    for suf in ("_5M", "-5M", "_5MIN", "_INTRADAY"):
        s = s[:-len(suf)] if s.endswith(suf) else s
    return s


def load_history(path: str) -> Dict[str, pd.DataFrame]:
    p, out = Path(path), {}
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix.lower() in (".csv", ".parquet"))
    for f in files:
        raw = pd.read_parquet(f) if f.suffix.lower() == ".parquet" else pd.read_csv(f)
        sym_col = next((c for c in raw.columns if str(c).lower() in ("symbol", "ticker")), None)
        parts = [(_name(str(k)), g) for k, g in raw.groupby(sym_col)] if sym_col else [(_name(f.stem), raw)]
        for sym, g in parts:
            df = standardise(g)
            if df is not None and len(set(df.index.date)) >= WARMUP_SESSIONS + 5:
                out[sym] = df
    return out


# ------------------------------------------------------------------ replay
def daily_meta(df: pd.DataFrame, sc) -> Dict[Any, Dict[str, float]]:
    """As-of metadata per session using ONLY earlier sessions."""
    g = df.groupby(df.index.date)
    d = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(), "volume": g["volume"].sum()})
    d["turn"] = d["close"] * d["volume"]
    meta = {}
    for i, day in enumerate(d.index):
        if i < WARMUP_SESSIONS:
            continue
        prev = d.iloc[:i]
        c = prev["close"].values
        meta[day] = {"price": float(c[-1]), "previous_close": float(c[-1]),
                     "avg_turnover": float(prev["turn"].tail(sc.CFG["LIQUIDITY_LOOKBACK_DAYS"]).mean()),
                     "daily_ema20": float(sc.ema(c, 20)),
                     "daily_ema50": float(sc.ema(c, 50)) if len(c) >= 50 else float(sc.ema(c, 20)),
                     "avg_daily_range_pct": float(((prev["high"] - prev["low"]) / prev["close"].clip(lower=1e-6) * 100.0).tail(20).mean())}
    return meta


class FastVolume:
    """Exact, fast re-implementation of scanner.same_slot_relative_volume and
    scanner.session_cumulative_relative_volume for the trailing-window replay (verified equal in tests).
    Lets the replay skip bars the volume gate would reject anyway."""

    def __init__(self, df: pd.DataFrame, days: List[Any], bounds: Dict[Any, Any], window_sessions: int = 8):
        self.vol = df["volume"].values.astype(float)
        self.days, self.bounds, self.w = days, bounds, window_sessions
        self.minute = np.array([t.hour * 60 + t.minute for t in df.index])
        self.k = {d: i for i, d in enumerate(days)}
        self.cum = np.cumsum(self.vol)

    def _prior_sessions(self, k: int) -> List[int]:
        return list(range(max(0, k - self.w), k))

    def values(self, day, pos: int, min_sessions: int = 3):
        k, m = self.k[day], self.minute[pos]
        prior = self._prior_sessions(k)
        # same-slot: last 5 prior sessions in the window that HAVE this slot (tail(5) before the >0 filter)
        found = []
        for j in reversed(prior):
            s, e = self.bounds[self.days[j]]
            hit = np.nonzero(self.minute[s:e] == m)[0]
            if len(hit):
                found.append(self.vol[s + hit[-1]])
            if len(found) == 5:
                break
        vals = [v for v in found if v > 0]
        relvol = self.vol[pos] / float(np.mean(vals)) if len(vals) >= min_sessions and float(np.mean(vals)) > 0 else 0.0
        # cumulative: today up to this bar vs the last 5 prior sessions up to the same minute
        s0, _ = self.bounds[day]
        current = float(self.vol[s0:pos + 1].sum())
        totals = []
        for j in prior[-5:]:
            s, e = self.bounds[self.days[j]]
            sel = self.minute[s:e] <= m
            if sel.any():
                tot = float(self.vol[s:e][sel].sum())
                if tot > 0:
                    totals.append(tot)
        cumrel = current / float(np.mean(totals)) if len(totals) >= min_sessions and float(np.mean(totals)) > 0 else 0.0
        return relvol, cumrel


def volume_gate(cfg: Dict[str, Any], relvol: float, cumrel: float) -> bool:
    return bool((relvol >= cfg["MIN_RELATIVE_VOLUME"] and cumrel >= cfg["MIN_CUMULATIVE_RVOL"])
                or (relvol >= 2.00 and cumrel >= 1.05) or (cumrel >= 1.75 and relvol >= 0.95))


def grade_grid(side: str, entry: float, after: pd.DataFrame, cost_pct: float) -> Dict[str, float]:
    """Net % per trade for fixed stop%/target% pairs (stop wins ties; end-of-day exit if neither is hit)."""
    sign = 1.0 if side == "BUY" else -1.0
    hi, lo = after["high"].values.astype(float), after["low"].values.astype(float)
    fav = ((hi / entry - 1.0) if sign > 0 else (1.0 - lo / entry)) * 100.0
    adv = ((1.0 - lo / entry) if sign > 0 else (hi / entry - 1.0)) * 100.0
    final = sign * (float(after["close"].values[-1]) / entry - 1.0) * 100.0
    res = {}
    for s in STOPS_PCT:
        si = np.argmax(adv >= s) if (adv >= s).any() else None
        for t in TARGETS_PCT:
            ti = np.argmax(fav >= t) if (fav >= t).any() else None
            pct = final if (si is None and ti is None) else -s if ti is None else t if si is None else (-s if si <= ti else t)
            res[f"{s}|{t}"] = round(pct - cost_pct, 4)
    return res


def mfe_before_stop(side: str, entry: float, stop: float, after: pd.DataFrame) -> float:
    """Largest favourable move (%) reached strictly before the stop is touched (uncapped by any target)."""
    hi, lo = after["high"].values.astype(float), after["low"].values.astype(float)
    fav = ((hi / entry - 1.0) if side == "BUY" else (1.0 - lo / entry)) * 100.0
    adv = ((1.0 - lo / entry) if side == "BUY" else (hi / entry - 1.0)) * 100.0
    hit = np.nonzero(adv >= abs(entry - stop) / entry * 100.0)[0]
    end = int(hit[0]) if len(hit) else len(fav)
    return float(fav[:end].max()) if end > 0 else 0.0


def replay_symbol(job) -> Dict[str, Any]:
    symbol, df, opts = job
    import scanner as sc
    import features_pro
    import mirror_study
    sc.load_filters = lambda: {}          # baseline levels: original strategy, not live-learned tweaks
    meta = daily_meta(df, sc)
    days = sorted(set(df.index.date))
    dates = np.array([t.date() for t in df.index])
    bounds = {d: (int(np.argmax(dates == d)), int(len(dates) - np.argmax(dates[::-1] == d))) for d in days}
    fv = FastVolume(df, days, bounds)
    cost_bps = float(opts["cost_bps"])
    os.environ["PAPER_ROUND_TRIP_BPS"] = str(cost_bps)
    trades, ungradable, scans = [], 0, 0
    for di, day in enumerate(days):
        m = meta.get(day)
        if not m or m["avg_turnover"] < sc.CFG["MIN_AVG_TURNOVER_INR"] or m["price"] <= sc.CFG["PRICE_FLOOR"]:
            continue
        s, e = bounds[day]
        day_open, prev_close = float(df["open"].iloc[s]), m["previous_close"]
        if abs(day_open / prev_close - 1.0) > 0.20:      # probable split/bonus/data error
            continue
        win_start = bounds[days[max(0, di - 8)]][0]
        for pos in range(s, e):
            ts = df.index[pos]
            if ts.time() > dtime(15, 15) or (pos - s) % int(opts["scan_every"] // 5 or 1):
                continue
            px = float(df["close"].iloc[pos])
            om, pm = (px / day_open - 1.0) * 100.0, (px / prev_close - 1.0) * 100.0
            eff = pm if abs(pm) >= abs(om) else om
            if not (sc.CFG["MIN_DAY_MOVE_PCT"] <= abs(eff) <= sc.CFG["MAX_DAY_MOVE_PCT"]):
                continue
            if not opts.get("exact") and not volume_gate(sc.CFG, *fv.values(day, pos)):
                continue                              # the live volume gate would reject this bar too
            prefix, anchor = df.iloc[win_start:pos + 1], ts.to_pydatetime() + timedelta(minutes=5)
            scans += 1
            sig = sc.evaluate_symbol(symbol, prefix, m, anchor, pre_normalized=True, allow_tier2=False)
            if sig is None and sc.is_tier2_window(anchor):
                sig = sc.evaluate_symbol(symbol, prefix, m, anchor, pre_normalized=True, allow_tier2=True)
            if sig is None:
                continue
            sig = {**sig, "shadow_only": True}
            sig["pro"] = features_pro.compute(prefix, sig["side"])
            res = sc.resolve_outcome(sig, df)
            if res is None:
                ungradable += 1
                break
            risk = abs(sig["price"] - sig["stop"])
            cost_r = sig["price"] * cost_bps / 1e4 / risk
            mp = mirror_study.make_mirror_payload(sig)
            mres = sc.resolve_outcome(mp, df) if mp else None
            risk_pct = risk / sig["price"] * 100.0
            trades.append({
                "day": str(day), "time": ts.strftime("%H:%M"), "symbol": symbol, "side": sig["side"], "tier": sig["tier"],
                "score": sig["score"], "window": sc.window_of(anchor), "formulas": sig.get("pillars_matched", []),
                "price": sig["price"], "risk_pct": round(risk_pct, 3),
                "target_pct": round(abs(sig["target"] - sig["price"]) / sig["price"] * 100.0, 3),
                "result": res["result"], "gross_r": res["r_result"], "cost_r": round(cost_r, 4),
                "net_r": round(res["r_result"] - cost_r, 4), "net_pct": round((res["r_result"] - cost_r) * risk_pct, 4),
                "mfe_pct": res["mfe_pct"], "mae_pct": res["mae_pct"], "mfe_pre_pct": round(mfe_before_stop(sig["side"], sig["price"], sig["stop"], df.iloc[pos + 1:e]), 3),
                "mirror_net_r": None if not mres else round(mres["r_result"] - cost_r, 4),
                "mirror_result": None if not mres else mres["result"],
                "grid": grade_grid(sig["side"], sig["price"], df.iloc[pos + 1:e], cost_bps / 100.0),
                "payload": {k: sig.get(k) for k in ("side", "pro", "day_move_pct", "relative_volume", "candle_time", "tier",
                                                    "pillars_matched", "score")}})
            break                                       # one signal per symbol per day, like the live dedupe
    return {"symbol": symbol, "trades": trades, "ungradable": ungradable, "scans": scans, "sessions": len(days)}


# ------------------------------------------------------------------ analysis
def stats(rows: List[Dict[str, Any]], key: str = "net_r") -> Dict[str, Any]:
    v = [r[key] for r in rows]
    if not v:
        return {"n": 0}
    w, l = sum(x for x in v if x > 0), -sum(x for x in v if x < 0)
    return {"n": len(v), "mean": round(mean(v), 3), "median": round(median(v), 3),
            "win_pct": round(100 * sum(x > 0 for x in v) / len(v), 1), "profit_factor": round(w / l, 2) if l > 0 else None}


def drawdown(vals: List[float]) -> Dict[str, Any]:
    peak = cum = worst = streak = worst_streak = 0.0
    for x in vals:
        cum += x
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
        streak = streak + 1 if x < 0 else 0
        worst_streak = max(worst_streak, streak)
    return {"max_drawdown_r": round(worst, 2), "longest_losing_streak": int(worst_streak), "total_net_r": round(cum, 2)}


def labels_for(t: Dict[str, Any]) -> List[str]:
    import feature_labels
    labs = feature_labels.extra_labels(t["payload"])
    labs += [f"Side: {t['side']}", f"Tier: {t['tier']}", f"Window: {t['window']}"] + [f"Formula: {f}" for f in t["formulas"]]
    return labs


def welch(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    from statistics import stdev
    se = (stdev(a) ** 2 / len(a) + stdev(b) ** 2 / len(b)) ** 0.5
    return (mean(a) - mean(b)) / se if se > 0 else 0.0


def analyze(trades: List[Dict[str, Any]], min_target_pct: float = 1.5) -> Dict[str, Any]:
    import feature_learner, mirror_study
    trades = sorted(trades, key=lambda t: (t["day"], t["time"]))
    days = sorted({t["day"] for t in trades})
    rep: Dict[str, Any] = {"trades": len(trades), "sessions": len(days)}
    if len(days) < 10 or len(trades) < 30:
        return {**rep, "verdict": "Too few trades/sessions to judge. Need more history or a wider universe."}
    split = days[max(1, int(len(days) * 0.7))]
    train, hold = [t for t in trades if t["day"] < split], [t for t in trades if t["day"] >= split]
    rep["split_day"] = split
    rep["overall"] = {"all": stats(trades), "train": stats(train), "holdout": stats(hold)}
    rep["overall_pct_per_trade"] = {"all": stats(trades, "net_pct"), "holdout": stats(hold, "net_pct")}
    rep["avg_target_pct"] = round(mean(t["target_pct"] for t in trades), 2)
    rep["avg_stop_pct"] = round(mean(t["risk_pct"] for t in trades), 2)
    rep["risk"] = drawdown([t["net_r"] for t in trades])
    months = {}
    for t in trades:
        months.setdefault(t["day"][:7], []).append(t)
    rep["by_month"] = {m: stats(v) for m, v in sorted(months.items())}
    rep["months_positive"] = f"{sum(1 for v in months.values() if mean(x['net_r'] for x in v) > 0)}/{len(months)}"
    rep["meaningful_moves"] = {f">= {x}%": {"session_mfe_pct_of_signals": round(100 * sum(t["mfe_pct"] >= x for t in trades) / len(trades), 1),
                                            "before_stop_pct_of_signals": round(100 * sum(t["mfe_pre_pct"] >= x for t in trades) / len(trades), 1)}
                               for x in (1.5, 2.0, 3.0, 5.0)}
    # breakdowns (exploratory) ------------------------------------------------
    dims: Dict[str, List[str]] = {}
    for t in trades:
        for lab in labels_for(t):
            dims.setdefault(lab, []).append(t)
    rows_all = [t["net_r"] for t in trades]
    table = []
    for lab, rs in dims.items():
        if len(rs) >= 30:
            rest = [t["net_r"] for t in trades if t not in rs] if len(rs) < len(trades) else []
            table.append({"label": lab, **stats(rs), "t_vs_rest": round(welch([t["net_r"] for t in rs], rest), 2)})
    table.sort(key=lambda r: r["t_vs_rest"])
    rep["exploratory_worst"], rep["exploratory_best"] = table[:8], table[::-1][:8]
    # validated learning (train/holdout + multiple-testing control) ------------
    learn_rows = [{"day": t["day"], "net_r": t["net_r"], "labels": set(labels_for(t))} for t in trades]
    rep["validated_features"] = feature_learner.learn_from_rows(learn_rows)
    # opposite-direction control -------------------------------------------------
    pairs = [{"orig_net": t["net_r"], "mirror_net": t["mirror_net_r"], "orig_result": t["result"], "mirror_result": t["mirror_result"]}
             for t in trades if t["mirror_net_r"] is not None]
    rep["mirror"] = mirror_study.compare(pairs)
    # target/stop grid with meaningful-target floor -----------------------------
    grid = []
    for key in trades[0]["grid"]:
        s, tgt = map(float, key.split("|"))
        if tgt < min_target_pct:
            continue
        tr, ho = [t["grid"][key] for t in train], [t["grid"][key] for t in hold]
        grid.append({"stop_pct": s, "target_pct": tgt, "train_net_pct": round(mean(tr), 3), "holdout_net_pct": round(mean(ho), 3),
                     "holdout_net_r": round(mean(ho) / s, 3), "holdout_win_pct": round(100 * sum(x > 0 for x in ho) / len(ho), 1)})
    grid.sort(key=lambda g: g["train_net_pct"], reverse=True)
    rep["grid_top_by_train"] = grid[:5]
    rep["grid_note"] = (f"Targets below {min_target_pct}% are excluded on purpose. Picking the best of {len(grid)} combinations on the training "
                        "months is a selection step: trust the holdout columns, not the training ones.")
    # verdict ---------------------------------------------------------------------
    h = [t["net_r"] for t in hold]
    lcb = mirror_study._lcb(h)
    if len(h) < 50:
        verdict = "Holdout has fewer than 50 trades: indicative only."
    elif lcb is not None and lcb > 0:
        verdict = "Positive expectancy after costs on unseen sessions (lower bound above zero)."
    elif mean(h) > 0:
        verdict = "Positive on unseen sessions but not statistically solid yet."
    else:
        verdict = "No edge after costs on unseen sessions for the baseline strategy."
    rep["holdout_lower_bound_net_r"] = None if lcb is None else round(lcb, 3)
    rep["verdict"] = verdict
    return rep


def write_outputs(out: Path, trades: List[Dict[str, Any]], rep: Dict[str, Any], meta: Dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trades.jsonl", "w") as f:
        for t in trades:
            f.write(json.dumps(t, default=str) + "\n")
    flat = [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in t.items() if k not in ("payload", "grid")} for t in trades]
    pd.DataFrame(flat).to_csv(out / "trades.csv", index=False)
    (out / "report.json").write_text(json.dumps({**rep, "run": meta}, indent=2, default=str))
    L = ["# Backtest report (baseline strategy, costs included)", "", f"**Verdict:** {rep['verdict']}", ""]
    if "overall" in rep:
        o = rep["overall"]
        L += [f"Trades {rep['trades']} over {rep['sessions']} sessions | split at {rep['split_day']} (train before, holdout after)",
              f"Avg stop {rep['avg_stop_pct']}% | avg target {rep['avg_target_pct']}%",
              f"Net R per trade: all {o['all']['mean']} | train {o['train'].get('mean')} | holdout {o['holdout'].get('mean')} (lower bound {rep['holdout_lower_bound_net_r']})",
              f"Profit factor all {o['all']['profit_factor']} | win rate (information only) {o['all']['win_pct']}%",
              f"Max drawdown {rep['risk']['max_drawdown_r']}R | longest losing streak {rep['risk']['longest_losing_streak']} | months positive {rep['months_positive']}", "",
              "## Does price really travel far enough? (share of signals)"]
        L += [f"- move {k}: reached in session {v['session_mfe_pct_of_signals']}% | before the stop {v['before_stop_pct_of_signals']}%" for k, v in rep["meaningful_moves"].items()]
        vf = rep["validated_features"]
        L += ["", f"## What the learner VALIDATED (status: {vf['status']})", "Helps: " + (", ".join(vf["positive"]) or "none"), "Hurts: " + (", ".join(vf["negative"]) or "none")]
        mi = rep["mirror"]
        L += ["", "## Opposite-trade control", mi.get("reading", f"collecting ({mi['pairs']} pairs)"),
              "", "## Exploratory (NOT validated): best / worst groups", ]
        L += [f"- {r['label']}: n={r['n']}, net {r['mean']}R, t={r['t_vs_rest']}" for r in rep["exploratory_best"][:5]]
        L += ["  ...", *[f"- {r['label']}: n={r['n']}, net {r['mean']}R, t={r['t_vs_rest']}" for r in rep["exploratory_worst"][:5]]]
        L += ["", "## Target/stop grid (meaningful targets only)", rep["grid_note"]]
        L += [f"- stop {g['stop_pct']}% target {g['target_pct']}%: train {g['train_net_pct']}%/trade -> holdout {g['holdout_net_pct']}%/trade ({g['holdout_net_r']}R, win {g['holdout_win_pct']}%)" for g in rep["grid_top_by_train"]]
    (out / "report.md").write_text("\n".join(L))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="backtest_out")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--scan-every", type=int, default=5, help="minutes between scans (multiple of 5)")
    ap.add_argument("--cost-bps", type=float, default=float(os.getenv("PAPER_ROUND_TRIP_BPS", "20")))
    ap.add_argument("--min-target-pct", type=float, default=1.5)
    ap.add_argument("--exact", action="store_true", help="skip the fast volume prefilter (slow, identical results)")
    a = ap.parse_args(argv)
    hist = load_history(a.data)
    syms = sorted(hist)[: a.max_symbols or None]
    print(f"Loaded {len(hist)} symbols; replaying {len(syms)}", flush=True)
    opts = {"cost_bps": a.cost_bps, "scan_every": max(5, a.scan_every), "exact": a.exact}
    jobs = [(s, hist[s], opts) for s in syms]
    results = []
    if a.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, r in enumerate(ex.map(replay_symbol, jobs), 1):
                results.append(r); print(f"  {i}/{len(jobs)} {r['symbol']}: {len(r['trades'])} signals", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            r = replay_symbol(j); results.append(r); print(f"  {i}/{len(jobs)} {r['symbol']}: {len(r['trades'])} signals", flush=True)
    trades = [t for r in results for t in r["trades"]]
    rep = analyze(trades, a.min_target_pct)
    meta = {"symbols": len(syms), "ungradable_signals": sum(r["ungradable"] for r in results), "cost_bps": a.cost_bps,
            "scan_every_min": opts["scan_every"], "generated": datetime.now(TZ).isoformat()}
    write_outputs(Path(a.out), trades, rep, meta)
    print(f"\nVERDICT: {rep['verdict']}\nWrote {a.out}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
