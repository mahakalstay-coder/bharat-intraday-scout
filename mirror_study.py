"""Opposite-direction ("mirror") paper trades: a built-in control experiment.

For every logged signal the system also records the OPPOSITE trade at the same entry with the same
risk and the same R-multiple target (BUY at 100, stop 99, target 102.5  ->  SELL at 100, stop 101,
target 97.5). Mirrors live in their own table, are graded with the same resolver and cost model,
and are NEVER mixed into the real ledger, learning, promotion or alerts.

Why: if the original beats its mirror, the setup has *directional* edge. If both do equally badly
(or well), the setup mostly finds movement, not direction, or the market simply moved one way.
"""
from __future__ import annotations
import json
import math
import os
import random
from datetime import datetime, timedelta
from statistics import mean
from typing import Any, Callable, Dict, List, Optional

DDL = """CREATE TABLE IF NOT EXISTS mirror_trades (
    signal_id INTEGER PRIMARY KEY, day TEXT NOT NULL, symbol TEXT NOT NULL, orig_side TEXT NOT NULL,
    shadow_only INTEGER NOT NULL, payload TEXT NOT NULL, result TEXT, r_result REAL, net_r REAL, resolved_at TEXT)"""


def ensure(conn) -> None:
    conn.execute(DDL)


def make_mirror_payload(p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        side, entry, stop, t1 = p["side"], float(p["price"]), float(p["stop"]), float(p["target"])
        risk = abs(entry - stop)
        if side not in ("BUY", "SELL") or risk <= 0 or not all(math.isfinite(v) for v in (entry, stop, t1)):
            return None
        if not (stop < entry < t1 if side == "BUY" else t1 < entry < stop):
            return None
        reward = abs(t1 - entry)
        if side == "BUY":
            m_side, m_stop, m_t1 = "SELL", entry + risk, entry - reward
        else:
            m_side, m_stop, m_t1 = "BUY", entry - risk, entry + reward
        return {"symbol": p["symbol"], "side": m_side, "price": entry, "stop": round(m_stop, 4),
                "target": round(m_t1, 4), "candle_time": p["candle_time"],
                "paper_cost_bps": p.get("paper_cost_bps", os.getenv("PAPER_ROUND_TRIP_BPS", "20")),
                "stop_scale": p.get("stop_scale", 1.0), "mirror_of_side": side}
    except (KeyError, TypeError, ValueError):
        return None


def record(conn, signal_id: int, payload: Dict[str, Any], now: datetime) -> bool:
    if os.getenv("ENABLE_MIRROR_TRADES", "true").lower() != "true":
        return False
    mp = make_mirror_payload(payload)
    if mp is None:
        return False
    ensure(conn)
    cur = conn.execute("INSERT OR IGNORE INTO mirror_trades (signal_id,day,symbol,orig_side,shadow_only,payload) VALUES (?,?,?,?,?,?)",
                       (signal_id, now.date().isoformat(), payload["symbol"], payload["side"],
                        1 if payload.get("shadow_only", True) else 0, json.dumps(mp)))
    return cur.rowcount == 1


def backfill(conn, now: datetime, max_age_days: int = 9) -> int:
    """Create mirrors for recent signals that predate this feature (older ones have no 5m data left to grade)."""
    if os.getenv("ENABLE_MIRROR_TRADES", "true").lower() != "true":
        return 0
    ensure(conn)
    n = 0
    for sid, raw in conn.execute("SELECT s.id, s.payload FROM signals s LEFT JOIN mirror_trades m ON m.signal_id=s.id "
                                 "WHERE m.signal_id IS NULL").fetchall():
        try:
            p = json.loads(raw)
            created = datetime.fromisoformat(str(p["candle_time"]))
            if (now.date() - created.date()).days > max_age_days:
                continue
            n += record(conn, sid, p, created)
        except Exception:
            continue
    conn.commit()
    return n


def _cost_r(p: Dict[str, Any]) -> Optional[float]:
    risk = abs(float(p["price"]) - float(p["stop"]))
    return float(p["price"]) * float(p.get("paper_cost_bps", 20)) / 10000.0 / risk if risk > 0 else None


def resolve_pending(conn, frames: Dict[str, Any], resolver: Callable, normalize: Callable, now: datetime,
                    session_end=(15, 30)) -> int:
    """Grade unresolved mirrors with the real outcome resolver. Returns how many were resolved."""
    ensure(conn)
    done = 0
    for row in conn.execute("SELECT signal_id, day, payload FROM mirror_trades WHERE result IS NULL").fetchall():
        sid, day, raw = row[0], row[1], row[2]
        try:
            if day == now.date().isoformat() and (now.hour, now.minute) < session_end:
                continue
            p = json.loads(raw)
            frame = frames.get(p["symbol"])
            out = resolver(p, normalize(frame)) if frame is not None and len(frame) else None
            if out is None:
                if (now.date() - datetime.fromisoformat(day).date()).days > 9:
                    conn.execute("UPDATE mirror_trades SET result='NO_DATA', resolved_at=? WHERE signal_id=?", (now.isoformat(), sid))
                continue
            cost = _cost_r(p)
            net = out["r_result"] - cost if cost is not None else None
            conn.execute("UPDATE mirror_trades SET result=?, r_result=?, net_r=?, resolved_at=? WHERE signal_id=?",
                         (out["result"], out["r_result"], net, now.isoformat(), sid))
            done += 1
        except Exception:
            continue
    conn.commit()
    return done


def _lcb(diffs: List[float], pct: float = 20.0, n_boot: int = 2000, seed: int = 11) -> Optional[float]:
    if len(diffs) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    return means[int(n_boot * pct / 100.0)]


def compare(pairs: List[Dict[str, Any]], min_pairs: int = 30) -> Dict[str, Any]:
    """pairs: [{'orig_net': float, 'mirror_net': float, 'orig_result': str, 'mirror_result': str}]"""
    n = len(pairs)
    out: Dict[str, Any] = {"pairs": n, "status": "collecting", "min_pairs": min_pairs}
    if n == 0:
        return out
    o, m = [x["orig_net"] for x in pairs], [x["mirror_net"] for x in pairs]
    diffs = [a - b for a, b in zip(o, m)]
    lcb = _lcb(diffs)
    out.update({"original_mean_net_r": round(mean(o), 3), "mirror_mean_net_r": round(mean(m), 3),
                "direction_edge_r": round(mean(diffs), 3),
                "original_beats_mirror_pct": round(100 * sum(d > 0 for d in diffs) / n, 1),
                "both_stopped_pct": round(100 * sum(x["orig_result"] == "STOP" and x["mirror_result"] == "STOP" for x in pairs) / n, 1),
                "edge_lower_bound_r": None if lcb is None else round(lcb, 3)})
    if n >= min_pairs:
        if lcb is not None and lcb > 0:
            out["status"], out["reading"] = "evaluated", "Directional edge: original beats its opposite (lower bound > 0)."
        elif mean(diffs) <= 0:
            out["status"], out["reading"] = "evaluated", "No directional edge: the opposite trade does as well or better. The setup finds movement, not direction."
        else:
            out["status"], out["reading"] = "evaluated", "Inconclusive: original leads, but not by enough to rule out luck."
    return out


def report(conn) -> Dict[str, Any]:
    try:
        ensure(conn)
        rows = conn.execute("""SELECT s.payload AS sp, o.result AS orr, o.r_result AS orig_r, m.result AS mres, m.net_r AS mnet, m.shadow_only AS sh
                               FROM mirror_trades m JOIN signals s ON s.id=m.signal_id JOIN outcomes o ON o.signal_id=m.signal_id
                               WHERE m.net_r IS NOT NULL AND o.r_result IS NOT NULL""").fetchall()
        pairs, approved = [], []
        for r in rows:
            sp = json.loads(r[0])
            if "paper_cost_bps" not in sp:
                continue  # legacy signals without an entry cost snapshot
            cost = _cost_r(sp)
            if cost is None:
                continue
            item = {"orig_net": float(r[2]) - cost, "mirror_net": float(r[4]), "orig_result": r[1], "mirror_result": r[3]}
            pairs.append(item)
            if not r[5]:
                approved.append(item)
        rep = compare(pairs)
        rep["approved_only"] = compare(approved, min_pairs=15) if approved else {"pairs": 0, "status": "collecting"}
        return rep
    except Exception as exc:
        return {"status": "unavailable", "error": type(exc).__name__}


def digest_line(rep: Optional[Dict[str, Any]]) -> Optional[str]:
    if not rep or rep.get("status") in (None, "unavailable"):
        return None
    if rep["status"] == "collecting":
        return f"Mirror test (opposite trades): collecting | {rep.get('pairs', 0)} of {rep.get('min_pairs', 30)} pairs"
    return (f"Mirror test: {rep['pairs']} pairs | original {rep['original_mean_net_r']:+.2f}R vs opposite {rep['mirror_mean_net_r']:+.2f}R net "
            f"| original better {rep['original_beats_mirror_pct']}% | {rep['reading']}")
