"""Auto-learning of which logged features actually help, with the same discipline as the rest of the system.

A feature label is promoted only when, on cost-adjusted paper trades split chronologically by session:
  * enough sessions have passed,
  * it has enough training AND holdout trades,
  * its mean net R beats the all-trade average by a margin on BOTH the training and the later holdout part.
Promoted labels add a small bounded score adjustment (+3 per helpful label, -3 per harmful label, capped +/-6).
Until something is validated the adjustment is zero, so behaviour is unchanged.
"""
from __future__ import annotations
import json
import os
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from statistics import NormalDist, mean, stdev
from typing import Any, Dict, Iterable, List, Optional

import feature_labels

STEP, CAP = 3, 6
SEP = " && "   # joins the two labels of a discovered combination


def _family(label: str) -> str:
    """Coarse origin of a label, so a feature is never paired with a restatement of itself."""
    return label.split(":")[0] if ":" in label else label.split(" ")[0]


def _env(name, default):
    return type(default)(os.getenv(name, str(default)))


def learn_from_rows(rows: List[Dict[str, Any]], now: Optional[datetime] = None) -> Dict[str, Any]:
    """rows: [{'day': 'YYYY-MM-DD', 'net_r': float, 'labels': iterable[str]}]"""
    min_sessions = _env("FEATURE_MIN_SESSIONS", 20)
    min_train, min_hold = _env("FEATURE_MIN_TRAIN", 30), _env("FEATURE_MIN_HOLD", 15)
    lift_train, lift_hold = _env("FEATURE_LIFT_TRAIN", 0.10), _env("FEATURE_LIFT_HOLD", 0.05)
    max_each = _env("FEATURE_MAX_PROMOTED", 3)
    pair_total = _env("FEATURE_PAIR_MIN_TOTAL", 25)
    pair_min_train, pair_min_hold = _env("FEATURE_PAIR_MIN_TRAIN", 20), _env("FEATURE_PAIR_MIN_HOLD", 10)
    counts = Counter(l for r in rows for l in r["labels"])
    frequent = sorted(l for l, c in counts.items() if c >= pair_total)
    aug = []
    for r in rows:   # add "A && B" combination labels for pairs of frequent, unrelated labels
        present = [l for l in frequent if l in r["labels"]]
        extra = {a + SEP + b for a, b in combinations(present, 2) if _family(a) != _family(b)}
        aug.append({**r, "labels": set(r["labels"]) | extra})
    rows = aug
    days = sorted({r["day"] for r in rows})
    base = {"at": (now or datetime.now()).isoformat(), "sessions": len(days), "samples": len(rows),
            "positive": {}, "negative": {}, "method": "screen on train (singles + pairs), confirm on later holdout with multiple-testing-adjusted t threshold"}
    if len(days) < min_sessions:
        return {**base, "status": "collecting"}
    split_day = days[max(1, int(len(days) * 0.7))]
    train = [r for r in rows if r["day"] < split_day]
    hold = [r for r in rows if r["day"] >= split_day]
    if not train or not hold:
        return {**base, "status": "collecting"}
    tb, hb = mean(r["net_r"] for r in train), mean(r["net_r"] for r in hold)
    alpha = _env("FEATURE_ALPHA", 0.10)
    names = {l for r in rows for l in r["labels"]}

    def tstat(label, part, _unused=None):
        """Welch t for (trades WITH the label) vs (trades WITHOUT it) inside `part`."""
        a = [r["net_r"] for r in part if label in r["labels"]]
        b = [r["net_r"] for r in part if label not in r["labels"]]
        if len(a) < 2 or len(b) < 2:
            return 0.0
        se = (stdev(a) ** 2 / len(a) + stdev(b) ** 2 / len(b)) ** 0.5
        return (mean(a) - mean(b)) / se if se > 0 else 0.0

    def welch(a, b):
        if len(a) < 2 or len(b) < 2:
            return 0.0
        se = (stdev(a) ** 2 / len(a) + stdev(b) ** 2 / len(b)) ** 0.5
        return (mean(a) - mean(b)) / se if se > 0 else 0.0

    def inc_ts(name, part):
        """A combination only counts if it beats EACH of its own parts played alone
        (trades with the combo vs trades with that part but not the other)."""
        x, y = name.split(SEP)
        both = [r["net_r"] for r in part if x in r["labels"] and y in r["labels"]]
        return [welch(both, [r["net_r"] for r in part if one in r["labels"] and other not in r["labels"]])
                for one, other in ((x, y), (y, x))]

    # 1) screen on TRAIN only; 2) confirm survivors on the untouched HOLDOUT with a
    #    multiple-testing penalty: the more candidates survive the screen, the higher the bar.
    cands = {}
    for name in names:
        tr = [r["net_r"] for r in train if name in r["labels"]]
        ho = [r["net_r"] for r in hold if name in r["labels"]]
        need_tr, need_ho = (pair_min_train, pair_min_hold) if SEP in name else (min_train, min_hold)
        if len(tr) < need_tr or len(ho) < need_ho:
            continue
        lt = mean(tr) - tb
        if abs(lt) >= lift_train and abs(tstat(name, train, tb)) >= 1.5:
            if SEP in name:   # combos must already add value beyond their parts on the training sessions
                ts = inc_ts(name, train)
                if not ((lt > 0 and min(ts) >= 1.0) or (lt < 0 and max(ts) <= -1.0)):
                    continue
            cands[name] = (tr, ho, lt)
    z_needed = NormalDist().inv_cdf(1.0 - alpha / max(1, len(cands)))
    pos, neg = {}, {}
    for name, (tr, ho, lt) in cands.items():
        lh = mean(ho) - hb
        th = tstat(name, hold, hb)
        entry = {"train_n": len(tr), "holdout_n": len(ho), "train_lift_r": round(lt, 3), "holdout_lift_r": round(lh, 3),
                 "holdout_t": round(th, 2), "holdout_mean_net_r": round(mean(ho), 3), "t_required": round(z_needed, 2)}
        if SEP in name:   # on the holdout the combo must beat BOTH of its parts, at the multiple-testing bar
            its = inc_ts(name, hold)
            entry["holdout_t_vs_parts"] = [round(t, 2) for t in its]
            th_pos, th_neg = min(its), max(its)
        else:
            th_pos = th_neg = th
        if lt > 0 and lh >= lift_hold and th >= z_needed and th_pos >= z_needed:
            pos[name] = entry
        elif lt < 0 and lh <= -lift_hold and th <= -z_needed and th_neg <= -z_needed:
            neg[name] = entry
    top = lambda d, rev: dict(sorted(d.items(), key=lambda kv: kv[1]["holdout_lift_r"], reverse=rev)[:max_each])
    pos, neg = top(pos, True), top(neg, False)
    return {**base, "status": "validated" if (pos or neg) else "no_stable_features", "positive": pos, "negative": neg}


def learn(conn, root, now: Optional[datetime] = None) -> Dict[str, Any]:
    import market_learning as ml
    ledger = [r for r in ml.ledger(conn) if r["net_r"] is not None and r["cost_source"] == "entry snapshot"]
    payloads = {sid: json.loads(raw) for sid, raw in conn.execute("SELECT id,payload FROM signals")}
    rows = [{"day": r["day"], "net_r": float(r["net_r"]),
             "labels": set(feature_labels.extra_labels(payloads.get(r["id"], {})))} for r in ledger]
    result = learn_from_rows(rows, now)
    ml.atomic_json(Path(root) / "data" / "validated_features.json", result)
    return result


def load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if data.get("status") == "validated" else None
    except Exception:
        return None


def score_adjustment(validated: Optional[Dict[str, Any]], labels: Iterable[str]) -> int:
    if not validated:
        return 0
    have = set(labels)
    present = lambda k: all(part in have for part in k.split(SEP))
    adj = STEP * sum(1 for k in validated.get("positive", {}) if present(k))
    adj -= STEP * sum(1 for k in validated.get("negative", {}) if present(k))
    return max(-CAP, min(CAP, adj))
