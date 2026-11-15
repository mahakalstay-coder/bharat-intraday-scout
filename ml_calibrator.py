"""Trained on <N> paper trades. Past association only. Will underperform out of sample if the regime changes or the sample is <500 trades.

The two EOD classes use the requested fractional target 0.5 and weight 0.5.
Training and weighted AUC expand these into equally weighted binary examples.
Brier is weighted squared error against the stated fractional target.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from market_learning import atomic_json

DISCLAIMER = ("Gradient-boosted probability trained on paper outcomes. This is a research quantity, "
              "not a trading recommendation, and is NOT used to gate alerts.")
INDICATORS = ("rsi","mfi","adx","bb_bandwidth_pct","relative_volume","cumulative_relative_volume","avg_daily_range_pct")


def features(payload):
    out = {}
    def flatten(prefix,value):
        if isinstance(value,dict):
            for k,v in value.items():
                flatten(prefix+"."+k,v)
        elif isinstance(value,list):
            for v in value:
                if isinstance(v,str):
                    out[prefix+"="+v] = 1.
        elif isinstance(value,str):
            out[prefix+"="+value] = 1.
        elif isinstance(value,(float,int)) and math.isfinite(value):
            out[prefix] = float(value)
    flatten("pro",payload.get("pro",{}))
    for k in INDICATORS:
        flatten(k,payload.get(k))
    if payload.get("vwap") and payload.get("price"):
        flatten("vwap_distance_pct",100*(payload["price"]/payload["vwap"]-1))
    if payload.get("avg_daily_range_pct") and payload.get("day_move_pct") is not None:
        flatten("adr_used",abs(payload["day_move_pct"])/payload["avg_daily_range_pct"])
    return out


def _expanded(rows, names):
    x,y,w = [],[],[]
    for _,attrs,target,weight in rows:
        vals = [attrs.get(k,0 if "=" in k else np.nan) for k in names]
        for label in ([0,1] if target==.5 else [int(target)]):
            x.append(vals)
            y.append(label)
            w.append(weight/2 if target==.5 else weight)
    return np.asarray(x),np.asarray(y),np.asarray(w)


def train(records, path=None, now=None):
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit
    except (ImportError,OSError):
        return {"status":"missing_dependency","disclaimer":DISCLAIMER}
    rows = []
    targets = {"T1":1.,"STOP":0.,"EOD_LOSS":.5,"EOD_WIN":.5}
    for r in records:
        if r.get("result") in targets and r.get("net_r") is not None and r.get("cost_source")=="entry snapshot":
            y = targets[r["result"]]
            rows.append((r["day"],features(r["payload"]),y,.5 if y==.5 else 1.))
    rows.sort(key=lambda r:r[0])
    days = sorted({r[0] for r in rows})
    train_days = days[:max(1,int(.7*len(days)))]
    tr = [r for r in rows if r[0] in train_days]
    ho = rows[len(tr):]
    stamp = now or (days[-1]+"T23:59:59+05:30" if days else None)
    report = dict(trained_at=stamp,n_train=len(tr),n_holdout=len(ho),holdout_auc=None,holdout_brier=None,
                  baseline_auc=None,baseline_brier=None,feature_importance_top10=[],cv=[],status="collecting",disclaimer=DISCLAIMER)
    params = dict(objective="binary",metric="binary_logloss",max_depth=4,num_leaves=16,learning_rate=.05,
                  min_child_samples=20,verbosity=-1,num_threads=1,seed=17,deterministic=True,force_col_wise=True)

    def fit(part,rounds=200,validation=None):
        names = sorted({k for _,f,_,_ in part for k in f})
        x,y,w = _expanded(part,names)
        if not names or len(set(y))<2:
            return None,names
        train_set = lgb.Dataset(x,label=y,weight=w)
        kwargs = {}
        if validation:
            vx,vy,vw = _expanded(validation,names)
            kwargs = dict(valid_sets=[lgb.Dataset(vx,label=vy,weight=vw,reference=train_set)],
                          callbacks=[lgb.early_stopping(20,verbose=False)])
        return lgb.train(params,train_set,num_boost_round=rounds,**kwargs),names

    def metrics(model,names,part):
        x,y,w = _expanded(part,names)
        pred = model.predict(x,num_threads=1)
        auc = float(roc_auc_score(y,pred,sample_weight=w)) if len(set(y))==2 else None
        targets = np.array([r[2] for r in part])
        weights = np.array([r[3] for r in part])
        first = np.cumsum([0]+[2 if t==.5 else 1 for t in targets[:-1]])
        return auc,float(np.average((pred[first]-targets)**2,weights=weights))

    if len(ho)>=100 and len(train_days)>=6:
        report["status"] = "no_stable_change"
        for a,b in TimeSeriesSplit(5).split(train_days):
            start,end = train_days[b[0]],train_days[b[-1]]
            earlier = [r for r in tr if r[0]<start]
            later = [r for r in tr if start<=r[0]<=end]
            model,names = fit(earlier)
            auc,brier = metrics(model,names,later) if model else (None,None)
            report["cv"].append(dict(start=start,end=end,auc=auc,brier=brier))
        cut = train_days[int(.8*len(train_days))]
        pilot,_ = fit([r for r in tr if r[0]<cut],validation=[r for r in tr if r[0]>=cut])
        if pilot:
            model,names = fit(tr,rounds=pilot.best_iteration or 200)
            auc,brier = metrics(model,names,ho)
            baseline = float(np.average([(.5-r[2])**2 for r in ho],weights=[r[3] for r in ho]))
            gains = sorted(zip(names,model.feature_importance("gain")),key=lambda v:-v[1])
            report.update(holdout_auc=auc,baseline_auc=.5 if auc is not None else None,holdout_brier=brier,baseline_brier=baseline,
                          feature_importance_top10=[[k,float(v)] for k,v in gains[:10]])
            if auc is not None and auc>=.6 and brier<baseline-.02:
                report.update(status="validated",columns=names,booster=model.model_to_string())
    if path:
        atomic_json(path,report)
    return report


def predict(payload, report):
    if report.get("status")!="validated":
        return None
    try:
        import lightgbm as lgb
        x,_,_ = _expanded([("",features(payload),0,1)],report["columns"])
        return float(lgb.Booster(model_str=report["booster"]).predict(x,num_threads=1)[0])
    except Exception:
        return None


def annotate(conn,path):
    """At export, annotate v5 entries with models available at entry."""
    row = conn.execute("SELECT value FROM app_state WHERE key='v5_calibrator'").fetchone()
    report = json.loads(row[0]) if row else (json.loads(path.read_text()) if path.exists() else {})
    for sid,stamp,raw in conn.execute("SELECT id,created_at,payload FROM signals WHERE json_type(payload,'$.pro.renko') IS NOT NULL AND json_type(payload,'$.ml_target_probability') IS NULL").fetchall():
        p = json.loads(raw)
        if "ml_target_probability" in p or "renko" not in (p.get("pro") or {}):
            continue
        eligible = report.get("trained_at") and pd.Timestamp(report["trained_at"])<=pd.Timestamp(stamp)
        p["ml_target_probability"] = predict(p,report) if eligible else None
        conn.execute("UPDATE signals SET payload=? WHERE id=? AND payload=?",(json.dumps(p,ensure_ascii=False),sid,raw))


# Predeclared label definitions. Each individual association is interpreted only
# for this condition; an unselected label stays collecting.
CONDITIONS = {
    "renko_trend":("UP","eq"),"renko_distance_atr":(2,"ge"),"renko_reversal_within":(3,"le"),"renko_agrees_daily":(True,"eq"),
    "bos_recent":(True,"eq"),"choch_recent":(True,"eq"),"liquidity_sweep":(True,"eq"),"equal_highs":(2,"ge"),"equal_lows":(2,"ge"),
    "mitigation_block_near":(True,"eq"),"premium_discount_zone":(50,"ge"),
    "fft_dominant_period_bars":(16,"le"),"fft_cycle_strength":(1.5,"ge"),"hilbert_phase_deg":(180,"ge"),
    "hilbert_amplitude_pct":(1,"ge"),"hilbert_cycle_direction":("UP","eq"),
    "gann_fan_angle_deg":(45,"ge"),"gann_fan_atr_angle_deg":(45,"ge"),"gann_retracement_pct":(50,"ge"),
    "gann_nearest_retracement_pct":(50,"eq"),"gann_time_days_from_swing":(30,"ge"),
    "gann_nearest_cycle_days":(30,"eq"),"gann_within_cycle":(True,"eq"),"gann_sq9_nearest_deg":(0,"eq"),"gann_sq9_distance_deg":(5,"le"),
}


def study(root,records):
    import operator
    import feature_learner
    import feature_labels
    statuses = {k:dict(status="collecting",observed=0,condition=[op,val]) for k,(val,op) in CONDITIONS.items()}
    rows = []
    for r in records:
        p = r["payload"]
        pro = p.get("pro") or {}
        if r["cost_source"]!="entry snapshot" or r["net_r"] is None or r["result"]=="NO_DATA" or not any(pro.get(g) for g in ("smc","cycles","gann","renko")):
            continue
        labels = set(feature_labels.extra_labels(p,research=True))-set(feature_labels.extra_labels(p))
        for group in ("smc","cycles","gann","renko"):
            for k,v in (pro.get(group) or {}).items():
                if k in CONDITIONS and v is not None:
                    statuses[k]["observed"] += 1
                    val,op = CONDITIONS[k]
                    if getattr(operator,op)(v,val):
                        labels.add("V5: "+k)
        rows.append(dict(day=r["day"],net_r=r["net_r"],labels=labels))
    evidence = feature_learner.learn_from_rows(rows)
    for kind,status in [("positive","validated"),("negative","negative")]:
        for label,stats in evidence[kind].items():
            k = label.removeprefix("V5: ")
            if k in statuses:
                statuses[k].update(status=status,evidence=stats)
    report = dict(paper_only=True,features=statuses,learner=evidence,disclaimer=DISCLAIMER)
    report["interpretation"] = {
        "Gann": "No peer-reviewed evidence for Gann geometry. Logged for the learner, not trusted. A positive learner result on one small sample is not proof.",
        "Cycles": "The Laplace PDE does not apply to 1D price. This is the correct spectral replacement.",
        "Learner": "Each status applies to the specified condition on entry-snapshot costed paper trades. Collecting also covers conditions without stable evidence.",
        "Calibrator": "Past association only. Will underperform out of sample if the regime changes or the sample is <500 trades.",
    }
    report["What was NOT added and why"] = {
        "Neural nets":"The available paper outcomes do not establish a need for a high-capacity model.",
        "Deep RL":"There is no validated market simulator for learning an execution policy.",
        "Order-flow-from-OHLCV":"Aggregated bars do not identify signed orders or queue changes.",
        "Laplace-as-PDE":"A one-dimensional price series does not provide the spatial boundary conditions for this PDE."}
    atomic_json(Path(root)/"reports/features_v5_report.json",report)
    return report


def refresh(root,conn,ledger):
    payloads = dict(conn.execute("SELECT id,payload FROM signals"))
    records = [dict(r,payload=json.loads(payloads[r["id"]])) for r in ledger]
    study(root,records)
    path = Path(root)/"data/ml_calibrator.json"
    result = train(records,path,datetime.now(timezone.utc).isoformat())
    conn.execute("INSERT OR REPLACE INTO app_state(key,value) VALUES('v5_calibrator',?)",(json.dumps(result),))
    if result["status"]=="missing_dependency":
        atomic_json(path,result)
