"""Gann geometry has no peer-reviewed predictive evidence. These are logged for the feature learner to test, not trusted. A positive learner result on one small sample is not proof.
No peer-reviewed evidence for Gann geometry. Logged for the learner, not trusted.
Uses the latest confirmed 20-bar-radius pivot and preceding opposite pivot.
Angles are signed rupees/bar and ATR/bar, measured from horizontal.
"""
import math
import pandas as pd
from features_pro import _prep, _atr, _swings

KEYS = ("gann_fan_angle_deg","gann_fan_atr_angle_deg","gann_retracement_pct","gann_nearest_retracement_pct",
        "gann_time_days_from_swing","gann_nearest_cycle_days","gann_within_cycle",
        "gann_sq9_nearest_deg","gann_sq9_distance_deg")


def compute(df, side=None):
    r = dict.fromkeys(KEYS)
    d = _prep(df)
    if d is None or d.close.iloc[-1]<=0:
        return r
    price = d.close.iloc[-1]
    deg = math.sqrt(price)*180%360
    nearest = round(deg/45)*45
    r.update(gann_sq9_nearest_deg=nearest%360,gann_sq9_distance_deg=abs(nearest-deg))
    highs,lows = _swings(d.high.values,d.low.values,w=20)
    pivots = sorted([(i,"high") for i in highs]+[(i,"low") for i in lows])
    if not pivots:
        return r
    i,k = pivots[-1]
    anchor = d[k].iloc[i]
    slope = (price-anchor)/(len(d)-1-i)
    atr = _atr(d)
    r[KEYS[0]] = math.degrees(math.atan(slope))
    r[KEYS[1]] = math.degrees(math.atan(slope/atr)) if atr>0 else None
    for j,other in reversed(pivots[:-1]):
        if other!=k and anchor!=d[other].iloc[j] and j<i:
            pct = float(100*(anchor-price)/(anchor-d[other].iloc[j]))
            r[KEYS[2]] = pct
            r[KEYS[3]] = min((12.5,25,37.5,50,62.5,75,87.5), key=lambda n:abs(n-pct))
            break
    if isinstance(d.index,pd.DatetimeIndex):
        days = (d.index[-1].date()-d.index[i].date()).days
        cycle = min((30,45,60,90,144,180,270,360),key=lambda n:abs(n-days))
        r[KEYS[4]],r[KEYS[5]],r[KEYS[6]] = days,cycle,abs(cycle-days)<=1
    return r
