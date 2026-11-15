"""Confirmed three-sided pivots. Recent means the last three completed bars."""
import numpy as np
from features_pro import _prep, _swings, _atr

KEYS = ("bos_recent", "choch_recent", "liquidity_sweep", "equal_highs", "equal_lows",
        "mitigation_block_near", "premium_discount_zone")


def _cluster(values):
    """Largest repeated cluster with total range <= 0.1% of its low."""
    a = np.sort(values[-30:])
    count = max(np.searchsorted(a, a*1.001, side="right") - np.arange(len(a)))
    return int(count) if count > 1 else 0


def _near_block(d):
    o,h,l,c = (d[k].to_numpy() for k in ("open","high","low","close"))
    atr = _atr(d)
    for i in range(max(10,len(c)-50), len(c)-4):
        future = slice(i+1,i+4)
        bull = (c[i]<o[i] and max(h[future])-h[i]>=1.5*atr and
                max(c[future])>max(h[i-10:i]) and min(c[i+1:])>=l[i])
        bear = (c[i]>o[i] and l[i]-min(l[future])>=1.5*atr and
                min(c[future])<min(l[i-10:i]) and max(c[i+1:])<=h[i])
        if (bull or bear) and l[i]-.3*atr <= c[-1] <= h[i]+.3*atr:
            return True
    return False


def compute(df, side=None):
    d = _prep(df)
    if d is None:
        return dict.fromkeys(KEYS)
    h,l,c = (d[k].to_numpy() for k in ("high","low","close"))
    highs,lows = _swings(h,l)
    hi = lo = None
    prior = 0
    broken = set()
    bos = choch = sweep = False
    highs,lows = set(highs),set(lows)
    for i in range(4,len(c)):
        if i-4 in highs:
            hi = i-4
        if i-4 in lows:
            lo = i-4
        recent = i >= len(c)-3
        if i == len(c)-1:
            sweep = bool((hi is not None and h[i]>h[hi]>=c[i]) or (lo is not None and l[i]<l[lo]<=c[i]))
        for direction, pivot, crossed in ((1,hi,hi is not None and c[i]>h[hi]),
                                          (-1,lo,lo is not None and c[i]<l[lo])):
            if crossed and (direction,pivot) not in broken:
                bos |= recent
                choch |= recent and prior == -direction
                prior = direction
                broken.add((direction,pivot))
    zone = None if hi is None or lo is None or h[hi]<=l[lo] else float(np.clip(100*(c[-1]-l[lo])/(h[hi]-l[lo]),0,100))
    return dict(zip(KEYS,(bos,choch,sweep,_cluster(h),_cluster(l),_near_block(d),zone)))
