"""Close-only Renko, with causal ATR14 bricks and a two-brick reversal.
Classic mode counts a brick after N consecutive directional closes.
"""
import numpy as np
import pandas as pd
from features_pro import _prep

KEYS = ("renko_trend", "renko_distance_atr", "renko_reversal_within", "renko_agrees_daily")


def compute(df, side=None, mode="atr", n=3):
    result = dict.fromkeys(KEYS)
    d = _prep(df)
    if d is None:
        return result
    c = d.close
    atr = pd.concat([d.high-d.low, (d.high-c.shift()).abs(), (d.low-c.shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
    anchor = c.iloc[13]
    bricks = []
    reversal = level = None
    streak = prev = 0
    for i in range(14, len(d)):
        size = atr.iloc[i]
        price = c.iloc[i]
        if mode == "classic":
            direction = int(np.sign(price-c.iloc[i-1]))
            streak = streak + 1 if direction == prev else 1
            prev = direction
            if not direction or streak < n:
                continue
            streak = 0
            count = 1
            new_anchor = price
        else:
            if size <= 0:
                continue
            direction = int(np.sign(price-anchor))
            steps = int(abs(price-anchor)/size)
            turning = bool(bricks and bricks[-1] != direction)
            count = steps - int(turning)
            if count < 1:
                continue
            new_anchor = anchor + direction*size*steps
        if bricks and direction != bricks[-1]:
            reversal, level = i, anchor
        anchor = new_anchor
        bricks = (bricks + [direction]*min(count, 3))[-3:]
    trend = "UP" if bricks == [1]*3 else "DOWN" if bricks == [-1]*3 else "FLAT"
    result[KEYS[0]] = trend
    if reversal is not None:
        result[KEYS[1]] = float(abs(c.iloc[-1]-level)/atr.iloc[-1]) if atr.iloc[-1] > 0 else None
        result[KEYS[2]] = len(d)-1-reversal
    if isinstance(c.index, pd.DatetimeIndex):
        daily = c.groupby(c.index.date).last().iloc[:-1]
        cached = df.attrs.get("prior_daily_ema20")
        slope = None
        if cached and cached[0] < str(c.index[-1].date()):
            slope = cached[1]
        elif len(daily) >= 21:
            slope = daily.ewm(span=20, adjust=False).mean().diff().iloc[-1]
        if slope is not None:
            result[KEYS[3]] = bool((trend == "UP" and slope > 0) or (trend == "DOWN" and slope < 0))
    return result
