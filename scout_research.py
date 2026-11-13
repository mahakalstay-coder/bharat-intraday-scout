"""Research preview forecasting and observed sector leadership for Bharat Scout."""
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd

def drift_forecast(close, candle_time, horizon):
    """Random walk with drift, independently implemented from the StatsForecast model.

    https://github.com/Nixtla/statsforecast/blob/main/python/statsforecast/models.py
    Use up to twelve contiguous completed five-minute closes from this session.
    """
    end = pd.Timestamp(candle_time)
    if end.tzinfo is None or horizon < 1:
        return {}
    first = end.normalize() + pd.Timedelta(hours=9, minutes=15)
    index = pd.date_range(max(first, end-pd.Timedelta(minutes=55)), end, freq='5min')
    if len(index) < 4 or end + pd.Timedelta(minutes=5*horizon) > first + pd.Timedelta(hours=6,minutes=10):
        return {}
    prices = close.reindex(index)
    if not all(pd.notna(x) and math.isfinite(float(x)) and float(x)>0 for x in prices):
        return {}
    start, last = float(prices.iloc[0]), float(prices.iloc[-1])
    change = 100*horizon*(last-start)/((len(prices)-1)*last)
    return {'model':'RandomWalkWithDrift', 'forecast_change_pct':round(change,4),
            'horizon_bars':horizon, 'asof':end.isoformat(), 'bars':len(prices)}


def alternative_support(signal):
    fore = signal.get('forecast') or {}
    alt = signal.get('drift_forecast') or {}
    sign = 1 if signal.get('side') == 'BUY' else -1
    if fore.get('model') != 'Kronos-mini' or alt.get('model') != 'RandomWalkWithDrift':
        return False
    kr, drift = fore.get('forecast_change_pct'), alt.get('forecast_change_pct')
    return (kr is not None and drift is not None and math.isfinite(kr) and math.isfinite(drift)
            and alt.get('asof') == signal.get('candle_time')
            and alt.get('horizon_bars') == fore.get('horizon_bars')
            and sign*kr > -0.08 and sign*drift >= 0.15)


def sector_update(conn, market, now, send):
    """Persist the last alert and confirmation in app_state across scheduled starts."""
    bucket = now.replace(minute=now.minute//5*5,second=0,microsecond=0)
    expected = (bucket-timedelta(minutes=5)).isoformat()
    groups = defaultdict(list)
    for sym, stock in market.get('stocks', {}).items():
        if not math.isfinite(stock['change_pct']):
            continue
        sector = stock.get('sector')
        if sector and sector != 'Unknown':
            groups[sector].append((sym,stock))
    returns, eligible = {}, []
    for sector in market.get('sectors', []):
        name = sector['sector']
        members = groups.get(name, [])
        if len(members) < 3 or sector['count'] != len(members):
            continue
        if sum(s.get('asof') == expected for _,s in members) < len(members)*0.6:
            continue
        returns[name] = sector['change_pct']
        breadth = sector['advancing_pct']
        if returns[name] >= 0.6 and breadth >= 60:
            eligible.append((returns[name],name,breadth))
    key = 'sector_rotation_v1'
    conn.execute('BEGIN IMMEDIATE')
    row = conn.execute('SELECT value FROM app_state WHERE key=?',(key,)).fetchone()
    state = json.loads(row[0]) if row else {}
    if state.get('day') != now.date().isoformat():
        state = {'day':now.date().isoformat(),'count':0}
    if state.get('slot','') >= expected:
        conn.rollback()
        return False
    last_slot = state.get('slot')
    state['slot'] = expected
    prior = state.get('leader')
    gain,name,breadth = max(eligible,default=(0,None,0))
    margin = float(os.getenv('SECTOR_ROTATION_MARGIN_PP','0.50'))
    candidate = (name and name != prior and state['count'] < 4 and
                 (not prior or (prior in returns and gain-returns[prior]>=margin)))
    consecutive = (candidate and state.get('candidate')==name and last_slot ==
                   (bucket-timedelta(minutes=10)).isoformat())
    state['candidate'] = name if candidate else None
    cooled = not state.get('sent') or now-datetime.fromisoformat(state['sent'])>=timedelta(minutes=30)
    sent = False
    if consecutive and cooled:
        members = sorted(groups[name],key=lambda m:(-m[1]['change_pct'],m[0]))
        symbol,stock = members[0]
        peers = [(s,p) for s,p in members[1:] if (stock.get('industry') not in (None,'Unknown') and p.get('industry')==stock['industry'])]
        peer = ', '.join(f"{s} {p['change_pct']:+.2f}%" for s,p in peers[:2]) or 'none in current sample'
        msg = (f"📊 SECTOR {'ROTATION' if prior else 'UPDATE'} | {now:%H:%M} IST\n"
               f"{name} {gain:+.2f}% vs previous close | {len(members)} sampled | {breadth:.0f}% advancing\n"
               f"Top mover: {symbol} {stock['change_pct']:+.2f}% | Peers: {peer}")
        if prior:
            msg += f"\nPrevious leader: {prior} {returns[prior]:+.2f}% | lead {gain-returns[prior]:.2f} pp"
        if send(msg+'\nMarket observation only; no broker order placed.'):
            state.update(leader=name,sent=now.isoformat(),count=state['count']+1)
            sent = True
    conn.execute('INSERT OR REPLACE INTO app_state(key,value) VALUES(?,?)',(key,json.dumps(state)))
    conn.commit()
    return sent


def watch_reason(signal, market):
    """Return the single missed preview check; missing evidence is ineligible."""
    fore = signal.get('forecast') or {}
    if fore.get('model') != 'Kronos-mini' or not (fore.get('direction_aligned') or alternative_support(signal)):
        return None
    score, tier = signal.get('score',0), signal.get('tier')
    if tier not in ('TIER-1','TIER-2') or score < (85 if tier=='TIER-1' else 90):
        return None
    stock = signal.get('market_context') or {}
    name = stock.get('sector')
    sector = next((s for s in market['sectors'] if s['sector']==name), {})
    if not name or name=='Unknown' or sector.get('count',0)<3 or stock.get('peer_count',0)<2:
        return None
    side = signal['side']
    values = [signal.get('relative_volume'),signal.get('cumulative_relative_volume'),
              sector.get('change_pct'),sector.get('advancing_pct'),stock.get('peer_relative_pct')]
    if side not in ('BUY','SELL') or any(v is None or not math.isfinite(v) for v in values):
        return None
    rv, cum, move, breadth, peer = values
    buy = side=='BUY'
    tests = [(rv>=1.8, f'5m RVOL {rv:.2f}x (minimum 1.80x)'),
             (cum>=1.2, f'Session RVOL {cum:.2f}x (minimum 1.20x)'),
             ((move>0 and breadth>=55) if buy else (move<0 and breadth<=45),
              f'{name} sector alignment: move {move:+.2f}%, advancing {breadth:.1f}% '
              f'(requires {"positive move and >=55" if buy else "negative move and <=45"}% advancing)'),
             (peer>=0 if buy else peer<=0, f'Peer strength {peer:+.2f} pp (requires {">=0" if buy else "<=0"})')]
    missed = [desc for ok,desc in tests if not ok]
    return missed[0] if len(missed)==1 else None


def send_research_watch(conn, candidates, now, send):
    if not candidates:
        return False
    day = now.date().isoformat()
    conn.execute('BEGIN IMMEDIATE')
    key = 'research_watch_last_day'
    row = conn.execute('SELECT value FROM app_state WHERE key=?',(key,)).fetchone()
    if row and row[0]==day:
        conn.rollback()
        return False
    p, reason = max(candidates,key=lambda item:(item[0]['score'],item[0]['relative_volume'],item[0]['symbol']))
    msg = (f"👀 RESEARCH WATCH — {p['symbol']} {p['side']} | {now:%H:%M} IST\n"
           f"Score {p['score']}/100 | observed price ₹{p['price']}\nMissed check: {reason}\n"
           "Observation only. No entry alert or broker order.")
    if not send(msg):
        conn.rollback()
        return False
    conn.execute('INSERT OR REPLACE INTO app_state(key,value) VALUES(?,?)',(key,day))
    conn.commit()
    return True
