"""Auditable paper learning and sampled sector breadth; no broker execution."""
from __future__ import annotations
import csv
import html
import json
import math
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from zoneinfo import ZoneInfo
import pandas as pd

TZ = ZoneInfo('Asia/Kolkata')

def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8'); tmp.replace(path)

def refresh_classifications(root, symbols, fetch_info):
    """Refresh bounded metadata batches, retaining failures for a later attempt."""
    path = Path(root)/'data/classifications.json'
    cache = json.loads(path.read_text()) if path.exists() else {}
    now = datetime.now(TZ)
    pending = []
    for symbol in symbols:
        old = cache.get(symbol, {})
        try: age = (now - datetime.fromisoformat(old['updated_at'])).days
        except (KeyError, ValueError): age = 999
        if age >= 30 or not old.get('industry'): pending.append(symbol)
    errors = []
    for symbol in pending[:int(os.getenv('METADATA_BATCH_SIZE', '50'))]:
        try:
            info = fetch_info(symbol)
            if info.get('sector') and info.get('industry'):
                cache[symbol] = {'sector':info['sector'], 'industry':info['industry'],
                                 'source':'Yahoo Finance metadata', 'updated_at':now.isoformat()}
            else: errors.append(symbol)
        except Exception: errors.append(symbol)
    atomic_json(path, cache)
    return {'classified':len(cache), 'attempted':min(len(pending),int(os.getenv('METADATA_BATCH_SIZE','50'))),
            'pending':max(0,len(pending)-int(os.getenv('METADATA_BATCH_SIZE','50')))+len(errors), 'failed':errors}

def snapshot(frames, metadata, now):
    stocks = {}
    for symbol, raw in frames.items():
        if raw is None or raw.empty: continue
        df = raw.copy(); idx = pd.to_datetime(df.index)
        df.index = idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)
        df = df[~df.index.duplicated()].sort_index()
        df = df[df.index + pd.Timedelta(minutes=5) <= now].between_time('09:15','15:25')
        today = df[df.index.date == now.date()]; prior = df[df.index.date < now.date()]
        if today.empty or prior.empty: continue
        ts = today.index[-1]
        if (now-ts).total_seconds() > 900: continue
        close, prev = float(today.iloc[-1]['close']), float(prior.iloc[-1]['close'])
        if not math.isfinite(close) or not math.isfinite(prev) or prev <= 0: continue
        meta = metadata.get(symbol,{})
        try: valid_meta = (now-datetime.fromisoformat(meta['updated_at'])).days < 90
        except (KeyError,ValueError): valid_meta = False
        sector = meta.get('sector','Unknown') if valid_meta else 'Unknown'
        industry = meta.get('industry','Unknown') if valid_meta else 'Unknown'
        stocks[symbol] = {'symbol':symbol,'sector':sector,'industry':industry,'price':close,
            'change_pct':round((close/prev-1)*100,3),'asof':ts.isoformat()}
    sectors = []
    for sector in sorted({s['sector'] for s in stocks.values()}):
        members = [s for s in stocks.values() if s['sector']==sector]
        sectors.append({'sector':sector,'count':len(members),
            'change_pct':round(mean(s['change_pct'] for s in members),3),
            'advancing_pct':round(100*sum(s['change_pct']>0 for s in members)/len(members),1)})
    for symbol, stock in stocks.items():
        peers = sorted([s for s in stocks.values() if s['symbol']!=symbol and s['industry']==stock['industry']
                        and stock['industry']!='Unknown'], key=lambda s:s['change_pct'], reverse=True)
        stock['peers'] = [{'symbol':s['symbol'],'change_pct':s['change_pct']} for s in peers[:8]]
        stock['peer_count'] = len(peers)
        stock['peer_relative_pct'] = round(stock['change_pct']-mean(s['change_pct'] for s in peers),3) if peers else None
    return {'generated_at':now.isoformat(),'scope':'Sampled liquid universe; equal-weight breadth, not official sector indices',
            'requested':len(frames),'fresh':len(stocks),'classified':sum(s['sector']!='Unknown' for s in stocks.values()),
            'stocks':stocks,'sectors':sectors}

def init(conn):
    conn.execute('CREATE TABLE IF NOT EXISTS market_snapshots (asof TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS learning_audit (id INTEGER PRIMARY KEY, asof TEXT, payload TEXT)')

def record_snapshot(conn, snap):
    init(conn)
    old=conn.execute('SELECT payload FROM market_snapshots ORDER BY asof DESC LIMIT 1').fetchone()
    if old:
        previous=json.loads(old[0])
        if previous['generated_at'][:10] == snap['generated_at'][:10]:
            before={x['sector']:x['change_pct'] for x in previous['sectors']}
            for sector in snap['sectors']:
                sector['change_since_previous_pp']=round(sector['change_pct']-before[sector['sector']],3) if sector['sector'] in before else None
            snap['previous_snapshot_at']=previous['generated_at']
    conn.execute('INSERT OR REPLACE INTO market_snapshots VALUES (?,?)',(snap['generated_at'],json.dumps(snap)))
    conn.execute("DELETE FROM market_snapshots WHERE asof < ?", ((pd.Timestamp(snap['generated_at'])-pd.Timedelta(days=30)).isoformat(),))
    conn.commit()

def ledger(conn):
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT s.id,s.created_at,s.payload,o.result,o.r_result FROM signals s LEFT JOIN outcomes o ON o.signal_id=s.id ORDER BY s.id').fetchall()
    result = []
    for row in rows:
        p=json.loads(row['payload']); risk=abs(p['price']-p['stop'])
        bps=float(p.get('paper_cost_bps',os.getenv('PAPER_ROUND_TRIP_BPS','20')))
        cost = p['price']*bps/10000/risk if risk>0 else None
        gross=row['r_result']; net=gross-cost if gross is not None and cost is not None else None
        result.append({'id':row['id'],'day':row['created_at'][:10],'symbol':p['symbol'],'side':p['side'],
            'entry':p['price'],'stop':p['stop'],'target':p['target'],'result':row['result'] or 'PENDING',
            'gross_r':gross,'cost_r':cost,'net_r':net,'cost_bps':bps,
            'cost_source':'entry snapshot' if 'paper_cost_bps' in p else 'legacy estimated',
            'sector':p.get('market_context',{}).get('sector','Unknown'),
            'industry':p.get('market_context',{}).get('industry','Unknown'),
            'formulas':p.get('pillars_matched',[]), 'paper_only':True})
    return result

def tune(conn, rows, defaults, path, now):
    """Bounded formula adaptation, chronological holdout by full session, no neural training.
    The holdout checks individual formula stability, not counterfactual portfolio returns.
    """
    good=[r for r in rows if r['net_r'] is not None and r['cost_source']=='entry snapshot']
    days=sorted({r['day'] for r in good})
    report={'at':now.isoformat(),'status':'collecting','samples':len(good),'sessions':len(days),
            'method':'chronological session holdout, formula conditional net expectancy','formulas':{}}
    weights=dict(defaults)
    if len(days)>=20:
        split=days[max(1,int(len(days)*.7))]
        train=[r for r in good if r['day']<split]; val=[r for r in good if r['day']>=split]
        report.update({'validation_start':split,'train_n':len(train),'validation_n':len(val)})
        for f, base in defaults.items():
            a=[r['net_r'] for r in train if f in r['formulas']]
            b=[r['net_r'] for r in val if f in r['formulas']]
            item={'train_n':len(a),'validation_n':len(b),'promoted':False}
            if len(a)>=30 and len(b)>=15:
                ma,mb=mean(a),mean(b)
                # Conservative 3-SE stability screen; correlated trades remain a limitation.
                margin=3*stdev(b)/math.sqrt(len(b))
                stable=(ma>0 and mb-margin>0) or (ma<0 and mb+margin<0)
                item.update({'train_net_r':ma,'validation_net_r':mb,'validation_margin':margin,'promoted':stable})
                if stable: weights[f]=round(base*(1.15 if ma>0 else .85),4)
            report['formulas'][f]=item
        report['status']='validated' if any(i['promoted'] for i in report['formulas'].values()) else 'no_stable_change'
    report['formula_weights']=weights
    atomic_json(path,report)
    init(conn); conn.execute('INSERT INTO learning_audit(asof,payload) VALUES (?,?)',(now.isoformat(),json.dumps(report))); conn.commit()
    return report

def export_reports(root, conn, snap=None):
    out=Path(root)/'reports'; out.mkdir(exist_ok=True)
    rows=ledger(conn)
    if rows:
        with (out/'paper_trades.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    closed=[r for r in rows if r['net_r'] is not None]
    stats={'paper_only':True,'signals':len(rows),'resolved':len(closed),
           'wins':sum(r['net_r']>0 for r in closed),'losses':sum(r['net_r']<0 for r in closed),
           'flat':sum(r['net_r']==0 for r in closed),'pending':sum(r['result']=='PENDING' for r in rows),
           'no_data':sum(r['result']=='NO_DATA' for r in rows),
           'mean_net_r':mean(r['net_r'] for r in closed) if closed else None}
    atomic_json(out/'performance.json',stats)
    if snap is None:
        init(conn); latest=conn.execute('SELECT payload FROM market_snapshots ORDER BY asof DESC LIMIT 1').fetchone()
        snap=json.loads(latest[0]) if latest else {'generated_at':'No market snapshot yet','sectors':[],'stocks':{},'scope':'Run scan during market hours.'}
    atomic_json(out/'market_snapshot.json',snap)
    cards=''.join('<div class="card" style="background:'+('#164e3c' if s['change_pct']>=0 else '#702a38')+'"><b>'+html.escape(s['sector'])+'</b><h2>'+f"{s['change_pct']:+.2f}%"+'</h2>'+f"{s['count']} stocks · {s['advancing_pct']}% advancing · Change since previous: {s.get('change_since_previous_pp', 'N/A')} pp"+'</div>' for s in snap['sectors'])
    tr=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [s['symbol'],s['sector'],s['industry'],s['change_pct'],s['peer_relative_pct'],', '.join(p['symbol'] for p in s['peers'])])+'</tr>' for s in sorted(snap['stocks'].values(),key=lambda x:x['change_pct'],reverse=True))
    page='''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Bharat market monitor</title><style>body{font:16px system-ui;background:#101827;color:#eef3ff;margin:30px}small{color:#bac8de}.grid{display:flex;flex-wrap:wrap;gap:12px}.card{padding:18px;border-radius:12px;min-width:180px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #344055}input{padding:12px;margin:20px 0}a{color:#80ccff}</style><h1>Bharat market monitor</h1>'''
    page+='<p>Coverage: '+str(snap.get('fresh',0))+'/'+str(snap.get('requested',0))+' fresh stocks; '+str(snap.get('classified',0))+' classified.</p>'
    page+='<p>Snapshot: '+html.escape(snap['generated_at'])+'</p><small>'+html.escape(snap['scope'])+' · Downloaded HTML does not stream live updates.</small><div class="grid">'+cards+'</div>'
    page+='<h2>Paper outcomes after estimated costs</h2><p>'+html.escape(str(stats))+'</p><a href="paper_trades.csv">Trade ledger CSV</a><h2>Industry peers</h2><input id="q" placeholder="Filter symbol / industry"><table><thead><tr><th>Symbol</th><th>Sector</th><th>Industry</th><th>Day %</th><th>Vs peers pp</th><th>Peers</th></tr></thead><tbody>'+tr+'</tbody></table><script>document.querySelector("#q").oninput=e=>document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()));</script></html>'
    (out/'heatmap.html').write_text(page,encoding='utf-8')
    return stats
