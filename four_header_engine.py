"""Four original strategies integrated alongside (not replacing) scanner.py.
Independent ledger/delivery quotas prevent contamination of the legacy ledger.
All outputs are unvalidated PAPER RESEARCH. Fixed strategy evaluators are copied
byte-for-byte in strategy_rules_original.py; only LEARNING_SETUP's original
online update changes its own weights. Challengers never modify baseline rules.
"""
from __future__ import annotations
import hashlib
import csv
import html
import json
import logging
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
import numpy as np
import pandas as pd
import strategy_rules_original as rules
import strategy_learning

LOG=logging.getLogger(__name__)
HEADERS=rules.STRATEGY_HEADERS
EVALUATORS=dict(EARLIER_SETUP=rules.evaluate_earlier_setup, LEARNING_SETUP=rules.evaluate_learning_setup,
                **{'90PCT_SETUP_V9':rules.evaluate_setup_v9}, BEST_SETTING_CALL=rules.evaluate_best_setting_call)
CONFIGS={'EARLIER_SETUP':(2.,1.,False,0.),'LEARNING_SETUP':(1.5,1.,False,0.),
         '90PCT_SETUP_V9':(2.,1.5,True,.3),'BEST_SETTING_CALL':(1.5,1.,False,0.)}


def init(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS four_header_calls(
      id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, day TEXT NOT NULL, candle_time TEXT NOT NULL,
      strategy_header TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
      price REAL NOT NULL, stop REAL NOT NULL, target REAL NOT NULL, stop_pct REAL NOT NULL,
      target_pct REAL NOT NULL, trailing INTEGER NOT NULL, trigger_pct REAL NOT NULL,
      approved INTEGER NOT NULL, kind TEXT NOT NULL, regime TEXT NOT NULL, payload TEXT NOT NULL,
      record_key TEXT NOT NULL UNIQUE, status TEXT NOT NULL DEFAULT 'OPEN', outcome TEXT,
      gross_r REAL, cost_r REAL, net_r REAL, mfe_pct REAL, mae_pct REAL, minutes_to_exit INTEGER,
      ambiguity INTEGER NOT NULL DEFAULT 0, resolved_at TEXT, learned INTEGER NOT NULL DEFAULT 0,
      delivery TEXT NOT NULL DEFAULT 'BACKGROUND', telegram_message_id TEXT, delivery_attempts INTEGER NOT NULL DEFAULT 0,
      delivery_error TEXT)''')
    conn.execute('CREATE INDEX IF NOT EXISTS four_calls_open ON four_header_calls(status,symbol)')
    conn.execute('''CREATE TABLE IF NOT EXISTS four_header_cycles(
      asof TEXT PRIMARY KEY, payload TEXT NOT NULL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS four_weight_audit(
      call_id INTEGER PRIMARY KEY, at TEXT, before_weights TEXT, after_weights TEXT, outcome_net_r REAL)''')
    strategy_learning.init(conn)
    conn.commit()


def configure(db_path):
    # Original evaluator reads its own adaptive state from the same existing DB.
    rules.DB_PATH=Path(db_path)


def _insert(conn, cand, result, context, now, kind, approved):
    header=result['strategy_header']; entry=float(cand['price']);side=cand['side']
    target_pct=float(result['target_pct']);stop_pct=float(result['stop_pct'])
    if not math.isfinite(entry) or entry<=0 or side not in ('BUY','SELL'):return False
    signed=1 if side=='BUY' else -1
    stop=round(entry*(1-signed*stop_pct/100),2);target=round(entry*(1+signed*target_pct/100),2)
    if abs(stop-entry)<=0:return False
    candle=cand['payload']['candle_time']
    key=f"{now.date()}|{header}|{cand['symbol']}|{pd.Timestamp(candle).floor('30min').isoformat()}"
    p=dict(candidate=cand,evaluation=result,context=context,cost_bps=float(os.getenv('PAPER_ROUND_TRIP_BPS','20')),
           reference_price_only=True, rules_sha256=hashlib.sha256(Path(rules.__file__).read_bytes()).hexdigest())
    regime=context.get('intraday_regime','UNKNOWN')
    p['companion']=strategy_learning.annotate(conn,header,side,regime,str(now.date()),p)
    if header=='LEARNING_SETUP':p['entry_weights']=rules.get_learning_weights().tolist()
    cur=conn.execute('''INSERT OR IGNORE INTO four_header_calls(
      created_at,day,candle_time,strategy_header,symbol,side,price,stop,target,stop_pct,target_pct,trailing,
      trigger_pct,approved,kind,regime,payload,record_key,delivery)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
      (now.isoformat(),str(now.date()),candle,header,cand['symbol'],side,entry,stop,target,stop_pct,target_pct,
       int(result.get('trailing_stop',False)),float(result.get('profit_lock_trig_pct',0)),int(approved),kind,
       regime,json.dumps(p,allow_nan=False),key,'PENDING' if kind=='PAPER_ALERT' else 'BACKGROUND'))
    return bool(cur.rowcount)


def collect(conn, frames, entries, market, context, now, db_path):
    configure(db_path);init(conn)
    diagnostics=dict(at=now.isoformat(),candidates=0,errors=0,headers={},paper_only=True)
    if not rules.scan_window_open(now):return diagnostics
    pro=[]
    for entry in entries[:rules.CFG['INTRADAY_UNIVERSE_SIZE']]:
        symbol=entry['symbol'];frame=frames.get(symbol)
        if frame is None or frame.empty:continue
        try:
            d=rules.completed_bars(rules.normalize_intraday(frame),now)
            if len(d)<rules.CFG['MIN_HISTORY_BARS'] or d.index[-1].date()!=now.date():continue
            if (pd.Timestamp(now)-d.index[-1]).total_seconds()>rules.CFG['MAX_BAR_AGE_MIN']*60:continue
            prev=float(entry.get('previous_close') or d['close'].iloc[0])
            candidate=rules.build_pro_candidate_payload(symbol,d,entry,prev)
            if candidate:pro.append(candidate)
        except Exception as exc:
            diagnostics['errors']+=1
            LOG.warning('Four-header candidate %s failed: %s',symbol,type(exc).__name__)
    diagnostics['candidates']=len(pro)
    limit=rules.CFG['MAX_SIGNALS_PER_STRATEGY_DAY']
    for header,fn in EVALUATORS.items():
        matched=[];rejected=[];errors=0
        for candidate in pro:
            try:
                res=fn(candidate)
                if res.get('approved'):matched.append((res['conviction'],candidate,res))
                else:rejected.append((res.get('conviction',0.),candidate,res))
            except Exception as exc:
                errors+=1;LOG.warning('Four-header evaluation %s: %s',header,type(exc).__name__)
        matched.sort(key=lambda x:(-x[0],x[1]['time'],x[1]['symbol']))
        added=0;background=0
        for _,cand,res in matched:
            count=conn.execute("SELECT count(*) FROM four_header_calls WHERE day=? AND strategy_header=? AND kind='PAPER_ALERT'",(str(now.date()),header)).fetchone()[0]
            prior=conn.execute("SELECT created_at FROM four_header_calls WHERE day=? AND strategy_header=? AND symbol=? AND kind='PAPER_ALERT'",(str(now.date()),header,cand['symbol'])).fetchall()
            eligible=count<limit and len(prior)<2 and all((pd.Timestamp(now)-pd.Timestamp(r[0])).total_seconds()>=1800 for r in prior)
            if not eligible and background>=2:continue
            kind='PAPER_ALERT' if eligible else 'APPROVED_BACKGROUND'
            local=dict(context);local['peers']=context.get('stocks',{}).get(cand['symbol'],{})
            local.pop('stocks',None)
            if _insert(conn,cand,res,local,now,kind,True):
                added+=int(eligible);background+=int(not eligible)
        # Small fixed rejected sample, not every stock x every candle. No alert.
        for _,cand,res in sorted(rejected,key=lambda x:(-x[0],x[1]['symbol']))[:2]:
            target,stop,trail,trig=CONFIGS[header]
            hypothetical=dict(strategy_header=header,target_pct=target,stop_pct=stop,trailing_stop=trail,
                              profit_lock_trig_pct=trig,conviction=res.get('conviction',0),setup_desc='Rejected original setup; hypothetical outcome')
            local=dict(context);local['peers']=context.get('stocks',{}).get(cand['symbol'],{});local.pop('stocks',None)
            _insert(conn,cand,hypothetical,local,now,'REJECTED_STUDY',False)
        diagnostics['headers'][header]=dict(approved=len(matched),rejected=len(rejected),errors=errors,
                                             queued=added,quota=limit,background=background)
    conn.execute('INSERT OR REPLACE INTO four_header_cycles VALUES(?,?)',(now.isoformat(),json.dumps(diagnostics)))
    conn.execute('DELETE FROM four_header_cycles WHERE asof<?',((pd.Timestamp(now)-pd.Timedelta(days=40)).isoformat(),))
    conn.commit();return diagnostics


def grade(row, frame, now):
    """Reference-price simulation, full bars starting AFTER recording. Stop first
    on ambiguous OHLC. A trail armed inside a bar only applies from the next bar;
    no profit is invented from the unknown high/low order of a single candle.
    """
    d=rules.completed_bars(rules.normalize_intraday(frame),now)
    created=pd.Timestamp(row['created_at']).tz_convert(rules.TZ);first=created.ceil('5min')
    end=first.normalize()+pd.Timedelta(hours=15,minutes=15)
    if first>end:return None
    after=d[(d.index>=first)&(d.index<=end)&(d.index.date==first.date())]
    if after.empty:return None
    expected=pd.date_range(first,after.index[-1],freq='5min')
    if not expected.isin(after.index).all():return None
    cols=['open','high','low','close']
    if not np.isfinite(after[cols].to_numpy(dtype=float)).all():return None
    entry=row['price'];direction=1 if row['side']=='BUY' else -1
    stop=row['stop'];target=row['target'];armed=False;mfe=mae=0.;ambiguity=0
    exit_price=None;outcome=None;exit_at=None
    for ts,b in after.iterrows():
        hi,lo,op,close=map(float,(b['high'],b['low'],b['open'],b['close']))
        fav=(hi-entry if direction==1 else entry-lo)/entry*100
        adv=(entry-lo if direction==1 else hi-entry)/entry*100
        mfe=max(mfe,fav);mae=max(mae,adv)
        hit_stop=lo<=stop if direction==1 else hi>=stop
        hit_target=hi>=target if direction==1 else lo<=target
        ambiguity|=int(hit_stop and hit_target)
        if hit_stop:
            exit_price=min(op,stop) if direction==1 else max(op,stop)
            outcome='TRAIL_PROFIT_LOCK' if armed and direction*(exit_price-entry)>0 else 'STOP_LOSS'
            exit_at=ts;break
        if hit_target:exit_price=target;outcome='TARGET_HIT';exit_at=ts;break
        if row['trailing'] and not armed and fav>=row['trigger_pct']:
            lock=entry*(1+direction*row['trigger_pct']/100)
            # Price may already have crossed the new lock in this candle.
            ambiguity|=int(lo<=lock if direction==1 else hi>=lock)
            stop=lock;armed=True
        if ts>=end:
            exit_price=close;outcome='EOD';exit_at=ts;break
    if exit_price is None:return None
    risk=abs(row['price']-row['stop']);gross=direction*(exit_price-entry)/risk
    payload=json.loads(row['payload']);cost=entry*payload['cost_bps']/10000/risk;net=gross-cost
    return dict(outcome=outcome,gross_r=gross,cost_r=cost,net_r=net,mfe_pct=mfe,mae_pct=mae,
                minutes_to_exit=int((exit_at-created).total_seconds()/60),ambiguity=ambiguity,
                quality='GREAT' if outcome=='TARGET_HIT' and net>0 and mae<row['stop_pct']/2 else 'GOOD' if net>0 else 'BAD')


def resolve(conn, frames, now, db_path):
    configure(db_path);init(conn);closed=0
    for row in conn.execute("SELECT * FROM four_header_calls WHERE status='OPEN' ORDER BY id").fetchall():
        frame=frames.get(row['symbol'])
        result=grade(row,frame,now) if frame is not None and not frame.empty else None
        if result:
            payload=json.loads(row['payload']);payload['quality']=result.pop('quality')
            columns=','.join(key+'=?' for key in result)
            conn.execute('UPDATE four_header_calls SET status=\'CLOSED\',resolved_at=?,payload=?,'+columns+' WHERE id=?',
                         (now.isoformat(),json.dumps(payload),*result.values(),row['id']))
            closed+=1
        elif (now.date()-pd.Timestamp(row['created_at']).date()).days>9:
            conn.execute("UPDATE four_header_calls SET status='NO_DATA',resolved_at=? WHERE id=?",(now.isoformat(),row['id']))
    conn.commit()
    # Exactly-once original logistic update. Frozen fixed-rule strategies never change.
    for row in conn.execute("SELECT * FROM four_header_calls WHERE status='CLOSED' AND strategy_header='LEARNING_SETUP' AND kind='PAPER_ALERT' AND learned=0 ORDER BY id").fetchall():
        raw=conn.execute("SELECT value FROM app_state WHERE key='learning_weights_30d'").fetchone()
        w=np.array(json.loads(raw[0]),dtype=float) if raw else rules.LOCKED_WEIGHTS_V9.copy()
        payload=json.loads(row['payload']);x=rules.extract_live_30d_features(payload['candidate'])
        pred=1/(1+math.exp(-float(np.clip(np.dot(w,x),-15,15))))
        after=np.clip(w+.04*((1. if row['net_r']>0 else 0.)-pred)*x,-3,3)
        after=np.array([round(float(value),5) for value in after])
        conn.execute("INSERT OR REPLACE INTO app_state VALUES('learning_weights_30d',?)",(json.dumps(after.tolist()),))
        conn.execute('INSERT INTO four_weight_audit VALUES(?,?,?,?,?)',(row['id'],now.isoformat(),json.dumps(w.tolist()),json.dumps(after.tolist()),row['net_r']))
        conn.execute('UPDATE four_header_calls SET learned=1 WHERE id=?',(row['id'],))
        conn.commit()
    return closed


def deliver(conn, now, send, limit=8):
    init(conn);sent=0
    for row in conn.execute("SELECT * FROM four_header_calls WHERE delivery='PENDING' ORDER BY id LIMIT ?",(limit,)).fetchall():
        age=(pd.Timestamp(now)-pd.Timestamp(row['created_at'])).total_seconds()
        if age<0 or age>900 or row['day']!=str(now.date()):
            conn.execute("UPDATE four_header_calls SET delivery='EXPIRED' WHERE id=?",(row['id'],));continue
        p=json.loads(row['payload']);context=p['context'];regime=context.get('intraday_regime','UNKNOWN')
        msg=(f"[{row['strategy_header']}] {row['symbol']} {row['side']}\n"
             f"PAPER RESEARCH — strategy not validated\n"
             f"Reference ₹{row['price']:.2f} | Stop ₹{row['stop']:.2f} | Target ₹{row['target']:.2f}\n"
             f"Target {row['target_pct']}% | Stop {row['stop_pct']}%"
             +(f" | Lock trigger {row['trigger_pct']}%" if row['trailing'] else ' | No trail')+'\n'
             f"Intraday market: {regime} | peers: {p['context'].get('peers',{}).get('peer_count',0)}\n"
             f"RVOL {p['candidate'].get('relative_volume')}x | model score {p['evaluation'].get('conviction',0):.3f} (not win probability)\n"
             'No broker order. Results use estimated costs and conservative candle grading.')
        conn.execute('UPDATE four_header_calls SET delivery_attempts=delivery_attempts+1 WHERE id=?',(row['id'],));conn.commit()
        try:
            result=send(msg)
            if result is not True:raise RuntimeError('Sender did not confirm success')
            conn.execute("UPDATE four_header_calls SET delivery='SENT',delivery_error=NULL WHERE id=?",(row['id'],));conn.commit();sent+=1
        except Exception as exc:
            conn.execute('UPDATE four_header_calls SET delivery_error=? WHERE id=?',(type(exc).__name__,row['id']));conn.commit()
            LOG.warning('Four-header Telegram failed: %s',type(exc).__name__);break
    conn.commit();return sent


def _stats(rows):
    closed=[r for r in rows if r['status']=='CLOSED'];vals=[r['net_r'] for r in closed]
    profit=sum(max(0,x) for x in vals);loss=-sum(min(0,x) for x in vals)
    equity=peak=drawdown=0.
    for r in sorted(closed,key=lambda r:(r.get('resolved_at') or '',r['id'])):
        equity+=r['net_r'];peak=max(peak,equity);drawdown=max(drawdown,peak-equity)
    return dict(recorded=len(rows),closed=len(closed),pending=sum(r['status']=='OPEN' for r in rows),
                no_data=sum(r['status']=='NO_DATA' for r in rows),sent=sum(r['delivery']=='SENT' for r in rows),
                wins=sum(x>0 for x in vals),losses=sum(x<0 for x in vals),
                win_rate_pct=100*sum(x>0 for x in vals)/len(vals) if vals else None,
                net_r=sum(vals),mean_net_r=mean(vals) if vals else None,profit_factor=profit/loss if loss else None,
                max_closed_trade_drawdown_r=drawdown,ambiguous=sum(r['ambiguity'] for r in closed),
                quality=dict(Counter(json.loads(r['payload']).get('quality','UNKNOWN') for r in closed)))


def report(conn, root, now, learn=True):
    init(conn)
    policies=strategy_learning.refresh(conn,now) if learn else [dict(r) for r in conn.execute('SELECT * FROM four_learning_policies')]
    rows=[dict(r) for r in conn.execute('SELECT * FROM four_header_calls ORDER BY day,id')]
    report=dict(asof=now.isoformat(),paper_only=True,baseline_rules='original; unchanged',
                note='Reference-price studies, not executable fills; correlated trades are not independent. Background samples are capped and are not an exhaustive opportunity set.',headers={},policies=policies)
    for header in HEADERS:
        group=[r for r in rows if r['strategy_header']==header]
        real=[r for r in group if r['kind']=='PAPER_ALERT']
        by_regime={}
        for regime in sorted({r['regime'] for r in real}):
            for side in ('BUY','SELL'):
                subset=[r for r in real if r['regime']==regime and r['side']==side]
                if subset:by_regime[regime+'|'+side]=_stats(subset)
        lessons=defaultdict(list)
        for r in real:
            if r['status']!='CLOSED':continue
            p=json.loads(r['payload']);c=p['context'];peers=c.get('peers',{})
            labels=[f"intraday:{r['regime']}",f"side:{r['side']}",f"background:{c.get('nifty',{}).get('background','UNKNOWN')}",
                    f"VIX:{c.get('vix_bucket','UNKNOWN')}"]
            labels.append('RVOL>=2.2' if p['candidate'].get('relative_volume',0)>=2.2 else 'RVOL<2.2')
            peer=peers.get('stock_vs_peers_pp')
            if peer is not None:labels.append('peer_aligned' if (peer>=0 if r['side']=='BUY' else peer<=0) else 'peer_opposed')
            for label in labels:lessons[label].append(r['net_r'])
        report['headers'][header]=dict(paper_alerts=_stats(real),sent_alerts=_stats([r for r in real if r['delivery']=='SENT']),
             approved_background=_stats([r for r in group if r['kind']=='APPROVED_BACKGROUND']),
             rejected_study=_stats([r for r in group if r['kind']=='REJECTED_STUDY']),by_regime=by_regime,
             observed_associations={k:dict(n=len(v),mean_net_r=mean(v),note='association, not causation') for k,v in lessons.items()},
             latest_calls=[dict(symbol=r['symbol'],side=r['side'],day=r['day'],outcome=r['outcome'],net_r=r['net_r'],
                                quality=json.loads(r['payload']).get('quality'),regime=r['regime']) for r in real[-5:]])
    latest=conn.execute('SELECT payload FROM four_header_cycles ORDER BY asof DESC LIMIT 1').fetchone()
    report['latest_cycle']=json.loads(latest[0]) if latest else None
    path=Path(root)/'reports/four_header_performance.json';path.parent.mkdir(exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False));tmp.replace(path)
    fields=['id','created_at','strategy_header','symbol','side','kind','regime','price','stop','target','status','outcome','gross_r','cost_r','net_r','ambiguity','delivery']
    with (path.parent/'four_header_ledger.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    body=['<!doctype html><meta charset="utf-8"><title>Four-header paper results</title>',
          '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px;background:#101827;color:#eef3ff}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:12px;border-bottom:1px solid #526078}a{color:#87ccff}</style>',
          '<h1>Four-header paper results</h1><p>'+html.escape(report['asof'])+'</p>',
          '<p>Original fixed strategy rules retained. Estimated costs. No broker fills and no claimed 90% accuracy.</p>',
          '<table><tr><th>Header</th><th>Sent</th><th>Graded</th><th>Win rate</th><th>Net R</th><th>Pending / no data</th></tr>']
    for header,item in report['headers'].items():
        stat=item['sent_alerts'];rate=f"{stat['win_rate_pct']:.1f}%" if stat['win_rate_pct'] is not None else 'Not measured'
        cells=[header,stat['sent'],stat['closed'],rate,f"{stat['net_r']:+.2f}",f"{item['paper_alerts']['pending']} / {item['paper_alerts']['no_data']}"]
        body.append('<tr>'+''.join('<td>'+html.escape(str(cell))+'</td>' for cell in cells)+'</tr>')
    body+=['</table><h2>Forward learning</h2><p>'+html.escape(str(dict(Counter(p['status'] for p in policies)) or 'Collecting evidence; no setting promoted.'))+'</p>',
           '<p>Market-regime and side breakdowns, observations and policy audit are in the JSON report.</p>',
           '<p><a href="four_header_ledger.csv">Trade ledger CSV</a> · <a href="four_header_performance.json">Full report JSON</a></p>']
    (path.parent/'four_header_performance.html').write_text(''.join(body))
    return report


def digest(report):
    lines=['FOUR-HEADER PAPER LEARNING — '+report['asof'][:10]]
    for header,item in report['headers'].items():
        s=item['sent_alerts'];wr=f"{s['win_rate_pct']:.1f}%" if s['win_rate_pct'] is not None else 'not measured'
        lines.append(f"[{header}] sent {s['sent']} | graded {s['closed']} | W/L {s['wins']}/{s['losses']} | win {wr} | net {s['net_r']:+.2f}R")
        states=dict(Counter(p['status'] for p in report['policies'] if p['header']==header))
        lines.append(f"  Pending {item['paper_alerts']['pending']} | missing {item['paper_alerts']['no_data']} | settings {states or 'collecting'}")
        lines.append(f"  Graded quality: {item['paper_alerts']['quality']}")
        cycle=(report.get('latest_cycle') or {}).get('headers',{}).get(header,{})
        lines.append(f"  Latest cycle: approved {cycle.get('approved',0)}, rejected {cycle.get('rejected',0)}, errors {cycle.get('errors',0)}")
        top=sorted(item['observed_associations'].items(),key=lambda x:x[1]['mean_net_r'])
        if top:lines.append(f"  Weakest association: {top[0][0]} ({top[0][1]['n']} samples, {top[0][1]['mean_net_r']:+.2f}R)")
        for call in item['latest_calls'][-2:]:
            value=f"{call['net_r']:+.2f}R" if call['net_r'] is not None else 'pending'
            lines.append(f"  {call['symbol']} {call['side']} {call['quality'] or 'UNRESOLVED'} {value} ({call['regime']})")
    lines+=['Fixed strategies unchanged. New settings are forward-tested in a separate paper companion.',
            'No 90% accuracy claim. Missing data excluded; costs estimated.']
    return '\n'.join(lines)
