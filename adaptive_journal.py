"""Evidence-first shadow validation for Bharat Intraday Scout.

No broker execution, neural training, or claims of causal attribution.
All statistics are paper-trade estimates and depend on completed market bars.
"""
from __future__ import annotations
import json
import math
import os
import sqlite3
from datetime import datetime, time as dtime
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo
import pandas as pd
import market_learning as ml

TZ = ZoneInfo('Asia/Kolkata')


def _i(name, default):
    return int(os.getenv(name, str(default)))


def init(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS watchlist (
        day TEXT NOT NULL, symbol TEXT NOT NULL, observed_at TEXT NOT NULL,
        price REAL NOT NULL, side TEXT NOT NULL, payload TEXT NOT NULL,
        result TEXT, move_pct REAL,
        PRIMARY KEY(day, symbol)
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS kronos_accuracy (
        signal_id INTEGER PRIMARY KEY, target_at TEXT NOT NULL,
        prediction_pct REAL NOT NULL, observed_pct REAL NOT NULL,
        correct_direction INTEGER NOT NULL, abs_error_pct REAL NOT NULL
    )''')
    conn.commit()


def record_watch(conn, watch, now):
    init(conn)
    for p in watch:
        conn.execute('''INSERT OR IGNORE INTO watchlist
            (day,symbol,observed_at,price,side,payload) VALUES (?,?,?,?,?,?)''',
            (now.date().isoformat(),p['symbol'],now.isoformat(),float(p['price']),p['side'],json.dumps(p)))
    conn.commit()


def resolve_watch(conn, frames, now):
    """Record forward maximum favorable excursion; NOT a simulated executable trade."""
    init(conn)
    results=[]
    for day,symbol,seen,price,side in conn.execute('''SELECT day,symbol,observed_at,price,side
        FROM watchlist WHERE result IS NULL''').fetchall():
        frame=frames.get(symbol)
        if frame is None or frame.empty: continue
        df=frame.copy()
        idx=pd.to_datetime(df.index)
        df.index=idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)
        df=df[~df.index.duplicated()].sort_index()
        ts=pd.Timestamp(seen)
        ts=ts.tz_localize(TZ) if ts.tz is None else ts.tz_convert(TZ)
        if ts.date() >= now.date() and now.time() < dtime(15,30): continue
        after=df[(df.index>ts) & (df.index.date==ts.date())].between_time('09:15','15:25')
        # Missing any intermediate 5m candle makes the watch ungradable, not a failure.
        expected=pd.date_range(ts.floor('5min')+pd.Timedelta(minutes=5),
                               ts.normalize()+pd.Timedelta(hours=15,minutes=25), freq='5min')
        if after.empty or len(expected)==0 or not expected.isin(after.index).all(): continue
        if side=='BUY': favorable=float(after.high.max()/price-1)*100
        else: favorable=float(1-after.low.min()/price)*100
        result='REACHED_2PCT' if favorable>=2.0 else 'DID_NOT_REACH_2PCT'
        conn.execute('UPDATE watchlist SET result=?,move_pct=? WHERE day=? AND symbol=?',
                     (result,round(favorable,3),day,symbol))
        results.append({'symbol':symbol,'day':day,'outcome':result,'max_favorable_pct':round(favorable,2)})
    conn.commit()
    return results


def resolve_kronos(conn, frames):
    """Score frozen 30-minute closing forecasts against observed later candles.
    Forecasts with unavailable/incomplete follow-up are excluded, never scored wrong.
    """
    init(conn)
    rows=conn.execute('''SELECT s.id,s.symbol,s.payload FROM signals s
       LEFT JOIN kronos_accuracy k ON k.signal_id=s.id WHERE k.signal_id IS NULL''').fetchall()
    graded=[]
    for rec in rows:
        sid,symbol,raw=rec
        p=json.loads(raw)
        forecast=p.get('forecast') or {}
        if forecast.get('model')!='Kronos-mini' or forecast.get('forecast_change_pct') is None: continue
        df=frames.get(symbol)
        if df is None or df.empty: continue
        df=df.copy(); idx=pd.to_datetime(df.index)
        df.index=idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)
        df=df[~df.index.duplicated()].sort_index()
        ct=pd.Timestamp(p['candle_time']); ct=ct.tz_localize(TZ) if ct.tz is None else ct.tz_convert(TZ)
        horizon=int(forecast.get('horizon_bars',6))
        if horizon <= 0 or horizon > 24: continue
        target=ct+pd.Timedelta(minutes=5*horizon)
        if target.date()!=ct.date() or target.time()>datetime.strptime('15:25','%H:%M').time(): continue
        if target not in df.index: continue
        actual=(float(df.loc[target,'close'])/float(p['price'])-1)*100
        predicted=float(forecast['forecast_change_pct'])
        correct=int((actual>0 and predicted>0) or (actual<0 and predicted<0))
        err=abs(actual-predicted)
        conn.execute('''INSERT OR IGNORE INTO kronos_accuracy
           (signal_id,target_at,prediction_pct,observed_pct,correct_direction,abs_error_pct)
           VALUES (?,?,?,?,?,?)''',(sid,target.isoformat(),predicted,round(actual,4),correct,round(err,4)))
        graded.append({'symbol':symbol,'actual_pct':round(actual,3),'predicted_pct':predicted,'correct':bool(correct)})
    conn.commit()
    return graded


def _confidence_report(conn, now):
    """Evidence gate. Strictly uses *chronologically held-out*, cost-adjusted paper trades.
    A 14-day review is not automatic promotion. No-data, old unverified cost
    assumptions, and pending trades never count towards promotion.
    """
    try:
        all_costed=[r for r in ml.ledger(conn) if r['net_r'] is not None and r['cost_source']=='entry snapshot']
        payloads={sid:json.loads(raw) for sid,raw in conn.execute('SELECT id,payload FROM signals')}
        # Promotion must be based on the SAME high-confluence criteria that
        # would be delivered tomorrow, not on loose/failed shadow setups.
        records=[r for r in all_costed if confirmed_gate(
            payloads.get(r['id'],{}),
            {'sectors':[payloads.get(r['id'],{}).get('sector_context') or {}]},
            {'status':'VALIDATED_PAPER_MODE','minimum_rvol_for_validated_alert':1.8})[0]]
    except sqlite3.OperationalError:
        all_costed=[];records=[];payloads={}
    days=sorted({r['day'] for r in records if r['day'] <= now.date().isoformat()})
    min_sessions=_i('OBSERVATION_MIN_SESSIONS',12)
    min_samples=_i('ACTIVATION_MIN_TRADES',35)
    val_min=_i('ACTIVATION_MIN_HOLDOUT',12)
    min_net=float(os.getenv('ACTIVATION_MIN_NET_R','0.10'))
    train=[]; hold=[]
    if len(days)>=2:
        split=days[max(1,int(len(days)*0.7))]
        train=[r for r in records if r['day']<split]
        hold=[r for r in records if r['day']>=split]
    # Tune ONLY on the chronological training segment. Test candidate thresholds on
    # a later untouched holdout; revert to the safe baseline when uncertain.
    min_rvol=1.8
    suggested=None
    if len(train)>=30 and len(hold)>=15:
        def get_rv(row):
            return float(payloads.get(row['id'],{}).get('relative_volume') or 0)
        baseline_hold=[r['net_r'] for r in hold if get_rv(r)>=min_rvol]
        base_avg=mean(baseline_hold) if len(baseline_hold)>=10 else None
        choices=[]
        for floor in (1.8,2.0,2.2):
            a=[r['net_r'] for r in train if get_rv(r)>=floor]
            if len(a)>=25:
                choices.append((mean(a),floor,len(a)))
        if choices:
            best=max(choices,key=lambda t:t[0])
            test=[r['net_r'] for r in hold if get_rv(r)>=best[1]]
            if (best[1]>1.8 and len(test)>=10 and base_avg is not None
                    and mean(test)>=base_avg+0.05 and mean(test)>=0.10):
                suggested={'rvol_floor':best[1], 'train_n':best[2],
                           'holdout_n':len(test),
                           'holdout_net_r':round(mean(test),3),
                           'base_holdout_net_r':round(base_avg,3)}
                min_rvol=best[1]
    kr=conn.execute('SELECT COUNT(*),SUM(correct_direction),AVG(abs_error_pct) FROM kronos_accuracy').fetchone()
    kn=int(kr[0]); direction_accuracy=float(kr[1] or 0)/kn if kn else None
    net=mean([r['net_r'] for r in hold]) if hold else None
    wins=sum(r['net_r']>0 for r in hold)
    # Conservative: positive paper expectancy, nontrivial sample size, and Kronos
    # evidence must ALL be present. Positive past expectancy does not assure future gains.
    checks={
        'min_sessions':len(days)>=min_sessions,
        'min_trades':len(records)>=min_samples,
        'holdout_size':len(hold)>=val_min,
        'holdout_net_expectancy':net is not None and net>=min_net,
        'holdout_win_rate':len(hold)>0 and wins/len(hold)>=0.40,
        'kronos_evidence':kn>=_i('ACTIVATION_MIN_KRONOS',15) and direction_accuracy is not None and direction_accuracy>=float(os.getenv('ACTIVATION_MIN_KRONOS_ACCURACY','0.53')),
    }
    return {'status':'VALIDATED_PAPER_MODE' if all(checks.values()) else 'RESEARCH_ONLY',
        'sessions':len(days),'completed_costed_trades':len(records),
        'all_costed_research_trades':len(all_costed),
        'holdout_trades':len(hold),'holdout_mean_net_r':round(net,4) if net is not None else None,
        'holdout_win_rate_pct':round(100*wins/len(hold),1) if hold else None,
        'kronos_evaluated':kn,'kronos_direction_accuracy_pct':round(100*direction_accuracy,1) if direction_accuracy is not None else None,
        'kronos_mae_pct':round(float(kr[2]),3) if kr[2] is not None else None,
        'minimum_rvol_for_validated_alert':min_rvol,'rvol_tuning':suggested,
        'checks':checks,'notes':'Paper evidence, not certainty. 10-14 trading days is a review milestone, not a guaranteed activation.'}


def mode(conn, now):
    init(conn)
    return _confidence_report(conn, now)


from scout_research import alternative_support


def confirmed_gate(signal, market, activation):
    """Independent high-confluence live alert gate; rejected setups still become shadow signals."""
    score = int(signal.get('score', 0))
    if activation['status'] != 'VALIDATED_PAPER_MODE':
        fore = signal.get('forecast') or {}
        rvol = float(signal.get('relative_volume', 0))
        cum = float(signal.get('cumulative_relative_volume', 0))
        stock = signal.get('market_context') or {}
        sector = stock.get('sector')
        sec = next((s for s in market.get('sectors', []) if s['sector'] == sector), None)
        if signal.get('tier') != 'TIER-1' and not (signal.get('tier') == 'TIER-2' and score >= 90):
            return False, 'research: tier/score'
        if score < 85: return False, 'research: score < 85'
        if rvol < 1.8 or cum < 1.2: return False, 'research: rvol floor'
        if fore.get('model') != 'Kronos-mini' or not (fore.get('direction_aligned') or alternative_support(signal)):
            return False, 'research: Kronos disagreed'
        if not sector or sector == 'Unknown': return False, 'research: sector missing'
        if not sec or sec.get('count', 0) < 3: return False, 'research: thin sector'
        peers = stock.get('peer_relative_pct')
        if peers is None or stock.get('peer_count', 0) < 2:
            return False, 'research: not enough same-industry peers'
        if signal['side'] == 'BUY' and not (sec['change_pct'] > 0 and sec['advancing_pct'] >= 55 and peers >= 0):
            return False, 'research: weak sector/peer'
        if signal['side'] == 'SELL' and not (sec['change_pct'] < 0 and sec['advancing_pct'] <= 45 and peers <= 0):
            return False, 'research: weak sector/peer'
        if not fore.get('direction_aligned'):
            return True, 'RESEARCH PREVIEW — drift-supported, Kronos inconclusive (paper-only)'
        return True, 'RESEARCH PREVIEW — not validated (paper-only)'
    if not (signal.get('tier') == 'TIER-1' and score >= 82) and score < 90:
        return False, 'tier/score'
    if signal.get('relative_volume',0)<activation.get('minimum_rvol_for_validated_alert',1.8) or signal.get('cumulative_relative_volume',0)<1.3:
        return False,'insufficient relative volume'
    fore=signal.get('forecast') or {}
    if fore.get('model')!='Kronos-mini' or not fore.get('direction_aligned'):
        return False,'Kronos unavailable/disagreed'
    stock=signal.get('market_context') or {}
    sector=stock.get('sector')
    if not sector or sector=='Unknown': return False,'sector metadata missing'
    peers=stock.get('peer_relative_pct')
    if peers is None or stock.get('peer_count',0)<2: return False,'not enough same-industry peers'
    sec=next((s for s in market.get('sectors',[]) if s['sector']==sector),None)
    if not sec or sec.get('count',0)<3: return False,'thin sector coverage'
    if signal['side']=='BUY':
        if not (sec['change_pct']>0 and sec['advancing_pct']>=55 and peers>=0):
            return False,'weak sector/peer relative strength'
    else:
        if not (sec['change_pct']<0 and sec['advancing_pct']<=45 and peers<=0):
            return False,'weak sector/peer relative weakness'
    return True,'validated high-confluence paper setup'


def _evidence_tags(p, net, forecast_correct):
    tags=[]
    rvol=float(p.get('relative_volume') or 0)
    if rvol>=2: tags.append('strong same-time relative volume')
    if float(p.get('cumulative_relative_volume') or 0)>=1.5: tags.append('broad session-volume participation')
    bw=p.get('bb_bandwidth_pct')
    if bw is not None and float(bw)<=1.6: tags.append('tight Bollinger compression')
    market=p.get('market_context') or {}
    peer=market.get('peer_relative_pct')
    if peer is not None: tags.append('sector peer outperformance' if (float(peer)>0 and p['side']=='BUY') or (float(peer)<0 and p['side']=='SELL') else 'peer relative weakness')
    if forecast_correct is not None: tags.append('Kronos direction correct' if forecast_correct else 'Kronos direction wrong')
    if p.get('news_headlines'): tags.append('recent news available; catalyst not independently confirmed')
    if p.get('research_rejection'): tags.append('shadow setup rejected: '+str(p['research_rejection']))
    if net<=0: tags.append('net-of-cost paper loss; review stop, regime, volume follow-through')
    else: tags.append('net-of-cost paper gain; test repeated setup before weight promotion')
    return tags


def export(root,conn,now,frames=None):
    init(conn)
    if frames is not None:
        watch=resolve_watch(conn,frames,now)
        preds=resolve_kronos(conn,frames)
    else: watch=[];preds=[]
    activation=_confidence_report(conn,now)
    ledger=ml.ledger(conn)
    eval_map={r[0]:r[1] for r in conn.execute('SELECT signal_id,correct_direction FROM kronos_accuracy')}
    signal_map={r[0]:json.loads(r[1]) for r in conn.execute('SELECT id,payload FROM signals')}
    from scout_target_study import evaluate
    try:
        target_study = evaluate(ledger, signal_map)
    except Exception as exc:
        target_study = {'status':'unavailable', 'error':type(exc).__name__, 'paper_only':True}
    paired = []
    for sid, actual, kronos_pct in conn.execute(
        'SELECT signal_id,observed_pct,prediction_pct FROM kronos_accuracy'
    ):
        payload = signal_map.get(sid, {})
        drift = payload.get('drift_forecast') or {}
        if (actual is None or kronos_pct is None or drift.get('forecast_change_pct') is None
                or drift.get('asof') != payload.get('candle_time')
                or drift.get('horizon_bars') != (payload.get('forecast') or {}).get('horizon_bars')):
            continue
        paired.append((actual, kronos_pct, drift['forecast_change_pct']))
    comparison = {'n':len(paired)}
    if paired:
        for i, model in [(1,'kronos'),(2,'drift')]:
            comparison[model+'_direction_pct'] = round(100*mean(r[0]*r[i]>0 for r in paired),1)
            comparison[model+'_mae_pct'] = round(mean(abs(r[0]-r[i]) for r in paired),3)
    items=[]
    for r in ledger:
        if r['net_r'] is None:continue
        p=signal_map[r['id']]
        notes=_evidence_tags(p,r['net_r'],eval_map.get(r['id']))
        items.append({'date':r['day'],'symbol':r['symbol'],'side':r['side'],
                      'result':r['result'],'net_r':round(r['net_r'],3),'evidence_tags':notes,
                      'paper_only':True})
    watches=conn.execute("SELECT result,COUNT(*) FROM watchlist GROUP BY result").fetchall()
    by_formula={}
    for f in sorted({x for r in ledger if r['net_r'] is not None for x in r['formulas']}):
        vals=[r['net_r'] for r in ledger if r['net_r'] is not None and f in r['formulas']]
        by_formula[f]={'n':len(vals),'mean_net_r':round(mean(vals),3) if vals else None,
                       'win_pct':round(100*sum(v>0 for v in vals)/len(vals),1) if vals else None,
                       'note':'conditional association only; not proof that the formula caused returns'}
    # Indicator breakdowns include all closed observations, not just winning trades.
    def dimensions(p):
        labels=[]
        rvol=float(p.get('relative_volume') or 0)
        if rvol>=2: labels.append('RVOL >= 2x')
        elif rvol>=1.5: labels.append('RVOL 1.5-2x')
        else: labels.append('RVOL below 1.5x')
        cumulative=float(p.get('cumulative_relative_volume') or 0)
        labels.append('Session RVOL >= 1.5x' if cumulative>=1.5 else 'Session RVOL < 1.5x')
        bb=float(p.get('bb_bandwidth_pct') or 99)
        labels.append('BB compression <= 1.6%' if bb<=1.6 else 'BB width > 1.6%')
        adx=float(p.get('adx') or 0)
        labels.append('ADX >= 25' if adx>=25 else 'ADX < 25')
        rsi=float(p.get('rsi') or 50)
        labels.append('RSI >= 65' if rsi>=65 else 'RSI < 65')
        mfi=float(p.get('mfi') or 50)
        labels.append('MFI >= 65' if mfi>=65 else 'MFI < 65')
        fore=p.get('forecast') or {}
        labels.append('Kronos aligned' if fore.get('direction_aligned') else 'Kronos unavailable/misaligned')
        ctx=p.get('market_context') or {}
        peer=ctx.get('peer_relative_pct')
        if peer is not None:
            leader=(float(peer)>0 and p.get('side')=='BUY') or (float(peer)<0 and p.get('side')=='SELL')
            labels.append('Ahead of industry peers' if leader else 'Behind industry peers')
        try:
            hour=pd.Timestamp(p['candle_time']).hour
            labels.append('Morning setup' if hour<11 else 'Afternoon setup')
        except (KeyError,ValueError): pass
        return labels
    feature_values={}
    for r in ledger:
        if r['net_r'] is None: continue
        for key in dimensions(signal_map[r['id']]):
            feature_values.setdefault(key,[]).append(r['net_r'])
    feature_scorecard={k:{'n':len(v),'mean_net_r':round(mean(v),3),
                          'win_pct':round(100*sum(x>0 for x in v)/len(v),1)}
                       for k,v in sorted(feature_values.items())}
    report={'generated_at':now.isoformat(),'activation':activation,'graded_watch_today':watch,
            'graded_kronos_today':preds, 'paired_forecasts':comparison, 'target_hit_study':target_study,
            'watch_totals':{str(k) if k is not None else 'PENDING':v for k,v in watches},
            'indicator_scorecard':by_formula,'feature_scorecard':feature_scorecard,
            'latest_trade_explanations':items[-25:],
            'disclaimer':'Research/paper data only. Observed associations do not establish why market prices moved.'}
    out=Path(root)/'reports';out.mkdir(exist_ok=True)
    ml.atomic_json(out/'adaptive_report.json',report)
    return report


def telegram_digest(report):
    a=report['activation'];lines=['🧠 SCOUT ADAPTIVE VALIDATION — DAILY',
        f"Mode: {a['status']} | observed sessions {a['sessions']} / 14 minimum",
        f"Costed closed paper trades: {a['completed_costed_trades']} | holdout {a['holdout_trades']}",
        f"Holdout mean net: {a['holdout_mean_net_r']} R | win: {a['holdout_win_rate_pct']}%",
        f"Kronos direction: {a['kronos_direction_accuracy_pct']}% across {a['kronos_evaluated']} resolved forecasts",
        'Promotion checks: '+', '.join(k for k,v in a['checks'].items() if not v) if a['status']!='VALIDATED_PAPER_MODE' else 'Promotion checks passed (paper evidence only).']
    study = report.get('target_hit_study') or {}
    if study.get('status')=='collecting':
        lines.append(f"Target-hit study: collecting | {study['hits']} T1 hits / {study['samples']} costed outcomes across {study['sessions']} sessions")
    elif study.get('status')=='evaluated':
        lines.append(f"Target-hit study: {study['holdout']} later trades | Brier {study['brier']} vs baseline {study['baseline_brier']} (lower is better) | top quarter {study['top_quarter_mean_net_r']:+.2f}R vs all {study['all_holdout_mean_net_r']:+.2f}R net")
    elif study.get('status')=='unavailable':
        lines.append('Target-hit study unavailable: '+study['error'])
    paired = report.get('paired_forecasts') or {}
    if paired.get('n'):
        lines.append(f"Same {paired['n']} graded forecasts: Kronos {paired['kronos_direction_pct']}% vs drift {paired['drift_direction_pct']}% direction correct")
    if report['latest_trade_explanations']:
        lines.append('LATEST TRADE REVIEW (evidence, not causation):')
        for p in report['latest_trade_explanations'][-5:]:
            lines.append(f"{p['symbol']} {p['side']} {p['net_r']:+.2f}R | "+'; '.join(p['evidence_tags'][:3]))
    lines.append('Validated alerts minimum 5m RVOL: '+str(a['minimum_rvol_for_validated_alert'])+'x')
    if a.get('rvol_tuning'):
        lines.append('Holdout-supported RVOL threshold proposal: '+str(a['rvol_tuning']))
    lines.append('Prebreakout watch studies: '+str(report['watch_totals']))
    return '\n'.join(lines)
