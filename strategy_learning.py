"""Forward-only challenger settings. Baseline strategy decisions are never changed.
Propose from past sessions, freeze a policy, measure on NEW sessions, promote only
its paper companion; withdraw it when later evidence deteriorates.
"""
import json
import math
from collections import defaultdict
from statistics import mean, stdev


def init(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS four_learning_policies(
      id INTEGER PRIMARY KEY, header TEXT, regime TEXT, side TEXT, created_day TEXT,
      rule TEXT, status TEXT, evidence TEXT, UNIQUE(header,regime,side,created_day))''')
    conn.execute('''CREATE TABLE IF NOT EXISTS four_policy_audit(
      id INTEGER PRIMARY KEY, at TEXT, policy_id INTEGER, event TEXT, payload TEXT)''')


def accepts(rule, payload):
    p=payload['candidate']; context=payload.get('context',{})
    if rule=='rvol_1.8': return float(p.get('relative_volume',0))>=1.8
    if rule=='rvol_2.2': return float(p.get('relative_volume',0))>=2.2
    if rule=='peer_aligned':
        value=context.get('peers',{}).get('stock_vs_peers_pp')
        return value is not None and (value>=0 if p['side']=='BUY' else value<=0)
    if rule=='intraday_aligned':
        return context.get('intraday_regime')==('UP' if p['side']=='BUY' else 'DOWN')
    raise ValueError('Unknown challenger rule')


def annotate(conn, header, side, regime, day, payload):
    row=conn.execute('''SELECT * FROM four_learning_policies WHERE header=? AND side=? AND regime=?
       AND created_day<? AND status IN ('FORWARD_PAPER','VALIDATED_PAPER') ORDER BY id DESC LIMIT 1''',
       (header,side,regime,day)).fetchone()
    if not row:return {}
    return dict(policy_id=row['id'], rule=row['rule'], status=row['status'], accepted=accepts(row['rule'],payload))


def _audit(conn, now, pid, event, data):
    conn.execute('INSERT INTO four_policy_audit(at,policy_id,event,payload) VALUES(?,?,?,?)',
                 (now.isoformat(),pid,event,json.dumps(data)))


def refresh(conn, now):
    init(conn)
    rows=[dict(r) for r in conn.execute("SELECT * FROM four_header_calls WHERE status='CLOSED' AND ambiguity=0 AND approved=1 ORDER BY day,id")]
    # No same-day learning updates can leak into today's entry decisions.
    rows=[r for r in rows if r['day']<now.date().isoformat()]
    groups=defaultdict(list)
    for r in rows: groups[(r['strategy_header'],r['regime'],r['side'])].append(r)
    for (header,regime,side), group in groups.items():
        previous=conn.execute('SELECT * FROM four_learning_policies WHERE header=? AND regime=? AND side=? ORDER BY id DESC LIMIT 1',
                              (header,regime,side)).fetchone()
        if previous:
            if previous['status'] in ('FORWARD_PAPER','VALIDATED_PAPER'):continue
            # A failed generation does not stop learning forever. Require an
            # entirely new training window before proposing its successor.
            evidence=json.loads(previous['evidence'])
            after=evidence.get('finalized_day',previous['created_day'])
            group=[r for r in group if r['day']>after]
        days=sorted({r['day'] for r in group})
        if regime=='UNKNOWN' or len(days)<20 or len(group)<60: continue
        scored=[]
        for rule in ('rvol_1.8','rvol_2.2','peer_aligned','intraday_aligned'):
            selected=[r for r in group if accepts(rule,json.loads(r['payload']))]
            if len(selected)<30 or len(selected)==len(group):continue
            delta=defaultdict(float)
            selected_ids={r['id'] for r in selected}
            for r in group:delta[r['day']]+=0 if r['id'] in selected_ids else -r['net_r']
            values=list(delta.values())
            margin=3*stdev(values)/math.sqrt(len(values)) if len(values)>1 else math.inf
            # Includes opportunity cost of skipped winners, with four tested rules.
            if mean(values)>margin and mean(r['net_r'] for r in selected)>0:
                scored.append((mean(values),rule,len(selected)))
        if scored:
            lift,rule,n=max(scored)
            evidence=dict(train_sessions=len(days),train_n=len(group),selected_n=n,train_daily_lift_r=lift,
                          note='Discovered on history; requires entirely new forward sessions; no baseline change')
            cur=conn.execute('''INSERT INTO four_learning_policies(header,regime,side,created_day,rule,status,evidence)
                  VALUES(?,?,?,?,?,'FORWARD_PAPER',?)''',
                  (header,regime,side,now.date().isoformat(),rule,json.dumps(evidence)))
            _audit(conn,now,cur.lastrowid,'PROPOSED',evidence)
    for policy in conn.execute("SELECT * FROM four_learning_policies WHERE status IN ('FORWARD_PAPER','VALIDATED_PAPER')").fetchall():
        forward=[]
        for r in rows:
            payload=json.loads(r['payload']); companion=payload.get('companion',{})
            if companion.get('policy_id')==policy['id'] and r['day']>policy['created_day']:forward.append((r,companion['accepted']))
        daily=defaultdict(float)
        for r,accepted in forward: daily[r['day']]+=0 if accepted else -r['net_r']
        selected=[r for r,a in forward if a]
        values=list(daily.values())
        evidence=json.loads(policy['evidence'])
        evidence.update(forward_sessions=len(daily),forward_opportunities=len(forward),forward_selected=len(selected),
                        forward_daily_lift_r=mean(values) if values else None,
                        forward_selected_mean_net_r=mean(r['net_r'] for r in selected) if selected else None)
        status=policy['status'];event=None
        if status=='FORWARD_PAPER' and len(daily)>=15 and len(forward)>=45 and len(selected)>=20:
            lcb=mean(values)-3*stdev(values)/math.sqrt(len(values))
            evidence['forward_daily_lift_lcb']=lcb
            status='VALIDATED_PAPER' if lcb>0 and mean(r['net_r'] for r in selected)>0 else 'REJECTED'
            if status=='VALIDATED_PAPER':evidence['validated_day']=now.date().isoformat()
            event=status
        elif status=='VALIDATED_PAPER':
            later_days=sorted({r['day'] for r,_ in forward if r['day']>evidence.get('validated_day','9999')})[-5:]
            later=[r for r,a in forward if a and r['day'] in later_days]
            if len(later_days)>=5 and len(later)>=10 and sum(r['net_r'] for r in later)<0 and sum(daily[d] for d in later_days)<0:
                status='ROLLED_BACK';event=status
        conn.execute('UPDATE four_learning_policies SET status=?,evidence=? WHERE id=?',(status,json.dumps(evidence),policy['id']))
        if event:
            if status in ('REJECTED','ROLLED_BACK'):
                evidence['finalized_day']=now.date().isoformat()
                conn.execute('UPDATE four_learning_policies SET evidence=? WHERE id=?',(json.dumps(evidence),policy['id']))
            _audit(conn,now,policy['id'],event,evidence)
    conn.commit()
    return [dict(r) for r in conn.execute('SELECT * FROM four_learning_policies')]
