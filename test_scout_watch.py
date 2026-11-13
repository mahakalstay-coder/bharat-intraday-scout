import sqlite3
from datetime import datetime,timedelta
from unittest.mock import Mock
from scout_research import watch_reason, send_research_watch
from scout_target_study import evaluate


def setup():
    p=dict(side='BUY',symbol='TEST',price=100,stop=99,target=102,score=90,tier='TIER-1',
           forecast=dict(model='Kronos-mini',direction_aligned=True), relative_volume=2.1,
           cumulative_relative_volume=1.08,market_context=dict(sector='Tech',peer_count=2,peer_relative_pct=1))
    return p, dict(sectors=[dict(sector='Tech',change_pct=1,advancing_pct=60,count=4)])


def test_single_failure_and_forecast_required():
    p,m=setup()
    assert 'Session RVOL' in watch_reason(p,m)
    m['sectors'][0]['advancing_pct']=40
    assert watch_reason(p,m) is None
    p['cumulative_relative_volume']=2
    assert 'advancing' in watch_reason(p,m)
    p['forecast']['direction_aligned']=False
    assert watch_reason(p,m) is None


def test_dedup_restart_ranking_and_failed_send(tmp_path):
    now=datetime(2026,11,13,10)
    p,m=setup()
    choices=[(p,'volume'),(dict(p,score=97,symbol='BEST'),'volume')]
    path=tmp_path/'state.db'
    with sqlite3.connect(path) as c:
        c.execute('CREATE TABLE app_state(key TEXT PRIMARY KEY,value TEXT)')
        assert not send_research_watch(c,choices,now,lambda x:False)
        sender=Mock(return_value=True)
        assert send_research_watch(c,choices,now,sender)
        assert 'BEST' in sender.call_args.args[0]
    with sqlite3.connect(path) as c:
        assert not send_research_watch(c,choices,now,sender)
        assert send_research_watch(c,choices,now+timedelta(days=1),sender)


def test_target_labels_and_chronological_model():
    p,_=setup(); rows=[];payloads={}
    for i in range(80):
        day=f'2026-10-{1+i//4:02d}'
        rows.append(dict(id=i,result=('T1','STOP','EOD_WIN','NO_DATA')[i%4],
                         cost_source='entry snapshot',day=day,net_r=[1.8,-1.2,.2,None][i%4]))
        payloads[i]=dict(p,score=80+i%9)
    r=evaluate(rows,payloads)
    assert r['status']=='evaluated' and r['holdout_from']=='2026-10-15'
    assert (r['hits'],r['samples'],r['train'],r['holdout'])==(20,60,42,18)
    assert 0<=r['brier']<=1
