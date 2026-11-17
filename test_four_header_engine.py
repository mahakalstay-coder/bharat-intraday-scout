import hashlib
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime,timedelta
from pathlib import Path
from unittest.mock import Mock,patch
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import four_header_engine as e
import learning_context as context
import strategy_learning as learning

TZ=ZoneInfo('Asia/Kolkata')
NOW=datetime(2026,11,17,10,0,12,tzinfo=TZ)


def candidate(symbol='X'):
    p=dict(candle_time='2026-11-17T09:55:00+05:30',relative_volume=2.5,day_move_pct=1.,
           pro=dict(confluence=6,conflict_count=0,indicators=dict(atr14_pct=.5),anatomy={}))
    return dict(symbol=symbol,side='BUY',day='2026-11-17',time='09:55',price=100.,relative_volume=2.5,payload=p)


def result(header='EARLIER_SETUP'):
    t,s,tr,trig=e.CONFIGS[header]
    return dict(strategy_header=header,target_pct=t,stop_pct=s,trailing_stop=tr,
                profit_lock_trig_pct=trig,conviction=.8,setup_desc='fixture',approved=True)


def bars(values,start='2026-11-17 10:05'):
    d=pd.DataFrame(values,columns=['open','high','low','close'],index=pd.date_range(start,periods=len(values),freq='5min',tz=TZ))
    d['volume']=1000.
    return d


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.db=self.root/'state.sqlite'
        self.c=sqlite3.connect(self.db);self.c.row_factory=sqlite3.Row
        self.c.execute('CREATE TABLE app_state(key TEXT PRIMARY KEY,value TEXT)');e.init(self.c);e.configure(self.db)
    def tearDown(self):self.c.close();self.tmp.cleanup()
    def insert(self,header='EARLIER_SETUP',kind='PAPER_ALERT',now=NOW,symbol='X'):
        e._insert(self.c,candidate(symbol),result(header),dict(intraday_regime='DOWN',peers={}),now,kind,True)
        self.c.commit()
        return self.c.execute('SELECT * FROM four_header_calls ORDER BY id DESC LIMIT 1').fetchone()
    def test_original_strategy_file_is_byte_for_byte_preserved(self):
        self.assertEqual(hashlib.sha256(Path(e.rules.__file__).read_bytes()).hexdigest(),
                         '3158467038110023cd2d1ea912b6374d56c14d8f3d7ce8f2488fe578975b52f5')
    def test_policy_forward_validation_and_later_rollback(self):
        self.c.execute("INSERT INTO four_learning_policies VALUES(1,'EARLIER_SETUP','DOWN','BUY','2026-11-17','rvol_2.2','FORWARD_PAPER','{}')")
        def add_day(day, deteriorated=False):
            for i in range(3):
                cand=candidate(str(i));cand['relative_volume']=3. if i<2 else 1.
                moment=NOW+timedelta(days=day)
                e._insert(self.c,cand,result(),dict(intraday_regime='DOWN',peers={}),moment,'APPROVED_BACKGROUND',True)
                net=(-1 if i<2 else 1) if deteriorated else (1 if i<2 else -1)
                self.c.execute("UPDATE four_header_calls SET status='CLOSED',net_r=? WHERE day=? AND symbol=?",(net,str(moment.date()),str(i)))
            self.c.commit()
        for day in range(1,16):add_day(day)
        self.assertEqual(learning.refresh(self.c,NOW+timedelta(days=16))[0]['status'],'VALIDATED_PAPER')
        for day in range(17,22):add_day(day,True)
        self.assertEqual(learning.refresh(self.c,NOW+timedelta(days=22))[0]['status'],'ROLLED_BACK')
    def test_no_history_contamination_and_dedup(self):
        self.insert();self.insert()
        self.assertEqual(self.c.execute('SELECT count(*) FROM four_header_calls').fetchone()[0],1)
        self.assertFalse(self.c.execute("SELECT name FROM sqlite_master WHERE name='signals'").fetchone())
    def test_failure_is_not_sent_and_retry_once(self):
        self.insert();send=Mock(return_value=False)
        self.assertEqual(e.deliver(self.c,NOW,send),0)
        self.assertEqual(self.c.execute('SELECT delivery FROM four_header_calls').fetchone()[0],'PENDING')
        send.return_value=True
        self.assertEqual(e.deliver(self.c,NOW,send),1)
        self.assertEqual(e.deliver(self.c,NOW,send),0)
        self.assertTrue(send.call_args.args[0].startswith('[EARLIER_SETUP]'))
    def test_background_and_stale_never_delivered(self):
        self.insert(kind='REJECTED_STUDY');self.insert(symbol='Y')
        send=Mock(return_value=True)
        self.assertEqual(e.deliver(self.c,NOW+timedelta(minutes=20),send),0);send.assert_not_called()
    def test_candle_before_alert_cannot_win(self):
        row=self.insert()
        d=bars([(100,110,90,100)],start='2026-11-17 10:00')
        self.assertIsNone(e.grade(row,d,NOW+timedelta(minutes=10)))
    def test_target_and_stop_same_bar_stop_first(self):
        row=self.insert();d=bars([(100,103,98,101)])
        out=e.grade(row,d,NOW+timedelta(minutes=10))
        self.assertEqual(out['outcome'],'STOP_LOSS');self.assertEqual(out['ambiguity'],1)
        self.assertAlmostEqual(out['net_r'],-1.2)
    def test_sell_gap_stop_uses_worse_open(self):
        row=dict(self.insert());row.update(side='SELL',stop=101,target=98)
        out=e.grade(row,bars([(102,103,100,102)]),NOW+timedelta(minutes=10))
        self.assertAlmostEqual(out['gross_r'],-2.)
    def test_trail_cannot_turn_same_bar_stop_into_profit(self):
        row=self.insert('90PCT_SETUP_V9')
        out=e.grade(row,bars([(100,100.5,98,99)]),NOW+timedelta(minutes=10))
        self.assertEqual(out['outcome'],'STOP_LOSS');self.assertLess(out['net_r'],0)
    def test_trail_applies_next_bar_and_ambiguity_flagged(self):
        row=self.insert('90PCT_SETUP_V9')
        out=e.grade(row,bars([(100,100.5,99.9,100.4),(100.4,100.5,100.2,100.3)]),NOW+timedelta(minutes=15))
        self.assertEqual(out['outcome'],'TRAIL_PROFIT_LOCK');self.assertEqual(out['ambiguity'],1)
    def test_missing_intermediate_bar_is_pending(self):
        row=self.insert();d=bars([(100,100.5,99.9,100)]*3);d=d.drop(d.index[1]);d.loc[d.index[-1],'high']=103
        self.assertIsNone(e.grade(row,d,NOW+timedelta(minutes=20)))
    def test_online_learning_exactly_once_and_fixed_weights_unchanged(self):
        row=self.insert('LEARNING_SETUP');before=e.rules.LOCKED_WEIGHTS_V9.copy()
        d=bars([(100,102,99.9,101.5)])
        e.resolve(self.c,{'X':d},NOW+timedelta(minutes=10),self.db)
        first=self.c.execute("SELECT value FROM app_state WHERE key='learning_weights_30d'").fetchone()[0]
        e.resolve(self.c,{'X':d},NOW+timedelta(minutes=10),self.db)
        self.assertEqual(first,self.c.execute("SELECT value FROM app_state WHERE key='learning_weights_30d'").fetchone()[0])
        self.assertEqual(self.c.execute('SELECT count(*) FROM four_weight_audit').fetchone()[0],1)
        np.testing.assert_array_equal(before,e.rules.LOCKED_WEIGHTS_V9)
    def test_no_data_not_loss_and_zero_samples_not_zero_accuracy(self):
        self.insert();e.resolve(self.c,{},NOW+timedelta(days=10),self.db)
        rep=e.report(self.c,self.root,NOW+timedelta(days=10))
        self.assertEqual(rep['headers']['EARLIER_SETUP']['paper_alerts']['no_data'],1)
        self.assertIsNone(rep['headers']['EARLIER_SETUP']['paper_alerts']['win_rate_pct'])
    def test_all_headers_independent_of_legacy_quota(self):
        d=bars([(100,101,99,100)]*61,start='2026-11-17 04:55')
        with patch.object(e.rules,'build_pro_candidate_payload',return_value=candidate()),patch.dict(e.EVALUATORS,{h:Mock(return_value=result(h)) for h in e.HEADERS},clear=True):
            out=e.collect(self.c,{'X':d},[dict(symbol='X')],{},dict(intraday_regime='DOWN',stocks={}),NOW,self.db)
        self.assertEqual(set(out['headers']),set(e.HEADERS))
        self.assertEqual(self.c.execute("SELECT count(*) FROM four_header_calls WHERE kind='PAPER_ALERT'").fetchone()[0],4)
    def test_daily_quota_and_thirty_minute_spacing(self):
        d=bars([(100,101,99,100)]*100,start='2026-11-17 04:55')
        with patch.object(e.rules,'build_pro_candidate_payload',side_effect=lambda *args:candidate()),patch.dict(e.EVALUATORS,{'EARLIER_SETUP':Mock(return_value=result())},clear=True):
            for minute in [0,5,30,60]:
                # Candidate's fixed source candle deliberately dedups repeated snapshots.
                e.collect(self.c,{'X':d},[dict(symbol='X')],{},dict(intraday_regime='DOWN',stocks={}),NOW+timedelta(minutes=minute),self.db)
        self.assertLessEqual(self.c.execute("SELECT count(*) FROM four_header_calls WHERE kind='PAPER_ALERT'").fetchone()[0],2)
    def test_no_policy_changes_fixed_evaluator(self):
        self.c.execute("INSERT INTO four_learning_policies VALUES(1,'EARLIER_SETUP','DOWN','BUY','2026-11-16','rvol_2.2','VALIDATED_PAPER','{}')")
        c=candidate();before=e.rules.evaluate_earlier_setup(c)
        self.insert();self.assertEqual(e.rules.evaluate_earlier_setup(c),before)
    def test_forward_promotion_uses_only_post_proposal_frozen_decisions(self):
        self.c.execute("INSERT INTO four_learning_policies VALUES(1,'EARLIER_SETUP','DOWN','BUY','2026-11-17','rvol_2.2','FORWARD_PAPER','{}')")
        # Historical good results do not count as forward validation.
        row=self.insert();payload=json.loads(row['payload']);payload['companion']={'policy_id':1,'accepted':False}
        self.c.execute("UPDATE four_header_calls SET status='CLOSED',net_r=-1,payload=?",(json.dumps(payload),));self.c.commit()
        policies=learning.refresh(self.c,NOW+timedelta(days=1))
        self.assertEqual(policies[0]['status'],'FORWARD_PAPER')
        self.assertEqual(json.loads(policies[0]['evidence'])['forward_opportunities'],0)


class ContextTests(unittest.TestCase):
    def test_intraday_recovery_separate_from_negative_day_return(self):
        prior=bars([(110,110.1,109.9,110)],start='2026-11-16 15:25')
        today=bars([(100,101,99.9,100.5),(100.5,101.2,100.4,101),(101,102,100.9,101.8),(101.8,103,101.7,102.8)],start='2026-11-17 09:15')
        out=context.measure(pd.concat([prior,today]),datetime(2026,11,17,9,35,12,tzinfo=TZ))
        self.assertLess(out['change_pct'],0);self.assertEqual(out['intraday_raw'],'UP')
    def test_future_bars_do_not_change_context(self):
        d=bars([(100+i,101+i,99+i,100.5+i) for i in range(8)],start='2026-11-17 09:15')
        now=datetime(2026,11,17,9,35,12,tzinfo=TZ)
        a=context.measure(d,now);d.iloc[-1]=[1,10000,1,5000,100]
        self.assertEqual(a,context.measure(d,now))
    def test_stale_and_no_volume_are_unknown(self):
        d=bars([(100,101,99,100)]*4,start='2026-11-17 09:15');d['volume']=0
        self.assertEqual(context.measure(d,NOW)['status'],'UNKNOWN')
    def test_official_membership_fetch_failure_retains_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            now=NOW;data=context.refresh_memberships(tmp,now,fetch=lambda u:'Symbol,Industry\nA,Bank\nB,Bank\nC,Bank',batch=1)
            self.assertIn('Nifty Bank',data)
            old=data['Nifty Bank']['asof']
            later=now+timedelta(days=8)
            data=context.refresh_memberships(tmp,later,fetch=Mock(side_effect=RuntimeError('offline')),batch=12)
            self.assertEqual(data['Nifty Bank']['asof'],old)
    def test_regime_persistence_and_peer_self_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=sqlite3.connect(':memory:');c.execute('CREATE TABLE app_state(key TEXT PRIMARY KEY,value TEXT)')
            prior=bars([(110,110.1,109.9,110)],start='2026-11-16 15:25')
            d=bars([(100+i,101+i,99.9+i,100.5+i) for i in range(6)],start='2026-11-17 09:15')
            f=pd.concat([prior,d]);m={'stocks':{s:{'industry':'Bank'} for s in ['A','B','C']}}
            start=datetime(2026,11,17,9,35,12,tzinfo=TZ)
            one=context.snapshot(tmp,c,dict(A=f,B=f,C=f),m,start,lambda syms:{'^NSEI':f})
            two=context.snapshot(tmp,c,dict(A=f,B=f,C=f),m,start,lambda syms:{'^NSEI':f})
            self.assertEqual(one['intraday_regime'],'UNKNOWN');self.assertEqual(two['intraday_regime'],'UNKNOWN')
            three=context.snapshot(tmp,c,dict(A=f,B=f,C=f),m,start+timedelta(minutes=5),lambda syms:{'^NSEI':f})
            self.assertEqual(three['intraday_regime'],'UP')
            self.assertEqual(three['stocks']['A']['peer_count'],2);self.assertNotIn('A',three['stocks']['A']['peer_symbols'])


if __name__=='__main__':unittest.main()
